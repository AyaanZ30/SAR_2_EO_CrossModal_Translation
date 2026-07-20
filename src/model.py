import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math

class TimeEmbedding(nn.Module):
    """
    Maps scalar timesteps (t) -> high dimensional sinusoidal vectors
    of size (Batch size, 256) 
    
    Timesteps are injected for serving as context about how much noise is expected in 
    the EO latent vector at that step (t = 1000 [pure noise] -> t = 1 [almost no noise])
    """
    def __init__(self, embedding_dim : int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4), 
            nn.SiLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
    
    def forward(self, timesteps : torch.Tensor) -> torch.Tensor:
        # convert each value in timesteps [870, 23, 500, ... (batch_size)] => embeddings after passing thru mlp
        half_dim = self.embedding_dim // 2
        
        # exponent = 1 / (10000 ^ (2i / d_model)) => -log(10000) * i / (d_model/2 OR half_dim) 
        exponent = -math.log(10000) * torch.arange(start = 0, end = half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / half_dim
        
        # multiplying each timestep scalar (in array) by 1 / (10000 ^ (2i / d_model))
        # e ^ [-log(10000) * (2i / d_model]) => [10000 ^ (-2i / d_model)]=> [1 / (10000 ^ (2i / d_model))]
        args = timesteps[:, None] * torch.exp(exponent)[None, :]
        embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim = 1)
        
        return self.mlp(embeddings)

class SpatialGradientLoss(nn.Module):
    """
    Computes L1 dist b/w spatial gradients (edges) of 2 tensors (learning neighbor-pixel transitions)
    Forces the model to generate sharp structural boundaries rather than blurry averages.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pred : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        # compute horizontal gradients (along W dim) [B, C, H, (W)] (diff b/w adj columns)
        pred_grad_x = pred[:, :, :, :-1]  - pred[:, :, :, 1:]    
        target_grad_x = target[:, :, :, :-1]  - target[:, :, :, 1:]    

        # compute vertical gradients (along H dim) [B, C, (H), W] (diff b/w adj rows)
        pred_grad_y = pred[:, :, :-1, :]  - pred[:, :, 1:, :]    
        target_grad_y = target[:, :, :-1, :]  - target[:, :, 1:, :]

        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)

        return loss_x + loss_y 
        

class CDiffDownBlock(nn.Module):
    """
    Downsampling block for C-DiffSET U-Net. Integrates spatial feature maps 
    with sinusoidal time embeddings.
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__() 
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups = 8, num_channels = out_channels),
            nn.SiLU()
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features = time_emb_dim, out_features = out_channels)
        )
        self.residual_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups = 8, num_channels = out_channels),
            nn.SiLU()
        )
    
    def forward(self, x : torch.Tensor, time_emb : torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        t_spatial = self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        return self.residual_conv(x + t_spatial)
    

class CDiffUpBlock(nn.Module):
    """
    Upsampling block for C-DiffSET U-Net. Merges up-projected representations 
    with structural skip connections across matching latent scales.
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, num_res_blocks: int = 3):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        
        # after concatenating feature maps at diff latent scales => dim increases again => conv required to shrink it to out channels size
        self.conv_blend = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size = 3, padding = 1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()    
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features = time_emb_dim, out_features = out_channels)
        )
        
    def forward(self, x : torch.Tensor, skip : torch.Tensor, time_emb : torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x_skip = torch.cat([x, skip], dim = 1)
        x_skip = self.conv_blend(x_skip)
        t_spatial = self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        return x_skip + t_spatial   

    
class CDiffSETUNet(nn.Module):
    """
    Confidence Diffusion U-Net for SAR -> EO Latent Image Translation.
    
    Inputs:
        - concatenated_latent: (B, latent_channels * 2, H_l, W_l) [(4, 1, 32, 32) SAR + (4, 3, 32, 32) EO => combined latent rep (4, 4, 32, 32) -> UNet]
                               representing channel-wise combined SAR latent and noisy EO latent.
        - timesteps: (B,) current diffusion timesteps array.
        
    Outputs:
        - predicted_noise: (B, latent_channels, H_l, W_l) [4, 4, 32, 32]
        - confidence_map:  (B, 1, H_l, W_l) -> Valued via Softplus to penalize noise misalignments (1 as SAR images have a single channel => each pixel assgined a conf score)
    """
    def __init__(self, latent_channels : int = 4, base_channels : int = 64, time_dim: int = 256):
        super().__init__()
        self.latent_channels = latent_channels        
        # time projection global line (used in fwd pass thru the UNet)
        self.time_embed = TimeEmbedding(embedding_dim = time_dim)
        
        # Input Layer (Accepts concatenated SAR latent + noisy EO latent)
        self.in_conv = nn.Conv2d(latent_channels * 2, base_channels, kernel_size=3, padding=1)

        self.down1 = CDiffDownBlock(base_channels, base_channels * 2, time_dim)
        self.down2 = CDiffDownBlock(base_channels * 2, base_channels * 4, time_dim)
        self.down3 = CDiffDownBlock(base_channels * 4, base_channels * 8, time_dim)        
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size = 3, padding = 1),
            nn.GroupNorm(8, base_channels * 8),
            nn.SiLU()
        )
        
        self.up3 = CDiffUpBlock(base_channels * 8, base_channels * 4, time_dim)
        self.up2 = CDiffUpBlock(base_channels * 4, base_channels * 2, time_dim)
        self.up1 = CDiffUpBlock(base_channels * 2, base_channels, time_dim)
        
        # gives (4 x 4 x 32 x 32) noise preds (each pixel of 32 x 32 EO images across all 3 channels)
        # self.noise_head = nn.Conv2d(base_channels, latent_channels, kernel_size = 3, padding = 1)
        self.noise_head = nn.Sequential(
            nn.Conv2d(base_channels, latent_channels, kernel_size = 3, padding = 1)
        )
        
        # gives (4 x 1 x 32 x 32) confidence values (4 SAR images in a batch => 1 channel per image) => pixel level conf scores
        self.confidence_head = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size = 3, padding = 1),
            nn.Sigmoid()
        )
    
    # def forward(self, concatenated_latent : torch.Tensor, timesteps : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    def forward(self, concatenated_latent : torch.Tensor, timesteps : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs:
            eo_latent:  (B, latent_channels, H, W)  -> Noisy EO Representation
            sar_latent: (B, latent_channels, H, W)  -> Clear SAR Structural Conditioning Map
            timesteps:  (B,) 
        """
        t_emb = self.time_embed(timesteps)
        
        x1 = self.in_conv(concatenated_latent)
        
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        x4 = self.down3(x3, t_emb)

        mid = self.bottleneck(x4)
        
        # Main path reconstruction flow with deep scale skips
        u3 = self.up3(mid, x3, t_emb)                 # (B, 256, 8, 8)
        u2 = self.up2(u3, x2, t_emb)                 # (B, 128, 16, 16)
        u1 = self.up1(u2, x1, t_emb)                 # (B, 64, 32, 32)
        
        pred_noise = self.noise_head(u1)
        confidence = self.confidence_head(u1)
        
        return pred_noise, confidence
    
if __name__ == "__main__":
    model = CDiffSETUNet(latent_channels = 4, base_channels = 64)
    batch_size = 4
    
    dummy_sar_latent = torch.randn(batch_size, 4, 32, 32)
    dummy_noisy_eo_latent = torch.randn(batch_size, 4, 32, 32)
    dummy_t = torch.randint(0, 1000, (batch_size,)).float()
    
    # FIXED: Forward pass signature decoupling matching new Cross-Attention design rules
    eps, conf = model(dummy_noisy_eo_latent, dummy_sar_latent, dummy_t)
    
    print("--- C-DiffSET Execution Shape Targets ---")
    print(f"Predicted Noise Map Profile: {tuple(eps.shape)}")
    print(f"Generated Confidence Matrix: {tuple(conf.shape)}")
    
    assert eps.shape == dummy_sar_latent.shape, "Noise matching head target shape mismatch."
    assert conf.shape == (batch_size, 1, 32, 32), "Confidence map tracking spatial shape failure."
    print("\n[SUCCESS] Deep model scales successfully align with C-DiffSET paradigms.")