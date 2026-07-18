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

class ResidualBlock2D(nn.Module):
    """
    Refines features at the same spatial resolution while 
    deeply integrating time embeddings over multiple convolutions.
    """
    def __init__(self, channels : int, time_emb_dim : int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=time_emb_dim, out_features=channels)
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x : torch.Tensor, time_emb : torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)

        t_spatial = self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        x = x + t_spatial

        x = self.conv2(x)
        return residual + x

class CDiffDownBlock(nn.Module):
    """
    Downsampling block for C-DiffSET U-Net. Integrates spatial feature maps 
    with sinusoidal time embeddings.
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, num_res_blocks : int = 3):
        super().__init__() 
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups = 8, num_channels = out_channels),
            nn.SiLU()
        )
        
        # to deeply integrate the time information with spatial SAR/EO features (no shape shrinking just blending)
        self.res_blocks = nn.ModuleList([
            ResidualBlock2D(out_channels, time_emb_dim) for _ in range(num_res_blocks)
        ])
    
    def forward(self, x : torch.Tensor, time_emb : torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        for block in self.res_blocks:
            x = block(x, time_emb)
        return x
    

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
        self.res_blocks = nn.ModuleList([
            ResidualBlock2D(out_channels, time_emb_dim) for _ in range(num_res_blocks)
        ])
        
    def forward(self, x : torch.Tensor, skip : torch.Tensor, time_emb : torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x_skip = torch.cat([x, skip], dim = 1)
        x_skip = self.conv_blend(x_skip)

        for block in self.res_blocks:
            x_skip = block(x_skip, time_emb)
        return x_skip

class SelfAttention2D(nn.Module):
    """
    Standard multi-head self-attention over spatial positions, applied at
    low-resolution feature maps where the token count (H*W) is small.
    """
    def __init__(self, channels : int, num_heads : int):
        super().__init__()
        self.channels = channels 
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size = 1)    # skeleton of q, k, v
        self.proj = nn.Conv2d(channels, channels, kernel_size = 1)       # output proj layer

        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape  
        residual = x

        x = self.norm(x)
        qkv = self.qkv(x)                     # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim = 1)       # split into q, k, v

        head_dim = C // self.num_heads
        def reshape_heads(t):
            return t.view(B, self.num_heads, head_dim, H*W).transpose(2, 3)

        q, k, v = reshape_heads(q), reshape_heads(k), reshape_heads(v)
        
        attn = torch.softmax((q @ k.transpose(-2, -1)) / (head_dim ** 0.5), dim = -1)
        out = attn @ v     # (B, num_heads, H*W, head_dim)
        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        out = self.proj(out)

        return (residual + out)
    

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

        num_residual_blocks = 3
        
        # time projection global line (used in fwd pass thru the UNet)
        self.time_embed = TimeEmbedding(embedding_dim = time_dim)
        
        # Input Layer (Accepts concatenated SAR latent + noisy EO latent)
        self.in_conv = nn.Conv2d(latent_channels * 2, base_channels, kernel_size=3, padding=1)

        self.down1 = CDiffDownBlock(base_channels, base_channels * 2, time_dim, num_residual_blocks)
        self.down2 = CDiffDownBlock(base_channels * 2, base_channels * 4, time_dim, num_residual_blocks)
        self.down3 = CDiffDownBlock(base_channels * 4, base_channels * 8, time_dim, num_residual_blocks)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size = 3, padding = 1),
            nn.GroupNorm(8, base_channels * 8),
            nn.SiLU()
        )
        self.bottleneck_attn = SelfAttention2D(base_channels * 8, num_heads = 8)

        self.mid_attn = SelfAttention2D(base_channels * 8, num_heads=8)  # after down3, before bottleneck
        
        self.up3 = CDiffUpBlock(base_channels * 8, base_channels * 4, time_dim, num_residual_blocks)
        self.up2 = CDiffUpBlock(base_channels * 4, base_channels * 2, time_dim, num_residual_blocks)
        self.up1 = CDiffUpBlock(base_channels * 2, base_channels, time_dim, num_residual_blocks)
        
        
        # gives (4 x 1 x 32 x 32) confidence values (4 SAR images in a batch => 1 channel per image) => pixel level conf scores
        self.confidence_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size = 3, padding = 1),
            nn.SiLU(),
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
            # nn.Softplus()
            nn.Sigmoid()
        )
        
        # gives (4 x 4 x 32 x 32) noise preds (each pixel of 32 x 32 EO images across all 3 channels)
        self.noise_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size = 3, padding = 1),
            nn.SiLU(),
            nn.Conv2d(base_channels, latent_channels, kernel_size = 3, padding = 1),
        )
    
    def forward(self, concatenated_latent : torch.Tensor, timesteps : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t_emb = self.time_embed(timesteps)
        
        x1 = self.in_conv(concatenated_latent)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        x4 = self.down3(x3, t_emb)

        x4 = self.mid_attn(x4)         # new
        
        mid = self.bottleneck(x4)

        mid = self.bottleneck_attn(mid)    # new
        
        u3 = self.up3(mid, x3, t_emb)
        u2 = self.up2(u3, x2, t_emb)
        u1 = self.up1(u2, x1, t_emb)
        
        pred_noise = self.noise_head(u1)
        confidence = self.confidence_head(u1)
        
        return pred_noise, confidence
    
if __name__ == "__main__":
    model = CDiffSETUNet(latent_channels = 4, base_channels = 64)
    
    batch_size = 4
    
    print('VAE stage skipped [where (4, 3, 32, 32)(EO) & (4, 1, 32, 32)(SAR) => (4, 4, 32, 32) each]')
    dummy_sar_latent = zy = torch.randn(batch_size, 4, 32, 32)
    dummy_noisy_eo_latent = zx_t = torch.randn(batch_size, 4, 32, 32)
    dummy_input = torch.cat([zy, zx_t], dim=1)
    
    dummy_t = torch.randint(0, 1000, (batch_size,)).float()
    
    eps, conf = model(dummy_input, dummy_t)
    
    print("--- C-DiffSET Execution Shape Targets ---")
    print(f"Combined Inputs Shapes:    {tuple(dummy_input.shape)}")
    print(f"Predicted Noise Map Profile: {tuple(eps.shape)}")
    print(f"Generated Confidence Matrix: {tuple(conf.shape)}")
    
    assert eps.shape == dummy_sar_latent.shape, "Noise matching head target shape mismatch."
    assert conf.shape == (batch_size, 1, 32, 32), "Confidence map tracking spatial shape failure."
    print("\n[SUCCESS] Model shapes successfully align with C-DiffSET design paradigms.")
