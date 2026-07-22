import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .timesteps import TimeEmbedding
from .losses import SpatialGradientLoss
from .blocks import CDiffUpBlock, CDiffDownBlock

class SAREncoder(nn.Module):
    """
    Produces SAR feature maps at each resolution the main U-Net operates at,
    so conditioning isn't only available at the input — it's reinforced at every scale.
    """
    def __init__(self, latent_ch : int = 4, base_ch : int = 96):
        super().__init__()
        self.in_conv = nn.Conv2d(latent_ch, base_ch, kernel_size=3, padding=1)
        self.down1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch * 2),
            nn.SiLU()
        ) 
        self.down2 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch * 4),
            nn.SiLU()
        ) 
        self.down3 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 8, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch * 8),
            nn.SiLU()
        ) 

    def forward(self, z_sar):
        f0 = self.in_conv(z_sar)      # (B, base, 32, 32)
        f1 = self.down1(f0)           # (B, base*2, 16, 16)
        f2 = self.down2(f1)           # (B, base*4, 8, 8)
        f3 = self.down3(f2)           # (B, base*8, 4, 4)
        return f0, f1, f2, f3         # one feature map per U-Net resolution

class CDiffSETUNet(nn.Module):
    """
    Confidence Diffusion U-Net for SAR -> EO Latent Image Translation.
    
    Inputs:
        - concatenated_latent: (B, latent_ch * 2, H_l, W_l) [(4, 1, 32, 32) SAR + (4, 3, 32, 32) EO => combined latent rep (4, 4, 32, 32) -> UNet]
                               representing channel-wise combined SAR latent and noisy EO latent.
        - timesteps: (B,) current diffusion timesteps array.
        
    Outputs:
        - predicted_noise: (B, latent_ch, H_l, W_l) [4, 4, 32, 32]
        - confidence_map:  (B, 1, H_l, W_l) -> Valued via Softplus to penalize noise misalignments (1 as SAR images have a single channel => each pixel assgined a conf score)
    """
    def __init__(self, latent_ch : int = 4, base_ch : int = 96, time_dim: int = 256, num_res_blocks: int = 2):
        super().__init__()
        self.latent_ch = latent_ch        
        self.time_embed = TimeEmbedding(embedding_dim = time_dim)

        self.sar_encoder = SAREncoder(latent_ch, base_ch)   # seperate path for extracting features at diff sar image resolutions
        # self.in_conv = nn.Conv2d(latent_ch * 2, base_ch, kernel_size=3, padding=1)   # (Accepts concatenated SAR latent + noisy EO latent)
        self.in_conv = nn.Conv2d(latent_ch , base_ch, kernel_size=3, padding=1)   # (Accepts concatenated SAR latent + noisy EO latent)

        self.down1 = CDiffDownBlock(base_ch, base_ch * 2, time_dim, num_res_blocks)
        self.down2 = CDiffDownBlock(base_ch * 2, base_ch * 4, time_dim, num_res_blocks)
        self.down3 = CDiffDownBlock(base_ch * 4, base_ch * 8, time_dim, num_res_blocks)        
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 8, kernel_size = 3, padding = 1),
            nn.GroupNorm(8, base_ch * 8),
            nn.SiLU()
        )
        
        self.up3 = CDiffUpBlock(base_ch * 8, base_ch * 4, time_dim, num_res_blocks)
        self.up2 = CDiffUpBlock(base_ch * 4, base_ch * 2, time_dim, num_res_blocks)
        self.up1 = CDiffUpBlock(base_ch * 2, base_ch, time_dim, num_res_blocks)
        
        # gives (4 x 4 x 32 x 32) noise preds (each pixel of 32 x 32 EO images across all 3 channels)
        self.noise_head = nn.Conv2d(base_ch, latent_ch, kernel_size = 3, padding = 1)
        
        # gives (4 x 1 x 32 x 32) confidence values (4 SAR images in a batch => 1 channel per image) => pixel level conf scores
        self.confidence_head = nn.Sequential(
            nn.Conv2d(base_ch, 1, kernel_size = 3, padding = 1),
            nn.Sigmoid()
        )
    
    def forward(self, z_sar, z_y_noisy, timesteps) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs:
            eo_latent:  (B, latent_ch, H, W)  -> Noisy EO Representation
            sar_latent: (B, latent_ch, H, W)  -> Clear SAR Structural Conditioning Map
            timesteps:  (B,) 
        """
        t_emb = self.time_embed(timesteps)
        sar_f0, sar_f1, sar_f2, sar_f3 = self.sar_encoder(z_sar)
        
        x1 = self.in_conv(z_y_noisy)
        x1 = x1 + sar_f0
        
        x2 = self.down1(x1, t_emb)
        x2 = x2 + sar_f1

        x3 = self.down2(x2, t_emb)
        x3 = x3 + sar_f2

        x4 = self.down3(x3, t_emb)
        x4 = x4 + sar_f3

        mid = self.bottleneck(x4)
        
        # Main path reconstruction flow with deep scale skips
        u3 = self.up3(mid, x3, t_emb)                 # (B, 256, 8, 8)
        u2 = self.up2(u3, x2, t_emb)                 # (B, 128, 16, 16)
        u1 = self.up1(u2, x1, t_emb)                 # (B, 64, 32, 32)
        
        pred_noise = self.noise_head(u1)
        confidence = self.confidence_head(u1)
        
        return pred_noise, confidence
    