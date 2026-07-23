import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F

class CDiffDownBlock(nn.Module):
    """
    Downsampling block for C-DiffSET U-Net. Integrates spatial feature maps 
    with sinusoidal time embeddings.
    """
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, num_residual_blocks : int = 2):
        super().__init__() 
        self.downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU(),
                nn.Dropout2d(0.1),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.GroupNorm(8, out_ch),
            ) for _ in range(num_residual_blocks)
        ])
    
    def forward(self, x, time_emb) -> torch.Tensor:
        x = self.downsample(x)
        t_spatial = self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        x = x + t_spatial 
        for block in self.res_blocks:
            x = x + F.silu(block(x))
        return x

    
class CDiffUpBlock(nn.Module):
    """
    Upsampling block for C-DiffSET U-Net. Merges up-projected representations 
    with structural skip connections across matching latent scales.
    """
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, num_residual_blocks : int = 2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        
        # after concatenating feature maps at diff latent scales => dim increases again => conv required to shrink it to out channels size
        self.conv_blend = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size = 3, padding = 1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()    
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU(),
                nn.Dropout2d(0.1),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.GroupNorm(8, out_ch),
            ) for _ in range(num_residual_blocks)
        ])
        
    def forward(self, x : torch.Tensor, skip : torch.Tensor, time_emb : torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv_blend(torch.cat([x, skip], dim = 1))
        t_spatial = self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        x = x + t_spatial
        for block in self.res_blocks:
            x = x + F.silu(block(x))
        return x

class DilatedBottleneck(nn.Module):
    """
    Plain bottlenech conv <--> Parallel Dilated convolutons (for broader receptive field)
    Captures more global multi-scale context cheaply (without attention cost)
    """
    def __init__(self, ch):
        super().__init__()
        self.branch1 = nn.Conv2d(ch, ch // 4, kernel_size=3, padding=1, dilation=1)
        self.branch2 = nn.Conv2d(ch, ch // 4, kernel_size=3, padding=2, dilation=2)
        self.branch3 = nn.Conv2d(ch, ch // 4, kernel_size=3, padding=4, dilation=4)
        self.branch4 = nn.Conv2d(ch, ch // 4, kernel_size=3, padding=8, dilation=8)
        self.proj = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=1),
            nn.GroupNorm(8, ch),
            nn.SiLU()
        )

    def forward(self, x):
        out = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
        return x + self.proj(out)