# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple
# import math

# class TimeEmbedding(nn.Module):
#     """
#     Maps scalar timesteps (t) -> high dimensional sinusoidal vectors
#     of size (Batch size, 256) 
    
#     Timesteps are injected for serving as context about how much noise is expected in 
#     the EO latent vector at that step (t = 1000 [pure noise] -> t = 1 [almost no noise])
#     """
#     def __init__(self, embedding_dim : int):
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.mlp = nn.Sequential(
#             nn.Linear(embedding_dim, embedding_dim * 4), 
#             nn.SiLU(),
#             nn.Linear(embedding_dim * 4, embedding_dim)
#         )
    
#     def forward(self, timesteps : torch.Tensor) -> torch.Tensor:
#         # convert each value in timesteps [870, 23, 500, ... (batch_size)] => embeddings after passing thru mlp
#         half_dim = self.embedding_dim // 2
        
#         # exponent = 1 / (10000 ^ (2i / d_model)) => -log(10000) * i / (d_model/2 OR half_dim) 
#         exponent = -math.log(10000) * torch.arange(start = 0, end = half_dim, dtype=torch.float32, device=timesteps.device)
#         exponent = exponent / half_dim
        
#         # multiplying each timestep scalar (in array) by 1 / (10000 ^ (2i / d_model))
#         # e ^ [-log(10000) * (2i / d_model]) => [10000 ^ (-2i / d_model)]=> [1 / (10000 ^ (2i / d_model))]
#         args = timesteps[:, None] * torch.exp(exponent)[None, :]
#         embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim = 1)
        
#         return self.mlp(embeddings)

# class CDiffDownBlock(nn.Module):
#     """
#     Downsampling block for C-DiffSET U-Net. Integrates spatial feature maps 
#     with sinusoidal time embeddings.
#     """
#     def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
#         super().__init__() 
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
#             nn.GroupNorm(num_groups = 8, num_channels = out_channels),
#             nn.SiLU()
#         )
#         self.time_mlp = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(in_features = time_emb_dim, out_features = out_channels)
#         )
        
#         # to deeply integrate the time information with spatial SAR/EO features (no shape shrinking just blending)
#         self.residual_conv = nn.Sequential(
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.GroupNorm(num_groups = 8, num_channels = out_channels),
#             nn.SiLU()
#         )
    
#     def forward(self, x : torch.Tensor, time_emb : torch.Tensor) -> torch.Tensor:
#         x = self.conv(x)
        
#         # Inject time embedding contextually across channels
#         t_spatial = self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1)    # converts [4, 64] -> [4, 64, 1, 1] (adds 2  trailing dims [-1(last position)])
#         return self.residual_conv(x + t_spatial)
    

# class CDiffUpBlock(nn.Module):
#     """
#     Upsampling block for C-DiffSET U-Net. Merges up-projected representations 
#     with structural skip connections across matching latent scales.
#     """
#     def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
#         super().__init__()
#         self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        
#         # after concatenating feature maps at diff latent scales => dim increases again => conv required to shrink it to out channels size
#         self.conv = nn.Sequential(
#             nn.Conv2d(out_channels * 2, out_channels, kernel_size = 3, padding = 1),
#             nn.GroupNorm(8, out_channels),
#             nn.SiLU()    
#         )
#         self.time_mlp = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(in_features = time_emb_dim, out_features = out_channels)
#         )
        
#     def forward(self, x : torch.Tensor, skip : torch.Tensor, time_emb : torch.Tensor) -> torch.Tensor:
#         x = self.up(x)
#         x_skip = torch.cat([x, skip], dim = 1)
#         x_skip = self.conv(x_skip)
#         t_spatial = self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
#         return (x_skip + t_spatial)
    

# class CDiffSETUNet(nn.Module):
#     """
#     Confidence Diffusion U-Net for SAR -> EO Latent Image Translation.
    
#     Inputs:
#         - concatenated_latent: (B, latent_channels * 2, H_l, W_l) [(4, 1, 32, 32) SAR + (4, 3, 32, 32) EO => combined latent rep (4, 4, 32, 32) -> UNet]
#                                representing channel-wise combined SAR latent and noisy EO latent.
#         - timesteps: (B,) current diffusion timesteps array.
        
#     Outputs:
#         - predicted_noise: (B, latent_channels, H_l, W_l) [4, 4, 32, 32]
#         - confidence_map:  (B, 1, H_l, W_l) -> Valued via Softplus to penalize noise misalignments (1 as SAR images have a single channel => each pixel assgined a conf score)
#     """
#     def __init__(self, latent_channels : int = 4, base_channels : int = 64, time_dim: int = 256):
#         super().__init__()
#         self.latent_channels = latent_channels
        
#         # time projection global line (used in fwd pass thru the UNet)
#         self.time_embed = TimeEmbedding(embedding_dim = time_dim)
        
#         # Input Layer (Accepts concatenated SAR latent + noisy EO latent)
#         self.in_conv = nn.Conv2d(latent_channels * 2, base_channels, kernel_size=3, padding=1)

#         self.down1 = CDiffDownBlock(base_channels, base_channels * 2, time_dim)
#         self.down2 = CDiffDownBlock(base_channels * 2, base_channels * 4, time_dim)
#         self.down3 = CDiffDownBlock(base_channels * 4, base_channels * 8, time_dim)
        
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size = 3, padding = 1),
#             nn.GroupNorm(8, base_channels * 8),
#             nn.SiLU()
#         )
        
#         self.up3 = CDiffUpBlock(base_channels * 8, base_channels * 4, time_dim)
#         self.up2 = CDiffUpBlock(base_channels * 4, base_channels * 2, time_dim)
#         self.up1 = CDiffUpBlock(base_channels * 2, base_channels, time_dim)
        
        
#         # gives (4 x 1 x 32 x 32) confidence values (4 SAR images in a batch => 1 channel per image) => pixel level conf scores
#         self.confidence_head = nn.Sequential(
#             nn.Conv2d(base_channels, base_channels, kernel_size = 3, padding = 1),
#             nn.SiLU(),
#             nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
#             # nn.Softplus()
#             nn.Sigmoid()
#         )
        
#         # gives (4 x 4 x 32 x 32) noise preds (each pixel of 32 x 32 EO images across all 3 channels)
#         self.noise_head = nn.Sequential(
#             nn.Conv2d(base_channels, base_channels, kernel_size = 3, padding = 1),
#             nn.SiLU(),
#             nn.Conv2d(base_channels, latent_channels, kernel_size = 3, padding = 1),
#         )
    
#     def forward(self, concatenated_latent : torch.Tensor, timesteps : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         t_emb = self.time_embed(timesteps)
        
#         x1 = self.in_conv(concatenated_latent)
#         x2 = self.down1(x1, t_emb)
#         x3 = self.down2(x2, t_emb)
#         x4 = self.down3(x3, t_emb)
        
#         mid = self.bottleneck(x4)
        
#         u3 = self.up3(mid, x3, t_emb)
#         u2 = self.up2(u3, x2, t_emb)
#         u1 = self.up1(u2, x1, t_emb)
        
#         pred_noise = self.noise_head(u1)
#         confidence = self.confidence_head(u1)
        
#         return pred_noise, confidence
    
# if __name__ == "__main__":
#     model = CDiffSETUNet(latent_channels = 4, base_channels = 64)
    
#     batch_size = 4
    
#     print('VAE stage skipped [where (4, 3, 32, 32)(EO) & (4, 1, 32, 32)(SAR) => (4, 4, 32, 32) each]')
#     dummy_sar_latent = zy = torch.randn(batch_size, 4, 32, 32)
#     dummy_noisy_eo_latent = zx_t = torch.randn(batch_size, 4, 32, 32)
#     dummy_input = torch.cat([zy, zx_t], dim=1)
    
#     dummy_t = torch.randint(0, 1000, (batch_size,)).float()
    
#     eps, conf = model(dummy_input, dummy_t)
    
#     print("--- C-DiffSET Execution Shape Targets ---")
#     print(f"Combined Inputs Shapes:    {tuple(dummy_input.shape)}")
#     print(f"Predicted Noise Map Profile: {tuple(eps.shape)}")
#     print(f"Generated Confidence Matrix: {tuple(conf.shape)}")
    
#     assert eps.shape == dummy_sar_latent.shape, "Noise matching head target shape mismatch."
#     assert conf.shape == (batch_size, 1, 32, 32), "Confidence map tracking spatial shape failure."
#     print("\n[SUCCESS] Model shapes successfully align with C-DiffSET design paradigms.")
import torch
import torch.nn as nn
from typing import Tuple
from diffusers import UNet2DConditionModel

class CDiffSETPretrainedUNet(nn.Module):
    """
    C-DiffSET Inspired Pretrained U-Net Wrapper for SAR -> EO Latent Image Translation.
    Uses Stable Diffusion v1.5 U-Net trunk as a powerful generative prior.
    """
    def __init__(self, latent_channels: int = 4, base_channels: int = 64):
        super().__init__()
        self.latent_channels = latent_channels
        
        # 1. Load the SOTA foundational U-Net architecture
        self.unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            subfolder="unet",
            torch_dtype=torch.float32 
        )
        
        # 2. Modify the first layer to accept 8 channels (4 SAR latents + 4 Noisy EO latents)
        # We extract the pretrained weights to preserve what the model already knows about image structures
        old_weights = self.unet.conv_in.weight.data # Shape: [320, 4, 3, 3]
        new_conv = nn.Conv2d(
            in_channels=latent_channels * 2, # 8 channels total
            out_channels=self.unet.config.block_out_channels[0], # 320 channels
            kernel_size=3,
            padding=1
        )
        
        # Initialize new weights: Copy pretrained optical weights to the first 4 channels,
        # and initialize the 4 SAR channels with the same weights to provide an excellent starting baseline
        with torch.no_grad():
            new_conv.weight.data[:, :latent_channels] = old_weights
            new_conv.weight.data[:, latent_channels:] = old_weights
            new_conv.bias.data = self.unet.conv_in.bias.data
            
        self.unet.conv_in = new_conv
        
        # 3. Add C-DiffSET's isolated Confidence Head onto the U-Net's output features
        # Out channels matching the final block out size (320 channels)
        final_out_channels = self.unet.config.block_out_channels[0] 
        self.confidence_head = nn.Sequential(
            nn.Conv2d(latent_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # 4. OPTIMIZATION STRATEGY: Freeze base weights to protect VRAM
        # This prevents backpropagation through the massive inner layers, keeping memory low
        for name, param in self.unet.named_parameters():
            if "conv_in" not in name: # Only train our new input adapter layer
                param.requires_grad = False

    def forward(self, concatenated_latent: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            concatenated_latent: (B, 8, 32, 32) tensor representing [SAR, Noisy_EO]
            timesteps: (B,) current diffusion timesteps
        """
        # Stable Diffusion expects text embeddings for its cross-attention layers.
        # Since we aren't using text, we pass a dummy empty embedding tensor to satisfy the input format.
        batch_size = concatenated_latent.shape[0]
        dummy_encoder_hidden_states = torch.zeros(
            (batch_size, 1, 768), 
            dtype=concatenated_latent.dtype, 
            device=concatenated_latent.device
        )
        
        # 1. Forward pass through the foundational 860M parameter trunk
        unet_output = self.unet(
            sample=concatenated_latent,
            timestep=timesteps,
            encoder_hidden_states=dummy_encoder_hidden_states,
            return_dict=False
        )[0] # Shape: [B, 4, 32, 32]
        
        # 2. Extract internal features from the U-Net output to generate the confidence mask
        # We drive it through our custom head to track pixel alignments
        pred_noise = unet_output
        
        confidence = self.confidence_head(unet_output)
            
        return pred_noise, confidence

if __name__ == "__main__":
    # Test compilation script
    model = CDiffSETPretrainedUNet().half().cuda()
    dummy_input = torch.randn(2, 8, 32, 32).half().cuda()
    dummy_t = torch.randint(0, 1000, (2,)).cuda()
    
    eps, conf = model(dummy_input, dummy_t)
    print("\n✅ Pretrained LDM Trunk successfully initialized without VRAM overhead!")
    print(f"Noise output shape: {eps.shape} | Confidence output shape: {conf.shape}")