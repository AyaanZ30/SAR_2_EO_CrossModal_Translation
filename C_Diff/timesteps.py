import torch
import torch.nn as nn 
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
