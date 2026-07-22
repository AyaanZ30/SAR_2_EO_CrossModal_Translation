import torch
import torch.nn as nn
import torch.nn.functional as F

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