import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVF

def gaussian_blur(x, kernel_size = 5, sigma = 1.5):
    """Blurs out fine structure while preserving overall color distribution."""
    return TVF.gaussian_blur(x, kernel_size = [kernel_size, kernel_size], sigma = [sigma, sigma])

def reconstruct_image_x0(z_y_noisy, pred_noise, timesteps, noise_scheduler):
    """
    Algebraically invert the forward noising equation to get an estimate of
    the clean latent from the current noisy latent + predicted noise.
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
    alpha_bar_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)

    x0_pred = (z_y_noisy - ((1 - alpha_bar_t).sqrt()) * pred_noise)  / alpha_bar_t.sqrt()
    return x0_pred

def color_supervision_loss(z_y_noisy, pred_noise, z_y_clean, timesteps, noise_scheduler, kernel_size = 5, sigma = 3.0):
    x0_pred = reconstruct_image_x0(z_y_noisy, pred_noise, timesteps, noise_scheduler) 

    x0_pred_blur = gaussian_blur(x0_pred, kernel_size, sigma)
    x0_gt_blur = gaussian_blur(z_y_clean, kernel_size, sigma)

    return F.mse_loss(x0_pred_blur, x0_gt_blur)

def edge_preservation_loss(z_y_noisy, pred_noise, z_y_clean, timesteps, noise_scheduler):
    x0_pred = reconstruct_image_x0(z_y_noisy, pred_noise, timesteps, noise_scheduler)  # reuse from color loss

    pred_grad_x = x0_pred[:, :, :, :-1] - x0_pred[:, :, :, 1:]
    target_grad_x = z_y_clean[:, :, :, :-1] - z_y_clean[:, :, :, 1:]
    pred_grad_y = x0_pred[:, :, :-1, :] - x0_pred[:, :, 1:, :]
    target_grad_y = z_y_clean[:, :, :-1, :] - z_y_clean[:, :, 1:, :]

    return F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)