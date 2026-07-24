import numpy as np
import matplotlib.pyplot as plt
import os

def plot_performance(extreme_samples, output_dir, tier_name = "best"):
    """
    Plots a row matrix containing the SAR input, generated EO target, and Ground Truth EO.
    """
    num_samples = len(extreme_samples)
    if num_samples == 0:
        return
        
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 2 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)
        
    for idx, sample in enumerate(extreme_samples):
        # Denormalize images from [-1, 1] to [0, 1] for visual display
        sar = ((sample['sar'] + 1.0) / 2.0).permute(1, 2, 0).cpu().numpy()
        if sar.shape[-1] == 2 or sar.shape[-1] == 3:
            sar = sar[:, :, 0]                          # Extract the primary polarization channel (VV)
            
        pred = ((sample['pred'] + 1.0) / 2.0).permute(1, 2, 0).cpu().numpy()
        gt = ((sample['gt'] + 1.0) / 2.0).permute(1, 2, 0).cpu().numpy()
        
        axes[idx, 0].imshow(sar, cmap='gray')
        axes[idx, 0].set_title(f"SAR (L1: {sample['loss']:.3f})", fontsize=8)
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(np.clip(pred, 0, 1))
        axes[idx, 1].set_title("Generated EO", fontsize=8)
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(np.clip(gt, 0, 1))
        axes[idx, 2].set_title("Ground Truth", fontsize=8)
        axes[idx, 2].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"performance_{tier_name}.png"), dpi=150)
    plt.close()