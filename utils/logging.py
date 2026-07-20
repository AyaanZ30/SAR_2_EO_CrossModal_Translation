import numpy as np 
import os

def log_epoch(epoch, total_epochs, elapsed, train_stats, val_result=None):
    val_str = f"{val_result['mean']:.4f}" if val_result else "Skipped"
    img_str = f"{val_result['image_l1']:.4f}" if val_result else "Skipped"

    print(f"Epoch {epoch:03d}/{total_epochs} | Train Loss: {train_stats['train_loss']:.4f} | "
          f"Val L1 Error: {val_str} | Time: {elapsed:.1f}s")
    print(f"  ├── Raw Residual Mean: {train_stats['raw_residual']:.4f}")
    print(f"  ├── Confidence Status   -> Mean: {train_stats['conf_mean']:.4f} | "
          f"Min: {train_stats['conf_min']:.4f} | Max: {train_stats['conf_max']:.4f}")
    print(f"  └── Eval Noise L1 (Avg): {val_str} | Image Pixel Space L1: {img_str}")

    if val_result:
        t_diag = ", ".join(f"t{t}:{v:.4f}" for t, v in sorted(val_result["per_t"].items()))
        print(f"        └──► Timestep Grid Loss: {t_diag}")