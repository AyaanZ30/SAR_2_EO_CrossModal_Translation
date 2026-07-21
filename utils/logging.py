import numpy as np 
import os

def log_epoch(epoch, total_epochs, elapsed, train_stats, val_result=None):
    val_str = f"{val_result['mean']:.4f}" if val_result else "Skipped"
    img_str = f"{val_result['image_l1']:.4f}" if val_result else "Skipped"

    print(f"Epoch {epoch:03d}/{total_epochs} | Train Loss: {train_stats['train_loss']:.4f} | "
          f"Val L1 Error: {val_str} | Image L1: {img_str} | Time: {elapsed:.1f}s")

    if val_result:
        t_diag = ", ".join(f"t{t}:{v:.4f}" for t, v in sorted(val_result["per_t"].items()))
        print(f"        └──► Timestep Grid Loss: {t_diag}")