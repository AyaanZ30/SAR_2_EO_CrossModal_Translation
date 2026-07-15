"""
Training script for SAR-to-EO pix2pix.
 
Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --resume   # resume from last full checkpoint
"""
import argparse
import csv
import os
import random
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler
 
from src.dataset import list_roi_ids, split_roi_ids, PrecomputedLatentDataset
from src.model import CDiffSETUNet

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def save_checkpoint(path, epoch, model, optimizer, scaler, history):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "history": history,
    }, path)

def load_checkpoint(path, model, optimizer, scaler, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No training checkpoint found at {path}")
    ckpt = torch.load(path, map_location = device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt["epoch"] + 1, ckpt["history"]

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
        sar = ((sample['sar'] + 1.0) / 2.0).squeeze(0).cpu().numpy()
        pred = ((sample['pred'] + 1.0) / 2.0).permute(1, 2, 0).cpu().numpy()
        gt = ((sample['gt'] + 1.0) / 2.0).permute(1, 2, 0).cpu().numpy()
        
        # Matplotlib handles (H, W, 3) automatically, but if you pass cmap='gray' it prefers (H, W)
        # Take just the first channel for pure grayscale plotting layout:
        if sar.shape[-1] == 3:
            sar = sar[:, :, 0]
        
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

def train_one_epoch(train_loader : DataLoader, val_loader : DataLoader):
    pass
    

def main(cfg_path, resume):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    
    set_seed(cfg["seed"])
    
    accelerator = Accelerator(mixed_precision = "fp16" if torch.cuda.is_available() else "no")
    device = accelerator.device
    
    ckpt_dir = cfg["train"]["checkpoint_dir"]
    log_dir = cfg["train"]["log_dir"]
    
    if accelerator.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
    full_ckpt_path = os.path.join(ckpt_dir, "cdiffset_state.pt")
    
    roi_ids = list_roi_ids(cfg["data"]["seasons_dir"])
    train_ids, val_ids = split_roi_ids(roi_ids, val_frac = cfg["data"]["val_frac"], seed = cfg["seed"])
    
    if accelerator.is_main_process:
        print(f"{len(roi_ids)} ROI scenes total -> {len(train_ids)} train / {len(val_ids)} val")
    
    train_ds = PrecomputedLatentDataset(cfg["data"]["seasons_dir"], train_ids, cfg["train"]["vae_output_dir"])
    val_ds = PrecomputedLatentDataset(cfg["data"]["seasons_dir"], val_ids, cfg["train"]["vae_output_dir"])
    
    if accelerator.is_main_process:
        print(f"Train latent pairs: {len(train_ds)}, Val latent pairs: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"], pin_memory=True)
    
    noise_scheduler = DDPMScheduler(num_train_timesteps = 1000)
    model = CDiffSETUNet(latent_channels = cfg["model"]["latent_channels"], base_channels = cfg["model"]["base_channels"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=1e-4)
    
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to("cpu")
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
        
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    start_epoch = 1
    history = []  
    
    if resume and os.path.exists(full_ckpt_path):
        # start_epoch, history = load_checkpoint(full_ckpt_path, model, optimizer, scaler, device)
        accelerator.wait_for_everyone()
        ckpt = torch.load(full_ckpt_path, map_location=device, weights_only=False)
        
        accelerator.unwrap_model(model).load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt["history"]

    total_epochs = cfg["train"]["epochs"]
    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.perf_counter()
        
        mean_val_l1 = float('nan')
        
        model.train()
        train_losses = []
 
        for z_x, z_y in train_loader:
            # z_x (sar latent vector) & z_y (eo latent vec) precomputed and directly loaded from memory
             
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (z_x.shape[0],), device = device).long()
            noise = torch.randn_like(z_y)
            
            z_y_noisy = noise_scheduler.add_noise(z_y, noise, timesteps)
            
            # combined input : Shape: (B, 4*2, 32, 32)
            u_net_input = torch.cat([z_x, z_y_noisy], dim = 1)
            
            optimizer.zero_grad()
            # Forward pass implicitly maps mixed-precision configurations
            pred_noise, confidence = model(u_net_input, timesteps) 
            
            noise_residual = torch.square(noise - pred_noise)
            loss_map = noise_residual * confidence - torch.log(confidence + 1e-6) #
            loss = loss_map.mean()
            
            accelerator.backward(loss)
            optimizer.step()
            
            train_losses.append(loss.item())
            
        if epoch % 10 == 0 or epoch == total_epochs:
            model.eval()
            all_val_records = []
            l1_metric = nn.L1Loss(reduction='none')
            
            with torch.no_grad():
                for z_x, z_y in val_loader:
                    # perform an abbreviated reverse denoising loop step for eval verification
                    z_t = torch.rand_like(z_x)    
                    eval_t = torch.full((z_x.shape[0],), 50, device = device).long()
                    
                    noisy_sar_latent_input = torch.concat([z_x, z_t], dim = 1)
                    pred_noise, _ = model(noisy_sar_latent_input, eval_t)
                    
                    # direct step reconstruction (removing pred noise from noisy sar => denoising step)
                    z_denoised = (z_t - pred_noise)
                    
                    # decoded_output = vae.decode(z_denoised / 0.18215).sample
                    
                    # Gather individual tensors back across process boundaries safely
                    gathered_z_x, gathered_z_y, gathered_pred = accelerator.gather_for_metrics((z_x, z_y, z_denoised))
                    
                    # decoded_sar = vae.decode(gathered_z_x / 0.18215).sample
                    # decoded_gt = vae.decode(gathered_z_y / 0.18215).sample
                    
                    # directly compute l1 pixel loss [netween generated eo (decoded o/p) and eo]
                    batch_l1 = l1_metric(gathered_pred, gathered_z_y).mean(dim = [1, 2, 3])
                
                    for idx in range(gathered_z_x.shape[0]):
                        all_val_records.append({
                            'loss': batch_l1[idx].item(),
                            'sar_lat': gathered_z_x[idx].cpu(),
                            'pred_lat': gathered_pred[idx].cpu(),
                            'gt_lat': gathered_z_y[idx].cpu()
                        })
                    
            
            if accelerator.is_main_process:
                all_val_records.sort(key = lambda item : item['loss'])
                best_5_samples = all_val_records[:5]    
                worst_5_samples = all_val_records[-5:]   
                 
                mean_val_l1 = np.mean([item['loss'] for item in all_val_records])
                
                def decode_records(records_list):
                    decoded_samples = []
                    for item in records_list:
                        # Add batch axis, push to device, decode to pixels, and strip batch axis
                        z_sar = item['sar_lat'].unsqueeze(0).to("cpu")
                        z_pred = item['pred_lat'].unsqueeze(0).to("cpu")
                        z_gt = item['gt_lat'].unsqueeze(0).to("cpu")
                        
                        decoded_samples.append({
                            'loss': item['loss'],
                            'sar': vae.decode(z_sar / 0.18215).sample.squeeze(0),
                            'pred': vae.decode(z_pred / 0.18215).sample.squeeze(0),
                            'gt': vae.decode(z_gt / 0.18215).sample.squeeze(0)
                        })
                    return decoded_samples

                best_5_samples = decode_records(best_5_samples)
                worst_5_samples = decode_records(worst_5_samples)
                
                plot_performance(best_5_samples, log_dir, tier_name=f"best_epoch_{epoch}")
                plot_performance(worst_5_samples, log_dir, tier_name=f"worst_epoch_{epoch}")                
            
        if accelerator.is_main_process:    
            elapsed_time = time.perf_counter() - epoch_start
            mean_train_loss = np.mean(train_losses)
            
            val_str = f"{mean_val_l1:.4f}" if not np.isnan(mean_val_l1) else "Skipped"
            print(f"Epoch {epoch:03d}/{total_epochs} | Train Loss: {mean_train_loss:.4f} | Val L1 Error: {val_str} | Time: {elapsed_time:.1f}s")
            
            history.append({"epoch": epoch, "train_loss": mean_train_loss, "val_l1": mean_val_l1})
            
            # save_checkpoint(full_ckpt_path, epoch, model, optimizer, scaler, history)
            torch.save({
                "epoch": epoch,
                "model_state": accelerator.unwrap_model(model).state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
            }, full_ckpt_path)
        
        accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        # --- persist raw loss values + plot, as required by the assignment ---
        csv_path = os.path.join(log_dir, "loss_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_l1"])
            writer.writeheader()
            writer.writerows(history)
    
        epochs = [h["epoch"] for h in history]
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, [h["train_loss"] for h in history], label="Train C-Diff Loss")
        plt.plot(epochs, [h["val_l1"] for h in history], label="Val L1 Reconstruction Pixel Loss")
        plt.xlabel("Epoch Count")
        plt.ylabel("Loss Index")
        plt.title("C-DiffSET Training Convergence Diagnostics Summary")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "loss_diagnostic_curve.png"))
        print(f"[PROCESS COMPLETED] Logs safely cataloged to {log_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint_dir/training_state.pt")
    args = parser.parse_args()
    main(args.config, args.resume)
    