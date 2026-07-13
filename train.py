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
 
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import yaml
from torch.utils.data import DataLoader
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from src.dataset import SAR2EODataset, list_roi_ids, split_roi_ids
from src.model import UNetGenerator, PatchGANDiscriminator

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_perceptual_loss_fn(device):
    from torchvision.models import VGG16_Weights, vgg16
    
    vgg = vgg16(weights = VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
    for p in vgg.parameters():
        p.requires_grad = False
    layer_ids = {8: "relu2_2", 15: "relu3_3"}
    
    def extract(x):
        # x is in [-1, 1] (3-channel); VGG expects ImageNet-normalized [0, 1] input.
        x = (x + 1.0) / 2.0
        mean = torch.tensor([0.485, 0.456, 0.406], device = x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device = x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        
        feats = []
        for i, layer in enumerate(vgg):
            x = layer(x)
            if i in layer_ids:
                feats.append(x)
            if i == max(layer_ids):
                break  
        return feats

    def loss_fn(fake, real):
        feats_fake, feats_real = extract(fake), extract(real)
        return sum(nn.functional.l1_loss(a, b) for a, b in zip(feats_fake, feats_real)) 

    return loss_fn

def save_checkpoint(path, epoch, G, D, opt_G, opt_D, history):
    torch.save({
        "epoch": epoch,
        "G_state": G.state_dict(),
        "D_state": D.state_dict(),
        "opt_G_state": opt_G.state_dict(),
        "opt_D_state": opt_D.state_dict(),
        "history": history,
    }, path)

def get_loaders(cfg, train_ds, val_ds):
    train_loader = DataLoader(
        train_ds, batch_size = cfg["train"]["batch_size"], shuffle = True,
        num_workers=cfg["train"]["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size = cfg["train"]["batch_size"], shuffle = True,
        num_workers=cfg["train"]["num_workers"], pin_memory=True, drop_last=True,
    )
    return train_loader, val_loader
    
def initModels(cfg, device):
    G = UNetGenerator(cfg["model"]["in_ch"], cfg["model"]["out_ch"], cfg["model"]["base_channels"]).to(device)
    D = PatchGANDiscriminator(cfg["model"]["in_ch"], cfg["model"]["out_ch"], cfg["model"]["base_channels"]).to(device)
    return G, D

def initOptimizers(cfg, G, D):
    opt_G = torch.optim.Adam(G.parameters(), cfg["train"]["lr"], betas=(cfg["train"]["beta1"], cfg["train"]["beta2"]))
    opt_D = torch.optim.Adam(D.parameters(), cfg["train"]["lr"], betas=(cfg["train"]["beta1"], cfg["train"]["beta2"]))
    return opt_G, opt_D
    
def continue_training_from_last_ckpt(full_ckpt_path, G, D, opt_G, opt_D, scaler_G, scaler_D, device):
    if not os.path.exists(full_ckpt_path):
        raise FileNotFoundError(f"no training checkpoint found at {full_ckpt_path}")
    
    ckpt = torch.load(full_ckpt_path, map_location = device)
    G.load_state_dict(ckpt["G_state"])
    D.load_state_dict(ckpt["D_state"])
    opt_G.load_state_dict(ckpt["opt_G_state"])
    opt_D.load_state_dict(ckpt["opt_D_state"])
    if "scaler_G_state" in ckpt:
        scaler_G.load_state_dict(ckpt["scaler_G_state"])
        scaler_D.load_state_dict(ckpt["scaler_D_state"])
    history = ckpt["history"]
    start_epoch = ckpt["epoch"] + 1
    print(f"resumed from epoch {ckpt['epoch']}, continuing at epoch {start_epoch}")

def train_one_epoch(train_loader : DataLoader, val_loader : DataLoader):
    pass
    

def main(cfg_path, resume):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    
    ckpt_dir = cfg["train"]["checkpoint_dir"]
    log_dir = cfg["train"]["log_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    full_ckpt_path = os.path.join(ckpt_dir, "training_state.pt")
    
    roi_ids = list_roi_ids(cfg["data"]["seasons_dir"])
    train_ids, val_ids = split_roi_ids(roi_ids, val_frac=cfg["data"]["val_frac"], seed=cfg["seed"])
    
    max_train_rois = cfg["data"].get("max_train_rois")
    if max_train_rois is not None:
        train_ids = train_ids[:max_train_rois]
    
    print(f"{len(roi_ids)} ROI scenes total -> {len(train_ids)} train / {len(val_ids)} val")
    
    train_ds = SAR2EODataset(cfg["data"]["seasons_dir"], train_ids)
    val_ds = SAR2EODataset(cfg["data"]["seasons_dir"], val_ids)
    print(f"train pairs: {len(train_ds)}, val pairs: {len(val_ds)}")
    
    train_loader, val_loader = get_loaders(cfg, train_ds, val_ds)
    
    G, D = initModels(cfg, device)
    opt_G, opt_D = initOptimizers(cfg, G, D)
    
    l1_loss = nn.L1Loss()
    gan_loss = nn.MSELoss() if cfg["train"]["use_lsgan"] else nn.BCEWithLogitsLoss()
    
    use_amp = (device.type == "cuda")
    scaler_G = GradScaler(device.type, enabled = use_amp)
    scaler_D = GradScaler(device.type, enabled = use_amp)
    
    lambda_l1 = cfg["train"]["lambda_l1"]
    lambda_perc = cfg["train"]["lambda_perceptual"]
    perceptual_loss_fn = get_perceptual_loss_fn(device) if lambda_perc > 0 else None
    
    start_epoch = 1
    history = []  # per-epoch: epoch, g_loss, d_loss, val_l1
    
    if resume:
        continue_training_from_last_ckpt(full_ckpt_path, G, D, opt_G, opt_D, scaler_G, scaler_D, device)
    
    total_epochs = cfg["train"]["epochs"]
    if(start_epoch > total_epochs):
        print(f"checkpoint is already at epoch {start_epoch - 1} >= configured epochs {total_epochs}. Nothing to do.")
        return

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.perf_counter()
        
        G.train()
        D.train()
        g_losses, d_losses = [], []
 
        for sar, eo in train_loader:
            sar, eo = sar.to(device), eo.to(device)
            with autocast(device_type = device.type, enabled = use_amp):
                fake_eo = G(sar)
 
            # --- train discriminator ---
            opt_D.zero_grad()
            with autocast(device_type = device.type, enabled = use_amp):
                pred_real = D(sar, eo)
                pred_fake_for_d = D(sar, fake_eo.detach())
                valid = torch.ones_like(pred_real)
                fake_label = torch.zeros_like(pred_real)
                d_loss = 0.5 * (gan_loss(pred_real, valid) + gan_loss(pred_fake_for_d, fake_label))

            scaler_D.scale(d_loss).backward()
            scaler_D.step(opt_D)
            scaler_D.update()
 
            # --- train generator ---
            opt_G.zero_grad()
            with autocast(device_type = device.type, enabled = use_amp):
                pred_fake_for_g = D(sar, fake_eo)
                g_adv = gan_loss(pred_fake_for_g, valid)
                g_l1 = l1_loss(fake_eo, eo) * lambda_l1
                g_loss = g_adv + g_l1
                if perceptual_loss_fn is not None:
                    g_loss = g_loss + perceptual_loss_fn(fake_eo, eo) * lambda_perc
                    
            scaler_G.scale(g_loss).backward()
            scaler_G.step(opt_G)
            scaler_G.update()
 
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
 
        # --- validation (L1 only here; run eval.py separately for LPIPS/FID/SSIM/PSNR) ---
        G.eval()
        val_l1s = []
        with torch.no_grad():
            for sar, eo in val_loader:
                sar, eo = sar.to(device), eo.to(device)
                fake_eo = G(sar)
                val_l1s.append(l1_loss(fake_eo, eo).item())
            
        elapsed_time = time.perf_counter() - epoch_start
 
        # cast to native floats -- numpy scalars aren't in torch.load's weights_only=True
        # allowlist (default since PyTorch 2.6) and would break --resume otherwise
        mean_g, mean_d, mean_val = float(np.mean(g_losses)), float(np.mean(d_losses)), float(np.mean(val_l1s))
        print(f"epoch {epoch:03d}/{total_epochs} | G {mean_g:.4f} | D {mean_d:.4f} | val_L1 {mean_val:.4f} (time : {elapsed_time:.2f}s)")
        history.append({"epoch": epoch, "g_loss": mean_g, "d_loss": mean_d, "val_l1": mean_val})
 
        # full state (for resuming) + plain generator weights (for eval.py / infer.py)
        save_checkpoint(full_ckpt_path, epoch, G, D, opt_G, opt_D, history)
        torch.save(G.state_dict(), os.path.join(ckpt_dir, "generator_latest.pt"))
 
    # --- persist raw loss values + plot, as required by the assignment ---
    csv_path = os.path.join(log_dir, "loss_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "g_loss", "d_loss", "val_l1"])
        writer.writeheader()
        writer.writerows(history)
 
    epochs = [h["epoch"] for h in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h["g_loss"] for h in history], label="Generator loss (train)")
    plt.plot(epochs, [h["d_loss"] for h in history], label="Discriminator loss (train)")
    plt.plot(epochs, [h["val_l1"] for h in history], label="Validation L1")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "loss_curve.png"))
    print(f"saved loss history to {csv_path} and loss_curve.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint_dir/training_state.pt")
    args = parser.parse_args()
    main(args.config, args.resume)
    