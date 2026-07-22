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
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
 
from C_Diff.utils.plots import plot_performance
from C_Diff.utils.logging import log_epoch

from C_Diff.image_datasets import list_roi_ids, split_roi_ids, PrecomputedLatentDataset
from C_Diff.model import CDiffSETUNet

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def min_snr_weighted_mse(pred_noise, noise, timesteps, noise_scheduler, gamma = 5.0):
    """
    In a DDPM noise schedule, (alpha-bar) represents the cumulative amount of original clean image structure preserved at time t
    SNR (at time t) = (alpha_bar_t / 1 - alpha_bar_t)   [at t:0, no noise & at t:1, pure noise]
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
    alpha_bar_t = alphas_cumprod[timesteps]
    snr = alpha_bar_t / (1 - alpha_bar_t)
    min_snr_gamma = torch.clamp(snr, max = gamma)
    weight = min_snr_gamma / snr  # per-sample scaling factor 

    per_sample_mse = torch.square(noise - pred_noise).mean(dim=[1, 2, 3])
    weighted_loss = (weight * per_sample_mse).mean()
    return weighted_loss

def update_ema(ema_model, model, decay = 0.999):
    with torch.no_grad():
        msd = model.state_dict()
        for k, v in ema_model.state_dict().items():
            v.copy_(v * decay + msd[k].detach().to(v.device) * (1 - decay)) 

def diffusion_step(model, noise_scheduler, z_x, z_y, timesteps = None):
    """Runs one forward pass of noise prediction + confidence, returns loss and raw components."""
    if timesteps is None: 
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (z_x.shape[0],), device = z_x.device).long()
    noise = torch.randn_like(z_y)
    z_y_noisy = noise_scheduler.add_noise(z_y, noise, timesteps)

    u_net_input = torch.cat([z_x, z_y_noisy], dim=1)
    pred_noise, _ = model(u_net_input, timesteps)

    loss = min_snr_weighted_mse(pred_noise, noise, timesteps, noise_scheduler, gamma = 5.0)
    return loss, pred_noise, noise


def train_one_epoch(model, ema_model, loader, optimizer, noise_scheduler, accelerator, ema_decay=0.999):
    model.train()
    losses = []

    for z_x, z_y in loader:
        optimizer.zero_grad()
        loss, _, _, = diffusion_step(model, noise_scheduler, z_x, z_y)
        accelerator.backward(loss)
        optimizer.step()

        if accelerator.is_main_process:
            update_ema(ema_model, accelerator.unwrap_model(model), decay = ema_decay) 
        
        losses.append(loss.item())

    return {"train_loss" : np.mean(losses)}

def _compute_image_space_l1(model, noise_scheduler, vae, sample_fn, z_x, z_y, device, accelerator):
    unwrapped = accelerator.unwrap_model(model)
    z_gen = sample_fn(unwrapped, noise_scheduler, z_x, device, num_inference_steps = 1000)

    img_pred = vae.decode((z_gen.cpu()) / 0.18215).sample
    img_gt = vae.decode((z_y.cpu()) / 0.18215).sample

    img_l1 = torch.abs(img_pred - img_gt).mean(dim = [1, 2, 3])
    img_l1 = img_l1.to(device)

    gathered = accelerator.gather_for_metrics(img_l1)
    return gathered.mean().item()


def _accumulate_timestep_records(model, noise_scheduler, l1_metric, z_x, z_y, t_val, device, accelerator, all_val_records):
    noise = torch.randn_like(z_y)
    eval_t = torch.full((z_x.shape[0],), t_val, device=device).long()
    z_y_noisy = noise_scheduler.add_noise(z_y, noise, eval_t)
    combined = torch.cat([z_x, z_y_noisy], dim=1)
    pred_noise, _ = model(combined, eval_t)

    g_zx, g_zy, g_pred, g_noise = accelerator.gather_for_metrics((z_x, z_y, pred_noise, noise))
    batch_l1 = l1_metric(g_pred, g_noise).mean(dim=[1, 2, 3])

    for idx in range(g_zx.shape[0]):
        all_val_records[t_val].append({
            'loss': batch_l1[idx].item(),
            'sar_lat': g_zx[idx].cpu(),
            'gt_lat': g_zy[idx].cpu(),
        })


EVAL_TIMESTEPS = [25, 50, 250, 500, 750, 999]
def run_validation(model, val_loader, noise_scheduler, vae, sample_fn, device, accelerator):
    model.eval()
    l1_metric = nn.L1Loss(reduction='none')
    all_val_records = {t: [] for t in EVAL_TIMESTEPS}
    val_image_l1_errors = []

    with torch.no_grad():
        for i, (z_x, z_y) in enumerate(val_loader):
            if i == 0:
                val_image_l1_errors.append(
                    _compute_image_space_l1(model, noise_scheduler, vae, sample_fn, z_x, z_y, device, accelerator)
                ) 
            for t_val in EVAL_TIMESTEPS:
                _accumulate_timestep_records(model, noise_scheduler, l1_metric, z_x, z_y, t_val, device, accelerator, all_val_records)
    
    mean_val_l1_per_t = {t: np.mean([r['loss'] for r in recs]) for t, recs in all_val_records.items()}
    return {
        "per_t": mean_val_l1_per_t,
        "mean": np.mean(list(mean_val_l1_per_t.values())),
        "image_l1": np.mean(val_image_l1_errors) if val_image_l1_errors else float('nan'),
        "records": all_val_records,   # needed downstream for best/worst plotting
    }

def decode_records(records_list, model, noise_scheduler, vae, sample_fn, device, accelerator):
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.eval()
    decoded = []
    for item in records_list:
        z_sar = item['sar_lat'].unsqueeze(0).to(device)
        z_gt = item['gt_lat'].unsqueeze(0).to("cpu")
        z_gen = sample_fn(unwrapped, noise_scheduler, z_sar, device, num_inference_steps=50)
        decoded.append({
            'loss': item['loss'],
            'sar': vae.decode(z_sar.cpu() / 0.18215).sample.squeeze(0),
            'pred': vae.decode(z_gen.cpu() / 0.18215).sample.squeeze(0),
            'gt': vae.decode(z_gt / 0.18215).sample.squeeze(0),
        })
    return decoded


def plot_best_worst(val_result, epoch, model, noise_scheduler, vae, sample_fn, device, accelerator, log_dir):
    plot_records = sorted(val_result["records"][500], key=lambda r: r['loss'])
    best_5 = decode_records(plot_records[:5], model, noise_scheduler, vae, sample_fn, device, accelerator)
    worst_5 = decode_records(plot_records[-5:], model, noise_scheduler, vae, sample_fn, device, accelerator)
    plot_performance(best_5, log_dir, tier_name=f"best_epoch_{epoch}")
    plot_performance(worst_5, log_dir, tier_name=f"worst_epoch_{epoch}")


@torch.no_grad()    
def sample(model, scheduler, z_sar, device, num_inference_steps = 1000):
    """Iterative DDPM reverse sampling conditioned on a SAR latent (more sophisticated than subtraction of noise)"""
    z_sar = z_sar.to(device)

    # sched = DDPMScheduler(num_train_timesteps = scheduler.config.num_train_timesteps)
    sched = DDIMScheduler(num_train_timesteps = scheduler.config.num_train_timesteps)
    sched.set_timesteps(num_inference_steps, device=device)
    
    z_t = torch.randn_like(z_sar, device = device)
    for t in sched.timesteps:
        t_batch = torch.full((z_sar.shape[0],), t, device=device, dtype=torch.long)
        concatenated_latent = torch.cat([z_sar, z_t], dim = 1)
        pred_noise, _ = model(concatenated_latent, timesteps = t_batch)
        z_t = sched.step(pred_noise, t, z_t).prev_sample
    return z_t 
    

def main(cfg_path, resume):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    
    set_seed(cfg["seed"])
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters = True)
    accelerator = Accelerator(
        mixed_precision = "fp16" if torch.cuda.is_available() else "no",
        kwargs_handlers = [ddp_kwargs]
    )

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

    ema_model = None
    if accelerator.is_main_process:
        from copy import deepcopy
        ema_model = deepcopy(accelerator.unwrap_model(model)).eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
    
    start_epoch = 1
    history = []  
    
    if resume and os.path.exists(full_ckpt_path):
        accelerator.wait_for_everyone()

        ckpt = torch.load(full_ckpt_path, map_location=device, weights_only=False)
        accelerator.unwrap_model(model).load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt["history"]

        if accelerator.is_main_process and "ema_model_state" in ckpt:
            ema_model.load_state_dict(ckpt["ema_model_state"])


    total_epochs = cfg["train"]["epochs"]
    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.perf_counter()
        
        train_stats = train_one_epoch(model, ema_model, train_loader, optimizer, noise_scheduler, accelerator)
            
        val_result = None
        if epoch % 10 == 0 or epoch == total_epochs:
            eval_model = ema_model if accelerator.is_main_process else model
            val_result = run_validation(eval_model, val_loader, noise_scheduler, vae, sample, device, accelerator)
            
            if accelerator.is_main_process:       
                plot_best_worst(val_result, epoch, eval_model, noise_scheduler, vae, sample, device, accelerator, log_dir)

        if accelerator.is_main_process:    
            elapsed = time.perf_counter() - epoch_start
            log_epoch(epoch, total_epochs, elapsed, train_stats, val_result)

            history.append({
                "epoch": epoch, 
                "train_loss": train_stats["train_loss"], 
                "val_l1": dict(val_result["per_t"]) if val_result else None,
            })
            
            # save_checkpoint(full_ckpt_path, epoch, model, optimizer, scaler, history)
            torch.save({
                "epoch": epoch,
                "model_state": accelerator.unwrap_model(model).state_dict(),
                "ema_model_state" : ema_model.state_dict() if ema_model is not None else None,
                "optimizer_state": optimizer.state_dict(),
                "history": history,
            }, full_ckpt_path)
        
        accelerator.wait_for_everyone()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint_dir/  ")
    args = parser.parse_args()
    main(args.config, args.resume)
    
