import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusers import AutoencoderKL
from accelerate import Accelerator

from src.dataset import SAR2EODataset, list_roi_ids, split_roi_ids

def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    
    accelerator = Accelerator(mixed_precision="fp16" if torch.cuda.is_available() else "no")
    device = accelerator.device
    print(f"Targeting compute device for VAE processing: {device}")
    
    output_base_dir = cfg["train"].get("vae_output_dir", "/kaggle/working/latent_vecs")
    if accelerator.is_main_process:
        os.makedirs(output_base_dir, exist_ok = True)
        
    # Synchronize so GPU 1 waits until GPU 0 creates the output folder [2 t4 gpus]
    accelerator.wait_for_everyone()
    
    # Load all unique matching scenes
    roi_ids = list_roi_ids(cfg["data"]["seasons_dir"])
    train_ids, val_ids = split_roi_ids(roi_ids, val_frac=cfg["data"]["val_frac"], seed=cfg["seed"])
    all_ids = sorted(train_ids + val_ids)
    
    dataset = SAR2EODataset(cfg["data"]["seasons_dir"], all_ids)
    
    # Standard DataLoader (Accelerate will convert this into a Distributed Dataloader)  
    loader = DataLoader(dataset, batch_size = cfg["train"]["batch_size"], shuffle=False, num_workers = cfg["train"]["num_workers"], pin_memory=True)
    
    # Initialize frozen VAE engine
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    
    loader, vae = accelerator.prepare(loader, vae)
        
    if accelerator.is_main_process:
        print(f"Beginning VAE latent rep generation across entire data [{len(dataset)} pairs]..")
    
    with torch.no_grad():
        for batch_idx, (sar, eo) in enumerate(tqdm(loader)):            
            sar_rgb = torch.cat([sar, sar, sar], dim = 1) if sar.shape[1] == 1 else sar
            
            # compress inputs down into standard 4-channel latents
            z_x = vae.encode(sar_rgb).latent_dist.sample() * 0.18215 # (B, 4, 32, 32)
            z_y = vae.encode(eo).latent_dist.sample() * 0.18215      # (B, 4, 32, 32)
            
            # unpack batch instances and save them individually to disk
            for idx in range(sar.shape[0]):
                # Reconstruct the absolute dataset index across distributed worker spaces
                # accelerator.num_processes tells us how many GPUs are running (2)
                # accelerator.process_index tells us if this is GPU 0 or GPU 1
                data_index = (batch_idx * 64 * accelerator.num_processes) + (idx * accelerator.num_processes) + accelerator.process_index
                
                # Boundary check safety verification handling for final batch remainder tails
                if data_index >= len(dataset):
                    continue
                
                pair_paths = dataset.pairs[data_index]  
               
                sar_filename = os.path.splitext(os.path.basename(pair_paths[0]))[0]
                eo_filename = os.path.splitext(os.path.basename(pair_paths[1]))[0]
                
                torch.save(z_x[idx].cpu(), os.path.join(output_base_dir, f"{sar_filename}.pt"))
                torch.save(z_y[idx].cpu(), os.path.join(output_base_dir, f"{eo_filename}.pt"))
    
    # until all processes wrap up completely
    accelerator.wait_for_everyone()
        
    if accelerator.is_main_process:
        print(f"[PRECOMPUTATION SUCCESSFUL] All compressed configurations saved cleanly to {output_base_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)