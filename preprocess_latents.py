import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusers import AutoencoderKL
from accelerate import Accelerator

sys.path.append(os.path.join(os.path.dirname(__file__), "eo-vae"))

from eo_vae.models.new_autoencoder import EOFluxVAE
from C_Diff.image_datasets import SAR2EODataset, list_roi_ids, split_roi_ids

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
    loader = DataLoader(dataset, batch_size = 16, shuffle=False, num_workers = cfg["train"]["num_workers"], pin_memory=True)
    
    # Initialize frozen VAE engine
    # vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae = EOFluxVAE.from_pretrained(
        repo_id="nilsleh/eo-vae",
        ckpt_filename="eo-vae.ckpt",
        config_filename="model_config.yaml",
        device=device,
    )
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    s1_wvs = torch.tensor([5.4, 5.6], dtype=torch.float32, device=device)           # 2 channels (VV, VH)
    s2_wvs = torch.tensor([0.665, 0.56, 0.49], dtype=torch.float32, device=device)  # 3 channels (R, G, B)
    
    loader, vae = accelerator.prepare(loader, vae)
        
    if accelerator.is_main_process:
        print(f"Beginning EO-VAE latent rep generation across entire data [{len(dataset)} pairs]..")
    
    with torch.no_grad():
        for batch_idx, (sar, eo) in enumerate(tqdm(loader, disable = not accelerator.is_local_main_process)):       
            with torch.amp.autocast(device_type = "cuda", dtype = torch.float16):     
                # sar_rgb = torch.cat([sar, sar, sar], dim = 1) if sar.shape[1] == 1 else sar
                if sar.shape[1] == 1:
                    sar = sar.repeat(1, 2, 1, 1)    # Repeat 1-channel SAR to 2-channels (VV, VH dummy duplication)
                elif sar.shape[1] > 1:
                    sar = sar[:, :2, :, :]          # If 3-channel SAR loaded prev, slice it to 2 channe;

                # compress inputs down into standard 4-channel latents
                # z_x = vae.encode_spatial_normalized(sar).latent_dist.sample() * 0.18215 # (B, 4, 32, 32)
                # z_y = vae.encode_spatial_normalized(eo).latent_dist.sample() * 0.18215      # (B, 4, 32, 32)
            
                z_x = vae.encode_spatial_normalized(sar, s1_wvs)
                z_y = vae.encode_spatial_normalized(eo, s2_wvs)
            # unpack batch instances and save them individually to disk
            for idx in range(sar.shape[0]):
                # batch size : 16
                data_index = (batch_idx * cfg["train"]["batch_size"] + idx) * accelerator.num_processes + accelerator.process_index
                
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