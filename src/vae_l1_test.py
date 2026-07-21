import torch
import torch.nn as nn
import numpy as np
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms

# reuse your existing dataset.py utilities directly
from dataset import list_roi_ids, split_roi_ids, build_pairs

device = "cuda" if torch.cuda.is_available() else "cpu"
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
vae.eval()

SEASON_DIR = "/kaggle/input/datasets/ayaanrzaverigmailcom/data-pairs/ROIs1868_summer"

# same split logic your training script already uses -> real val-set patches, not one random file
roi_ids = list_roi_ids(SEASON_DIR)
train_ids, val_ids = split_roi_ids(roi_ids)   # same seed=42 default as your training code
val_pairs = build_pairs(SEASON_DIR, val_ids)  # list of (sar_path, eo_path)

print(f"Total val pairs available: {len(val_pairs)}")

N_SAMPLES = 100  # bump this up/down depending on how long you're willing to wait
rng = np.random.default_rng(seed=0)
sample_idxs = rng.choice(len(val_pairs), size=min(N_SAMPLES, len(val_pairs)), replace=False)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

l1_fn = nn.L1Loss()

def encode_decode(img_tensor, deterministic=False):
    with torch.no_grad():
        dist = vae.encode(img_tensor).latent_dist
        z = dist.mode() if deterministic else dist.sample()
        z = z * 0.18215
        recon = vae.decode(z / 0.18215).sample
    return recon

sample_losses, mode_losses = [], []

for i, idx in enumerate(sample_idxs):
    _, eo_path = val_pairs[idx]
    image = Image.open(eo_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    recon_sample = encode_decode(img_tensor, deterministic=False)
    recon_mode   = encode_decode(img_tensor, deterministic=True)

    sample_losses.append(l1_fn(recon_sample, img_tensor).item())
    mode_losses.append(l1_fn(recon_mode, img_tensor).item())

    if (i + 1) % 20 == 0:
        print(f"  processed {i+1}/{len(sample_idxs)}")

sample_losses = np.array(sample_losses)
mode_losses = np.array(mode_losses)

print("\n--- VAE Reconstruction Floor (val split, N={}) ---".format(len(sample_idxs)))
print(f"Stochastic (.sample()) -> mean: {sample_losses.mean():.5f} | std: {sample_losses.std():.5f} | "
      f"min: {sample_losses.min():.5f} | max: {sample_losses.max():.5f}")
print(f"Deterministic (.mode()) -> mean: {mode_losses.mean():.5f} | std: {mode_losses.std():.5f} | "
      f"min: {mode_losses.min():.5f} | max: {mode_losses.max():.5f}")
print(f"\nYour current model's image-space L1: 0.433")
print(f"Gap (model / floor): {0.433 / mode_losses.mean():.2f}x")