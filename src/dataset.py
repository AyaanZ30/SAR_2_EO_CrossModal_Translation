import os
import glob
import random
 
import numpy as np
from typing import List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset

ROI_SUMMER_DATASET_PATH = 'ROIs1868_summer'

# do in main
# SENTINEL_1_DATA_DIR = os.path.join(ROI_SUMMER_DATASET_PATH, 's1_0')
# SENTINEL_2_DATA_DIR = os.path.join(ROI_SUMMER_DATASET_PATH, 's2_0')

SAR_IMAGES_FOLDER_PREFIX = "s1_";   SAR_FILENAME_TAG = "_s1_";
EO_IMAGES_FOLDER_PREFIX = "s2_";    EO_FILENAME_TAG = "_s2_";

def list_roi_ids(season_dir : str) -> List[str]:
    """
    Scan `season_dir` for SAR subfolders (e.g. "s1_0", "s1_1", ...) and return
    the bare ROI ids found (e.g. ["0", "1", ...]), sorted for determinism.
    """
    sar_glob = os.path.join(season_dir, f"{SAR_IMAGES_FOLDER_PREFIX}*")
    sar_dirs = glob.glob(sar_glob)
    roi_ids = sorted({
        os.path.basename(d)[len(SAR_IMAGES_FOLDER_PREFIX):] for d in sar_dirs
    })
    return roi_ids

def split_roi_ids(roi_ids : List[str], val_frac : float = 0.2, seed : int = 42):
    """
    Create train-val split at the ROI-id level itself 
    [ex: s1_0 -> s1_5 dirs(each with ~ 2k SAR imgs) & s2_0 -> s2_5 dirs(each with ~ 2k EO imgs)] : TRAINING SET
    [ex: s1_5 -> s1_9 dirs(each with ~ 2k SAR imgs) & s2_5 -> s2_9 dirs(each with ~ 2k EO imgs)] : VALIDATION SET
    
    treat each unique ROI ID as a single atomic unit (avoid splitting at individual patch level to avoid spatial data leakage)
    """
    shuffled = list(roi_ids)
    random.Random(seed).shuffle(shuffled)
    n_vals = max(1, int(len(roi_ids) * val_frac))
    
    val_ids, train_ids = shuffled[:n_vals], shuffled[n_vals:]
    return sorted(train_ids), sorted(val_ids)

def sar_filename_to_eo_filename(sar_filename: str) -> str:
    """
    Convert a SAR patch filename to its expected EO counterpart, e.g.:
        "ROIs1868_summer_s1_0_p1.png" -> "ROIs1868_summer_s2_0_p1.png"
    """
    return sar_filename.replace(SAR_FILENAME_TAG, EO_FILENAME_TAG)

def find_roi_pairs(season_dir : str, roi_id : str) -> List[Tuple[str, str]]:
    sar_dir = os.path.join(season_dir, f"{SAR_IMAGES_FOLDER_PREFIX}{roi_id}")
    eo_dir = os.path.join(season_dir, f"{EO_IMAGES_FOLDER_PREFIX}{roi_id}")
    
    if not (os.path.isdir(sar_dir) and os.path.isdir(eo_dir)): return [] 
    
    eo_files_in_dir = set(os.listdir(eo_dir))
    
    pairs = []
    for sar_path in sorted(glob.glob(os.path.join(sar_dir, "*.png"))):
        sar_filename = os.path.basename(sar_path)
        eo_filename = sar_filename_to_eo_filename(sar_filename)
        if eo_filename in eo_files_in_dir:
            eo_path = os.path.join(eo_dir, eo_filename)
            pairs.append((sar_path, eo_path))
            
    return pairs

def build_pairs(season_dir : str, roi_ids : List[str]) -> List[Tuple[str, str]]:
    """Aggregate find_pairs_for_roi across every ROI id in `roi_ids`."""
    all_pairs = []
    for roi_id in roi_ids:
        all_pairs.extend(find_roi_pairs(season_dir, roi_id))
    return all_pairs

def load_sar_patch(path : str) -> torch.Tensor:
    """
    Load a single-channel SAR patch and normalize it to [-1, 1].
    (data is already dB-scaled and 8-bit quantized by the dataset's creators)
 
    Returns a (1, H, W) float32 tensor.
    """
    grayscale = np.array(Image.open(path).convert("L"), dtype = np.float32) 
    normalized_0_1 = grayscale / 255.0
    normalized_neg1_1 = normalized_0_1 * 2.0 - 1.0
    return torch.from_numpy(normalized_neg1_1).unsqueeze(0).float()
    

def load_eo_patch(path : str) -> torch.Tensor:
    """
    Load a 3-channel EO (RGB) patch and normalize it to [-1, 1], matching the
    generator's Tanh output range.
 
    Returns a (3, H, W) float32 tensor.
    """
    rgb = np.array(Image.open(path).convert('RGB'), dtype = np.float32) / 255.0
    normalized_neg1_1 = rgb * 2.0 - 1.0
    image_values = normalized_neg1_1.transpose(2, 0, 1)    # (H, W, 3) => (3, H, W)
    return torch.from_numpy(image_values).float()    
    


class SAR2EODataset(Dataset):
    """
    Class deliberately written to orchestrate the logic by calling above functions
    Handles the way dataset is created [train / val]
    """
    def __init__(self, season_dir : str, roi_ids : List[str]):
        super().__init__()
        
        self.pairs = build_pairs(season_dir, roi_ids)
        if len(self.pairs) == 0:
            raise RuntimeError(f"No SAR/EO pairs found under {season_dir} for ROIs {roi_ids}")
    
    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        sar_path, eo_path = self.pairs[idx]
        return load_sar_patch(sar_path), load_eo_patch(eo_path) 

class PrecomputedLatentDataset(Dataset):
    """
    High-performance dataset engine that bypasses raw images completely.
    Directly loads pre-computed 4-channel latents (B, 4, 32, 32) from memory caches 
    """
    def __init__(self, seasons_dir : str, roi_ids : List[str], latent_dir : str):
        self.latent_dir = latent_dir 
        raw_pairs = build_pairs(seasons_dir, roi_ids)
        
        self.latent_pairs = []
        for sar_path, eo_path in self.raw_pairs:
            sar_name = os.path.splitext(os.path.basename(sar_path))[0]
            eo_name = os.path.splitext(os.path.basename(eo_path))[0]
            
            sar_latent_path = os.path.join(latent_dir, f"{sar_name}.pt")
            eo_latent_path = os.path.join(latent_dir, f"{eo_name}.pt")
            
            if os.path.exists(sar_latent_path) and os.path.exists(eo_latent_path):
                self.latent_pairs.append((sar_latent_path, eo_latent_path))
        
        if len(self.latent_pairs) == 0:
            raise RuntimeError(f"No matching precomputed latent tensors found under: {latent_dir}")
        
    def __len__(self) -> int:
        return len(self.latent_pairs) 
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        sar_latent_path, eo_latent_path = self.latent_pairs[idx] 
        
        # load only the precomputed latent vector representations for current pair
        z_x = torch.load(sar_latent_path, weights_only = True)  # (4, 32, 32)
        z_y = torch.load(eo_latent_path, weights_only = True)   # (4, 32, 32)
        return z_x, z_y
    
    
if __name__ == "__main__":
    import sys
 
    # quick unit check of the pure string function, no filesystem needed
    example = "ROIs1868_summer_s1_0_p1.png"
    converted = sar_filename_to_eo_filename(example)
    assert converted == "ROIs1868_summer_s2_0_p1.png", f"unexpected conversion: {converted}"
    print(f"filename conversion check passed: {example} -> {converted}")
 
    season_dir = sys.argv[1] if len(sys.argv) > 1 else "./ROIs1868_summer"
 
    roi_ids = list_roi_ids(season_dir)
    print(f"found {len(roi_ids)} ROI scene(s): {roi_ids}")
 
    train_ids, val_ids = split_roi_ids(roi_ids)
    print(f"train scenes: {train_ids}")
    print(f"val scenes:   {val_ids}")
 
    all_ids = roi_ids  # use every available ROI for this smoke test, ignoring the split
    dataset = SAR2EODataset(season_dir, all_ids)
    print(f"total pairs found: {len(dataset)}")
 
    sar_tensor, eo_tensor = dataset[0]
    print(f"sar tensor -> shape {tuple(sar_tensor.shape)}, "
          f"range [{sar_tensor.min().item():.3f}, {sar_tensor.max().item():.3f}]")
    print(f"eo  tensor -> shape {tuple(eo_tensor.shape)}, "
          f"range [{eo_tensor.min().item():.3f}, {eo_tensor.max().item():.3f}]")
 
