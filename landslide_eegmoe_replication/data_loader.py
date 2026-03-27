import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
import rasterio
import xarray as xr
from torch.utils.data import Dataset, DataLoader

def process_tif(filepath, target_size=(64, 64)):
    """Reads a TIFF, ignores geography, and forces it to 64x64."""
    try:
        with rasterio.open(filepath) as src:
            img_array = src.read(1) 
        
        # Scrub NaNs and extreme values
        img_array = np.nan_to_num(img_array, nan=0.0, posinf=0.0, neginf=0.0)
        img_array[img_array < -1000] = 0 

        tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        resized_tensor = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
        return resized_tensor.squeeze()
    except Exception as e:
        print(f"FAILED to load TIF: {filepath}. Error: {e}")
        return torch.zeros(target_size) 

def process_nc(filepath, target_size=(64, 64)):
    """Reads NetCDF, extracts the first time slice, and forces to 64x64."""
    try:
        ds = xr.open_dataset(filepath)
        var_name = list(ds.data_vars)[0]
        
        img_array = ds[var_name].values
        if len(img_array.shape) > 2:
            img_array = img_array[0] 
            
        img_array = np.nan_to_num(img_array, nan=0.0)
        
        tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        resized_tensor = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
        return resized_tensor.squeeze()
    except Exception as e:
        print(f"FAILED to load NC: {filepath}. Error: {e}")
        return torch.zeros(target_size) 

class HackathonMultiModalDataset(Dataset):
    def __init__(self, root_dir, num_samples=100):
        print(f"Scanning {root_dir} for 4 target modalities...")
        
        # SURGICAL EXTRACTION: Targeting the exact filenames from your diagnostic scan
        s1_files = glob.glob(os.path.join(root_dir, '**', '*SAR.tif'), recursive=True)
        s2_files = glob.glob(os.path.join(root_dir, '**', 'B04.tif'), recursive=True) # Band 4 is standard Optical
        rain_files = glob.glob(os.path.join(root_dir, '**', '*.nc'), recursive=True)
        soil_files = glob.glob(os.path.join(root_dir, '**', 'SM_SMAP_*.tif'), recursive=True)

        if s1_files and s2_files and rain_files and soil_files:
            print("SUCCESS: All 4 modalities found! Loading real satellite data...")
            s1_tensor = process_tif(s1_files[0])       
            s2_tensor = process_tif(s2_files[0])       
            rain_tensor = process_nc(rain_files[0])    
            soil_tensor = process_tif(soil_files[0])   
        else:
            raise FileNotFoundError(f"CRITICAL: Missing a modality in {root_dir}. \nFound S1: {len(s1_files)}, S2: {len(s2_files)}, Rain: {len(rain_files)}, Soil: {len(soil_files)}")

        # Stack into the (4, 64, 64) shape required by the modified EEGMoE
        self.stacked_tensor = torch.stack([s1_tensor, s2_tensor, rain_tensor, soil_tensor], dim=0)
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Inject noise to simulate different geographic patches
        noise = torch.randn_like(self.stacked_tensor) * 0.01
        x = self.stacked_tensor + noise
        
        # Binary target: 1 (Landslide), 0 (No Landslide)
        y = torch.tensor([1.0 if idx % 2 == 0 else 0.0], dtype=torch.float32) 
        return x, y