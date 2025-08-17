"""
Data loading utilities for various image datasets.
Extracted from train.py for better modularity.
"""

import os
import glob
import numpy as np
from scipy.io import loadmat
from skimage import io
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DataLoader(Dataset):
    """Simple DataLoader for .tif files based on infer.py pattern."""

    def __init__(self, data_dir, patch=256):
        super(DataLoader, self).__init__()
        self.data_dir = data_dir
        self.patch = patch

        # Get all .tif files
        self.train_fns = glob.glob(os.path.join(self.data_dir, "*.tif"))
        self.train_fns.sort()
        print(f"fetch {len(self.train_fns)} .tif samples for training")

    def __getitem__(self, index):
        # fetch image
        fn = self.train_fns[index]
        img_array = io.imread(fn).astype(np.float32)

        # Handle dimensions like in infer.py
        if img_array.ndim == 2:
            img_array = img_array[:, :, np.newaxis]
        elif img_array.ndim == 3 and img_array.shape[2] == 1:
            pass  # Already has single channel dimension
        else:

            img_array = np.mean(img_array, axis=2, keepdims=True)

        # Random crop
        H, W = img_array.shape[:2]
        CSize = self.patch
        rnd_h = np.random.randint(0, max(0, H - CSize))
        rnd_w = np.random.randint(0, max(0, W - CSize))
        img_array = img_array[rnd_h : rnd_h + CSize, rnd_w : rnd_w + CSize]

        # Convert to tensor format (C, H, W)
        if img_array.ndim == 3:
            img_array = img_array.transpose(2, 0, 1)
        else:
            img_array = img_array[np.newaxis, :, :]

        # Convert to torch tensor
        img_tensor = torch.from_numpy(img_array.copy())
        return img_tensor

    def __len__(self):
        return len(self.train_fns)


def validation_data(dataset_dir):
    """Load .tif validation data based on infer.py pattern."""
    if not os.path.exists(dataset_dir):
        print(f"Validation directory {dataset_dir} does not exist")
        return []
        
    tif_files = glob.glob(os.path.join(dataset_dir, "*.tif"))
    tif_files.sort()
    
    print(f"Found {len(tif_files)} .tif files in {dataset_dir}")

    data_images = []

    for img_path in tif_files:
        try:
            img_array = io.imread(img_path).astype(np.float32)
            print(f"Loaded {img_path}: shape={img_array.shape}, dtype={img_array.dtype}")

            # Handle dimensions like in infer.py
            if img_array.ndim == 2:
                img_array = img_array[:, :, np.newaxis]
            elif img_array.ndim == 3 and img_array.shape[2] == 1:
                pass  # Already has single channel dimension
            else:
                img_array = np.mean(img_array, axis=2, keepdims=True)

            data_images.append(img_array)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

    print(f"Successfully loaded {len(data_images)} validation images")
    return data_images
