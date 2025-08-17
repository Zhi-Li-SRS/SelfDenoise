import os
import glob
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from skimage import io
from tqdm import tqdm


class ImageDataset(Dataset):
    """Image dataset optimized for TIFF files with scikit-image loading"""

    def __init__(self, data_dir, patch=128):
        super().__init__()
        self.data_dir = data_dir
        self.patch = patch
        # Prioritize TIFF files, fallback to other formats
        extensions = ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp"]
        self.image_files = []
        for ext in extensions:
            self.image_files.extend(glob.glob(os.path.join(self.data_dir, ext)))
            self.image_files.extend(glob.glob(os.path.join(self.data_dir, ext.upper())))
        self.image_files.sort()
        print(f"Found {len(self.image_files)} samples for training")

    def __getitem__(self, index):
        fn = self.image_files[index]  # get image file name
        file_ext = os.path.splitext(fn)[1].lower()
        
        try:
            # Use scikit-image for TIFF files for better handling
            if file_ext in ['.tif', '.tiff']:
                im = io.imread(fn)
                # Ensure RGB format for TIFF files
                if len(im.shape) == 2:  # Grayscale
                    im = np.stack([im, im, im], axis=-1)
                elif len(im.shape) == 3 and im.shape[2] == 4:  # RGBA
                    im = im[:, :, :3]
                elif len(im.shape) == 3 and im.shape[2] == 1:  # Single channel
                    im = np.repeat(im, 3, axis=2)
                im = im.astype(np.float32)
            else:
                # Use PIL for other formats
                im = Image.open(fn)
                if im.mode != "RGB":
                    im = im.convert("RGB")
                im = np.array(im, dtype=np.float32)
                
        except Exception as e:
            print(f"Error loading image {fn}: {e}")
            # Fallback to cv2
            im = cv2.imread(fn, cv2.IMREAD_COLOR)
            if im is None:
                raise ValueError(f"Cannot load image: {fn}")
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im.astype(np.float32)
            
        H, W = im.shape[:2]
        
        # Random crop to patch size
        if H - self.patch > 0:
            xx = np.random.randint(0, H - self.patch)
            im = im[xx : xx + self.patch, :, :]
        if W - self.patch > 0:
            yy = np.random.randint(0, W - self.patch)
            im = im[:, yy : yy + self.patch, :]

        # Convert to tensor
        transformer = transforms.Compose([transforms.ToTensor()])
        im = transformer(im)
        return im

    def __len__(self):
        return len(self.image_files)


def load_validation_data(dataset_dir):
    """Load validation dataset optimized for TIFF files"""
    # Support for .tif, .tiff, .png, .jpg, .jpeg files
    extensions = ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp"]
    fns = []
    for ext in extensions:
        fns.extend(glob.glob(os.path.join(dataset_dir, ext)))
        fns.extend(glob.glob(os.path.join(dataset_dir, ext.upper())))
    fns.sort()
    images = []

    # Use progress bar for loading images
    pbar = tqdm(fns, desc="📂 Loading validation images", unit="img", ncols=80)
    
    for fn in pbar:
        file_ext = os.path.splitext(fn)[1].lower()
        pbar.set_postfix({'format': file_ext.upper()})

        try:
            if file_ext in [".tif", ".tiff"]:
                im = io.imread(fn)
                # Ensure RGB format
                if len(im.shape) == 2:  # Grayscale
                    im = np.stack([im, im, im], axis=-1)
                elif len(im.shape) == 3 and im.shape[2] == 4:  # RGBA
                    im = im[:, :, :3]
                elif len(im.shape) == 3 and im.shape[2] == 1:  # Single channel
                    im = np.repeat(im, 3, axis=2)
                im = im.astype(np.float32)
                # TIFF files are kept in their original range (no normalization)
            else:
                im = Image.open(fn)
                if im.mode != "RGB":
                    im = im.convert("RGB")
                im = np.array(im, dtype=np.float32)
                # For 8-bit images (PNG/JPG), normalize to 0-1 if needed
                if im.max() > 1.0:
                    im = im / 255.0

        except Exception as e:
            print(f"Error loading validation image {fn}: {e}")
            # Try with cv2 as fallback
            im = cv2.imread(fn, cv2.IMREAD_COLOR)
            if im is None:
                print(f"Skipping image: {fn}")
                continue
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im.astype(np.float32)
            if im.max() > 1.0:
                im = im / 255.0

        images.append(im)
    
    pbar.close()

    print(f"Loaded {len(images)} validation images")
    return images
