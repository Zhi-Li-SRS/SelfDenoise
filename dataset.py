import os
import glob
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """General image dataset for training"""

    def __init__(self, data_dir, patch=256):
        super().__init__()
        self.data_dir = data_dir
        self.patch = patch
        extensions = ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp"]
        self.image_files = []
        for ext in extensions:
            self.image_files.extend(glob.glob(os.path.join(self.data_dir, ext)))
            self.image_files.extend(glob.glob(os.path.join(self.data_dir, ext.upper())))
        self.image_files.sort()
        print(f"Found {len(self.image_files)} samples for training")

    def __getitem__(self, index):
        fn = self.image_files[index]  # get image file name
        try:
            im = Image.open(fn)
            if im.mode != "RGB":
                im = im.convert("RGB")
            im = np.array(im, dtype=np.float32)
        except Exception as e:
            print(f"Error loading image {fn}: {e}")
            im = cv2.imread(fn, cv2.IMREAD_COLOR)
            if im is None:
                raise ValueError(f"Cannot load image: {fn}")
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im.astype(np.float32)
        H, W = im.shape[:2]
        if H - self.patch > 0:
            xx = np.random.randint(0, H - self.patch)
            im = im[xx : xx + self.patch, :, :]
        if W - self.patch > 0:
            yy = np.random.randint(0, W - self.patch)
            im = im[:, yy : yy + self.patch, :]

        transformer = transforms.Compose([transforms.ToTensor()])
        im = transformer(im)
        return im

    def __len__(self):
        return len(self.image_files)


def load_validation_data(dataset_dir):
    """Load validation dataset"""
    # Support for .tif, .tiff, .png, .jpg, .jpeg files
    extensions = ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp"]
    fns = []
    for ext in extensions:
        fns.extend(glob.glob(os.path.join(dataset_dir, ext)))
        fns.extend(glob.glob(os.path.join(dataset_dir, ext.upper())))
    fns.sort()
    images = []
    for fn in fns:
        try:
            im = Image.open(fn)
            # Convert to RGB if needed (handle different TIFF modes)
            if im.mode != "RGB":
                im = im.convert("RGB")
            im = np.array(im, dtype=np.float32)
        except Exception as e:
            print(f"Error loading validation image {fn}: {e}")
            # Try with cv2 as fallback for problematic TIFF files
            im = cv2.imread(fn, cv2.IMREAD_COLOR)
            if im is None:
                print(f"Skipping image: {fn}")
                continue
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im.astype(np.float32)
        images.append(im)
    print(f"Loaded {len(images)} validation images")
    return images
