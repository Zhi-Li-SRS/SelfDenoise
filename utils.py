import logging
import os
import random
import sys
import time
from collections import OrderedDict
from datetime import datetime
from shutil import get_terminal_size
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import glob
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import yaml

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader


def OrderedYaml():
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def get_timestamp():
    return datetime.now().strftime("%y%m%d-%H%M%S")


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + "_archived_" + get_timestamp()
        print("Path already exists. Rename it to [{:s}]".format(new_name))
        logger = logging.getLogger("base")
        logger.info("Path already exists. Rename it to [{:s}]".format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(
    logger_name, root, phase, level=logging.INFO, screen=False, tofile=False
):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + "_{}.log".format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


# Network utilities
def save_network(network, save_path, epoch, name, logger=None):
    """Save network checkpoint"""
    model_dir = os.path.join(save_path, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_name = "epoch_{}_{:03d}.pth".format(name, epoch)
    model_path = os.path.join(model_dir, model_name)
    
    if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
        network = network.module
    
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    
    torch.save(state_dict, model_path)
    if logger:
        logger.info("Checkpoint saved to {}".format(model_path))
    else:
        print("Checkpoint saved to {}".format(model_path))
    
    return model_path


def load_network(load_path, network, strict=True, logger=None):
    """Load network checkpoint"""
    assert load_path is not None
    if logger:
        logger.info("Loading model from [{:s}] ...".format(load_path))
    else:
        print("Loading model from [{:s}] ...".format(load_path))
    
    if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
        network = network.module
    
    load_net = torch.load(load_path)
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith("module."):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    
    network.load_state_dict(load_net_clean, strict=strict)
    return network


def resume_state(load_path, optimizer, scheduler):
    """Resume the optimizers and schedulers for training"""
    resume_state = torch.load(load_path)
    epoch = resume_state["epoch"]
    resume_optimizer = resume_state["optimizer"]
    resume_scheduler = resume_state["scheduler"]
    optimizer.load_state_dict(resume_optimizer)
    scheduler.load_state_dict(resume_scheduler)
    return epoch, optimizer, scheduler


# Global operation counter for reproducible random generation
operation_seed_counter = 0


def get_generator(device="cuda"):
    """Get a reproducible random generator"""
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device=device)
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


# Tensor manipulation utilities
def space_to_depth(x, block_size):
    """Convert space to depth"""
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size, w // block_size)


def depth_to_space(x, block_size):
    """Convert depth to space"""
    return torch.nn.functional.pixel_shuffle(x, block_size)


# Masking utilities
def generate_mask(img, width=4, mask_type="all"):
    """Generate random masks for blind spot training"""
    n, c, h, w = img.shape
    mask = torch.zeros(size=(n * h // width * w // width * width**2,), dtype=torch.int64, device=img.device)
    idx_list = torch.arange(0, width**2, 1, dtype=torch.int64, device=img.device)
    rd_idx = torch.zeros(size=(n * h // width * w // width,), dtype=torch.int64, device=img.device)

    if mask_type == "all":
        rd_idx = torch.randint(
            low=0,
            high=len(idx_list),
            size=(1,),
            device=img.device,
            generator=get_generator(device=img.device),
        ).repeat(n * h // width * w // width)
    elif "fix" in mask_type:
        index = mask_type.split("_")[-1]
        index = torch.from_numpy(np.array(index).astype(np.int64)).type(torch.int64)
        rd_idx = index.repeat(n * h // width * w // width).to(img.device)

    rd_pair_idx = idx_list[rd_idx]
    rd_pair_idx += torch.arange(
        start=0,
        end=n * h // width * w // width * width**2,
        step=width**2,
        dtype=torch.int64,
        device=img.device,
    )

    mask[rd_pair_idx] = 1

    mask = depth_to_space(
        mask.type_as(img).view(n, h // width, w // width, width**2).permute(0, 3, 1, 2), block_size=width
    ).type(torch.int64)

    return mask


def interpolate_mask(tensor, mask, mask_inv):
    """Apply interpolation masking"""
    n, c, h, w = tensor.shape
    device = tensor.device
    mask = mask.to(device)
    kernel_size = 3

    filtered_tensor = F.max_pool2d(tensor.view(n * c, 1, h, w), kernel_size=kernel_size, stride=1, padding=1)

    return filtered_tensor.view_as(tensor) * mask + tensor * mask_inv


class Masker(object):
    """Blind spot masker for self-supervised training"""
    
    def __init__(self, width=4, mode="interpolate", mask_type="all"):
        self.width = width
        self.mode = mode
        self.mask_type = mask_type

    def mask(self, img, mask_type=None, mode=None):
        """Generate masked images given random masks"""
        if mode is None:
            mode = self.mode
        if mask_type is None:
            mask_type = self.mask_type

        n, c, h, w = img.shape
        mask = generate_mask(img, width=self.width, mask_type=mask_type)
        mask_inv = torch.ones(mask.shape).to(img.device) - mask
        if mode == "interpolate":
            masked = interpolate_mask(img, mask, mask_inv)
        else:
            raise NotImplementedError

        net_input = masked
        return net_input, mask

    def train(self, img):
        """Generate training masks for all possible positions"""
        n, c, h, w = img.shape
        tensors = torch.zeros((n, self.width**2, c, h, w), device=img.device)
        masks = torch.zeros((n, self.width**2, 1, h, w), device=img.device)
        for i in range(self.width**2):
            x, mask = self.mask(img, mask_type="fix_{}".format(i))
            tensors[:, i, ...] = x
            masks[:, i, ...] = mask
        tensors = tensors.view(-1, c, h, w)
        masks = masks.view(-1, 1, h, w)
        return tensors, masks


# Dataset utilities
class ImageDataset(Dataset):
    """General image dataset for training"""
    
    def __init__(self, data_dir, patch=256):
        super(ImageDataset, self).__init__()
        self.data_dir = data_dir
        self.patch = patch
        self.image_files = glob.glob(os.path.join(self.data_dir, "*"))
        self.image_files.sort()
        print("Found {} samples for training".format(len(self.image_files)))

    def __getitem__(self, index):
        # Fetch image
        fn = self.image_files[index]
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        
        # Random crop
        H, W = im.shape[:2]
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
    """Load validation dataset"""
    fns = glob.glob(os.path.join(dataset_dir, "*"))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images


# Evaluation metrics
def ssim(prediction, target):
    """Calculate SSIM between two images"""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def calculate_ssim(target, ref):
    """
    Calculate SSIM - same outputs as MATLAB's
    img1, img2: [0, 255]
    """
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")


def calculate_psnr(target, ref, data_range=255.0):
    """Calculate PSNR between two images"""
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(data_range**2 / np.mean(np.square(diff)))
    return psnr


