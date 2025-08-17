"""
Utility functions for image processing, evaluation, and model management.
"""
import os
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
from collections import OrderedDict


def ssim(prediction, target):
    """Calculate SSIM between two single channel images."""
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
    Calculate SSIM between two images.
    Compatible with MATLAB's implementation.
    
    Args:
        target: Target image [0, 255]
        ref: Reference image [0, 255]
    
    Returns:
        float: SSIM value
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
    """
    Calculate PSNR between two images.
    
    Args:
        target: Target image
        ref: Reference image  
        data_range: Maximum possible pixel value (default 255.0)
    
    Returns:
        float: PSNR value in dB
    """
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(data_range**2 / np.mean(np.square(diff)))
    return psnr


def setup_logger(logger_name, save_dir, phase, level=logging.INFO, screen=False, tofile=False):
    """
    Set up logger configuration.
    
    Args:
        logger_name: Name of the logger
        save_dir: Directory to save log files
        phase: Phase name for log file
        level: Logging level
        screen: Whether to log to screen
        tofile: Whether to log to file
    """
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(save_dir, f'{phase}_{logger_name}.log')
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def save_network(network, epoch, name, save_path):
    """
    Save network weights to checkpoint file.
    
    Args:
        network: PyTorch network
        epoch: Current epoch number
        name: Model name
        save_path: Directory to save the model
    """
    save_dir = os.path.join(save_path, "models")
    os.makedirs(save_dir, exist_ok=True)
    model_name = f"epoch_{name}_{epoch:03d}.pth"
    save_file = os.path.join(save_dir, model_name)
    
    if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
        network = network.module
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_file)
    
    logger = logging.getLogger("train")
    logger.info(f"Checkpoint saved to {save_file}")


def load_network(load_path, network, strict=True):
    """
    Load network weights from checkpoint file.
    
    Args:
        load_path: Path to checkpoint file
        network: PyTorch network to load weights into
        strict: Whether to strictly enforce state_dict keys match
        
    Returns:
        Network with loaded weights
    """
    assert load_path is not None
    logger = logging.getLogger("train")
    logger.info(f"Loading model from [{load_path}] ...")
    
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


def load_network_with_mapping(load_path, network, strict=True):
    """
    Load network weights with parameter name mapping for compatibility.
    Based on infer.py implementation for loading old checkpoint format.
    
    Args:
        load_path: Path to checkpoint file
        network: PyTorch network to load weights into
        strict: Whether to strictly enforce state_dict keys match
        
    Returns:
        Network with loaded weights
    """
    assert load_path is not None
    print(f"Loading model from {load_path}")

    if isinstance(network, nn.DataParallel):
        network = network.module

    load_net = torch.load(load_path, map_location="cpu")
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith("module."):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v

    # Map old parameter names to new parameter names
    param_mapping = {
        "head.0.block.0.weight": "input_conv.0.conv_block.0.weight",
        "head.0.block.0.bias": "input_conv.0.conv_block.0.bias",
        "head.1.block.0.weight": "input_conv.1.conv_block.0.weight",
        "head.1.block.0.bias": "input_conv.1.conv_block.0.bias",
        "last.0.block.0.weight": "output_conv.0.conv_block.0.weight",
        "last.0.block.0.bias": "output_conv.0.conv_block.0.bias",
        "last.1.block.0.weight": "output_conv.1.conv_block.0.weight",
        "last.1.block.0.bias": "output_conv.1.conv_block.0.bias",
        "last.2.weight": "output_conv.2.weight",
        "last.2.bias": "output_conv.2.bias",
    }

    # Map down_path to encoder_blocks
    for i in range(5):
        param_mapping[f"down_path.{i}.block.0.weight"] = f"encoder_blocks.{i}.conv_block.0.weight"
        param_mapping[f"down_path.{i}.block.0.bias"] = f"encoder_blocks.{i}.conv_block.0.bias"

    # Map up_path to decoder_blocks
    for i in range(5):
        param_mapping[f"up_path.{i}.conv_1.block.0.weight"] = f"decoder_blocks.{i}.conv1.conv_block.0.weight"
        param_mapping[f"up_path.{i}.conv_1.block.0.bias"] = f"decoder_blocks.{i}.conv1.conv_block.0.bias"
        param_mapping[f"up_path.{i}.conv_2.block.0.weight"] = f"decoder_blocks.{i}.conv2.conv_block.0.weight"
        param_mapping[f"up_path.{i}.conv_2.block.0.bias"] = f"decoder_blocks.{i}.conv2.conv_block.0.bias"

    # Apply parameter name mapping
    mapped_state_dict = OrderedDict()
    for old_key, tensor in load_net_clean.items():
        new_key = param_mapping.get(old_key, old_key)
        mapped_state_dict[new_key] = tensor

    network.load_state_dict(mapped_state_dict, strict=strict)
    return network


def save_state(epoch, optimizer, scheduler, save_path):
    """
    Save training state during training for resuming.
    
    Args:
        epoch: Current epoch number
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        save_path: Directory to save training state
    """
    save_dir = os.path.join(save_path, "training_states")
    os.makedirs(save_dir, exist_ok=True)
    state = {"epoch": epoch, "scheduler": scheduler.state_dict(), "optimizer": optimizer.state_dict()}
    save_filename = f"{epoch}.state"
    save_file = os.path.join(save_dir, save_filename)
    torch.save(state, save_file)


def resume_state(load_path, optimizer, scheduler):
    """
    Resume optimizers and schedulers for training.
    
    Args:
        load_path: Path to training state file
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        
    Returns:
        Tuple of (epoch, optimizer, scheduler)
    """
    resume_state = torch.load(load_path)
    epoch = resume_state["epoch"]
    resume_optimizer = resume_state["optimizer"]
    resume_scheduler = resume_state["scheduler"]
    optimizer.load_state_dict(resume_optimizer)
    scheduler.load_state_dict(resume_scheduler)
    return epoch, optimizer, scheduler