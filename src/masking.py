import numpy as np
import torch
import torch.nn as nn


operation_seed_counter = 0


def get_generator(device="cuda"):
    """Generate a CUDA generator with manual seed for reproducible randomness."""
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device=device)
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


class AugmentNoise(object):
    """Noise augmentation for training data."""

    def __init__(self, style):
        print(style)
        if style.startswith("gauss"):
            self.params = [float(p) / 255.0 for p in style.replace("gauss", "").split("_")]
            if len(self.params) == 1:
                self.style = "gauss_fix"
            elif len(self.params) == 2:
                self.style = "gauss_range"
        elif style.startswith("poisson"):
            self.params = [float(p) for p in style.replace("poisson", "").split("_")]
            if len(self.params) == 1:
                self.style = "poisson_fix"
            elif len(self.params) == 2:
                self.style = "poisson_range"

    def add_train_noise(self, x):
        """Add noise for training."""
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            std = std * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0.0, std=std, generator=get_generator(device=x.device), out=noise)
            return x + noise
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = torch.rand(size=(shape[0], 1, 1, 1), device=x.device) * (max_std - min_std) + min_std
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0, std=std, generator=get_generator(device=x.device), out=noise)
            return x + noise
        elif self.style == "poisson_fix":
            lam = self.params[0]
            lam = lam * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noised = torch.poisson(lam * x, generator=get_generator(device=x.device)) / lam
            return noised
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = torch.rand(size=(shape[0], 1, 1, 1), device=x.device) * (max_lam - min_lam) + min_lam
            noised = torch.poisson(lam * x, generator=get_generator(device=x.device)) / lam
            return noised

    def add_valid_noise(self, x):
        """Add noise for validation."""
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            return np.array(x + np.random.normal(size=shape) * std, dtype=np.float32)
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = np.random.uniform(low=min_std, high=max_std, size=(1, 1, 1))
            return np.array(x + np.random.normal(size=shape) * std, dtype=np.float32)
        elif self.style == "poisson_fix":
            lam = self.params[0]
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = np.random.uniform(low=min_lam, high=max_lam, size=(1, 1, 1))
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)


def space_to_depth(x, block_size):
    """Convert spatial dimensions to depth (channel) dimension."""
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size, w // block_size)


def depth_to_space(x, block_size):
    """Convert depth (channel) dimension to spatial dimensions."""
    return torch.nn.functional.pixel_shuffle(x, block_size)


def generate_mask(img, width=4, mask_type="random"):
    """Generate random masks with shape (N x C x H/2 x W/2)."""
    n, c, h, w = img.shape
    mask = torch.zeros(size=(n * h // width * w // width * width**2,), dtype=torch.int64, device=img.device)
    idx_list = torch.arange(0, width**2, 1, dtype=torch.int64, device=img.device)
    rd_idx = torch.zeros(size=(n * h // width * w // width,), dtype=torch.int64, device=img.device)

    if mask_type == "random":
        torch.randint(
            low=0,
            high=len(idx_list),
            size=(n * h // width * w // width,),
            device=img.device,
            generator=get_generator(device=img.device),
            out=rd_idx,
        )
    elif mask_type == "batch":
        rd_idx = torch.randint(
            low=0,
            high=len(idx_list),
            size=(n,),
            device=img.device,
            generator=get_generator(device=img.device),
        ).repeat(h // width * w // width)
    elif mask_type == "all":
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
    """Apply interpolation masking."""
    n, c, h, w = tensor.shape
    device = tensor.device
    mask = mask.to(device)
    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]])

    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = torch.Tensor(kernel).to(device)
    kernel = kernel / kernel.sum()

    filtered_tensor = torch.nn.functional.conv2d(tensor.view(n * c, 1, h, w), kernel, stride=1, padding=1)

    return filtered_tensor.view_as(tensor) * mask + tensor * mask_inv


class Masker(object):
    """Masking utility for self-supervised denoising."""

    def __init__(self, width=4, mode="interpolate", mask_type="all"):
        self.width = width
        self.mode = mode
        self.mask_type = mask_type

    def mask(self, img, mask_type=None, mode=None):
        """Generate masked images given random masks."""
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
        """Generate training data with all possible masks."""
        n, c, h, w = img.shape
        tensors = torch.zeros((n, self.width**2, c, h, w), device=img.device)
        masks = torch.zeros((n, self.width**2, 1, h, w), device=img.device)
        for i in range(self.width**2):
            x, mask = self.mask(img, mask_type=f"fix_{i}")
            tensors[:, i, ...] = x
            masks[:, i, ...] = mask
        tensors = tensors.view(-1, c, h, w)
        masks = masks.view(-1, 1, h, w)
        return tensors, masks
