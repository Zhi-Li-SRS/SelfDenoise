import os
import glob
import numpy as np
from skimage import io
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from collections import OrderedDict

from model import UNet


class Masker(object):
    """Masking utility for self-supervised denoising."""

    def __init__(self, width=4, mode="interpolate", mask_type="all"):
        self.width = width
        self.mode = mode
        self.mask_type = mask_type

    def generate_mask(self, img, width=4, mask_type="random"):
        """Generate random masks for self-supervised learning."""
        n, c, h, w = img.shape
        mask = torch.zeros(
            size=(n * h // width * w // width * width**2,), dtype=torch.int64, device=img.device
        )
        idx_list = torch.arange(0, width**2, 1, dtype=torch.int64, device=img.device)
        rd_idx = torch.zeros(size=(n * h // width * w // width,), dtype=torch.int64, device=img.device)

        if mask_type == "random":
            torch.randint(
                low=0, high=len(idx_list), size=(n * h // width * w // width,), device=img.device, out=rd_idx
            )
        elif mask_type == "all":
            rd_idx = torch.randint(low=0, high=len(idx_list), size=(1,), device=img.device).repeat(
                n * h // width * w // width
            )
        elif "fix" in mask_type:
            index = int(mask_type.split("_")[-1])
            rd_idx.fill_(index)

        rd_pair_idx = idx_list[rd_idx]
        rd_pair_idx += torch.arange(
            start=0,
            end=n * h // width * w // width * width**2,
            step=width**2,
            dtype=torch.int64,
            device=img.device,
        )

        mask[rd_pair_idx] = 1
        mask = torch.nn.functional.pixel_shuffle(
            mask.type_as(img).view(n, h // width, w // width, width**2).permute(0, 3, 1, 2),
            upscale_factor=width,
        ).type(torch.int64)

        return mask

    def interpolate_mask(self, tensor, mask, mask_inv):
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

    def mask(self, img, mask_type=None, mode=None):
        """Apply masking to input image."""
        if mode is None:
            mode = self.mode
        if mask_type is None:
            mask_type = self.mask_type

        mask = self.generate_mask(img, width=self.width, mask_type=mask_type)
        mask_inv = torch.ones(mask.shape).to(img.device) - mask

        if mode == "interpolate":
            masked = self.interpolate_mask(img, mask, mask_inv)
        else:
            raise NotImplementedError

        return masked, mask

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


def load_network(load_path, network, strict=True):
    """Load pre-trained network weights."""
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


def process_tif_images(
    test_dir="data/test", ckpt_path="ckpt/checkpoint.pth", output_dir="predict", beta=20.0
):
    """
    Process .tif images from test directory and save denoised results.

    Args:
        test_dir: Directory containing .tif test images
        ckpt_path: Path to model checkpoint
        output_dir: Directory to save processed images
        beta: Beta parameter for combining predictions
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    tif_files = glob.glob(os.path.join(test_dir, "*.tif"))
    tif_files.sort()

    if not tif_files:
        print(f"No .tif files found in {test_dir}")
        return

    print(f"Found {len(tif_files)} .tif files to process")

    # Initialize model for single channel (grayscale) images
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use new parameter names but map from old checkpoint
    model = UNet(in_channels=1, out_channels=1, depth=5, base_filters=48).to(device)
    model = load_network(ckpt_path, model, strict=True)
    model.eval()

    # Initialize masker
    masker = Masker(width=4, mode="interpolate", mask_type="all")

    # Process each image
    for img_path in tqdm(tif_files, desc="Processing images"):
        filename = os.path.basename(img_path)
        save_name = os.path.splitext(filename)[0]

        img_array = io.imread(img_path).astype(np.float32)

        if img_array.ndim == 2:
            img_array = img_array[:, :, np.newaxis]
        elif img_array.ndim == 3 and img_array.shape[2] == 1:
            pass
        else:
            print(f"Unexpected image shape {img_array.shape} for {filename}, skipping...")
            continue

        h, w = img_array.shape[:2]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32

        if pad_h > 0 or pad_w > 0:
            img_array = np.pad(img_array, [[0, pad_h], [0, pad_w], [0, 0]], mode="reflect")

        # Normalize to [0,1] based on per-image min/max (consistent with training/validation)
        img_max = img_array.max()
        img_min = img_array.min()
        img_range = img_max - img_min
        if img_range > 0:
            img_normalized = (img_array - img_min) / img_range
        else:
            img_normalized = img_array - img_min

        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(img_normalized).unsqueeze(0).to(device)

        # Process with model
        with torch.no_grad():
            n, c, tensor_h, tensor_w = img_tensor.shape

            # Self-supervised prediction
            net_input, mask = masker.train(img_tensor)
            self_supervised_output = (model(net_input) * mask).view(n, -1, c, tensor_h, tensor_w).sum(dim=1)

            direct_output = model(img_tensor)

            # Combined prediction
            combined_output = (self_supervised_output + beta * direct_output) / (1 + beta)

        # Convert back to numpy and crop to original size
        result = combined_output[:, :, :h, :w]
        result = result.permute(0, 2, 3, 1).cpu().data.numpy().squeeze()

        # Denormalize back to original intensity range
        if img_range > 0:
            result_output = (result * img_range + img_min).astype(np.float32)
        else:
            result_output = (result + img_min).astype(np.float32)

        # Save as 32-bit float .tif using scikit-image
        output_path = os.path.join(output_dir, f"{save_name}_denoised.tif")
        io.imsave(output_path, result_output)

    print(f"Processing complete! Results saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Denoise .tif images")
    parser.add_argument(
        "--test_dir", type=str, default="data/test", help="Directory containing .tif test images"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="ckpt/checkpoint.pth", help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_dir", type=str, default="predict", help="Directory to save processed images"
    )
    parser.add_argument("--beta", type=float, default=20.0, help="Beta parameter for combining predictions")

    args = parser.parse_args()

    process_tif_images(
        test_dir=args.test_dir, ckpt_path=args.checkpoint, output_dir=args.output_dir, beta=args.beta
    )
