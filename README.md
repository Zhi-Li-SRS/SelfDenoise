# Self-Supervised Denoising of Tiff images by UNet-based Blind Spot model (Unet-Transfoermr is optional)

**Official Pytorch implementation of the model.**

## Preparing Training Dataset

The field of view of each image is 512 × 512 and the intensities of all images were normalized during the training.


## Installation

Install the package in editable mode for development:

```bash
pip install -e .
```

Or install with dependencies from `requirements.txt` first:

```bash
pip install -r requirements.txt
pip install -e .
```

Note: Installing PyTorch may require selecting the right extra index for your CUDA version. See https://pytorch.org/get-started/locally/ if the default wheel does not match your environment.

## CLI Usage

After installation, the following commands are available:

- `selfdenoise-train`: runs the training entrypoint (same args as `python train.py`).
- `selfdenoise-infer`: runs inference on a folder of `.tif` images.

Examples:
```bash
# Train
selfdenoise-train --data_dir ./data/train --val_dirs ./data/validation --n_epoch 100

# Inference
selfdenoise-infer --test_dir ./data/test --checkpoint ./ckpt/checkpoint.pth --output_dir ./predict
```
