# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch implementation of "Self-Supervised Denoising of TPF and SRS images by Transformer-based Blind Spot model". The project implements a U-Net style transformer architecture (biuformer) for image denoising using self-supervised learning with blind spot masking.

## Core Architecture

The project consists of four main Python files:

- **model.py**: Contains the complete transformer-based U-Net architecture (biuformer) with bidirectional transformer blocks, patch embedding, upsampling modules, and the main uformer wrapper class
- **train.py**: Main training script with masking strategies, loss functions, and validation loops
- **test.py**: Inference/testing script for evaluating trained models
- **utils.py**: Utility functions for logging, file operations, and random seed management

## Key Components

### Model Architecture (model.py)
- `biuformer`: Main U-Net style transformer with encoder-decoder structure
- `biTransformerBlock`: Core transformer block with window-based attention
- `Masker`: Implements blind spot masking strategy for self-supervised training
- Supports 256x256 input images with 4x4 patch embedding

### Training System (train.py)
- Uses blind spot masking with 4x4 blocks for self-supervised learning
- Implements dual loss: regularization loss (diff²) and reversibility loss
- Dynamic beta weighting that increases during training
- Supports noise types: 'gauss25', 'gauss5_50', 'poisson30', 'poisson5_50'
- Validation using PSNR and SSIM metrics

## Commands

### Training
```bash
python train.py
```

Optional arguments:
- `--data_dir`: Path to training dataset (default: ./data/train/Imagenet_val)
- `--val_dirs`: Path to validation dataset (default: ./data/validation)
- `--noisetype`: Noise distribution ['gauss25', 'gauss5_50', 'poisson30', 'poisson5_50']
- `--save_model_path`: Directory for saved models (default: ./experiments/results)
- `--log_name`: Experiment name (default: THG-2023-12)
- `--n_epoch`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 5e-7)
- `--patchsize`: Training patch size (default: 256)

### Testing
```bash
python test.py
```

Optional arguments:
- `--checkpoint`: Path to trained model (default: experiments/results/models/epoch_model_500.pth)
- `--test_dirs`: Path to test dataset (default: ./data/validation)
- `--save_test_path`: Output directory (default: ./test)
- `--beta`: Beta parameter for combining outputs (default: 20.0)

## Data Requirements

- Training images should be 512×512 pixels, divided into 256×256 patches
- Images are normalized to [0, 255] range
- Validation dataset should be in `./data/validation/data_THG/` directory
- Supports common image formats (PNG, JPG, TIFF)

## Model Outputs

The model produces three types of outputs:
1. **dn**: Denoised output from masked input
2. **exp**: Expected output from full input  
3. **mid**: Combined output using weighted average: (dn + beta*exp) / (1 + beta)

## Dependencies

Key dependencies (based on imports):
- torch, torchvision
- PIL (Image processing)
- opencv-python (cv2)
- numpy
- timm (for transformer components)
- thop (for model profiling)

## File Import Structure

- `train.py` imports from `TBS` module (should contain uformer class)
- `test.py` imports from `TBS` module
- Both training and testing scripts import utilities from `utils.py`
- The model architecture assumes `TBS.py` contains the uformer implementation

## Notes

- The project uses CUDA for GPU acceleration
- Models are saved every `n_snapshot` epochs during training
- Validation includes both PSNR and SSIM metrics
- The masking strategy uses 4×4 blocks with interpolation for blind spots