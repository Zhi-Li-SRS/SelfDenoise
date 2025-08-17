# Self-Supervised Denoising of TPF and SRS images by Transformer-based Blind Spot model

**Official Pytorch implementation of the model.**

## Preparing Training Dataset

The field of view of each THG image is 512 × 512 and the intensities of all images were scaled to [0, 255].Because of limited computational resource, each THG image having 512 × 512 pixels was divided into 2×2 smaller images having 256 × 256 pixels.

## Training

To train a network, run:

```bash
python train.py 
```
- selected optional arguments:
  - `data_dir` Path to the training set
  - `val_dirs` Path to the validation sets
  - `noisetype` Distribution of image noise, choosing from `gauss25`, `gauss5_50`, `poisson30`, or `poisson5_50`
  - `save_model_path` Base-path to the saved files
  - `log_name` Path to the saved files
  


