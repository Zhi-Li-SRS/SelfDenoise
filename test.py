import os
import logging
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import utils
import dataset
from model import uformer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test self-supervised denoising model")

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/results/models/epoch_model_500.pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--noisetype",
        type=str,
        default="poisson30",
        choices=["poisson30", "gauss5_50", "poisson30", "poisson5_50"],
        help="Noise type (for compatibility)",
    )

    # Data arguments
    parser.add_argument("--test_dirs", type=str, default="./data/validation", help="Path to test dataset")

    # Output arguments
    parser.add_argument(
        "--save_test_path", type=str, default="./test", help="Output directory for test results"
    )
    parser.add_argument("--log_name", type=str, default="SRS", help="Experiment name")

    # Testing arguments
    parser.add_argument("--beta", type=float, default=20.0, help="Beta parameter for combining outputs")

    # Hardware arguments
    parser.add_argument("--gpu_devices", default="0", type=str, help="GPU device IDs")
    parser.add_argument("--parallel", action="store_true", help="Use data parallel")

    return parser.parse_args()


def setup_testing(args):
    """Setup testing environment"""
    # Set up CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    torch.set_num_threads(8)

    # Create output directory
    os.makedirs(args.save_test_path, exist_ok=True)

    # Set up logger
    utils.setup_logger(
        "test", args.save_test_path, "test_" + args.log_name, level=logging.INFO, screen=True, tofile=True
    )
    logger = logging.getLogger("test")

    return logger


def create_model(args, logger):
    """Create and load trained model"""
    network = uformer(args)

    if args.parallel:
        network = torch.nn.DataParallel(network)
    network = network.cuda()

    # Load trained weights
    network = utils.load_network(args.checkpoint, network, strict=False, logger=logger)
    network.eval()

    return network


def load_test_data(args):
    """Load test dataset"""
    data_dir = os.path.join(args.test_dirs, "data_srs")
    test_dict = {"data_srs": dataset.load_validation_data(data_dir)}
    return test_dict


def process_image_patch_based(image, network, masker, beta, patch_size=128, overlap=32):
    """Process a large image using patch-based inference with overlap"""
    origin255 = image.copy().astype(np.uint8)
    im = np.array(image, dtype=np.float32) / 255.0
    
    H, W = im.shape[:2]
    
    # Initialize output arrays
    pred_dn_full = np.zeros_like(im)
    pred_exp_full = np.zeros_like(im)
    weight_map = np.zeros((H, W), dtype=np.float32)
    
    # Calculate stride (patch_size - overlap)
    stride = patch_size - overlap
    
    transformer = transforms.Compose([transforms.ToTensor()])
    
    with torch.no_grad():
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                # Calculate patch boundaries
                y_end = min(y + patch_size, H)
                x_end = min(x + patch_size, W)
                y_start = y_end - patch_size if y_end - y >= patch_size else max(0, y_end - patch_size)
                x_start = x_end - patch_size if x_end - x >= patch_size else max(0, x_end - patch_size)
                
                # Extract patch
                patch = im[y_start:y_start+patch_size, x_start:x_start+patch_size]
                
                # Pad if necessary
                if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                    pad_h = patch_size - patch.shape[0]
                    pad_w = patch_size - patch.shape[1]
                    patch = np.pad(patch, [[0, pad_h], [0, pad_w], [0, 0]], "reflect")
                
                # Convert to tensor
                patch_tensor = transformer(patch).unsqueeze(0).cuda()
                
                # Process patch
                n, c, h, w = patch_tensor.shape
                net_input, mask = masker.train(patch_tensor)
                patch_dn = (network(net_input) * mask).view(n, -1, c, h, w).sum(dim=1)
                patch_exp = network(patch_tensor)
                
                # Convert back to numpy
                patch_dn = patch_dn.permute(0, 2, 3, 1).cpu().data.clamp(0, 1).numpy().squeeze(0)
                patch_exp = patch_exp.permute(0, 2, 3, 1).cpu().data.clamp(0, 1).numpy().squeeze(0)
                
                # Calculate valid region (excluding padding)
                valid_h = min(patch_size, H - y_start)
                valid_w = min(patch_size, W - x_start)
                
                # Add to full image with weight
                weight = np.ones((valid_h, valid_w), dtype=np.float32)
                
                # Apply overlapping weight (cosine taper)
                if overlap > 0:
                    # Taper edges
                    taper_size = min(overlap // 2, valid_h // 4, valid_w // 4)
                    if taper_size > 0:
                        for i in range(taper_size):
                            weight_val = 0.5 * (1 + np.cos(np.pi * i / taper_size))
                            if y_start > 0:  # Top edge
                                weight[i, :] *= weight_val
                            if x_start > 0:  # Left edge
                                weight[:, i] *= weight_val
                            if y_start + valid_h < H:  # Bottom edge
                                weight[valid_h-1-i, :] *= weight_val
                            if x_start + valid_w < W:  # Right edge
                                weight[:, valid_w-1-i] *= weight_val
                
                # Accumulate results
                pred_dn_full[y_start:y_start+valid_h, x_start:x_start+valid_w] += patch_dn[:valid_h, :valid_w] * weight[:, :, np.newaxis]
                pred_exp_full[y_start:y_start+valid_h, x_start:x_start+valid_w] += patch_exp[:valid_h, :valid_w] * weight[:, :, np.newaxis]
                weight_map[y_start:y_start+valid_h, x_start:x_start+valid_w] += weight
    
    # Normalize by weight
    weight_map[weight_map == 0] = 1  # Avoid division by zero
    pred_dn_full = pred_dn_full / weight_map[:, :, np.newaxis]
    pred_exp_full = pred_exp_full / weight_map[:, :, np.newaxis]
    
    # Combine outputs
    pred_mid_full = (pred_dn_full + beta * pred_exp_full) / (1 + beta)
    
    return pred_dn_full, pred_exp_full, pred_mid_full


def process_image(image, network, masker, beta):
    """Process a single image through the model - updated for large images"""
    origin255 = image.copy().astype(np.uint8)
    H, W = origin255.shape[:2]
    
    # Use patch-based inference for large images
    if H > 256 or W > 256:
        pred_dn, pred_exp, pred_mid = process_image_patch_based(image, network, masker, beta)
    else:
        # Original method for small images
        im = np.array(image, dtype=np.float32) / 255.0
        noisy_im = im

        # Pad to training patch size
        patch_size = 128
        if H < patch_size or W < patch_size:
            pad_h = max(0, patch_size - H)
            pad_w = max(0, patch_size - W)
            noisy_im = np.pad(noisy_im, [[0, pad_h], [0, pad_w], [0, 0]], "reflect")

        # Convert to tensor
        transformer = transforms.Compose([transforms.ToTensor()])
        noisy_im = transformer(noisy_im).unsqueeze(0).cuda()

        with torch.no_grad():
            n, c, h, w = noisy_im.shape
            net_input, mask = masker.train(noisy_im)
            noisy_output = (network(net_input) * mask).view(n, -1, c, h, w).sum(dim=1)

            # Get expected output
            exp_output = network(noisy_im)

        # Crop back to original size
        pred_dn = noisy_output[:, :, :H, :W]
        pred_exp = exp_output[:, :, :H, :W]
        pred_mid = (pred_dn + beta * pred_exp) / (1 + beta)
        
        # Convert to numpy
        pred_dn = pred_dn.permute(0, 2, 3, 1).cpu().data.clamp(0, 1).numpy().squeeze(0)
        pred_exp = pred_exp.permute(0, 2, 3, 1).cpu().data.clamp(0, 1).numpy().squeeze(0)
        pred_mid = pred_mid.permute(0, 2, 3, 1).cpu().data.clamp(0, 1).numpy().squeeze(0)

    # pred_dn, pred_exp, pred_mid are already numpy arrays from either branch above

    # Convert to uint8
    pred255_dn = np.clip(pred_dn * 255.0 + 0.5, 0, 255).astype(np.uint8)
    pred255_exp = np.clip(pred_exp * 255.0 + 0.5, 0, 255).astype(np.uint8)
    pred255_mid = np.clip(pred_mid * 255.0 + 0.5, 0, 255).astype(np.uint8)

    # Calculate metrics
    psnr_dn = utils.calculate_psnr(origin255.astype(np.float32), pred255_dn.astype(np.float32))
    ssim_dn = utils.calculate_ssim(origin255.astype(np.float32), pred255_dn.astype(np.float32))

    psnr_exp = utils.calculate_psnr(origin255.astype(np.float32), pred255_exp.astype(np.float32))
    ssim_exp = utils.calculate_ssim(origin255.astype(np.float32), pred255_exp.astype(np.float32))

    psnr_mid = utils.calculate_psnr(origin255.astype(np.float32), pred255_mid.astype(np.float32))
    ssim_mid = utils.calculate_ssim(origin255.astype(np.float32), pred255_mid.astype(np.float32))

    return {
        "origin": origin255,
        "pred_dn": pred255_dn,
        "pred_exp": pred255_exp,
        "pred_mid": pred255_mid,
        "metrics": {
            "psnr_dn": psnr_dn,
            "ssim_dn": ssim_dn,
            "psnr_exp": psnr_exp,
            "ssim_exp": ssim_exp,
            "psnr_mid": psnr_mid,
            "ssim_mid": ssim_mid,
        },
    }


def test_dataset(network, test_dict, masker, args, logger):
    """Test the model on the dataset"""
    save_test_path = os.path.join(args.save_test_path, args.log_name)
    validation_path = os.path.join(save_test_path, "validation")
    os.makedirs(validation_path, exist_ok=True)

    np.random.seed(101)
    valid_repeat_times = {"data_THG": 1}  # Usually 1 for testing

    for valid_name, valid_images in test_dict.items():
        save_dir = os.path.join(validation_path, valid_name)
        os.makedirs(save_dir, exist_ok=True)
        logger.info("Processing {} dataset".format(valid_name))

        # Initialize metric collectors
        all_metrics = {
            "psnr_dn": [],
            "ssim_dn": [],
            "psnr_exp": [],
            "ssim_exp": [],
            "psnr_mid": [],
            "ssim_mid": [],
        }

        repeat_times = valid_repeat_times[valid_name]

        for i in range(repeat_times):
            for idx, image in enumerate(valid_images):
                # Process image
                results = process_image(image, network, masker, args.beta)

                # Collect metrics
                for key, value in results["metrics"].items():
                    all_metrics[key].append(value)

                # Log individual results
                logger.info(
                    "{} - img:{}_{:03d} - PSNR_DN: {:.6f} dB; SSIM_DN: {:.6f}; "
                    "PSNR_EXP: {:.6f} dB; SSIM_EXP: {:.6f}; PSNR_MID: {:.6f} dB; SSIM_MID: {:.6f}".format(
                        valid_name,
                        i,
                        idx,
                        results["metrics"]["psnr_dn"],
                        results["metrics"]["ssim_dn"],
                        results["metrics"]["psnr_exp"],
                        results["metrics"]["ssim_exp"],
                        results["metrics"]["psnr_mid"],
                        results["metrics"]["ssim_mid"],
                    )
                )

                # Save result images
                save_path = os.path.join(save_dir, "{:03d}-{:03d}_mid.png".format(i, idx))
                Image.fromarray(results["pred_mid"]).convert("RGB").save(save_path)

                # Optionally save other outputs
                if i == 0:  # Save detailed outputs for first repetition
                    Image.fromarray(results["pred_dn"]).convert("RGB").save(
                        os.path.join(save_dir, "{:03d}-{:03d}_dn.png".format(i, idx))
                    )
                    Image.fromarray(results["pred_exp"]).convert("RGB").save(
                        os.path.join(save_dir, "{:03d}-{:03d}_exp.png".format(i, idx))
                    )

        # Calculate and log average metrics
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}

        logger.info(
            "----Average PSNR/SSIM results for {}----\n"
            "\tPSNR_DN: {:.6f} dB; SSIM_DN: {:.6f}\n"
            "\tPSNR_EXP: {:.6f} dB; SSIM_EXP: {:.6f}\n"
            "\tPSNR_MID: {:.6f} dB; SSIM_MID: {:.6f}".format(
                valid_name,
                avg_metrics["psnr_dn"],
                avg_metrics["ssim_dn"],
                avg_metrics["psnr_exp"],
                avg_metrics["ssim_exp"],
                avg_metrics["psnr_mid"],
                avg_metrics["ssim_mid"],
            )
        )

        return avg_metrics


def main():
    """Main testing function"""
    args = parse_args()

    # Setup testing environment
    logger = setup_testing(args)

    # Create and load model
    model = create_model(args, logger)
    test_dict = load_test_data(args)

    # Create masker
    masker = utils.Masker(width=4, mode="interpolate", mask_type="all")

    logger.info("Starting testing...")

    # Test the model
    avg_metrics = test_dataset(model, test_dict, masker, args, logger)

    logger.info("Testing completed successfully")
    logger.info("Final average metrics: {}".format(avg_metrics))


if __name__ == "__main__":
    main()
