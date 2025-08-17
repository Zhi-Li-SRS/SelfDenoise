import os
import glob
import logging
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from skimage import io
from tqdm import tqdm

import src.utils as utils
import src.dataset as dataset
from src.model import uformer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test self-supervised denoising model")

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/results/checkpoints/epoch_model_200.pth",
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
    parser.add_argument("--test_dirs", type=str, default="./data/test", help="Path to test dataset")

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
    """Load test dataset with progress indication and validation"""
    # Try different possible directory structures
    possible_dirs = [
        os.path.join(args.test_dirs, "data_srs"),  # Original expected structure
        args.test_dirs,  # Direct test directory
        os.path.join(args.test_dirs, "data_THG"),  # Alternative naming
    ]

    data_dir = None
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            # Check if directory contains image files
            test_files = []
            extensions = ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp"]
            for ext in extensions:
                test_files.extend(glob.glob(os.path.join(dir_path, ext)))
                test_files.extend(glob.glob(os.path.join(dir_path, ext.upper())))

            if test_files:
                data_dir = dir_path
                print(f"📂 Found {len(test_files)} image files in: {data_dir}")
                break

    if data_dir is None:
        raise FileNotFoundError(
            f"No test directory with images found. Tried:\n" + "\n".join([f"  - {d}" for d in possible_dirs])
        )

    print(f"\n📂 Loading test dataset from: {data_dir}")
    images, extensions = dataset.load_validation_data(data_dir)

    if len(images) == 0:
        raise ValueError(f"No images found in {data_dir}. Please check the directory path and file formats.")

    test_dict = {"data_srs": (images, extensions)}
    print(f"✅ Loaded {len(images)} test images")

    return test_dict


def process_image_patch_based(image, network, masker, beta, patch_size=128, overlap=32):
    """Process a large image using patch-based inference with overlap and progress tracking"""
    origin255 = image.copy().astype(np.uint8)
    im = np.array(image, dtype=np.float32) / 255.0

    H, W = im.shape[:2]

    # Initialize output arrays
    pred_dn_full = np.zeros_like(im)
    pred_exp_full = np.zeros_like(im)
    weight_map = np.zeros((H, W), dtype=np.float32)

    # Calculate stride (patch_size - overlap)
    stride = patch_size - overlap

    # Calculate total number of patches
    y_positions = list(range(0, H, stride))
    x_positions = list(range(0, W, stride))
    total_patches = len(y_positions) * len(x_positions)

    transformer = transforms.Compose([transforms.ToTensor()])

    # Create progress bar for patch processing
    patch_pbar = tqdm(
        total=total_patches, desc=f"🧩 Processing patches ({H}x{W})", unit="patch", leave=False, ncols=80
    )

    with torch.no_grad():
        for y in y_positions:
            for x in x_positions:
                patch_pbar.update(1)
                # Calculate patch boundaries
                y_end = min(y + patch_size, H)
                x_end = min(x + patch_size, W)
                y_start = y_end - patch_size if y_end - y >= patch_size else max(0, y_end - patch_size)
                x_start = x_end - patch_size if x_end - x >= patch_size else max(0, x_end - patch_size)

                # Extract patch
                patch = im[y_start : y_start + patch_size, x_start : x_start + patch_size]

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
                                weight[valid_h - 1 - i, :] *= weight_val
                            if x_start + valid_w < W:  # Right edge
                                weight[:, valid_w - 1 - i] *= weight_val

                # Accumulate results
                pred_dn_full[y_start : y_start + valid_h, x_start : x_start + valid_w] += (
                    patch_dn[:valid_h, :valid_w] * weight[:, :, np.newaxis]
                )
                pred_exp_full[y_start : y_start + valid_h, x_start : x_start + valid_w] += (
                    patch_exp[:valid_h, :valid_w] * weight[:, :, np.newaxis]
                )
                weight_map[y_start : y_start + valid_h, x_start : x_start + valid_w] += weight

    patch_pbar.close()

    # Normalize by weight
    weight_map[weight_map == 0] = 1  # Avoid division by zero
    pred_dn_full = pred_dn_full / weight_map[:, :, np.newaxis]
    pred_exp_full = pred_exp_full / weight_map[:, :, np.newaxis]

    # Combine outputs
    pred_mid_full = (pred_dn_full + beta * pred_exp_full) / (1 + beta)

    return pred_dn_full, pred_exp_full, pred_mid_full


def save_image_with_extension(image_array: np.ndarray, file_path: str, extension: str):
    """Save image using appropriate method based on file extension"""
    if extension.lower() in [".tif", ".tiff"]:
        # Use scikit-image for TIFF files
        io.imsave(file_path, image_array)
    else:
        Image.fromarray(image_array).convert("RGB").save(file_path)


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
    """Test the model on the dataset with enhanced progress tracking and formatting"""
    save_test_path = os.path.join(args.save_test_path, args.log_name)
    validation_path = os.path.join(save_test_path, "validation")
    os.makedirs(validation_path, exist_ok=True)

    np.random.seed(101)
    valid_repeat_times = {"data_srs": 1}  # Usually 1 for testing

    logger.info("=" * 80)
    logger.info("🚀 STARTING MODEL TESTING")
    logger.info("=" * 80)

    overall_results = {}

    for valid_name, (valid_images, file_extensions) in test_dict.items():
        save_dir = os.path.join(validation_path, valid_name)
        os.makedirs(save_dir, exist_ok=True)

        total_images = len(valid_images)

        # Validate that we have images to process
        if total_images == 0:
            logger.warning(f"⚠️  No images found in {valid_name} dataset. Skipping...")
            continue

        logger.info(f"\n📁 Processing {valid_name} dataset ({total_images} images)")
        logger.info("-" * 60)

        # Initialize metric collectors
        all_metrics = {
            "psnr_dn": [],
            "ssim_dn": [],
            "psnr_exp": [],
            "ssim_exp": [],
            "psnr_mid": [],
            "ssim_mid": [],
        }

        repeat_times = valid_repeat_times.get(valid_name, 1)

        for i in range(repeat_times):
            logger.info(f"\n🔄 Repetition {i+1}/{repeat_times}")

            # Create progress bar for this repetition
            pbar = tqdm(
                zip(valid_images, file_extensions),
                total=total_images,
                desc=f"🖼️  Processing {valid_name}",
                unit="img",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                ncols=100,
            )

            for idx, (image, file_ext) in enumerate(pbar):
                # Update progress bar description with current image info
                pbar.set_description(f"🖼️  Processing {valid_name} ({file_ext.upper()})")

                # Process image
                results = process_image(image, network, masker, args.beta)

                # Collect metrics
                for key, value in results["metrics"].items():
                    all_metrics[key].append(value)

                # Update progress bar with current metrics
                metrics = results["metrics"]
                pbar.set_postfix(
                    {"PSNR": f"{metrics['psnr_mid']:.2f}dB", "SSIM": f"{metrics['ssim_mid']:.3f}"}
                )

                # Save result images with appropriate extension
                save_path = os.path.join(save_dir, f"{i:03d}-{idx:03d}_mid{file_ext}")
                save_image_with_extension(results["pred_mid"], save_path, file_ext)

                # Optionally save other outputs
                if i == 0:  # Save detailed outputs for first repetition
                    dn_path = os.path.join(save_dir, f"{i:03d}-{idx:03d}_dn{file_ext}")
                    exp_path = os.path.join(save_dir, f"{i:03d}-{idx:03d}_exp{file_ext}")
                    save_image_with_extension(results["pred_dn"], dn_path, file_ext)
                    save_image_with_extension(results["pred_exp"], exp_path, file_ext)

            # Close progress bar
            pbar.close()

            # Log summary for this repetition
            if len(all_metrics["psnr_mid"]) > 0:
                current_avg_psnr = np.mean(all_metrics["psnr_mid"])
                current_avg_ssim = np.mean(all_metrics["ssim_mid"])
                logger.info(
                    f"✅ Repetition {i+1} completed - Avg PSNR: {current_avg_psnr:.3f}dB, Avg SSIM: {current_avg_ssim:.4f}"
                )
            else:
                logger.warning(f"⚠️  Repetition {i+1} completed but no metrics collected")

        # Validate that we have metrics before calculating statistics
        if len(all_metrics["psnr_mid"]) == 0:
            logger.error(
                f"❌ No metrics collected for {valid_name}. Check if images were processed successfully."
            )
            continue

        # Calculate comprehensive statistics
        avg_metrics = {
            key: np.mean(values) if len(values) > 0 else 0.0 for key, values in all_metrics.items()
        }
        std_metrics = {key: np.std(values) if len(values) > 0 else 0.0 for key, values in all_metrics.items()}
        max_metrics = {key: np.max(values) if len(values) > 0 else 0.0 for key, values in all_metrics.items()}
        min_metrics = {key: np.min(values) if len(values) > 0 else 0.0 for key, values in all_metrics.items()}

        # Store results for overall summary
        overall_results[valid_name] = {
            "avg": avg_metrics,
            "std": std_metrics,
            "max": max_metrics,
            "min": min_metrics,
            "count": len(valid_images),
        }

        # Enhanced results display
        logger.info("\n" + "=" * 80)
        logger.info(f"📈 FINAL RESULTS SUMMARY FOR {valid_name.upper()}")
        logger.info("=" * 80)
        logger.info(
            f"\n🎯 DENOISED (DN) METRICS:\n"
            f"   ├─ PSNR: {avg_metrics['psnr_dn']:7.3f} ± {std_metrics['psnr_dn']:5.3f} dB  (range: {min_metrics['psnr_dn']:.2f} - {max_metrics['psnr_dn']:.2f})\n"
            f"   └─ SSIM: {avg_metrics['ssim_dn']:7.4f} ± {std_metrics['ssim_dn']:6.4f}     (range: {min_metrics['ssim_dn']:.3f} - {max_metrics['ssim_dn']:.3f})\n\n"
            f"📈 EXPECTED (EXP) METRICS:\n"
            f"   ├─ PSNR: {avg_metrics['psnr_exp']:7.3f} ± {std_metrics['psnr_exp']:5.3f} dB  (range: {min_metrics['psnr_exp']:.2f} - {max_metrics['psnr_exp']:.2f})\n"
            f"   └─ SSIM: {avg_metrics['ssim_exp']:7.4f} ± {std_metrics['ssim_exp']:6.4f}     (range: {min_metrics['ssim_exp']:.3f} - {max_metrics['ssim_exp']:.3f})\n\n"
            f"⭐ COMBINED (MID) METRICS:\n"
            f"   ├─ PSNR: {avg_metrics['psnr_mid']:7.3f} ± {std_metrics['psnr_mid']:5.3f} dB  (range: {min_metrics['psnr_mid']:.2f} - {max_metrics['psnr_mid']:.2f})\n"
            f"   └─ SSIM: {avg_metrics['ssim_mid']:7.4f} ± {std_metrics['ssim_mid']:6.4f}     (range: {min_metrics['ssim_mid']:.3f} - {max_metrics['ssim_mid']:.3f})"
        )
        logger.info(f"\n📊 Total images processed: {len(valid_images)}")
        logger.info(f"💾 Results saved to: {save_dir}")

    return overall_results


def main():
    """Main testing function with enhanced logging and summary"""
    args = parse_args()

    # Setup testing environment
    logger = setup_testing(args)

    # Log testing configuration
    logger.info("\n" + "=" * 80)
    logger.info("🎆 SELF-SUPERVISED DENOISING MODEL TESTING")
    logger.info("=" * 80)
    logger.info(f"📁 Test directory: {args.test_dirs}")
    logger.info(f"💾 Output directory: {args.save_test_path}")
    logger.info(f"🔧 Checkpoint: {args.checkpoint}")
    logger.info(f"⚙️ Beta parameter: {args.beta}")
    logger.info(f"💻 GPU devices: {args.gpu_devices}")

    # Create and load model
    logger.info("\n🤖 Loading model...")
    model = create_model(args, logger)
    test_dict = load_test_data(args)

    # Create masker
    masker = utils.Masker(width=4, mode="interpolate", mask_type="all")
    logger.info("✅ Model and data loaded successfully")

    # Test the model
    overall_results = test_dataset(model, test_dict, masker, args, logger)

    # Final comprehensive summary
    logger.info("\n" + "=" * 80)
    if overall_results:
        logger.info("🎉 TESTING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)

        for dataset_name, results in overall_results.items():
            logger.info(
                f"\n📈 {dataset_name.upper()} - BEST PERFORMING METRICS:\n"
                f"   ⭐ Highest PSNR (MID): {results['max']['psnr_mid']:.3f} dB\n"
                f"   ⭐ Highest SSIM (MID): {results['max']['ssim_mid']:.4f}\n"
                f"   📊 Average PSNR (MID): {results['avg']['psnr_mid']:.3f} dB\n"
                f"   📊 Average SSIM (MID): {results['avg']['ssim_mid']:.4f}"
            )

        logger.info(f"\n💾 All results saved to: {os.path.join(args.save_test_path, args.log_name)}")
        logger.info("✨ Testing session complete!")
    else:
        logger.error("❌ TESTING FAILED - No datasets were processed successfully!")
        logger.error("Please check:")
        logger.error("  1. Test directory path is correct")
        logger.error("  2. Images exist in the specified directory")
        logger.error("  3. Image formats are supported (.tif, .png, .jpg, etc.)")
        logger.error("  4. Model checkpoint exists and is valid")
        return


if __name__ == "__main__":
    main()
