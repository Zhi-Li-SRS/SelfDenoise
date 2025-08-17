import os
import logging
import time
import datetime
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
import wandb
from skimage import io
import psutil

try:
    import GPUtil

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

import src.utils as utils
import src.dataset as dataset
from src.model import uformer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train self-supervised denoising model")

    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./data/train", help="Path to training dataset")
    parser.add_argument(
        "--val_dirs", type=str, default="./data/validation", help="Path to validation dataset"
    )

    # Model arguments
    parser.add_argument(
        "--noisetype",
        type=str,
        default="poisson30",
        choices=["gauss25", "gauss5_50", "poisson30", "poisson5_50"],
        help="Noise type for training",
    )

    # Training arguments
    parser.add_argument("--n_epoch", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-6, help="Learning rate")
    parser.add_argument("--w_decay", type=float, default=1e-8, help="Weight decay")
    parser.add_argument("--gamma", type=float, default=0.5, help="Learning rate decay factor")
    parser.add_argument("--patchsize", type=int, default=128, help="Training patch size")
    parser.add_argument("--n_channel", type=int, default=3, help="Number of input channels")

    # Loss arguments
    parser.add_argument("--Lambda1", type=float, default=1.0, help="Regularization loss weight")
    parser.add_argument("--Lambda2", type=float, default=2.0, help="Initial reversibility weight")
    parser.add_argument("--increase_ratio", type=float, default=20.0, help="Final reversibility weight")

    # Checkpoint arguments
    parser.add_argument("--resume", type=str, help="Path to resume checkpoint")
    parser.add_argument("--checkpoint", type=str, help="Path to pretrained model")
    parser.add_argument("--n_snapshot", type=int, default=50, help="Save model every n epochs")

    # Output arguments
    parser.add_argument(
        "--save_model_path", type=str, default="./experiments/results", help="Base path for saving models"
    )
    parser.add_argument("--log_name", type=str, default="SRS", help="Experiment name")

    # Hardware arguments
    parser.add_argument("--gpu_devices", default="0", type=str, help="GPU device IDs")
    parser.add_argument("--parallel", action="store_true", help="Use data parallel")

    return parser.parse_args()


def setup_training(args):
    """Setup training environment with Weights & Biases"""
    # Set up timestamp and paths
    systime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    args.save_path = os.path.join(args.save_model_path, systime)
    os.makedirs(args.save_path, exist_ok=True)

    # Set up CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    torch.set_num_threads(6)

    # Initialize Weights & Biases
    wandb.init(
        project="self-denoise",
        name=f"{args.log_name}_{systime}",
        config={
            "data_dir": args.data_dir,
            "val_dirs": args.val_dirs,
            "noisetype": args.noisetype,
            "learning_rate": args.lr,
            "epochs": args.n_epoch,
            "patch_size": args.patchsize,
            "n_channel": args.n_channel,
            "lambda1": args.Lambda1,
            "lambda2": args.Lambda2,
            "increase_ratio": args.increase_ratio,
            "weight_decay": args.w_decay,
            "gamma": args.gamma,
            "batch_size": 2,
            "n_snapshot": args.n_snapshot,
            "optimizer": "Adam",
            "scheduler": "MultiStepLR",
            "masker_width": 4,
            "masker_mode": "interpolate",
            "architecture": "uformer",
            "save_path": args.save_path,
        },
        save_code=True,
        tags=[args.noisetype, "self-supervised"],
    )

    # Create models directory for wandb
    models_dir = os.path.join(args.save_path, "models")
    os.makedirs(models_dir, exist_ok=True)

    print(f"🚀 Started W&B run: {wandb.run.name}")
    print(f"📁 Saving to: {args.save_path}")
    print(f"🔗 View at: {wandb.run.url}")

    return systime


def create_model(args):
    """Create and initialize model"""
    network = uformer(args)

    if args.parallel:
        network = torch.nn.DataParallel(network)
    network = network.cuda()

    return network


def create_data_loaders(args):
    """Create training and validation data loaders"""
    # Training dataset
    training_dataset = dataset.ImageDataset(args.data_dir, patch=args.patchsize)
    training_loader = DataLoader(
        dataset=training_dataset, num_workers=0, batch_size=4, shuffle=True, pin_memory=False, drop_last=True
    )

    # Validation dataset
    data_dir = os.path.join(args.val_dirs, "data_srs")
    valid_dict = {"data_srs": dataset.load_validation_data(data_dir)}

    return training_loader, valid_dict


def create_optimizer_scheduler(network, args):
    """Create optimizer and learning rate scheduler"""
    optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.w_decay)

    num_epoch = args.n_epoch
    ratio = num_epoch / 100
    scheduler = lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(20 * ratio) - 1, int(40 * ratio) - 1, int(60 * ratio) - 1, int(80 * ratio) - 1],
        gamma=args.gamma,
    )

    return optimizer, scheduler


def get_loss_weights(epoch, args):
    """Calculate dynamic loss weights based on training progress"""
    if args.noisetype in ["gauss25", "poisson30"]:
        Thread1 = 0.8
        Thread2 = 1.0
    else:
        Thread1 = 0.4
        Thread2 = 1.0

    Lambda = epoch / args.n_epoch
    if Lambda <= Thread1:
        beta = args.Lambda2
    elif Thread1 <= Lambda <= Thread2:
        beta = args.Lambda2 + (Lambda - Thread1) * (args.increase_ratio - args.Lambda2) / (Thread2 - Thread1)
    else:
        beta = args.increase_ratio

    alpha = args.Lambda1
    return alpha, beta


def train_epoch(network, training_loader, optimizer, masker, args, epoch):
    """Train for one epoch with W&B logging"""
    network.train()

    for param_group in optimizer.param_groups:
        current_lr = param_group["lr"]
    print(f"Epoch {epoch} - Learning Rate: {current_lr:.2e}")

    alpha, beta = get_loss_weights(epoch, args)
    epoch_losses = {"reg": [], "rev": [], "total": [], "diff": [], "exp_diff": []}

    for iteration, clean in enumerate(training_loader):
        st = time.time()
        # TIFF images are already in proper float32 format from dataset
        # No need to divide by 255 as they're processed correctly in dataset.py
        noisy = clean.cuda()

        optimizer.zero_grad()

        # Generate masked input and get prediction
        net_input, mask = masker.train(noisy)
        noisy_output = network(net_input)
        n, c, h, w = noisy.shape
        noisy_output = (noisy_output * mask).view(n, -1, c, h, w).sum(dim=1)
        diff = noisy_output - noisy

        # Get expected output
        with torch.no_grad():
            exp_output = network(noisy)
        exp_diff = exp_output - noisy

        # Calculate losses
        revisible = diff + beta * exp_diff
        loss_reg = alpha * torch.mean(diff**2)
        loss_rev = torch.mean(revisible**2)
        loss_all = loss_reg + loss_rev

        # Backward pass
        loss_all.backward()
        optimizer.step()

        # Clear GPU cache to prevent memory accumulation
        torch.cuda.empty_cache()

        # Collect losses for epoch averaging
        epoch_losses["reg"].append(loss_reg.item())
        epoch_losses["rev"].append(loss_rev.item())
        epoch_losses["total"].append(loss_all.item())
        epoch_losses["diff"].append(torch.mean(diff**2).item())
        epoch_losses["exp_diff"].append(torch.mean(exp_diff**2).item())

    # Log epoch averages
    avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
    wandb.log(
        {
            "epoch_avg/loss_reg": avg_losses["reg"],
            "epoch_avg/loss_rev": avg_losses["rev"],
            "epoch_avg/loss_total": avg_losses["total"],
            "epoch_avg/diff_squared": avg_losses["diff"],
            "epoch_avg/exp_diff_squared": avg_losses["exp_diff"],
            "epoch": epoch,
        }
    )

    return avg_losses


def validate_model(network, valid_dict, masker, args, epoch, systime):
    """Validate the model"""
    network.eval()

    # Set up validation paths
    save_model_path = os.path.join(args.save_model_path, args.log_name, systime)
    validation_path = os.path.join(save_model_path, "validation")
    os.makedirs(validation_path, exist_ok=True)

    np.random.seed(101)
    valid_repeat_times = {"data_srs": 3}
    _, beta = get_loss_weights(epoch, args)

    for valid_name, valid_images in valid_dict.items():
        avg_psnr_dn = []
        avg_ssim_dn = []
        avg_psnr_exp = []
        avg_ssim_exp = []
        avg_psnr_mid = []
        avg_ssim_mid = []

        save_dir = os.path.join(validation_path, valid_name)
        os.makedirs(save_dir, exist_ok=True)
        repeat_times = valid_repeat_times[valid_name]

        for i in range(repeat_times):
            for idx, im in enumerate(valid_images):
                # Handle TIFF files properly - they're already in float32 format
                if im.max() > 1.0:  # TIFF format, original range preserved
                    # Normalize for processing while preserving precision
                    im_max = im.max()
                    noisy_im_raw = im / im_max if im_max > 0 else im
                    # For display, convert to 0-255 range
                    origin255 = (
                        np.clip(im * 255.0 / im_max, 0, 255).astype(np.uint8)
                        if im_max > 0
                        else im.astype(np.uint8)
                    )
                else:  # Already normalized (PNG/JPG)
                    noisy_im_raw = im
                    origin255 = np.clip(im * 255.0, 0, 255).astype(np.uint8)

                im = noisy_im_raw
                noisy_im = im

                if epoch == args.n_snapshot:
                    noisy255 = np.clip(noisy_im * 255.0, 0, 255).astype(np.uint8)

                # Prepare input - use same patch size as training
                H, W = noisy_im.shape[:2]
                patch_size = args.patchsize if hasattr(args, "patchsize") else 128

                # Center crop to patch_size x patch_size like in training
                start_h = max(0, (H - patch_size) // 2)
                start_w = max(0, (W - patch_size) // 2)
                end_h = min(H, start_h + patch_size)
                end_w = min(W, start_w + patch_size)

                noisy_im = noisy_im[start_h:end_h, start_w:end_w]
                # Also crop origin255 to match the patch
                origin255 = origin255[start_h:end_h, start_w:end_w]

                # Pad to patch_size if image is smaller
                if noisy_im.shape[0] < patch_size or noisy_im.shape[1] < patch_size:
                    pad_h = max(0, patch_size - noisy_im.shape[0])
                    pad_w = max(0, patch_size - noisy_im.shape[1])
                    noisy_im = np.pad(noisy_im, [[0, pad_h], [0, pad_w], [0, 0]], "reflect")
                    origin255 = np.pad(origin255, [[0, pad_h], [0, pad_w], [0, 0]], "reflect")

                # Convert to tensor

                transformer = transforms.Compose([transforms.ToTensor()])
                noisy_im = transformer(noisy_im).unsqueeze(0).cuda()

                # Inference
                with torch.no_grad():
                    n, c, h, w = noisy_im.shape
                    net_input, mask = masker.train(noisy_im)
                    noisy_output = (network(net_input) * mask).view(n, -1, c, h, w).sum(dim=1)
                    dn_output = noisy_output.detach().clone()

                    # Free memory
                    del net_input, mask, noisy_output
                    torch.cuda.empty_cache()

                    exp_output = network(noisy_im)

                # Process outputs
                pred_dn = (
                    dn_output[:, :, :H, :W].permute(0, 2, 3, 1).cpu().data.clamp(0, 1).numpy().squeeze(0)
                )
                pred_exp = (
                    exp_output.detach()
                    .clone()[:, :, :H, :W]
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .data.clamp(0, 1)
                    .numpy()
                    .squeeze(0)
                )
                pred_mid = (dn_output[:, :, :H, :W] + beta * exp_output.detach().clone()[:, :, :H, :W]) / (
                    1 + beta
                )
                pred_mid = pred_mid.permute(0, 2, 3, 1).cpu().data.clamp(0, 1).numpy().squeeze(0)

                # Free memory
                del exp_output
                torch.cuda.empty_cache()

                # Convert to uint8 for display (PNG format for wandb)
                pred255_dn = np.clip(pred_dn * 255.0, 0, 255).astype(np.uint8)
                pred255_exp = np.clip(pred_exp * 255.0, 0, 255).astype(np.uint8)
                pred255_mid = np.clip(pred_mid * 255.0, 0, 255).astype(np.uint8)

                # Calculate metrics
                psnr_dn = utils.calculate_psnr(origin255.astype(np.float32), pred255_dn.astype(np.float32))
                ssim_dn = utils.calculate_ssim(origin255.astype(np.float32), pred255_dn.astype(np.float32))
                avg_psnr_dn.append(psnr_dn)
                avg_ssim_dn.append(ssim_dn)

                psnr_exp = utils.calculate_psnr(origin255.astype(np.float32), pred255_exp.astype(np.float32))
                ssim_exp = utils.calculate_ssim(origin255.astype(np.float32), pred255_exp.astype(np.float32))
                avg_psnr_exp.append(psnr_exp)
                avg_ssim_exp.append(ssim_exp)

                psnr_mid = utils.calculate_psnr(origin255.astype(np.float32), pred255_mid.astype(np.float32))
                ssim_mid = utils.calculate_ssim(origin255.astype(np.float32), pred255_mid.astype(np.float32))
                avg_psnr_mid.append(psnr_mid)
                avg_ssim_mid.append(ssim_mid)

                # Save images
                if i == 0:
                    if epoch == args.n_snapshot:
                        # Save clean and noisy images (check if grayscale)
                        if (
                            len(origin255.shape) == 3
                            and np.allclose(origin255[:, :, 0], origin255[:, :, 1])
                            and np.allclose(origin255[:, :, 1], origin255[:, :, 2])
                        ):
                            Image.fromarray(origin255[:, :, 0], mode="L").save(
                                os.path.join(
                                    save_dir, "{}_{:03d}-{:03d}_clean.png".format(valid_name, idx, epoch)
                                )
                            )
                            Image.fromarray(noisy255[:, :, 0], mode="L").save(
                                os.path.join(
                                    save_dir, "{}_{:03d}-{:03d}_noisy.png".format(valid_name, idx, epoch)
                                )
                            )
                        else:
                            Image.fromarray(origin255).save(
                                os.path.join(
                                    save_dir, "{}_{:03d}-{:03d}_clean.png".format(valid_name, idx, epoch)
                                )
                            )
                            Image.fromarray(noisy255).save(
                                os.path.join(
                                    save_dir, "{}_{:03d}-{:03d}_noisy.png".format(valid_name, idx, epoch)
                                )
                            )

                    # Save as grayscale PNG if all channels are the same, otherwise RGB
                    if (
                        len(pred255_dn.shape) == 3
                        and np.allclose(pred255_dn[:, :, 0], pred255_dn[:, :, 1])
                        and np.allclose(pred255_dn[:, :, 1], pred255_dn[:, :, 2])
                    ):
                        Image.fromarray(pred255_dn[:, :, 0], mode="L").save(
                            os.path.join(save_dir, "{}_{:03d}-{:03d}_dn.png".format(valid_name, idx, epoch))
                        )
                        Image.fromarray(pred255_exp[:, :, 0], mode="L").save(
                            os.path.join(save_dir, "{}_{:03d}-{:03d}_exp.png".format(valid_name, idx, epoch))
                        )
                        Image.fromarray(pred255_mid[:, :, 0], mode="L").save(
                            os.path.join(save_dir, "{}_{:03d}-{:03d}_mid.png".format(valid_name, idx, epoch))
                        )
                    else:
                        Image.fromarray(pred255_dn).save(
                            os.path.join(save_dir, "{}_{:03d}-{:03d}_dn.png".format(valid_name, idx, epoch))
                        )
                        Image.fromarray(pred255_exp).save(
                            os.path.join(save_dir, "{}_{:03d}-{:03d}_exp.png".format(valid_name, idx, epoch))
                        )
                        Image.fromarray(pred255_mid).save(
                            os.path.join(save_dir, "{}_{:03d}-{:03d}_mid.png".format(valid_name, idx, epoch))
                        )

        # Calculate average metrics
        avg_psnr_dn = np.mean(avg_psnr_dn)
        avg_ssim_dn = np.mean(avg_ssim_dn)
        avg_psnr_exp = np.mean(avg_psnr_exp)
        avg_ssim_exp = np.mean(avg_ssim_exp)
        avg_psnr_mid = np.mean(avg_psnr_mid)
        avg_ssim_mid = np.mean(avg_ssim_mid)

        # Log validation results to W&B and CSV
        wandb.log(
            {
                f"val_{valid_name}/psnr_dn": avg_psnr_dn,
                f"val_{valid_name}/ssim_dn": avg_ssim_dn,
                f"val_{valid_name}/psnr_exp": avg_psnr_exp,
                f"val_{valid_name}/ssim_exp": avg_ssim_exp,
                f"val_{valid_name}/psnr_mid": avg_psnr_mid,
                f"val_{valid_name}/ssim_mid": avg_ssim_mid,
                f"val_{valid_name}/beta": beta,
                "epoch": epoch,
            }
        )

        # Save sample images to W&B
        if epoch % (args.n_snapshot * 2) == 0:  # Log images less frequently
            sample_images = []
            for idx in range(min(3, len(valid_images))):  # Log first 3 validation images
                if os.path.exists(os.path.join(save_dir, f"{valid_name}_{idx:03d}-{epoch:03d}_clean.png")):
                    sample_images.append(
                        wandb.Image(
                            os.path.join(save_dir, f"{valid_name}_{idx:03d}-{epoch:03d}_clean.png"),
                            caption=f"Clean {idx}",
                        )
                    )
                if os.path.exists(os.path.join(save_dir, f"{valid_name}_{idx:03d}-{epoch:03d}_dn.png")):
                    sample_images.append(
                        wandb.Image(
                            os.path.join(save_dir, f"{valid_name}_{idx:03d}-{epoch:03d}_dn.png"),
                            caption=f"Denoised {idx}",
                        )
                    )
                if os.path.exists(os.path.join(save_dir, f"{valid_name}_{idx:03d}-{epoch:03d}_mid.png")):
                    sample_images.append(
                        wandb.Image(
                            os.path.join(save_dir, f"{valid_name}_{idx:03d}-{epoch:03d}_mid.png"),
                            caption=f"Mid {idx}",
                        )
                    )

            if sample_images:
                wandb.log({f"val_{valid_name}/sample_images": sample_images, "epoch": epoch})

        print(
            f"📊 Validation E{epoch:03d} | PSNR_dn: {avg_psnr_dn:.2f} | SSIM_dn: {avg_ssim_dn:.4f} | PSNR_mid: {avg_psnr_mid:.2f} | SSIM_mid: {avg_ssim_mid:.4f}"
        )


def main():
    """Main training function"""
    args = parse_args()

    systime = setup_training(args)  # setup training environment with wandb

    model = create_model(args)  # create model

    training_loader, valid_dict = create_data_loaders(args)  # create data loaders

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_scheduler(model, args)

    # Create masker
    masker = utils.Masker(width=4, mode="interpolate", mask_type="all")

    # Resume training if specified
    epoch_init = 1
    if args.resume is not None:
        epoch_init, optimizer, scheduler = utils.resume_state(args.resume, optimizer, scheduler)
        print(f"🔄 Resumed training from epoch {epoch_init}")
        wandb.config.update({"resumed_from": args.resume, "resume_epoch": epoch_init})

    # Load pretrained model if specified
    if args.checkpoint is not None:
        model = utils.load_network(args.checkpoint, model, strict=True)
        print(f"📦 Loaded pretrained model from {args.checkpoint}")
        wandb.config.update({"pretrained_model": args.checkpoint})

        # Reset epoch for fine-tuning
        epoch_init = 1
        for _ in range(1, epoch_init):
            scheduler.step()
            new_lr = scheduler.get_lr()[0]
            print(f"==> Resuming Training with learning rate: {new_lr}")

    print("✅ Training initialized successfully")
    print("Batchsize={}, number of epoch={}".format(2, args.n_epoch))

    # Clear GPU cache before training
    torch.cuda.empty_cache()

    # Training loop
    for epoch in range(epoch_init, args.n_epoch + 1):
        # Train one epoch
        epoch_losses = train_epoch(model, training_loader, optimizer, masker, args, epoch)

        wandb.log(
            {
                "epoch_summary/avg_loss_total": epoch_losses["total"],
                "epoch_summary/avg_loss_reg": epoch_losses["reg"],
                "epoch_summary/avg_loss_rev": epoch_losses["rev"],
                "epoch": epoch,
            }
        )

        # Update learning rate
        scheduler.step()

        # Log learning rate
        current_lr = scheduler.get_lr()[0]
        wandb.log({"train/learning_rate_epoch": current_lr, "epoch": epoch})

        # Validation and checkpoint saving
        if epoch % args.n_snapshot == 0 or epoch == args.n_epoch:
            # Save model
            model_path = utils.save_network(model, args.save_path, epoch, "model")

            # Log model artifact to W&B
            model_artifact = wandb.Artifact(f"model-epoch-{epoch}", type="model")
            model_artifact.add_file(model_path)
            wandb.log_artifact(model_artifact)

            # Validate model
            validate_model(model, valid_dict, masker, args, epoch, systime)

    print("🎉 Training completed successfully!")
    wandb.finish()


if __name__ == "__main__":
    main()
