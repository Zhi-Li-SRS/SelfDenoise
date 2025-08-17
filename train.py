import os
import logging
import time
import glob
import datetime
import argparse
import numpy as np
from skimage import io
from PIL import Image
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm

from model import UNet
import utils as util
from masking import Masker
from data_loader import DataLoader, validation_data


def create_parser():
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description="Self-supervised denoising training script")

    # Model arguments
    parser.add_argument(
        "--noisetype",
        type=str,
        default="gauss25",
        choices=["gauss25", "gauss5_50", "poisson30", "poisson5_50"],
    )
    parser.add_argument("--n_feature", type=int, default=48, help="Base number of features in UNet")
    parser.add_argument("--n_channel", type=int, default=1, help="Number of input/output channels")
    parser.add_argument("--depth", type=int, default=5, help="Depth of UNet (number of downsampling layers)")

    # Training arguments
    parser.add_argument("--lr", type=float, default=1e-7, help="Learning rate")
    parser.add_argument("--w_decay", type=float, default=1e-9, help="Weight decay")
    parser.add_argument("--gamma", type=float, default=0.5, help="LR scheduler gamma")
    parser.add_argument("--n_epoch", type=int, default=500, help="Number of epochs")
    parser.add_argument("--batchsize", type=int, default=4, help="Batch size")
    parser.add_argument("--patchsize", type=int, default=256, help="Patch size for training")

    # Loss function arguments
    parser.add_argument("--Lambda1", type=float, default=1.0, help="Lambda1 parameter")
    parser.add_argument("--Lambda2", type=float, default=2.0, help="Lambda2 parameter")
    parser.add_argument("--increase_ratio", type=float, default=20.0, help="Increase ratio parameter")

    # Data and checkpoint arguments
    parser.add_argument("--data_dir", type=str, default="./data/train", help="Training data directory")
    parser.add_argument("--val_dirs", type=str, default="./data/validation", help="Validation data directory")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, help="Load model checkpoint")
    parser.add_argument(
        "--pretrained",
        type=str,
        default="./ckpt/checkpoint.pth",
        help="Path to pretrained model for fine-tuning",
    )

    # Output arguments
    parser.add_argument("--save_model_path", type=str, default="./experiments", help="Model save path")
    parser.add_argument("--log_name", type=str, default="selfdenoise_TPF", help="Log name")
    parser.add_argument("--n_snapshot", type=int, default=50, help="Save model every n epochs")

    # System arguments
    parser.add_argument("--gpu_devices", default="0", type=str, help="GPU devices")
    parser.add_argument("--parallel", action="store_true", help="Use data parallel")

    return parser


def setup_environment_and_logging(opt):
    """Setup environment variables and logging."""
    systime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_devices
    torch.set_num_threads(8)

    # config loggers
    opt.save_path = os.path.join(opt.save_model_path, opt.log_name, systime)
    os.makedirs(opt.save_path, exist_ok=True)
    util.setup_logger(
        "train", opt.save_path, "train_" + opt.log_name, level=logging.INFO, screen=True, tofile=True
    )
    logger = logging.getLogger("train")
    return logger, systime


def setup_data_loaders(opt):
    """Setup training and validation data loaders."""
    # Training Set - using new TIF DataLoader
    TrainingDataset = DataLoader(opt.data_dir, patch=opt.patchsize)
    print(f"Training dataset size: {len(TrainingDataset)}")
    print(f"Batch size: {opt.batchsize}")
    print(f"Number of batches: {len(TrainingDataset) // opt.batchsize}")

    TrainingLoader = TorchDataLoader(
        dataset=TrainingDataset,
        num_workers=8,
        batch_size=opt.batchsize,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    # Validation data
    valid_data = validation_data(opt.val_dirs)
    print(f"Validation dataset size: {len(valid_data) if valid_data else 0}")

    return TrainingLoader, valid_data


def setup_model_and_optimizer(opt, logger):
    """Setup model, optimizer, and scheduler."""
    # Masker
    masker = Masker(width=4, mode="interpolate", mask_type="all")

    # Network
    network = UNet(
        in_channels=opt.n_channel, out_channels=opt.n_channel, depth=opt.depth, base_filters=opt.n_feature
    )
    if opt.parallel:
        network = torch.nn.DataParallel(network)
    network = network.cuda()

    # optimizer and scheduler
    num_epoch = opt.n_epoch
    ratio = num_epoch / 100
    optimizer = optim.Adam(network.parameters(), lr=opt.lr, weight_decay=opt.w_decay)
    scheduler = lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(20 * ratio) - 1, int(40 * ratio) - 1, int(60 * ratio) - 1, int(80 * ratio) - 1],
        gamma=opt.gamma,
    )

    # Resume and load pre-trained model
    epoch_init = 1
    if opt.resume is not None:
        epoch_init, optimizer, scheduler = util.resume_state(opt.resume, optimizer, scheduler)
    if opt.checkpoint is not None:
        network = util.load_network(opt.checkpoint, network, strict=True)
    if opt.pretrained is not None:
        logger.info("Loading pretrained model for fine-tuning: {}".format(opt.pretrained))
        # Use the same parameter mapping as in infer.py
        from infer import load_network

        network = load_network(opt.pretrained, network, strict=True)

    logger.info("Batchsize={}, number of epoch={}".format(opt.batchsize, opt.n_epoch))
    logger.info("Setup finished")

    return network, optimizer, scheduler, masker, epoch_init


def train_epoch(network, optimizer, scheduler, masker, TrainingLoader, epoch, opt, logger):
    """Train one epoch with tqdm progress bar."""
    Thread1 = 0.4
    Thread2 = 1.0
    Lambda1 = opt.Lambda1
    Lambda2 = opt.Lambda2
    increase_ratio = opt.increase_ratio

    # Get current learning rate
    for param_group in optimizer.param_groups:
        current_lr = param_group["lr"]
    logger.info("LearningRate of Epoch {} = {}".format(epoch, current_lr))

    network.train()

    # Initialize loss tracking for epoch averages
    epoch_loss_all = []
    epoch_loss_reg = []
    epoch_loss_rev = []
    epoch_diff = []
    epoch_exp_diff = []

    # Use tqdm for progress bar
    pbar = tqdm(TrainingLoader, desc=f"Epoch {epoch:03d}")

    for iteration, noisy in enumerate(pbar):
        st = time.time()

        # Normalize input data to reasonable range for training stability
        # Calculate per-batch statistics
        batch_max = noisy.max()
        batch_min = noisy.min()
        batch_range = batch_max - batch_min

        # Normalize to [0, 1] range for training stability
        if batch_range > 0:
            noisy_normalized = (noisy - batch_min) / batch_range
        else:
            noisy_normalized = noisy - batch_min

        noisy_normalized = noisy_normalized.cuda()

        optimizer.zero_grad()

        net_input, mask = masker.train(noisy_normalized)
        noisy_output = network(net_input)
        n, c, h, w = noisy_normalized.shape
        noisy_output = (noisy_output * mask).view(n, -1, c, h, w).sum(dim=1)
        diff = noisy_output - noisy_normalized

        with torch.no_grad():
            exp_output = network(noisy_normalized)
        exp_diff = exp_output - noisy_normalized

        # Calculate lambda and beta parameters
        Lambda = epoch / opt.n_epoch
        if Lambda <= Thread1:
            beta = Lambda2
        elif Thread1 <= Lambda <= Thread2:
            beta = Lambda2 + (Lambda - Thread1) * (increase_ratio - Lambda2) / (Thread2 - Thread1)
        else:
            beta = increase_ratio
        alpha = Lambda1

        revisible = diff + beta * exp_diff
        loss_reg = alpha * torch.mean(diff**2)
        loss_rev = torch.mean(revisible**2)
        loss_all = loss_reg + loss_rev

        loss_all.backward()
        optimizer.step()

        # Track losses for epoch average
        epoch_loss_all.append(loss_all.item())
        epoch_loss_reg.append(loss_reg.item())
        epoch_loss_rev.append(loss_rev.item())
        epoch_diff.append(torch.mean(diff**2).item())
        epoch_exp_diff.append(torch.mean(exp_diff**2).item())

    # Calculate and log epoch averages
    avg_loss_all = np.mean(epoch_loss_all)
    avg_loss_reg = np.mean(epoch_loss_reg)
    avg_loss_rev = np.mean(epoch_loss_rev)
    avg_diff = np.mean(epoch_diff)
    avg_exp_diff = np.mean(epoch_exp_diff)

    logger.info(
        "Epoch {:04d} AVERAGES: Loss_All={:.6f}, Loss_Reg={:.6f}, Loss_Rev={:.6f}, "
        "Diff={:.6f}, Exp_Diff={:.6f}, Beta={:.2f}, Lambda={:.3f}".format(
            epoch, avg_loss_all, avg_loss_reg, avg_loss_rev, avg_diff, avg_exp_diff, beta, Lambda
        )
    )

    scheduler.step()

    return avg_loss_all, avg_loss_reg, avg_loss_rev


def validate_model(network, masker, valid_data, epoch, opt, logger, systime):
    """Validate model and save results."""
    # Check if validation data exists
    if not valid_data or len(valid_data) == 0:
        logger.warning("No validation data found, skipping validation")
        return

    Thread1 = 0.4
    Thread2 = 1.0
    Lambda2 = opt.Lambda2
    increase_ratio = opt.increase_ratio

    # Calculate beta for current epoch
    Lambda = epoch / opt.n_epoch
    if Lambda <= Thread1:
        beta = Lambda2
    elif Thread1 <= Lambda <= Thread2:
        beta = Lambda2 + (Lambda - Thread1) * (increase_ratio - Lambda2) / (Thread2 - Thread1)
    else:
        beta = increase_ratio

    network.eval()
    save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)
    validation_path = os.path.join(save_model_path, "validation")
    np.random.seed(101)

    avg_psnr_dn = []
    avg_ssim_dn = []
    avg_psnr_exp = []
    avg_ssim_exp = []
    avg_psnr_mid = []
    avg_ssim_mid = []

    save_dir = os.path.join(validation_path, "tif_validation")
    os.makedirs(save_dir, exist_ok=True)

    num_img = len(valid_data)
    logger.info(f"Validating {num_img} images")

    # Use tqdm for validation progress
    for idx in tqdm(range(num_img), desc="Validating"):

        im = valid_data[idx]  # Original image with meaningful intensity values
        noisy_im = valid_data[idx]
        noisy_im = np.array(noisy_im, dtype=np.float32)  # Keep original float32 values
        origin_float = im.copy().astype(np.float32)  # Keep original intensity values

        print(
            f"Image {idx}: shape={noisy_im.shape}, dtype={noisy_im.dtype}, min={noisy_im.min():.4f}, max={noisy_im.max():.4f}"
        )

        # padding to square
        H = noisy_im.shape[0]
        W = noisy_im.shape[1]
        val_size = (max(H, W) + 31) // 32 * 32
        noisy_im = np.pad(noisy_im, [[0, val_size - H], [0, val_size - W], [0, 0]], "reflect")

        # Normalize for model input (same as training)
        img_max = noisy_im.max()
        img_min = noisy_im.min()
        img_range = img_max - img_min

        print(
            f"After padding: shape={noisy_im.shape}, min={img_min:.4f}, max={img_max:.4f}, range={img_range:.4f}"
        )

        if img_range > 0:
            noisy_normalized = (noisy_im - img_min) / img_range
        else:
            noisy_normalized = noisy_im - img_min

        transformer = transforms.Compose([transforms.ToTensor()])
        noisy_normalized = transformer(noisy_normalized)
        noisy_normalized = torch.unsqueeze(noisy_normalized, 0)
        noisy_normalized = noisy_normalized.cuda()

        with torch.no_grad():
            n, c, h, w = noisy_normalized.shape
            net_input, mask = masker.train(noisy_normalized)
            noisy_output = (network(net_input) * mask).view(n, -1, c, h, w).sum(dim=1)
            dn_output = noisy_output.detach().clone()
            exp_output = network(noisy_normalized)

        pred_dn = dn_output[:, :, :H, :W]
        pred_exp = exp_output.detach().clone()[:, :, :H, :W]
        pred_mid = (pred_dn + beta * pred_exp) / (1 + beta)

        pred_dn = pred_dn.permute(0, 2, 3, 1)
        pred_exp = pred_exp.permute(0, 2, 3, 1)
        pred_mid = pred_mid.permute(0, 2, 3, 1)

        # Convert predictions back to original intensity range
        pred_dn_normalized = pred_dn.cpu().data.numpy().squeeze(0)
        pred_exp_normalized = pred_exp.cpu().data.numpy().squeeze(0)
        pred_mid_normalized = pred_mid.cpu().data.numpy().squeeze(0)

        # Denormalize back to original intensity range
        if img_range > 0:
            pred_dn_float = pred_dn_normalized * img_range + img_min
            pred_exp_float = pred_exp_normalized * img_range + img_min
            pred_mid_float = pred_mid_normalized * img_range + img_min
        else:
            pred_dn_float = pred_dn_normalized + img_min
            pred_exp_float = pred_exp_normalized + img_min
            pred_mid_float = pred_mid_normalized + img_min

        # calculate psnr using original float32 values with meaningful intensities
        psnr_dn = util.calculate_psnr(origin_float, pred_dn_float)
        avg_psnr_dn.append(psnr_dn)
        ssim_dn = util.calculate_ssim(origin_float, pred_dn_float)
        avg_ssim_dn.append(ssim_dn)

        psnr_exp = util.calculate_psnr(origin_float, pred_exp_float)
        avg_psnr_exp.append(psnr_exp)
        ssim_exp = util.calculate_ssim(origin_float, pred_exp_float)
        avg_ssim_exp.append(ssim_exp)

        psnr_mid = util.calculate_psnr(origin_float, pred_mid_float)
        avg_psnr_mid.append(psnr_mid)
        ssim_mid = util.calculate_ssim(origin_float, pred_mid_float)
        avg_ssim_mid.append(ssim_mid)

        # Save as float32 .tif files to preserve intensity values

        save_path = os.path.join(save_dir, "tif_{:03d}-{:03d}_clean.tif".format(idx, epoch))
        io.imsave(save_path, origin_float.squeeze())

        save_path = os.path.join(
            save_dir, "tif_{:03d}-{:03d}_dn_{:.6f}-{:.6f}.tif".format(idx, epoch, psnr_dn, ssim_dn)
        )
        io.imsave(save_path, pred_dn_float.squeeze())

        save_path = os.path.join(
            save_dir, "tif_{:03d}-{:03d}_exp_{:.6f}-{:.6f}.tif".format(idx, epoch, psnr_exp, ssim_exp)
        )
        io.imsave(save_path, pred_exp_float.squeeze())

        save_path = os.path.join(
            save_dir, "tif_{:03d}-{:03d}_mid_{:.6f}-{:.6f}.tif".format(idx, epoch, psnr_mid, ssim_mid)
        )
        io.imsave(save_path, pred_mid_float.squeeze())

    # Calculate average metrics
    avg_psnr_dn = np.mean(avg_psnr_dn)
    avg_ssim_dn = np.mean(avg_ssim_dn)
    avg_psnr_exp = np.mean(avg_psnr_exp)
    avg_ssim_exp = np.mean(avg_ssim_exp)
    avg_psnr_mid = np.mean(avg_psnr_mid)
    avg_ssim_mid = np.mean(avg_ssim_mid)

    # Log results
    log_path = os.path.join(validation_path, "A_log_tif.csv")
    with open(log_path, "a") as f:
        f.writelines(
            "epoch:{},dn:{:.6f}/{:.6f},exp:{:.6f}/{:.6f},mid:{:.6f}/{:.6f}\n".format(
                epoch, avg_psnr_dn, avg_ssim_dn, avg_psnr_exp, avg_ssim_exp, avg_psnr_mid, avg_ssim_mid
            )
        )

    logger.info(
        f"Validation - Epoch {epoch}: DN PSNR/SSIM: {avg_psnr_dn:.4f}/{avg_ssim_dn:.4f}, "
        f"EXP: {avg_psnr_exp:.4f}/{avg_ssim_exp:.4f}, MID: {avg_psnr_mid:.4f}/{avg_ssim_mid:.4f}"
    )


def main():
    """Main training function."""
    parser = create_parser()
    opt, _ = parser.parse_known_args()

    # Setup environment and logging
    logger, systime = setup_environment_and_logging(opt)

    # Setup data loaders
    TrainingLoader, valid_data = setup_data_loaders(opt)

    # Setup model and optimizer
    network, optimizer, scheduler, masker, epoch_init = setup_model_and_optimizer(opt, logger)

    # Training loop
    logger.info("Starting training...")
    for epoch in range(epoch_init, opt.n_epoch + 1):
        # Train one epoch
        avg_loss_all, avg_loss_reg, avg_loss_rev = train_epoch(
            network, optimizer, scheduler, masker, TrainingLoader, epoch, opt, logger
        )

        # Validation and save model
        if epoch % opt.n_snapshot == 0 or epoch == opt.n_epoch:
            # Save checkpoint
            util.save_network(network, epoch, "model", opt.save_path)
            # Validation
            validate_model(network, masker, valid_data, epoch, opt, logger, systime)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
