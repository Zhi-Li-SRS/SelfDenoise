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
import utils
import dataset
from model import uformer


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
    """Setup training environment"""
    # Set up timestamp and paths
    systime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    args.save_path = os.path.join(args.save_model_path, args.log_name, systime)
    os.makedirs(args.save_path, exist_ok=True)

    # Set up CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    torch.set_num_threads(6)

    # Set up logger
    utils.setup_logger(
        "train", args.save_path, "train_" + args.log_name, level=logging.INFO, screen=True, tofile=True
    )
    logger = logging.getLogger("train")

    return systime, logger


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
        dataset=training_dataset, num_workers=0, batch_size=2, shuffle=True, pin_memory=False, drop_last=True
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


def train_epoch(network, training_loader, optimizer, masker, args, epoch, logger):
    """Train for one epoch"""
    network.train()

    for param_group in optimizer.param_groups:
        current_lr = param_group["lr"]
    print("LearningRate of Epoch {} = {}".format(epoch, current_lr))

    alpha, beta = get_loss_weights(epoch, args)

    for iteration, clean in enumerate(training_loader):
        st = time.time()
        clean = clean / 255.0
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

        # Log progress
        logger.info(
            "{:04d} {:05d} diff={:.6f}, exp_diff={:.6f}, Loss_Reg={:.6f}, Lambda={:.3f}, Loss_Rev={:.6f}, Loss_All={:.6f}, Time={:.4f}".format(
                epoch,
                iteration,
                torch.mean(diff**2).item(),
                torch.mean(exp_diff**2).item(),
                loss_reg.item(),
                epoch / args.n_epoch,
                loss_rev.item(),
                loss_all.item(),
                time.time() - st,
            )
        )


def validate_model(network, valid_dict, masker, args, epoch, systime, logger):
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
                origin255 = im.copy().astype(np.uint8)
                im = np.array(im, dtype=np.float32) / 255.0
                noisy_im = im

                if epoch == args.n_snapshot:
                    noisy255 = np.clip(noisy_im * 255.0 + 0.5, 0, 255).astype(np.uint8)

                # Prepare input
                H, W = noisy_im.shape[:2]
                val_size = (max(H, W) + 31) // 32 * 32
                noisy_im = np.pad(noisy_im, [[0, val_size - H], [0, val_size - W], [0, 0]], "reflect")

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

                # Convert to uint8
                pred255_dn = np.clip(pred_dn * 255.0 + 0.5, 0, 255).astype(np.uint8)
                pred255_exp = np.clip(pred_exp * 255.0 + 0.5, 0, 255).astype(np.uint8)
                pred255_mid = np.clip(pred_mid * 255.0 + 0.5, 0, 255).astype(np.uint8)

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
                        Image.fromarray(origin255).convert("RGB").save(
                            os.path.join(
                                save_dir, "{}_{:03d}-{:03d}_clean.png".format(valid_name, idx, epoch)
                            )
                        )
                        Image.fromarray(noisy255).convert("RGB").save(
                            os.path.join(
                                save_dir, "{}_{:03d}-{:03d}_noisy.png".format(valid_name, idx, epoch)
                            )
                        )

                    Image.fromarray(pred255_dn).convert("RGB").save(
                        os.path.join(save_dir, "{}_{:03d}-{:03d}_dn.png".format(valid_name, idx, epoch))
                    )
                    Image.fromarray(pred255_exp).convert("RGB").save(
                        os.path.join(save_dir, "{}_{:03d}-{:03d}_exp.png".format(valid_name, idx, epoch))
                    )
                    Image.fromarray(pred255_mid).convert("RGB").save(
                        os.path.join(save_dir, "{}_{:03d}-{:03d}_mid.png".format(valid_name, idx, epoch))
                    )

        # Calculate average metrics
        avg_psnr_dn = np.mean(avg_psnr_dn)
        avg_ssim_dn = np.mean(avg_ssim_dn)
        avg_psnr_exp = np.mean(avg_psnr_exp)
        avg_ssim_exp = np.mean(avg_ssim_exp)
        avg_psnr_mid = np.mean(avg_psnr_mid)
        avg_ssim_mid = np.mean(avg_ssim_mid)

        # Log results
        log_path = os.path.join(validation_path, "A_log_{}.csv".format(valid_name))
        with open(log_path, "a") as f:
            f.write(
                "epoch:{},dn:{:.6f}/{:.6f},exp:{:.6f}/{:.6f},mid:{:.6f}/{:.6f}\n".format(
                    epoch, avg_psnr_dn, avg_ssim_dn, avg_psnr_exp, avg_ssim_exp, avg_psnr_mid, avg_ssim_mid
                )
            )


def main():
    """Main training function"""
    args = parse_args()

    systime, logger = setup_training(args)

    model = create_model(args)

    training_loader, valid_dict = create_data_loaders(args)

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_scheduler(model, args)

    # Create masker
    masker = utils.Masker(width=4, mode="interpolate", mask_type="all")

    # Resume training if specified
    epoch_init = 1
    if args.resume is not None:
        epoch_init, optimizer, scheduler = utils.resume_state(args.resume, optimizer, scheduler)

    # Load pretrained model if specified
    if args.checkpoint is not None:
        model = utils.load_network(args.checkpoint, model, strict=True, logger=logger)

        # Reset epoch for fine-tuning
        epoch_init = 1
        for i in range(1, epoch_init):
            scheduler.step()
            new_lr = scheduler.get_lr()[0]
            logger.info("----------------------------------------------------")
            logger.info("==> Resuming Training with learning rate:{}".format(new_lr))
            logger.info("----------------------------------------------------")

    logger.info("Training initialized successfully")
    print("Batchsize={}, number of epoch={}".format(2, args.n_epoch))

    # Clear GPU cache before training
    torch.cuda.empty_cache()
    
    # Training loop
    for epoch in range(epoch_init, args.n_epoch + 1):
        # Train one epoch
        train_epoch(model, training_loader, optimizer, masker, args, epoch, logger)

        # Update learning rate
        scheduler.step()

        # Validation and checkpoint saving
        if epoch % args.n_snapshot == 0 or epoch == args.n_epoch:
            # Save model
            utils.save_network(model, args.save_path, epoch, "model", logger)

            # Validate model
            validate_model(model, valid_dict, masker, args, epoch, systime, logger)

    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
