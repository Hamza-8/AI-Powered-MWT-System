import argparse
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.dataset_fcernn import DummyFCERNNData
from models.fcernn import FCERNN, select_block
from utils.io import load_config, seed_everything, save_json
from utils.metrics import compute_metrics, gradient_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train FCERNN on dummy data")
    parser.add_argument("--config", default="configs/fcernn.yaml", help="Path to config YAML")
    return parser.parse_args()


def compute_loss(pred, target, config):
    training_cfg = config["training"]
    mse_loss = nn.functional.mse_loss(pred, target)
    l1_loss = nn.functional.l1_loss(pred, target)
    loss = training_cfg["mse_weight"] * mse_loss + training_cfg["l1_weight"] * l1_loss
    if training_cfg.get("use_grad_loss", False):
        loss += training_cfg.get("grad_loss_weight", 0.0) * gradient_loss(pred, target)
    return loss


def run_epoch(model, loader, optimizer, scaler, device, config, train=True):
    model.train(train)
    total_loss = 0.0
    metrics_accum = {"mae": 0.0, "rmse": 0.0, "nrmse": 0.0}
    if config["training"].get("use_psnr", False):
        metrics_accum["psnr"] = 0.0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        if inputs.ndim in (4, 5):
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=config["training"].get("amp", False)):
            outputs = model(inputs)
            if config["output"].get("block_mode", False):
                outputs = select_block(
                    outputs,
                    config["output"]["block_splits"],
                    config["output"]["block_index"],
                )
            loss = compute_loss(outputs, targets, config)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        metrics = compute_metrics(outputs, targets, config["training"].get("use_psnr", False))
        for key in metrics_accum:
            metrics_accum[key] += metrics.get(key, 0.0)

    num_batches = len(loader)
    return total_loss / num_batches, {k: v / num_batches for k, v in metrics_accum.items()}


def main():
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.get("seed", 42))

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FCERNN(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["scheduler"]["t_max"]
    )

    train_data = DummyFCERNNData(config, split="train")
    val_data = DummyFCERNNData(config, split="val")
    train_loader = DataLoader(
        train_data,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    scaler = GradScaler(enabled=config["training"].get("amp", False))
    writer = SummaryWriter(log_dir=str(log_dir))

    best_loss = float("inf")
    for epoch in range(config["training"]["num_epochs"]):
        train_loss, train_metrics = run_epoch(
            model, train_loader, optimizer, scaler, device, config, train=True
        )
        val_loss, val_metrics = run_epoch(
            model, val_loader, optimizer, scaler, device, config, train=False
        )
        scheduler.step()

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        for key, value in train_metrics.items():
            writer.add_scalar(f"metrics/train_{key}", value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"metrics/val_{key}", value, epoch)

        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }
            torch.save(checkpoint, checkpoint_dir / "best.pt")

    save_json({"best_val_loss": best_loss}, output_dir / "summary.json")
    writer.close()


if __name__ == "__main__":
    main()
