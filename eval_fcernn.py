import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.dataset_fcernn import DummyFCERNNData
from models.fcernn import FCERNN, select_block
from utils.io import load_config, save_json
from utils.metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FCERNN on dummy data")
    parser.add_argument("--config", default="configs/fcernn.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt", help="Checkpoint path")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FCERNN(config).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    test_data = DummyFCERNNData(config, split="test")
    test_loader = DataLoader(
        test_data,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    metrics_accum = {"mae": 0.0, "rmse": 0.0, "nrmse": 0.0}
    if config["training"].get("use_psnr", False):
        metrics_accum["psnr"] = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            if inputs.ndim in (4, 5):
                inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
            outputs = model(inputs)
            if config["output"].get("block_mode", False):
                outputs = select_block(
                    outputs,
                    config["output"]["block_splits"],
                    config["output"]["block_index"],
                )
            metrics = compute_metrics(outputs, targets, config["training"].get("use_psnr", False))
            for key in metrics_accum:
                metrics_accum[key] += metrics.get(key, 0.0)

    num_batches = len(test_loader)
    metrics_avg = {k: v / num_batches for k, v in metrics_accum.items()}

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics_avg, output_dir / "eval_metrics.json")

    print(metrics_avg)


if __name__ == "__main__":
    main()
