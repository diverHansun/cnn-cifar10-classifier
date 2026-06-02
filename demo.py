from __future__ import annotations

import argparse
from pathlib import Path

from config import CLASS_NAMES, TrainConfig, ensure_checkpoint_exists, select_device
from evaluate import load_model


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Save a grid of CIFAR-10 demo predictions.")
    parser.add_argument("--checkpoint", type=Path, default=TrainConfig().best_checkpoint_path)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    return parser


def denormalize(image_tensor):
    import torch

    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=image_tensor.device).view(3, 1, 1)
    return (image_tensor * std + mean).clamp(0, 1)


def save_demo_predictions(model, loader, class_names, device: str, output_path: Path, samples: int) -> None:
    import math
    import torch
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if samples < 1:
        raise ValueError("samples must be at least 1")

    images, labels = next(iter(loader))
    images = images[:samples].to(device)
    labels = labels[:samples]
    with torch.no_grad():
        predictions = model(images).argmax(dim=1).cpu()

    cols = min(4, samples)
    rows = math.ceil(samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.6))
    axes = axes.reshape(-1) if hasattr(axes, "reshape") else [axes]

    for index, axis in enumerate(axes):
        axis.axis("off")
        if index >= len(images):
            continue
        image = denormalize(images[index]).cpu().permute(1, 2, 0).numpy()
        true_label = class_names[labels[index].item()]
        pred_label = class_names[predictions[index].item()]
        axis.imshow(image)
        axis.set_title(f"T: {true_label}\nP: {pred_label}", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    from dataset import build_dataloaders

    args = build_arg_parser().parse_args()
    cfg = TrainConfig()
    cfg.augment = False
    if args.device is not None:
        cfg.device = args.device
    cfg.ensure_directories()

    ensure_checkpoint_exists(args.checkpoint)
    device = select_device(cfg.device)
    model, class_names, _ = load_model(args.checkpoint, device)
    _, test_loader = build_dataloaders(cfg)
    save_demo_predictions(model, test_loader, class_names, device, cfg.demo_predictions_path, args.samples)
    print(f"saved demo predictions: {cfg.demo_predictions_path}")


if __name__ == "__main__":
    main()
