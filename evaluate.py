from __future__ import annotations

import argparse
from pathlib import Path

from config import CLASS_NAMES, TrainConfig, ensure_checkpoint_exists, select_device


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a CIFAR-10 CNN checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=TrainConfig().best_checkpoint_path)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    return parser


def load_model(checkpoint_path: Path, device: str):
    import torch

    from model import SimpleCNN

    checkpoint_path = ensure_checkpoint_exists(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("class_names", list(CLASS_NAMES))
    model = SimpleCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names, checkpoint


def collect_predictions(model, loader, device: str):
    import torch
    from tqdm import tqdm

    predictions = []
    targets = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="evaluate", leave=False):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())
    return targets, predictions


def save_confusion_matrix(targets, predictions, class_names, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    matrix = confusion_matrix(targets, predictions, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(8, 8))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    display.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45, colorbar=False)
    ax.set_title("CIFAR-10 Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    from dataset import build_dataloaders

    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.device is not None:
        cfg.device = args.device
    cfg.augment = False
    cfg.ensure_directories()

    ensure_checkpoint_exists(args.checkpoint)
    device = select_device(cfg.device)
    model, class_names, checkpoint = load_model(args.checkpoint, device)
    _, test_loader = build_dataloaders(cfg)
    targets, predictions = collect_predictions(model, test_loader, device)
    accuracy = sum(int(pred == target) for pred, target in zip(predictions, targets)) / len(targets)
    save_confusion_matrix(targets, predictions, class_names, cfg.confusion_matrix_path)

    print(f"checkpoint: {args.checkpoint}")
    print(f"checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"checkpoint best_acc: {checkpoint.get('best_acc')}")
    print(f"test accuracy: {accuracy:.4f}")
    print(f"saved confusion matrix: {cfg.confusion_matrix_path}")


if __name__ == "__main__":
    main()
