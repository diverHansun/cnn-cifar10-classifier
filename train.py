from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import json
import random
from pathlib import Path
from typing import Any

from config import CLASS_NAMES, TrainConfig, select_device


DEFAULT_BEST_CHECKPOINT_NAME = "best_model.pth"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a simple CNN on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()
    return apply_arg_overrides(cfg, args)


def apply_arg_overrides(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "device": args.device,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    if args.no_augment:
        cfg.augment = False
    if args.no_amp:
        cfg.use_amp = False
    return cfg


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def print_environment(torch_module: Any, device: str) -> None:
    print(f"torch: {torch_module.__version__}")
    print(f"cuda available: {torch_module.cuda.is_available()}")
    print(f"cuda version: {torch_module.version.cuda}")
    print(f"selected device: {device}")
    if torch_module.cuda.is_available():
        print(f"gpu count: {torch_module.cuda.device_count()}")
        for index in range(torch_module.cuda.device_count()):
            print(f"gpu {index}: {torch_module.cuda.get_device_name(index)}")


def probe_cuda_runtime(torch_module: Any, device: str) -> None:
    if device != "cuda":
        return

    current_device = torch_module.cuda.current_device()
    capability = torch_module.cuda.get_device_capability(current_device)
    arch_list = torch_module.cuda.get_arch_list()
    print(f"gpu compute capability: sm_{capability[0]}{capability[1]}")
    print(f"torch CUDA arch list: {arch_list}")

    try:
        probe = torch_module.ones((2, 2), device="cuda")
        result = (probe @ probe).sum().item()
    except Exception as exc:
        raise RuntimeError(
            "CUDA was detected, but a tiny CUDA tensor operation failed. "
            "Check that the NVIDIA driver and PyTorch CUDA wheel support this GPU."
        ) from exc
    print(f"cuda probe result: {result:.1f}")


def autocast_context(torch_module: Any, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch_module.amp.autocast(device_type="cuda", enabled=True)


def create_grad_scaler(torch_module: Any, enabled: bool):
    try:
        return torch_module.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch_module.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device: str, use_amp: bool, epoch: int):
    import torch
    from tqdm import tqdm

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    progress = tqdm(loader, desc=f"train epoch {epoch + 1}", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(torch, use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += batch_size
        progress.set_postfix(loss=total_loss / total_seen, acc=total_correct / total_seen)

    return total_loss / total_seen, total_correct / total_seen


def validate_one_epoch(model, loader, criterion, device: str, use_amp: bool):
    import torch
    from tqdm import tqdm

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    with torch.no_grad():
        progress = tqdm(loader, desc="validate", leave=False)
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast_context(torch, use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_seen += batch_size
            progress.set_postfix(loss=total_loss / total_seen, acc=total_correct / total_seen)

    return total_loss / total_seen, total_correct / total_seen


def save_checkpoint(
    path: Path,
    epoch: int,
    model,
    optimizer,
    scaler,
    best_acc: float,
    cfg: TrainConfig,
    history: dict[str, list[float]],
) -> None:
    import torch

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_acc": best_acc,
        "class_names": list(CLASS_NAMES),
        "config": cfg.as_dict(),
        "history": history,
    }
    torch.save(checkpoint, path)


def plot_training_curves(history: dict[str, list[float]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="test")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="test")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    import torch
    from torch import nn, optim
    from torch.utils.tensorboard import SummaryWriter

    from dataset import build_dataloaders
    from model import SimpleCNN, count_parameters

    parser = build_arg_parser()
    args = parser.parse_args()
    checkpoint = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")
        checkpoint_config = checkpoint.get("config", {})
        cfg = TrainConfig.from_dict(checkpoint_config) if checkpoint_config else TrainConfig()
        cfg = apply_arg_overrides(cfg, args)
    else:
        cfg = config_from_args(args)

    cfg.ensure_directories()
    set_seed(cfg.seed)

    device = select_device(cfg.device)
    use_amp = cfg.use_amp and device == "cuda"
    print_environment(torch, device)
    probe_cuda_runtime(torch, device)

    train_loader, test_loader = build_dataloaders(cfg)
    model = SimpleCNN(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    scaler = create_grad_scaler(torch, use_amp)

    start_epoch = 0
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_acc = float(checkpoint.get("best_acc", 0.0))
        history = checkpoint.get("history", history)
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    run_name = f"cifar10_cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_path = cfg.run_dir / run_name
    writer = SummaryWriter(log_dir=str(run_path))

    print(f"parameters: {count_parameters(model):,}")
    print(f"run: {run_path}")
    print(f"best checkpoint: {cfg.best_checkpoint_path}")
    print(f"last checkpoint: {cfg.last_checkpoint_path}")

    try:
        for epoch in range(start_epoch, cfg.epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, use_amp, epoch
            )
            val_loss, val_acc = validate_one_epoch(
                model, test_loader, criterion, device, use_amp
            )

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/test", val_loss, epoch)
            writer.add_scalar("accuracy/train", train_acc, epoch)
            writer.add_scalar("accuracy/test", val_acc, epoch)

            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc

            save_checkpoint(
                cfg.last_checkpoint_path,
                epoch,
                model,
                optimizer,
                scaler,
                best_acc,
                cfg,
                history,
            )
            if is_best:
                save_checkpoint(
                    cfg.best_checkpoint_path,
                    epoch,
                    model,
                    optimizer,
                    scaler,
                    best_acc,
                    cfg,
                    history,
                )

            print(
                f"epoch {epoch + 1}/{cfg.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"test_loss={val_loss:.4f} test_acc={val_acc:.4f} best={best_acc:.4f}"
            )
    finally:
        writer.close()

    plot_training_curves(history, cfg.training_curve_path)
    metrics_path = cfg.output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"saved training curves: {cfg.training_curve_path}")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
