from __future__ import annotations

import argparse
from pathlib import Path

from config import CLASS_NAMES, TrainConfig, ensure_checkpoint_exists, select_device
from metrics import (
    PER_CLASS_FIELDNAMES,
    PREDICTION_FIELDNAMES,
    TOP_CONFUSION_FIELDNAMES,
    collect_prediction_rows,
    summarize_predictions,
    write_csv_rows,
    write_json,
)
from report import generate_markdown_report
from visualize import (
    save_confidence_histogram,
    save_confusion_matrix_plot,
    save_per_class_bar,
    save_sample_grid,
    save_top_confusions_bar,
)


EVAL_FILENAMES = {
    "predictions": "predictions.csv",
    "per_class": "per_class_metrics.csv",
    "top_confusions": "top_confusions.csv",
    "summary": "eval_summary.json",
    "report": "eval_report.md",
    "confusion_matrix": "confusion_matrix.png",
    "confusion_matrix_normalized": "confusion_matrix_normalized.png",
    "per_class_precision": "per_class_precision.png",
    "per_class_recall": "per_class_recall.png",
    "per_class_f1": "per_class_f1.png",
    "top_confusions_plot": "top_confusions.png",
    "confidence_histogram": "confidence_histogram.png",
    "error_samples": "error_samples.png",
    "high_confidence_errors": "high_confidence_errors.png",
    "low_confidence_corrects": "low_confidence_corrects.png",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a CIFAR-10 CNN checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=TrainConfig().best_checkpoint_path)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    parser.add_argument("--output-dir", type=Path, default=TrainConfig().eval_output_dir)
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--top-confusions", type=int, default=12)
    parser.add_argument("--columns", type=int, default=4)
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


def _top_rows(rows: list[dict], samples: int) -> list[dict]:
    return rows[:samples]


def build_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        key: output_dir / filename
        for key, filename in EVAL_FILENAMES.items()
    }


def main() -> None:
    from dataset import build_dataloaders_from_dataset, load_raw_dataset

    parser = build_arg_parser()
    args = parser.parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be at least 1")
    if args.top_confusions < 1:
        raise ValueError("--top-confusions must be at least 1")

    cfg = TrainConfig()
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.device is not None:
        cfg.device = args.device
    cfg.augment = False
    cfg.ensure_directories()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = build_output_paths(output_dir)

    ensure_checkpoint_exists(args.checkpoint)
    device = select_device(cfg.device)
    model, class_names, checkpoint = load_model(args.checkpoint, device)

    raw_dataset = load_raw_dataset(cfg)
    _, test_loader = build_dataloaders_from_dataset(raw_dataset, cfg)
    test_split = raw_dataset["test"]

    prediction_rows = collect_prediction_rows(model, test_loader, device, list(class_names))
    summary, per_class_rows, matrix, normalized_matrix, top_confusion_rows = summarize_predictions(
        prediction_rows,
        list(class_names),
        args.top_confusions,
    )

    write_csv_rows(paths["predictions"], prediction_rows, PREDICTION_FIELDNAMES)
    write_csv_rows(paths["per_class"], per_class_rows, PER_CLASS_FIELDNAMES)
    write_csv_rows(paths["top_confusions"], top_confusion_rows, TOP_CONFUSION_FIELDNAMES)
    write_json(
        paths["summary"],
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_best_acc": checkpoint.get("best_acc"),
            "summary": summary,
            "per_class": per_class_rows,
            "top_confusions": top_confusion_rows,
        },
    )

    save_confusion_matrix_plot(
        matrix,
        list(class_names),
        paths["confusion_matrix"],
        normalized=False,
    )
    save_confusion_matrix_plot(
        normalized_matrix,
        list(class_names),
        paths["confusion_matrix_normalized"],
        normalized=True,
    )
    save_per_class_bar(
        per_class_rows,
        "precision",
        paths["per_class_precision"],
        "各类别精确率",
    )
    save_per_class_bar(
        per_class_rows,
        "recall",
        paths["per_class_recall"],
        "各类别召回率",
    )
    save_per_class_bar(
        per_class_rows,
        "f1",
        paths["per_class_f1"],
        "各类别 F1",
    )
    save_top_confusions_bar(top_confusion_rows, paths["top_confusions_plot"])
    save_confidence_histogram(prediction_rows, paths["confidence_histogram"])

    error_rows = sorted(
        [row for row in prediction_rows if not bool(row["correct"])],
        key=lambda row: float(row["loss"]),
        reverse=True,
    )
    high_confidence_error_rows = sorted(
        [row for row in prediction_rows if not bool(row["correct"])],
        key=lambda row: float(row["confidence"]),
        reverse=True,
    )
    low_confidence_correct_rows = sorted(
        [row for row in prediction_rows if bool(row["correct"])],
        key=lambda row: float(row["confidence"]),
    )
    save_sample_grid(
        test_split,
        _top_rows(error_rows, args.num_samples),
        paths["error_samples"],
        "错例样本",
        samples=args.num_samples,
        columns=args.columns,
    )
    save_sample_grid(
        test_split,
        _top_rows(high_confidence_error_rows, args.num_samples),
        paths["high_confidence_errors"],
        "高置信度错例",
        samples=args.num_samples,
        columns=args.columns,
    )
    save_sample_grid(
        test_split,
        _top_rows(low_confidence_correct_rows, args.num_samples),
        paths["low_confidence_corrects"],
        "低置信度正确样本",
        samples=args.num_samples,
        columns=args.columns,
    )

    generate_markdown_report(
        paths["report"],
        summary,
        per_class_rows,
        top_confusion_rows,
        args.checkpoint,
        paths["predictions"],
    )

    print(f"checkpoint: {args.checkpoint}")
    print(f"checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"checkpoint best_acc: {checkpoint.get('best_acc')}")
    print(f"test accuracy: {summary['accuracy']:.4f}")
    print(f"macro_f1: {summary['macro_f1']:.4f}")
    print(f"weighted_f1: {summary['weighted_f1']:.4f}")
    print(f"saved predictions: {paths['predictions']}")
    print(f"saved report: {paths['report']}")


if __name__ == "__main__":
    main()
