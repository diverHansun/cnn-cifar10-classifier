from __future__ import annotations

import argparse
import random
from pathlib import Path

from config import TrainConfig
from metrics import read_prediction_rows
from visualize import save_placeholder, save_sample_grid


DEFAULT_DEMO_OUTPUT_NOTE = "outputs/demo"
DEMO_MODES = (
    "random",
    "errors",
    "high-confidence-errors",
    "low-confidence-corrects",
    "confusion",
    "class",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create demo grids from evaluate.py predictions.csv.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=TrainConfig().eval_output_dir / "predictions.csv",
        help="Path to evaluate.py predictions.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TrainConfig().demo_output_dir,
        help=f"Directory for demo images, default {DEFAULT_DEMO_OUTPUT_NOTE}.",
    )
    parser.add_argument("--mode", choices=DEMO_MODES, default="random")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--true-label", type=str, default=None)
    parser.add_argument("--pred-label", type=str, default=None)
    parser.add_argument("--class-name", type=str, default=None)
    return parser


def ensure_predictions_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {path}. Run `python evaluate.py` first."
        )
    return path


def validate_sample_args(samples: int, columns: int) -> None:
    if samples < 1:
        raise ValueError("samples must be at least 1")
    if columns < 1:
        raise ValueError("columns must be at least 1")


def available_labels(rows: list[dict]) -> list[str]:
    labels = {row["true_label"] for row in rows}
    labels.update(row["pred_label"] for row in rows)
    return sorted(labels)


def validate_label(label: str, labels: list[str], option_name: str) -> str:
    if label not in labels:
        available = ", ".join(labels)
        raise ValueError(
            f"{option_name} must be one of the CIFAR-10 labels. Available labels: {available}"
        )
    return label


def output_filename_for_mode(args: argparse.Namespace) -> str:
    if args.mode == "confusion":
        true_label = args.true_label or "true"
        pred_label = args.pred_label or "pred"
        return f"demo_confusion_{true_label}_to_{pred_label}.png"
    if args.mode == "class":
        class_name = args.class_name or "class"
        return f"demo_class_{class_name}.png"
    return f"demo_{args.mode.replace('-', '_')}.png"


def select_rows(rows: list[dict], args: argparse.Namespace) -> tuple[list[dict], str]:
    labels = available_labels(rows)

    if args.mode == "random":
        shuffled_rows = list(rows)
        rng = random.Random(args.seed)
        rng.shuffle(shuffled_rows)
        return shuffled_rows[: args.samples], "随机演示样本"

    if args.mode == "errors":
        selected = [
            row for row in rows
            if not bool(row["correct"])
        ]
        selected.sort(key=lambda row: float(row["loss"]), reverse=True)
        return selected[: args.samples], "错例样本"

    if args.mode == "high-confidence-errors":
        selected = [
            row for row in rows
            if not bool(row["correct"])
        ]
        selected.sort(key=lambda row: float(row["confidence"]), reverse=True)
        return selected[: args.samples], "高置信度错例"

    if args.mode == "low-confidence-corrects":
        selected = [
            row for row in rows
            if bool(row["correct"])
        ]
        selected.sort(key=lambda row: float(row["confidence"]))
        return selected[: args.samples], "低置信度正确样本"

    if args.mode == "confusion":
        if not args.true_label or not args.pred_label:
            raise ValueError("--mode confusion requires --true-label and --pred-label")
        true_label = validate_label(args.true_label, labels, "--true-label")
        pred_label = validate_label(args.pred_label, labels, "--pred-label")
        selected = [
            row for row in rows
            if row["true_label"] == true_label and row["pred_label"] == pred_label
        ]
        selected.sort(key=lambda row: float(row["confidence"]), reverse=True)
        return selected[: args.samples], f"{true_label} -> {pred_label}"

    if args.mode == "class":
        if not args.class_name:
            raise ValueError("--mode class requires --class-name")
        class_name = validate_label(args.class_name, labels, "--class-name")
        selected = [
            row for row in rows
            if row["true_label"] == class_name
        ]
        selected.sort(
            key=lambda row: (not bool(row["correct"]), float(row["loss"])),
            reverse=True,
        )
        return selected[: args.samples], f"类别：{class_name}"

    raise ValueError(f"Unsupported demo mode: {args.mode}")


def main() -> None:
    from dataset import load_raw_dataset

    args = build_arg_parser().parse_args()
    validate_sample_args(args.samples, args.columns)

    cfg = TrainConfig()
    cfg.ensure_directories()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ensure_predictions_exists(args.predictions)
    prediction_rows = read_prediction_rows(args.predictions)
    selected_rows, title = select_rows(prediction_rows, args)
    output_path = args.output_dir / output_filename_for_mode(args)

    if not selected_rows:
        save_placeholder(output_path, title, "No samples matched this selection.")
        print(f"mode: {args.mode}")
        print(f"predictions: {args.predictions}")
        print(f"saved demo: {output_path}")
        return

    raw_dataset = load_raw_dataset(cfg)
    save_sample_grid(
        raw_dataset["test"],
        selected_rows,
        output_path,
        title,
        samples=args.samples,
        columns=args.columns,
    )

    print(f"mode: {args.mode}")
    print(f"predictions: {args.predictions}")
    print(f"saved demo: {output_path}")


if __name__ == "__main__":
    main()
