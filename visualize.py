from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def _prepare_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def save_placeholder(output_path: Path, title: str, message: str) -> None:
    plt = _prepare_matplotlib()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_confusion_matrix_plot(
    matrix,
    class_names: list[str],
    output_path: Path,
    normalized: bool = False,
) -> None:
    plt = _prepare_matplotlib()
    from sklearn.metrics import ConfusionMatrixDisplay

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 8.2))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    display.plot(
        ax=ax,
        cmap="Blues",
        values_format=".2f" if normalized else "d",
        xticks_rotation=45,
        colorbar=True,
    )
    ax.set_title("CIFAR-10 混淆矩阵" + ("（按真实类别归一化）" if normalized else ""))
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_per_class_bar(
    per_class_rows: list[dict[str, Any]],
    metric: str,
    output_path: Path,
    title: str,
) -> None:
    plt = _prepare_matplotlib()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = [row["class_name"] for row in per_class_rows]
    values = [float(row[metric]) for row in per_class_rows]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(names, values, color="#3b82f6")
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_xlabel("类别")
    ax.tick_params(axis="x", rotation=35)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.025, 0.98),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_top_confusions_bar(top_confusion_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not top_confusion_rows:
        save_placeholder(output_path, "主要混淆", "没有错误分类样本。")
        return

    plt = _prepare_matplotlib()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [
        f"{row['true_label']} -> {row['pred_label']}"
        for row in top_confusion_rows
    ]
    values = [int(row["count"]) for row in top_confusion_rows]

    fig_height = max(4.8, len(labels) * 0.42)
    fig, ax = plt.subplots(figsize=(8.6, fig_height))
    ax.barh(labels[::-1], values[::-1], color="#f97316")
    ax.set_xlabel("数量")
    ax.set_title("主要混淆")
    for index, value in enumerate(values[::-1]):
        ax.text(value + 1, index, str(value), va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_confidence_histogram(prediction_rows: list[dict[str, Any]], output_path: Path) -> None:
    plt = _prepare_matplotlib()
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)
    correct = [
        float(row["confidence"])
        for row in prediction_rows
        if bool(row["correct"])
    ]
    wrong = [
        float(row["confidence"])
        for row in prediction_rows
        if not bool(row["correct"])
    ]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bins = np.linspace(0, 1, 21)
    ax.hist(correct, bins=bins, alpha=0.75, label="正确", color="#16a34a")
    ax.hist(wrong, bins=bins, alpha=0.75, label="错误", color="#dc2626")
    ax.set_xlabel("Top-1 置信度")
    ax.set_ylabel("样本数")
    ax.set_title("预测置信度分布")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_sample_grid(
    raw_split,
    prediction_rows: list[dict[str, Any]],
    output_path: Path,
    title: str,
    samples: int = 24,
    columns: int = 4,
) -> None:
    if samples < 1:
        raise ValueError("samples must be at least 1")
    if columns < 1:
        raise ValueError("columns must be at least 1")

    selected_rows = prediction_rows[:samples]
    if not selected_rows:
        save_placeholder(output_path, title, "No samples matched this selection.")
        return

    plt = _prepare_matplotlib()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = min(columns, samples)
    rows = math.ceil(len(selected_rows) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.8, rows * 2.9))
    axes_list = axes.reshape(-1) if hasattr(axes, "reshape") else [axes]

    for axis in axes_list:
        axis.axis("off")

    for axis, row in zip(axes_list, selected_rows):
        item = raw_split[int(row["index"])]
        image = item["img"].convert("RGB")
        correct = bool(row["correct"])
        title_color = "#15803d" if correct else "#b91c1c"

        axis.imshow(image)
        axis.set_title(
            f"#{row['index']} 真实:{row['true_label']}\n"
            f"预测:{row['pred_label']} ({float(row['confidence']):.2f})",
            fontsize=8,
            color=title_color,
        )

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
