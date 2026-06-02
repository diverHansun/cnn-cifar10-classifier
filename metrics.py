from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PREDICTION_FIELDNAMES = [
    "index",
    "true_id",
    "true_label",
    "pred_id",
    "pred_label",
    "correct",
    "confidence",
    "true_prob",
    "loss",
    "top1_label",
    "top1_prob",
    "top2_label",
    "top2_prob",
    "top3_label",
    "top3_prob",
    "rank_of_true_label",
]

PER_CLASS_FIELDNAMES = [
    "class_id",
    "class_name",
    "precision",
    "recall",
    "f1",
    "support",
]

TOP_CONFUSION_FIELDNAMES = [
    "true_id",
    "true_label",
    "pred_id",
    "pred_label",
    "count",
    "rate_within_true",
]


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def normalize_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    for key in ("index", "true_id", "pred_id", "rank_of_true_label"):
        normalized[key] = int(normalized[key])
    for key in ("confidence", "true_prob", "loss", "top1_prob", "top2_prob", "top3_prob"):
        normalized[key] = float(normalized[key])
    normalized["correct"] = _to_bool(normalized["correct"])
    return normalized


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [normalize_prediction_row(row) for row in reader]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_prediction_rows(model, loader, device: str, class_names: list[str]) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as functional
    from tqdm import tqdm

    model.eval()
    rows: list[dict[str, Any]] = []
    sample_index = 0
    top_k = min(3, len(class_names))

    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate", leave=False):
            if len(batch) == 3:
                images, labels, indices = batch
            else:
                images, labels = batch
                indices = torch.arange(sample_index, sample_index + labels.size(0))
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            losses = functional.cross_entropy(logits, labels, reduction="none")
            top_probs, top_ids = probs.topk(k=top_k, dim=1)
            sorted_ids = probs.argsort(dim=1, descending=True)
            pred_ids = top_ids[:, 0]

            for batch_index in range(labels.size(0)):
                true_id = int(labels[batch_index].item())
                pred_id = int(pred_ids[batch_index].item())
                confidence = float(top_probs[batch_index, 0].item())
                true_prob = float(probs[batch_index, true_id].item())
                rank = int(
                    (sorted_ids[batch_index] == true_id)
                    .nonzero(as_tuple=True)[0]
                    .item()
                    + 1
                )
                top_labels = [
                    class_names[int(class_id)]
                    for class_id in top_ids[batch_index].cpu().tolist()
                ]
                top_values = [
                    float(value)
                    for value in top_probs[batch_index].cpu().tolist()
                ]

                rows.append(
                    {
                        "index": int(indices[batch_index].item()),
                        "true_id": true_id,
                        "true_label": class_names[true_id],
                        "pred_id": pred_id,
                        "pred_label": class_names[pred_id],
                        "correct": pred_id == true_id,
                        "confidence": confidence,
                        "true_prob": true_prob,
                        "loss": float(losses[batch_index].item()),
                        "top1_label": top_labels[0],
                        "top1_prob": top_values[0],
                        "top2_label": top_labels[1] if len(top_labels) > 1 else "",
                        "top2_prob": top_values[1] if len(top_values) > 1 else 0.0,
                        "top3_label": top_labels[2] if len(top_labels) > 2 else "",
                        "top3_prob": top_values[2] if len(top_values) > 2 else 0.0,
                        "rank_of_true_label": rank,
                    }
                )
                sample_index += 1

    return rows


def summarize_predictions(
    rows: list[dict[str, Any]],
    class_names: list[str],
    top_confusions: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], Any, Any, list[dict[str, Any]]]:
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    true_ids = [int(row["true_id"]) for row in rows]
    pred_ids = [int(row["pred_id"]) for row in rows]
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(true_ids, pred_ids, labels=labels)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        true_ids,
        pred_ids,
        labels=labels,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_ids,
        pred_ids,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        true_ids,
        pred_ids,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    per_class_rows = [
        {
            "class_id": class_id,
            "class_name": class_names[class_id],
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1": float(f1[class_id]),
            "support": int(support[class_id]),
        }
        for class_id in labels
    ]

    off_diagonal_rows: list[dict[str, Any]] = []
    for true_id in labels:
        true_total = int(matrix[true_id].sum())
        for pred_id in labels:
            if true_id == pred_id:
                continue
            count = int(matrix[true_id, pred_id])
            if count == 0:
                continue
            off_diagonal_rows.append(
                {
                    "true_id": true_id,
                    "true_label": class_names[true_id],
                    "pred_id": pred_id,
                    "pred_label": class_names[pred_id],
                    "count": count,
                    "rate_within_true": count / true_total if true_total else 0.0,
                }
            )
    top_confusion_rows = sorted(
        off_diagonal_rows,
        key=lambda row: (row["count"], row["rate_within_true"]),
        reverse=True,
    )[:top_confusions]

    sample_count = len(rows)
    average_loss = sum(float(row["loss"]) for row in rows) / sample_count if sample_count else 0.0
    top3_accuracy = (
        sum(1 for row in rows if int(row["rank_of_true_label"]) <= 3) / sample_count
        if sample_count
        else 0.0
    )
    average_confidence = (
        sum(float(row["confidence"]) for row in rows) / sample_count
        if sample_count
        else 0.0
    )

    summary = {
        "sample_count": sample_count,
        "accuracy": float(accuracy_score(true_ids, pred_ids)) if sample_count else 0.0,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "top3_accuracy": top3_accuracy,
        "average_loss": average_loss,
        "average_confidence": average_confidence,
    }

    return summary, per_class_rows, matrix, normalized_matrix, top_confusion_rows
