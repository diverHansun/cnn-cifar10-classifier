from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _per_class_table(per_class_rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| 类别 | 精确率 | 召回率 | F1 | 样本数 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in per_class_rows:
        lines.append(
            f"| {row['class_name']} | "
            f"{_pct(float(row['precision']))} | "
            f"{_pct(float(row['recall']))} | "
            f"{_pct(float(row['f1']))} | "
            f"{int(row['support'])} |"
        )
    return lines


def _top_confusion_table(top_confusion_rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| 真实类别 | 预测成 | 数量 | 类内占比 |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in top_confusion_rows:
        lines.append(
            f"| {row['true_label']} | {row['pred_label']} | "
            f"{int(row['count'])} | {_pct(float(row['rate_within_true']))} |"
        )
    return lines


def generate_markdown_report(
    output_path: Path,
    summary: dict[str, Any],
    per_class_rows: list[dict[str, Any]],
    top_confusion_rows: list[dict[str, Any]],
    checkpoint_path: Path,
    predictions_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_f1 = max(per_class_rows, key=lambda row: float(row["f1"]))
    weakest_f1 = min(per_class_rows, key=lambda row: float(row["f1"]))
    weakest_recall = min(per_class_rows, key=lambda row: float(row["recall"]))
    top_confusion = top_confusion_rows[0] if top_confusion_rows else None

    lines = [
        "# CIFAR-10 CNN 评估报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 权重文件：`{checkpoint_path}`",
        f"- 逐样本预测表：`{predictions_path.name}`",
        f"- 测试样本数：{int(summary['sample_count'])}",
        "",
        "## 总体指标",
        "",
        "| 指标 | 数值 | 作用 |",
        "| --- | ---: | --- |",
        f"| 准确率 Accuracy | {_pct(float(summary['accuracy']))} | 所有样本中预测正确的比例，是最直观的整体表现。 |",
        f"| 平均 Loss | {_fmt(float(summary['average_loss']))} | 交叉熵损失，越低代表模型对真实类别分配的概率越高。 |",
        f"| Macro Precision | {_pct(float(summary['macro_precision']))} | 各类别精确率的简单平均，观察模型预测某一类时是否可靠。 |",
        f"| Macro Recall | {_pct(float(summary['macro_recall']))} | 各类别召回率的简单平均，观察每个真实类别是否被充分识别。 |",
        f"| Macro F1 | {_pct(float(summary['macro_f1']))} | 精确率和召回率的均衡指标，每个类别权重相同。 |",
        f"| Weighted F1 | {_pct(float(summary['weighted_f1']))} | 按样本数加权的 F1，适合整体汇总但可能弱化少数类问题。 |",
        f"| Top-3 Accuracy | {_pct(float(summary['top3_accuracy']))} | 真实类别进入模型前三候选的比例，观察模型是否把正确答案排在前列。 |",
        f"| 平均置信度 | {_pct(float(summary['average_confidence']))} | 模型对 top-1 预测的平均概率，可用于判断是否过度自信。 |",
        "",
        "## 类别表现",
        "",
        f"- F1 最高类别：`{best_f1['class_name']}`，F1={_pct(float(best_f1['f1']))}",
        f"- F1 最低类别：`{weakest_f1['class_name']}`，F1={_pct(float(weakest_f1['f1']))}",
        f"- 召回率最低类别：`{weakest_recall['class_name']}`，召回率={_pct(float(weakest_recall['recall']))}",
        "",
        "![](per_class_precision.png)",
        "",
        "![](per_class_recall.png)",
        "",
        "![](per_class_f1.png)",
        "",
        *_per_class_table(per_class_rows),
        "",
        "## 混淆矩阵与主要混淆",
        "",
        "混淆矩阵用于观察“真实类别”和“预测类别”的对应关系；对角线越集中说明模型越稳定，非对角线较大的格子就是主要误判来源。",
        "",
        "![](confusion_matrix.png)",
        "",
        "![](confusion_matrix_normalized.png)",
        "",
    ]

    if top_confusion is not None:
        lines.extend(
            [
                f"当前最明显的混淆是 `{top_confusion['true_label']}` 被预测为 `{top_confusion['pred_label']}`，",
                f"数量为 {int(top_confusion['count'])}，占该真实类别样本的 {_pct(float(top_confusion['rate_within_true']))}。",
                "",
            ]
        )

    lines.extend(
        [
            "![](top_confusions.png)",
            "",
            *_top_confusion_table(top_confusion_rows),
            "",
            "## 置信度与错例",
            "",
            "置信度图用于判断模型是否“知道自己不知道”。高置信度错例尤其值得看，因为它们代表模型非常确定但判断错误，常能暴露数据增强、模型容量或类别边界的问题。",
            "",
            "![](confidence_histogram.png)",
            "",
            "下面三组图分别展示普通错例、高置信度错例、低置信度正确样本。默认每组 24 张，4x6 排布，便于直接放进报告。",
            "",
            "![](error_samples.png)",
            "",
            "![](high_confidence_errors.png)",
            "",
            "![](low_confidence_corrects.png)",
            "",
            "## 简要结论",
            "",
            "- 如果 `cat/dog/bird/deer` 等细粒度自然物类别混淆较多，优先考虑更深网络、学习率调度、更强数据增强或 ResNet 对照实验。",
            "- 如果训练准确率和测试准确率一起偏低，说明当前手写 CNN 容量可能不足；如果训练高但测试低，则优先处理过拟合。",
            "- 本报告基于 `predictions.csv` 自动生成，后续更换权重或模型后重新运行 `evaluate.py` 即可得到同格式图表。",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
