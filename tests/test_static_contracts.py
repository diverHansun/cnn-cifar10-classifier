import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class StaticProjectContractTests(unittest.TestCase):
    def test_expected_top_level_files_exist(self):
        expected_files = [
            "config.py",
            "dataset.py",
            "model.py",
            "train.py",
            "evaluate.py",
            "predict.py",
            "demo.py",
            "metrics.py",
            "visualize.py",
            "report.py",
            "requirements.txt",
            "README.md",
            "datasets/.gitkeep",
        ]

        for relative_path in expected_files:
            with self.subTest(relative_path=relative_path):
                self.assertTrue((ROOT / relative_path).exists(), f"Missing {relative_path}")

    def test_model_defines_simple_cnn_class(self):
        tree = ast.parse((ROOT / "model.py").read_text(encoding="utf-8"))

        class_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        }

        self.assertIn("SimpleCNN", class_names)

    def test_dataset_uses_hugging_face_cifar10_img_label_contract(self):
        source = (ROOT / "dataset.py").read_text(encoding="utf-8")

        self.assertIn("load_dataset", source)
        self.assertIn("uoft-cs/cifar10", source)
        self.assertIn("plain_text", source)
        self.assertIn('"img"', source)
        self.assertIn('"label"', source)
        self.assertNotIn("lambda examples", source)
        self.assertIn("ImageBatchTransform", source)

    def test_dataset_and_metrics_preserve_original_split_indices_for_demo_grids(self):
        dataset_source = (ROOT / "dataset.py").read_text(encoding="utf-8")
        metrics_source = (ROOT / "metrics.py").read_text(encoding="utf-8")
        train_source = (ROOT / "train.py").read_text(encoding="utf-8")

        self.assertIn("DATASET_INDEX_COLUMN", dataset_source)
        self.assertIn("with_indices=True", dataset_source)
        self.assertIn("dataset_index", dataset_source)
        self.assertIn("return images, labels, indices", dataset_source)
        self.assertIn("indices", metrics_source)
        self.assertIn('"index": int(indices[batch_index].item())', metrics_source)
        self.assertIn("unpack_batch", train_source)

    def test_train_saves_resume_friendly_checkpoint_dictionary(self):
        source = (ROOT / "train.py").read_text(encoding="utf-8")

        self.assertIn("torch.save", source)
        self.assertIn('"model_state_dict"', source)
        self.assertIn('"optimizer_state_dict"', source)
        self.assertIn('"scaler_state_dict"', source)
        self.assertIn('"best_acc"', source)
        self.assertIn("best_model.pth", source)

    def test_resume_restores_saved_config_before_training(self):
        source = (ROOT / "train.py").read_text(encoding="utf-8")

        self.assertIn("checkpoint_config", source)
        self.assertIn("TrainConfig.from_dict", source)
        self.assertIn("apply_arg_overrides", source)
        self.assertIn('scaler.load_state_dict', source)

    def test_predict_resizes_images_to_cifar10_shape(self):
        source = (ROOT / "predict.py").read_text(encoding="utf-8")

        self.assertIn("Resize((32, 32))", source)

    def test_checkpoint_users_validate_checkpoint_before_dataset_download(self):
        for filename in ("evaluate.py", "predict.py"):
            with self.subTest(filename=filename):
                source = (ROOT / filename).read_text(encoding="utf-8")
                self.assertIn("ensure_checkpoint_exists", source)

    def test_train_runs_cuda_fail_fast_probe(self):
        source = (ROOT / "train.py").read_text(encoding="utf-8")

        self.assertIn("probe_cuda_runtime", source)
        self.assertIn("get_device_capability", source)
        self.assertIn("get_arch_list", source)
        self.assertIn("ones((2, 2)", source)

    def test_train_logs_runtime_monitoring_metrics_to_tensorboard(self):
        source = (ROOT / "train.py").read_text(encoding="utf-8")

        expected_fragments = [
            "learning_rate",
            "accuracy/best",
            "performance/epoch_time_sec",
            "performance/images_per_second",
            "gpu/memory_allocated_mb",
            "gpu/memory_reserved_mb",
            "reset_peak_memory_stats",
        ]

        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)

    def test_evaluate_generates_report_ready_metrics_and_figures(self):
        source = (ROOT / "evaluate.py").read_text(encoding="utf-8")

        expected_fragments = [
            "predictions.csv",
            "per_class_metrics.csv",
            "top_confusions.csv",
            "eval_summary.json",
            "eval_report.md",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "per_class_precision.png",
            "per_class_recall.png",
            "per_class_f1.png",
            "top_confusions.png",
            "confidence_histogram.png",
            "error_samples.png",
            "high_confidence_errors.png",
            "low_confidence_corrects.png",
            "--num-samples",
            "--top-confusions",
            "--output-dir",
        ]

        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)

    def test_metrics_exports_prediction_rows_with_explainable_fields(self):
        source = (ROOT / "metrics.py").read_text(encoding="utf-8")

        expected_fields = [
            "true_label",
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
            "macro_f1",
            "weighted_f1",
        ]

        for field in expected_fields:
            with self.subTest(field=field):
                self.assertIn(field, source)

    def test_report_defaults_to_chinese_markdown_with_metric_explanations(self):
        source = (ROOT / "report.py").read_text(encoding="utf-8")

        expected_fragments = [
            "CIFAR-10 CNN 评估报告",
            "准确率",
            "精确率",
            "召回率",
            "F1",
            "混淆矩阵",
            "置信度",
            "错例",
        ]

        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)

    def test_visualize_uses_chinese_labels_for_report_figures(self):
        source = (ROOT / "visualize.py").read_text(encoding="utf-8")

        expected_fragments = [
            "混淆矩阵",
            "类别",
            "数量",
            "置信度",
            "正确",
            "错误",
            "真实",
            "预测",
        ]

        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)

    def test_demo_validates_sample_count(self):
        source = (ROOT / "demo.py").read_text(encoding="utf-8")

        self.assertIn("if samples < 1", source)

    def test_demo_reads_predictions_csv_and_supports_interpretable_modes(self):
        source = (ROOT / "demo.py").read_text(encoding="utf-8")

        expected_fragments = [
            "predictions.csv",
            "--predictions",
            "--mode",
            "random",
            "errors",
            "high-confidence-errors",
            "low-confidence-corrects",
            "confusion",
            "class",
            "--true-label",
            "--pred-label",
            "--class-name",
            "default=42",
            "outputs/demo",
        ]

        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)

    def test_demo_validates_labels_and_short_circuits_empty_selections(self):
        source = (ROOT / "demo.py").read_text(encoding="utf-8")

        expected_fragments = [
            "available_labels",
            "validate_label",
            "save_placeholder",
            "if not selected_rows",
            "Available labels",
        ]

        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)

    def test_readme_hf_download_wording_matches_unpinned_revision(self):
        source = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("latest published checkpoint", source)
        self.assertNotIn("published v0.1.0 checkpoint without retraining", source)

    def test_gitignore_keeps_dataset_folder_but_not_cached_data(self):
        source = (ROOT / ".gitignore").read_text(encoding="utf-8")

        self.assertIn("datasets/*", source)
        self.assertIn("!datasets/.gitkeep", source)
        self.assertNotIn("data/", source)


if __name__ == "__main__":
    unittest.main()
