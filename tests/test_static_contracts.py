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
            "requirements.txt",
            "README.md",
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

    def test_evaluate_and_demo_validate_checkpoint_before_dataset_download(self):
        for filename in ("evaluate.py", "demo.py", "predict.py"):
            with self.subTest(filename=filename):
                source = (ROOT / filename).read_text(encoding="utf-8")
                self.assertIn("ensure_checkpoint_exists", source)

    def test_train_runs_cuda_fail_fast_probe(self):
        source = (ROOT / "train.py").read_text(encoding="utf-8")

        self.assertIn("probe_cuda_runtime", source)
        self.assertIn("get_device_capability", source)
        self.assertIn("get_arch_list", source)
        self.assertIn("ones((2, 2)", source)

    def test_demo_validates_sample_count(self):
        source = (ROOT / "demo.py").read_text(encoding="utf-8")

        self.assertIn("if samples < 1", source)


if __name__ == "__main__":
    unittest.main()
