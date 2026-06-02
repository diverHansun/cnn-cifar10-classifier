import importlib
import unittest


class ConfigContractTests(unittest.TestCase):
    def test_default_config_has_server_friendly_paths_and_pth_checkpoints(self):
        config = importlib.import_module("config")

        cfg = config.TrainConfig()

        self.assertEqual(cfg.dataset_name, "uoft-cs/cifar10")
        self.assertEqual(cfg.dataset_config, "plain_text")
        self.assertEqual(cfg.device, "auto")
        self.assertEqual(cfg.checkpoint_dir.name, "checkpoints")
        self.assertEqual(cfg.output_dir.name, "outputs")
        self.assertEqual(cfg.run_dir.name, "runs")
        self.assertEqual(cfg.best_checkpoint_path.name, "best_model.pth")
        self.assertEqual(cfg.last_checkpoint_path.name, "last_model.pth")
        self.assertEqual(len(config.CLASS_NAMES), 10)
        self.assertEqual(config.CLASS_NAMES[0], "airplane")
        self.assertEqual(config.CLASS_NAMES[-1], "truck")

    def test_train_parser_can_override_core_server_parameters(self):
        train = importlib.import_module("train")

        parser = train.build_arg_parser()
        args = parser.parse_args(
            [
                "--epochs",
                "3",
                "--batch-size",
                "128",
                "--lr",
                "0.02",
                "--device",
                "cuda",
                "--num-workers",
                "8",
            ]
        )

        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.batch_size, 128)
        self.assertAlmostEqual(args.lr, 0.02)
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.num_workers, 8)


if __name__ == "__main__":
    unittest.main()
