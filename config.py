from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent

CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@dataclass
class TrainConfig:
    dataset_name: str = "uoft-cs/cifar10"
    dataset_config: str = "plain_text"
    batch_size: int = 256
    epochs: int = 20
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    num_workers: int = 8
    seed: int = 42
    device: str = "auto"
    use_amp: bool = True
    augment: bool = True
    pin_memory: bool = True
    data_dir: Path = PROJECT_ROOT / "data"
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints"
    output_dir: Path = PROJECT_ROOT / "outputs"
    run_dir: Path = PROJECT_ROOT / "runs"

    @property
    def best_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "best_model.pth"

    @property
    def last_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "last_model.pth"

    @property
    def training_curve_path(self) -> Path:
        return self.output_dir / "training_curves.png"

    @property
    def confusion_matrix_path(self) -> Path:
        return self.output_dir / "confusion_matrix.png"

    @property
    def demo_predictions_path(self) -> Path:
        return self.output_dir / "demo_predictions.png"

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.checkpoint_dir,
            self.output_dir,
            self.run_dir,
            PROJECT_ROOT / "demo_images",
        ):
            path.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in values.items()
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainConfig":
        path_fields = {"data_dir", "checkpoint_dir", "output_dir", "run_dir"}
        allowed_fields = {field.name for field in fields(cls)}
        clean_values = {}
        for key, value in values.items():
            if key not in allowed_fields:
                continue
            clean_values[key] = Path(value) if key in path_fields else value
        return cls(**clean_values)


def select_device(requested: str = "auto") -> str:
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("device must be one of: auto, cuda, cpu")

    if requested == "cpu":
        return "cpu"

    import torch

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return "cuda"

    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_checkpoint_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Train first with `python train.py`, "
            "or pass --checkpoint to an existing .pth file."
        )
    return path
