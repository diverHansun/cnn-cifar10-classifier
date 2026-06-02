from __future__ import annotations

from typing import Callable

from config import TrainConfig


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
DEFAULT_DATASET_NAME = "uoft-cs/cifar10"
DEFAULT_DATASET_CONFIG = "plain_text"
DATASET_INDEX_COLUMN = "_dataset_index"


def build_transforms(augment: bool = True) -> tuple[Callable, Callable]:
    from torchvision import transforms

    train_steps = []
    if augment:
        train_steps.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    test_steps = [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]

    return transforms.Compose(train_steps), transforms.Compose(test_steps)


def transform_examples(examples: dict, transform: Callable) -> dict:
    examples["pixel_values"] = [
        transform(image.convert("RGB"))
        for image in examples["img"]
    ]
    return examples


class ImageBatchTransform:
    def __init__(self, transform: Callable) -> None:
        self.transform = transform

    def __call__(self, examples: dict) -> dict:
        return transform_examples(examples, self.transform)


def add_dataset_index(example: dict, index: int) -> dict:
    return {DATASET_INDEX_COLUMN: index}


def ensure_dataset_indices(raw_dataset):
    indexed_splits = {}
    for split_name, split in raw_dataset.items():
        if DATASET_INDEX_COLUMN in split.column_names:
            indexed_splits[split_name] = split
            continue
        indexed_splits[split_name] = split.map(
            add_dataset_index,
            with_indices=True,
            desc=f"add {split_name} indices",
        )
    return indexed_splits


def collate_fn(examples: list[dict]):
    import torch

    images = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples], dtype=torch.long)
    indices = torch.tensor(
        [example["dataset_index"] if "dataset_index" in example else example[DATASET_INDEX_COLUMN] for example in examples],
        dtype=torch.long,
    )
    return images, labels, indices


def load_raw_dataset(cfg: TrainConfig):
    from datasets import load_dataset

    return load_dataset(
        cfg.dataset_name,
        cfg.dataset_config,
        cache_dir=str(cfg.data_dir / "hf_cache"),
    )


def build_dataloaders_from_dataset(raw_dataset, cfg: TrainConfig):
    from torch.utils.data import DataLoader

    raw_dataset = ensure_dataset_indices(raw_dataset)
    train_transform, test_transform = build_transforms(cfg.augment)

    train_dataset = raw_dataset["train"].with_transform(ImageBatchTransform(train_transform))
    test_dataset = raw_dataset["test"].with_transform(ImageBatchTransform(test_transform))

    common_kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "collate_fn": collate_fn,
    }
    if cfg.num_workers > 0:
        common_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **common_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_kwargs)
    return train_loader, test_loader


def build_dataloaders(cfg: TrainConfig):
    raw_dataset = load_raw_dataset(cfg)
    return build_dataloaders_from_dataset(raw_dataset, cfg)
