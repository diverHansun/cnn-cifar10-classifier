from __future__ import annotations

import argparse
from pathlib import Path

from config import CLASS_NAMES, TrainConfig, ensure_checkpoint_exists, select_device


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict a single CIFAR-10 style image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=TrainConfig().best_checkpoint_path)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--top-k", type=int, default=3)
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
    return model, class_names


def predict_image(image_path: Path, checkpoint_path: Path, device: str, top_k: int = 3):
    import torch
    from PIL import Image
    from torchvision import transforms

    from dataset import build_transforms

    _, test_transform = build_transforms(augment=False)
    predict_transform = transforms.Compose([transforms.Resize((32, 32)), test_transform])
    model, class_names = load_model(checkpoint_path, device)
    image = Image.open(image_path).convert("RGB")
    tensor = predict_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0)
        scores, indices = torch.topk(probabilities, k=min(top_k, len(class_names)))

    return [
        (class_names[index.item()], score.item())
        for score, index in zip(scores, indices)
    ]


def main() -> None:
    args = build_arg_parser().parse_args()
    ensure_checkpoint_exists(args.checkpoint)
    device = select_device(args.device)
    results = predict_image(args.image, args.checkpoint, device, args.top_k)
    print(f"image: {args.image}")
    for label, score in results:
        print(f"{label}: {score:.4f}")


if __name__ == "__main__":
    main()
