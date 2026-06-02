# cnn-cifar10-classifier

A small PyTorch CNN project for CIFAR-10 image classification using the Hugging Face `uoft-cs/cifar10` dataset.

The v0.1.0 goal is intentionally simple: hand-write a CNN, train it end to end, save `.pth` checkpoints, and generate report-friendly evaluation figures. ResNet and AdamW are planned as later comparison versions.

## Project Layout

```text
cnn-cifar10-classifier/
|-- config.py
|-- dataset.py
|-- model.py
|-- train.py
|-- evaluate.py
|-- predict.py
|-- demo.py
|-- checkpoints/
|   `-- best_model.pth
|-- outputs/
|   |-- training_curves.png
|   |-- confusion_matrix.png
|   `-- demo_predictions.png
|-- runs/
|-- demo_images/
|-- tests/
`-- requirements.txt
```

## Ubuntu GPU Server Setup

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a CUDA build of PyTorch that supports the server image. Use the current official PyTorch selector for Linux + Pip + Python and choose the CUDA platform suited to the rented machine. The RTX 5090 is a Blackwell GPU with compute capability 12.0, so avoid old CUDA wheels that do not support it.

```bash
# Example only. Prefer the command currently shown by https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Check the server driver before training. NVIDIA's CUDA release notes list minimum driver ranges by CUDA toolkit version; for example, CUDA 12.x requires a compatible 525+ driver range, while CUDA 13.x requires a 580+ driver range. If the server image is managed by the rental provider, choose a PyTorch wheel that matches the installed driver or ask for a newer image.

Before training, verify CUDA:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_arch_list() if torch.cuda.is_available() else [])
PY
```

`train.py` also runs a tiny CUDA tensor operation at startup so an unsupported wheel/driver combination fails before dataset download and training.

## Train

```bash
python train.py --epochs 20 --batch-size 256 --lr 0.01 --device cuda --num-workers 8
```

Training writes:

```text
checkpoints/best_model.pth
checkpoints/last_model.pth
outputs/training_curves.png
outputs/training_metrics.json
runs/cifar10_cnn_YYYYMMDD_HHMMSS/
```

The checkpoint is a dictionary with model weights, optimizer state, AMP scaler state, epoch, best accuracy, class names, config, and history. `.pth` files are generated artifacts and are not committed to Git. A fresh clone must train first, or you must provide an existing checkpoint path with `--checkpoint`.

## Evaluate

Evaluation requires a trained checkpoint:

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --device cuda
```

This saves:

```text
outputs/confusion_matrix.png
```

## Predict One Image

Put your own image under `demo_images/` first. Images are ignored by Git, so a fresh clone only contains the directory placeholder.
Prediction also requires a trained checkpoint. Input images are resized to 32x32 before inference.

```bash
python predict.py --image demo_images/your_image.png --checkpoint checkpoints/best_model.pth --device cuda
```

## Demo Predictions

```bash
python demo.py --checkpoint checkpoints/best_model.pth --samples 16 --device cuda
```

This saves:

```text
outputs/demo_predictions.png
```

## Local Checks

These checks do not require installing PyTorch locally:

```bash
python -m unittest discover -s tests
python -m compileall .
```

Real training/evaluation requires the server dependencies above.

## Version Roadmap

- v0.1.0: simple hand-written CNN, SGD momentum baseline, `.pth` checkpoints.
- v0.1.1: richer plots, wrong-sample inspection, report refinements.
- v0.2.0: AdamW and learning-rate schedule comparison.
- v0.3.0: CIFAR-style ResNet comparison.
