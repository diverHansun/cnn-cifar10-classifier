# cnn-cifar10-classifier

A small PyTorch CNN project for CIFAR-10 image classification using the Hugging Face `uoft-cs/cifar10` dataset.

The v0.1.0 goal is intentionally simple: hand-write a CNN, train it end to end, save `.pth` checkpoints, and generate report-friendly evaluation figures. ResNet and AdamW are planned as later comparison versions.

## v0.1.0 Release

The trained v0.1.0 baseline checkpoint and report artifacts are published on Hugging Face:

https://huggingface.co/diverWayne/cnn-cifar10-classifier

v0.1.0 reached `0.7857` test accuracy on the `uoft-cs/cifar10` `plain_text` test split. The Hugging Face release includes `best_model.pth`, `last_model.pth`, training metrics, training curves, a confusion matrix, demo predictions, logs, a TensorBoard event file, and a checksum manifest.

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
|-- metrics.py
|-- visualize.py
|-- report.py
|-- checkpoints/
|   |-- best_model.pth
|   `-- last_model.pth
|-- datasets/
|   `-- hf_cache/
|-- outputs/
|   |-- training_curves.png
|   |-- training_metrics.json
|   |-- eval/
|   |   |-- eval_report.md
|   |   |-- eval_summary.json
|   |   |-- predictions.csv
|   |   |-- per_class_metrics.csv
|   |   |-- top_confusions.csv
|   |   |-- confusion_matrix.png
|   |   |-- confusion_matrix_normalized.png
|   |   |-- per_class_precision.png
|   |   |-- per_class_recall.png
|   |   |-- per_class_f1.png
|   |   |-- top_confusions.png
|   |   |-- confidence_histogram.png
|   |   |-- error_samples.png
|   |   |-- high_confidence_errors.png
|   |   `-- low_confidence_corrects.png
|   `-- demo/
|       |-- demo_random.png
|       |-- demo_errors.png
|       `-- demo_confusion_cat_to_dog.png
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
datasets/hf_cache/
outputs/training_curves.png
outputs/training_metrics.json
runs/cifar10_cnn_YYYYMMDD_HHMMSS/
```

Dataset files are cached under `datasets/hf_cache/`. Model weights are saved under `checkpoints/`: `best_model.pth` is the best validation checkpoint, and `last_model.pth` is the latest epoch checkpoint for resume.

The checkpoint is a dictionary with model weights, optimizer state, AMP scaler state, epoch, best accuracy, class names, config, and history. `.pth` files are generated artifacts and are not committed to Git. A fresh clone must train first, or you must provide an existing checkpoint path with `--checkpoint`.

Monitor training with TensorBoard:

```bash
tensorboard --logdir runs --host 0.0.0.0 --port 8080
```

The training script logs train/test loss, train/test accuracy, best accuracy, learning rate, epoch time, images per second, and GPU memory peaks. On Vast.ai, keep the SSH tunnel mapping `-L 8080:localhost:8080`, then open `http://localhost:8080` on your local machine.

## Download Published Weights

If you want to evaluate the latest published checkpoint without retraining:

```bash
hf download diverWayne/cnn-cifar10-classifier checkpoints/best_model.pth --type model --local-dir .
```

## Evaluate

Evaluation requires a trained checkpoint:

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --device cuda
```

This saves:

```text
outputs/eval/eval_report.md
outputs/eval/eval_summary.json
outputs/eval/predictions.csv
outputs/eval/per_class_metrics.csv
outputs/eval/top_confusions.csv
outputs/eval/confusion_matrix.png
outputs/eval/confusion_matrix_normalized.png
outputs/eval/per_class_precision.png
outputs/eval/per_class_recall.png
outputs/eval/per_class_f1.png
outputs/eval/top_confusions.png
outputs/eval/confidence_histogram.png
outputs/eval/error_samples.png
outputs/eval/high_confidence_errors.png
outputs/eval/low_confidence_corrects.png
```

The Markdown report is generated in Chinese by default. It explains accuracy, precision, recall, F1, Macro F1, Weighted F1, Top-3 accuracy, loss, confidence, confusion pairs, and representative error samples.

## Predict One Image

Put your own image under `demo_images/` first. Images are ignored by Git, so a fresh clone only contains the directory placeholder.
Prediction also requires a trained checkpoint. Input images are resized to 32x32 before inference.

```bash
python predict.py --image demo_images/your_image.png --checkpoint checkpoints/best_model.pth --device cuda
```

## Demo Predictions

```bash
python demo.py --mode random --samples 24
```

This saves:

```text
outputs/demo/demo_random.png
```

`demo.py` reads `outputs/eval/predictions.csv`, so run `evaluate.py` first. It does not need to run inference again.

Useful demo modes:

```bash
python demo.py --mode random --seed 42
python demo.py --mode errors
python demo.py --mode high-confidence-errors
python demo.py --mode low-confidence-corrects
python demo.py --mode confusion --true-label cat --pred-label dog
python demo.py --mode class --class-name cat
```

Demo filenames follow the selected mode:

```text
outputs/demo/demo_random.png
outputs/demo/demo_errors.png
outputs/demo/demo_high_confidence_errors.png
outputs/demo/demo_low_confidence_corrects.png
outputs/demo/demo_confusion_<true>_to_<pred>.png
outputs/demo/demo_class_<class>.png
```

## Local Checks

These checks do not require installing PyTorch locally:

```bash
python -m unittest discover -s tests
python -m compileall .
```

Real training/evaluation requires the server dependencies above.

## Version Roadmap

- v0.1.0: released simple hand-written CNN, SGD momentum baseline, `.pth` checkpoints.
- v0.1.1: richer plots, wrong-sample inspection, report refinements.
- v0.2.0: AdamW and learning-rate schedule comparison.
- v0.3.0: CIFAR-style ResNet comparison.
