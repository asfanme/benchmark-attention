# Environment setup

This repository uses Python 3.11.

## Option 1 — pip

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate       # Windows

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Option 2 — Conda

```bash
conda env create -f environment.yml
conda activate convnext-attention-ham10000
```

## GPU note

The repository is compatible with CPU, Apple Silicon/MPS, and NVIDIA CUDA,
subject to the installed PyTorch build. For NVIDIA systems, use the official
PyTorch installation selector when a CUDA-specific wheel is required, while
keeping the `torch` and `torchvision` versions paired.

## Main dependencies

- PyTorch and torchvision for ConvNeXt-Tiny training and Grad-CAM
- scikit-learn for cross-validation and evaluation metrics
- SciPy and statsmodels for paired statistical tests and Holm correction
- pandas and NumPy for result processing
- matplotlib and Pillow for figures and image loading
