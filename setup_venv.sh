#!/usr/bin/env bash
set -e

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url "$TORCH_CUDA_INDEX"
python -m pip install opencv-python pillow ultralytics tqdm

python - <<'PY'
import torch
print("python ok")
print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
