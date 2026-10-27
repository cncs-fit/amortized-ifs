#!/usr/bin/env bash
# Build a separate conda env for the official code of Djeacoumar et al. (2025): Python 3.11, PyTorch 2.6
# with CUDA 12.4, and their CUDA extensions.  Run fetch_chaotic_fractals.sh first.
# Env: CHAOS_ENV (default chaotic-fractals); TORCH_CUDA_ARCH_LIST (default 8.6, an RTX 3090).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${REPO:-$ROOT/refs/chaotic-fractals}"
ENV="${CHAOS_ENV:-chaotic-fractals}"
eval "$(conda shell.bash hook)"
if ! conda env list | grep -q "^$ENV "; then conda create -y -n "$ENV" python=3.11; fi
conda activate "$ENV"
conda install -y -c nvidia/label/cuda-12.4.1 cuda-nvcc cuda-cudart-dev cuda-libraries-dev
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
cd "$REPO"
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
pip install easydict nbconvert plyfile numba tensorboard ninja glfw imageio imageio-ffmpeg lpips tqdm opencv-python matplotlib click scipy pyopengl freetype-py wheel setuptools
pip install --no-build-isolation lib/cf_base lib/simple-knn lib/diff-isogaussian-rasterization
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); import diff_isogaussian_rasterization, simple_knn; print('ext ok')"
