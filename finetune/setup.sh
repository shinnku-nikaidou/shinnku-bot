#!/bin/bash
# Run once on a fresh Colab VM before train.py: the packages Unsloth's own
# Qwen3.5 MoE notebook installs (Qwen3.6 uses the same code).
set -euo pipefail
pip install -q --upgrade uv
numpy=$(python -c 'import numpy; print(numpy.__version__)')
pillow=$(python -c 'import PIL; print(PIL.__version__)')
uv pip install --system -q \
    "torch==2.8.0" "triton>=3.3.0" "numpy==$numpy" "pillow==$pillow" torchvision "torchaudio==2.8.0" \
    bitsandbytes xformers==0.0.32.post2 \
    "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
    "unsloth[base] @ git+https://github.com/unslothai/unsloth"
uv pip install --system -q --upgrade --no-deps "tokenizers>=0.22.0,<=0.23.0" trl==0.22.2 unsloth unsloth_zoo
uv pip install --system -q transformers==5.2.0
unsloth install-kernels
uv pip install --system -q --no-deps --upgrade "torchao>=0.16.0"
# Their notebook drops these too: torchcodec does not import on Colab.
pip uninstall -y -q sentence-transformers torchcodec || true
python -c 'import torch, unsloth; print("ready: torch", torch.__version__, "cuda", torch.cuda.is_available())'
