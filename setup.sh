#!/usr/bin/env bash
# setup.sh — install aria2c, zstd, and Python deps for worker.py
# Tested on Ubuntu 20.04 / 22.04 with Python 3.8+

set -euo pipefail

sudo apt-get install -y \
    aria2 \
    zstd \
    python3-pip

python3 -m pip install --upgrade pip

python3 -m pip install \
    zstandard \
    tqdm

aria2c  --version | head -1
zstd    --version | head -1
python3 -c "import zstandard; print(f'zstandard {zstandard.__version__}')"
python3 -c "import tqdm; print(f'tqdm {tqdm.__version__}')"
