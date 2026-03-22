#!/usr/bin/env bash
# setup.sh — install aria2c, zstd, and Python deps for filter_worker.py
# Tested on Ubuntu 20.04 / 22.04 with Python 3.8+

set -euo pipefail

sudo apt-get update

sudo apt-get install -y \
    aria2 \
    zstd \
    python3-pip

python3 -m pip install --upgrade pip

python3 -m pip install \
    zstandard \
    tqdm \
    torf

aria2c  --version | head -1
zstd    --version | head -1
python3 -c "import zstandard; print(f'zstandard {zstandard.__version__}')"
python3 -c "import tqdm; print(f'tqdm {tqdm.__version__}')"
python3 -c "import torf; print(f'torf {torf.__version__}')"

# now copy the torrent you got from academictorrents
# https://academictorrents.com/details/3d426c47c767d40f82c7ef0f47c3acacedd2bf44
# (Run the next line with your ssh config & file locations etc. from your local machine)
# scp -P 30639 -i $env:USERPROFILE\.ssh\id_ed25519 "D:\Downloads\reddit-3d426c47c767d40f82c7ef0f47c3acacedd2bf44.torrent" root@213.173.111.104:/workspace/rsbookclub-datadumps/
