#!/usr/bin/env bash
set -euo pipefail
# Ubuntu 22.04 / Python 3.10 / NVIDIA GPU. Run as root in WSL.
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip
python3 -m venv /opt/hand-mamba
PY=/opt/hand-mamba/bin/python
"$PY" -m pip install --upgrade pip
"$PY" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
"$PY" -m pip install 'https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4%2Bcu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl' -r /mnt/c/Project/Model/requirements.txt
"$PY" -m pip check
cd /mnt/c/Project/Model
"$PY" verify_environment.py
