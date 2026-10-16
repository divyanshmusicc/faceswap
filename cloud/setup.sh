#!/usr/bin/env bash
# One-time FaceFusion setup on any Ubuntu + NVIDIA box.
# Works on Vast.ai, RunPod and TensorDock (any PyTorch/CUDA template).
# Run:  bash setup.sh
set -euo pipefail

nvidia-smi --query-gpu=name,memory.total --format=csv || { echo "No NVIDIA GPU visible. Attach a GPU first."; exit 1; }
python3 --version

# Persistent location first (/workspace survives pod stops on RunPod;
# instance disk persists on Vast.ai while the instance exists).
if [ -d /workspace ]; then FF_ROOT=/workspace/facefusion; else FF_ROOT="$HOME/facefusion"; fi

if [ ! -f "$FF_ROOT/facefusion.py" ]; then
  git clone https://github.com/facefusion/facefusion "$FF_ROOT"
else
  echo "facefusion already cloned at $FF_ROOT"
fi

cd "$FF_ROOT"
python3 install.py cuda@12 --skip-conda

# Pre-download models now so first swap doesn't stall mid-run (~1-2 GB).
python3 facefusion.py force-download

echo "Setup done. Launch the web UI with the launch-ui.sh next to this script."
