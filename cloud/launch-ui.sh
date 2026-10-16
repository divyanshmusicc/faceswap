#!/usr/bin/env bash
# Launch the full FaceFusion web UI (same panel as desktop).
# Gradio serves on 0.0.0.0:7860 by default, then:
#   Vast.ai / TensorDock: expose port 7860 on the instance, open http://<host-ip>:<mapped-port>
#   RunPod: expose 7860 as HTTP, open https://<pod-id>-7860.proxy.runpod.net
set -euo pipefail

if [ -d /workspace/facefusion ]; then FF_ROOT=/workspace/facefusion; else FF_ROOT="$HOME/facefusion"; fi
cd "$FF_ROOT"

if pgrep -f "facefusion.py run" > /dev/null; then
  echo "UI already running. Tail the log with:  tail -f ui.log"
  exit 0
fi

nohup python3 facefusion.py run > ui.log 2>&1 &
echo "UI starting in background (log: $FF_ROOT/ui.log)."
echo "Give it ~30s, then open your mapped port URL. Troubleshoot with:  tail -f ui.log"
