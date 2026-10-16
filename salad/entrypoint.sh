#!/usr/bin/env bash
# SaladCloud entrypoint: bridge IPv6 -> Gradio, then run the FaceFusion web UI.
# The container must keep a foreground process alive; `facefusion.py run` is it.
set -euo pipefail
cd /app

# Salad's Container Gateway reaches us over IPv6, but Gradio listens on
# IPv4-only 0.0.0.0:7860, so forward dual-stack [::]:7860 to it.
socat 'TCP6-LISTEN:7860,reuseaddr,fork,ipv6only=0' 'TCP:127.0.0.1:7860' &
echo "[entrypoint] IPv6 bridge up, starting FaceFusion UI..."

exec python3 facefusion.py run
