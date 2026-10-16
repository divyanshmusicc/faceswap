# FaceFusion web UI as a SaladCloud-ready container.
# - Ubuntu 24.04 + CUDA 12.8 runtime (matches FaceFusion's cuda@12 target)
# - FaceFusion installed via its official installer, models baked in at build
#   time so containers start swapping immediately (Salad storage is ephemeral).
# - Gradio binds IPv4-only; Salad's gateway needs IPv6/dual-stack, so the
#   entrypoint bridges [::]:7860 -> 127.0.0.1:7860 with socat.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git curl socat ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone --depth 1 https://github.com/facefusion/facefusion /app

RUN python3 install.py cuda@12 --skip-conda
RUN python3 facefusion.py force-download

COPY entrypoint.sh /app/entrypoint.sh

EXPOSE 7860
ENTRYPOINT ["bash", "/app/entrypoint.sh"]
