# Wan Animate 2 distilled ComfyUI worker for RunPod Serverless.
# Weights are about 25 GB. Set --build-arg DOWNLOAD_MODELS=false to skip them
# and download on first boot instead (use a network volume in that case).
ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    COMFYUI_VERSION=v0.37.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        git \
        wget \
        ca-certificates \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv \
    && pip install --upgrade pip setuptools wheel

# cu128 wheels run on driver >= 570 (CUDA 12.8+ hosts). Install torch before
# ComfyUI requirements so a bare `torch` pin does not replace them.
RUN pip install --no-cache-dir \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
        --index-url https://download.pytorch.org/whl/cu128

RUN git clone --depth 1 --branch ${COMFYUI_VERSION} https://github.com/Comfy-Org/ComfyUI.git /comfyui \
    && pip install --no-cache-dir -r /comfyui/requirements.txt \
    && pip install --no-cache-dir "transformers>=4.50.3,<5" "huggingface-hub<1.0" \
        runpod~=1.7.9 requests

WORKDIR /

COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml
COPY scripts/download_models.sh /download_models.sh
COPY scripts/start.sh /start.sh
COPY handler.py workflow.py / 
RUN chmod +x /download_models.sh /start.sh \
    && mkdir -p /comfyui/input /comfyui/output /comfyui/models

ARG DOWNLOAD_MODELS=true
RUN if [ "${DOWNLOAD_MODELS}" = "true" ]; then /download_models.sh; fi

CMD ["/start.sh"]
