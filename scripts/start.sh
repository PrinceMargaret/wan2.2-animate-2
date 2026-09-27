#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/venv/bin:${PATH}"
export PYTHONUNBUFFERED=1

if [[ "${DOWNLOAD_ON_START:-true}" == "true" ]]; then
  /download_models.sh
fi

cd /comfyui
python main.py \
  --listen 127.0.0.1 \
  --port 8188 \
  --disable-auto-launch \
  --preview-method none \
  --extra-model-paths-config /comfyui/extra_model_paths.yaml \
  ${COMFY_EXTRA_ARGS:-} &
echo $! > /tmp/comfyui.pid

cd /
exec python -u /handler.py
