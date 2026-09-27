#!/usr/bin/env bash
# Download the Wan Animate 2 distilled weights used by the workflow.
# Files that already exist under the network volume or the image are skipped.
set -euo pipefail

COMFY_MODELS="${COMFY_MODELS:-/comfyui/models}"
VOLUME_MODELS="${VOLUME_MODELS:-/runpod-volume/models}"
BASE_URL="${WAN_MODEL_BASE_URL:-https://huggingface.co/Comfy-Org/Wan-Animate-2/resolve/main}"

FILES=(
  "diffusion_models/wan_animate_2_distill_int8_convrot.safetensors"
  "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
  "clip_vision/clip_vision_h.safetensors"
  "vae/Wan2_1_VAE_bf16.safetensors"
)

if [[ -d /runpod-volume && -w /runpod-volume ]]; then
  DEST_ROOT="${VOLUME_MODELS}"
else
  DEST_ROOT="${COMFY_MODELS}"
fi

mkdir -p "${DEST_ROOT}"

auth=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  auth=(-H "Authorization: Bearer ${HF_TOKEN}")
fi

for relative in "${FILES[@]}"; do
  if [[ -s "${COMFY_MODELS}/${relative}" || -s "${VOLUME_MODELS}/${relative}" ]]; then
    echo "model present, skipping ${relative}"
    continue
  fi
  target="${DEST_ROOT}/${relative}"
  mkdir -p "$(dirname "${target}")"
  echo "downloading ${relative} -> ${target}"
  wget -q --show-progress --progress=dot:giga --tries=5 --timeout=60 \
    "${auth[@]}" \
    -O "${target}.partial" \
    "${BASE_URL}/${relative}"
  mv "${target}.partial" "${target}"
done

echo "model download complete"
