#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

sudo ln -sf "${ROOT}/.venv/bin/python" /usr/local/bin/python
sudo ln -sf "${ROOT}/.venv/bin/pip" /usr/local/bin/pip

mkdir -p "${ROOT}/.local/comfyui/input" "${ROOT}/.local/comfyui/output"
