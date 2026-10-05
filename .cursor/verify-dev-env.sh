#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "==> unit tests"
python -m unittest discover -s tests -p 'test_*.py' -v

echo "==> handler dry_run with built Wan Animate workflow graph"
python <<'PY'
import json
import os
import tempfile
from pathlib import Path

import handler as handler_module
from handler import handler
from workflow import build_prompt

prompt = build_prompt(
    {
        "reference_image_name": "reference.png",
        "pose_video_name": "pose.mp4",
        "width": 482,
        "height": 854,
    }
)

with tempfile.TemporaryDirectory() as tmp:
    handler_module.COMFY_INPUT_DIR = Path(tmp)
    os.environ["COMFY_INPUT_DIR"] = tmp
    result = handler(
        {
            "id": "cloud-agent-verify",
            "input": {"dry_run": True, "workflow": prompt},
        }
    )

assert result.get("status") == "dry_run", result
assert result["node_count"] > 20, result
assert result["workflow"]["911"]["class_type"] == "SaveVideo"
print(json.dumps({"status": result["status"], "node_count": result["node_count"]}, indent=2))
PY

echo "==> verification complete"
