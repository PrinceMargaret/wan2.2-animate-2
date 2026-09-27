"""RunPod serverless handler that executes any ComfyUI API workflow."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
import traceback
import uuid
from pathlib import Path
from urllib.parse import urlparse

import requests

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
COMFY_INPUT_DIR = Path(os.environ.get("COMFY_INPUT_DIR", "/comfyui/input"))
COMFY_TIMEOUT_S = int(os.environ.get("COMFY_TIMEOUT_S", "3600"))
MAX_INPUT_BYTES = int(os.environ.get("MAX_INPUT_MB", "200")) * 1024 * 1024
POLL_INTERVAL_S = float(os.environ.get("COMFY_POLL_INTERVAL_S", "1.0"))

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".gif", ".mov"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


class JobError(Exception):
    """A user-facing job failure."""


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_name(name: str, fallback: str) -> str:
    base = os.path.basename(str(name).replace("\\", "/").strip()) or fallback
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in base)
    return cleaned[:180] or fallback


def _decode_base64(payload: str) -> bytes:
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    try:
        return base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise JobError("Could not decode base64 media. Pass raw base64 or a data URI.") from exc


def _download_url(url: str) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise JobError("Media URLs must start with http:// or https://")
    try:
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            chunks = []
            total = 0
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_INPUT_BYTES:
                    raise JobError(f"Input media exceeds MAX_INPUT_MB={MAX_INPUT_BYTES // (1024 * 1024)}")
                chunks.append(chunk)
    except JobError:
        raise
    except Exception as exc:
        raise JobError(f"Failed to download media: {exc}") from exc
    return b"".join(chunks)


def _media_bytes(value: str) -> bytes:
    if not isinstance(value, str) or not value.strip():
        raise JobError("Media input must be a URL or base64 string")
    text = value.strip()
    if text.startswith("http://") or text.startswith("https://"):
        data = _download_url(text)
    else:
        data = _decode_base64(text)
    if len(data) > MAX_INPUT_BYTES:
        raise JobError(f"Input media exceeds MAX_INPUT_MB={MAX_INPUT_BYTES // (1024 * 1024)}")
    if not data:
        raise JobError("Input media is empty")
    return data


def _write_input(data: bytes, filename: str) -> str:
    COMFY_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = (COMFY_INPUT_DIR / filename).resolve()
    if COMFY_INPUT_DIR.resolve() not in target.parents and target != COMFY_INPUT_DIR.resolve():
        raise JobError(f"Refusing to write outside the ComfyUI input directory: {filename}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return filename


def _normalize_workflow(raw) -> dict:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise JobError("workflow is not valid JSON") from exc
    if not isinstance(raw, dict) or not raw:
        raise JobError("workflow must be a non-empty ComfyUI API prompt object")

    if isinstance(raw.get("prompt"), dict) and not any(
        isinstance(value, dict) and "class_type" in value for value in raw.values()
    ):
        raw = raw["prompt"]

    if "nodes" in raw and "links" in raw:
        raise JobError(
            "This is the ComfyUI editor workflow. In ComfyUI use Workflow > Export (API) and send that JSON as input.workflow."
        )

    prompt = {}
    for node_id, node in raw.items():
        if not isinstance(node, dict) or "class_type" not in node or "inputs" not in node:
            raise JobError(
                f"Node {node_id} is not an API workflow node. Each entry needs class_type and inputs. "
                "Export the graph with Workflow > Export (API)."
            )
        if not isinstance(node["inputs"], dict):
            raise JobError(f"Node {node_id} inputs must be an object")
        prompt[str(node_id)] = node
    return prompt


def _store_media_list(items, payload_key: str, fallback_ext: str) -> list[str]:
    if items is None:
        return []
    if not isinstance(items, list):
        raise JobError(f"{payload_key} must be a list")
    saved = []
    seen = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise JobError(f"{payload_key}[{index}] must be an object")
        payload = item.get(payload_key) or item.get("image") or item.get("video") or item.get("data") or item.get("url")
        if not payload:
            raise JobError(f"{payload_key}[{index}] needs a URL or base64 payload")
        fallback = f"input_{index}{fallback_ext}"
        name = _safe_name(item.get("name") or fallback, fallback)
        if name in seen:
            raise JobError(f"Duplicate input filename: {name}")
        seen.add(name)
        _write_input(_media_bytes(payload), name)
        saved.append(name)
    return saved


def _wait_for_server() -> None:
    deadline = time.time() + int(os.environ.get("COMFY_STARTUP_TIMEOUT_S", "300"))
    url = f"http://{COMFY_HOST}/system_stats"
    last_error = "not started"
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise JobError(f"ComfyUI at {COMFY_HOST} did not become ready ({last_error})")


def _queue_prompt(prompt: dict, comfy_org_api_key: str | None) -> str:
    client_id = str(uuid.uuid4())
    body: dict = {"prompt": prompt, "client_id": client_id}
    if comfy_org_api_key:
        body["extra_data"] = {"api_key_comfy_org": comfy_org_api_key}
    response = requests.post(f"http://{COMFY_HOST}/prompt", json=body, timeout=60)
    if response.status_code != 200:
        raise JobError(f"ComfyUI rejected the prompt ({response.status_code}): {response.text[:2000]}")
    payload = response.json()
    if payload.get("node_errors"):
        raise JobError(f"ComfyUI node errors: {json.dumps(payload['node_errors'])[:4000]}")
    prompt_id = payload.get("prompt_id")
    if not prompt_id:
        raise JobError(f"ComfyUI did not return a prompt_id: {payload}")
    return prompt_id


def _wait_for_prompt(prompt_id: str) -> dict:
    deadline = time.time() + COMFY_TIMEOUT_S
    url = f"http://{COMFY_HOST}/history/{prompt_id}"
    while time.time() < deadline:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        history = response.json()
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status") or {}
            if status.get("status_str") == "error":
                messages = status.get("messages") or entry.get("outputs")
                raise JobError(f"ComfyUI execution failed: {json.dumps(messages)[:4000]}")
            if status.get("completed") or entry.get("outputs"):
                return entry
        time.sleep(POLL_INTERVAL_S)
    raise JobError(f"ComfyUI job timed out after {COMFY_TIMEOUT_S}s")


def _iter_output_files(entry: dict):
    outputs = entry.get("outputs") or {}
    for node_outputs in outputs.values():
        if not isinstance(node_outputs, dict):
            continue
        for items in node_outputs.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict) and item.get("filename"):
                    yield item


def _fetch_output(item: dict) -> bytes:
    response = requests.get(
        f"http://{COMFY_HOST}/view",
        params={
            "filename": item["filename"],
            "subfolder": item.get("subfolder", ""),
            "type": item.get("type", "output"),
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.content


def _maybe_upload(job_id: str, filename: str, data: bytes) -> dict:
    suffix = Path(filename).suffix or ".bin"
    if os.environ.get("BUCKET_ENDPOINT_URL"):
        from runpod.serverless.utils import rp_upload

        temp_path = Path("/tmp") / f"{uuid.uuid4().hex}{suffix}"
        temp_path.write_bytes(data)
        try:
            url = rp_upload.upload_image(job_id, str(temp_path))
        finally:
            temp_path.unlink(missing_ok=True)
        return {"filename": filename, "type": "s3_url", "data": url}

    return {
        "filename": filename,
        "type": "base64",
        "mime": mimetypes.guess_type(filename)[0] or "application/octet-stream",
        "data": base64.b64encode(data).decode("utf-8"),
    }


def _bucket_for(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "images"
    if suffix in VIDEO_EXTENSIONS:
        return "videos"
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    return "files"


def handler(job: dict) -> dict:
    """Run the ComfyUI API workflow supplied in the job input."""
    job_input = job.get("input") or {}
    job_id = job.get("id") or uuid.uuid4().hex
    try:
        if "workflow" not in job_input:
            raise JobError("workflow is required. Pass the ComfyUI API JSON from Workflow > Export (API).")
        prompt = _normalize_workflow(job_input.get("workflow"))
        if _as_bool(job_input.get("dry_run"), False):
            return {"status": "dry_run", "node_count": len(prompt), "workflow": prompt}

        saved_images = _store_media_list(job_input.get("images"), "image", ".png")
        saved_videos = _store_media_list(job_input.get("videos"), "video", ".mp4")
        saved_files = _store_media_list(job_input.get("files"), "data", ".bin")

        _wait_for_server()
        prompt_id = _queue_prompt(prompt, job_input.get("comfy_org_api_key") or os.environ.get("COMFY_ORG_API_KEY"))
        entry = _wait_for_prompt(prompt_id)

        grouped = {"images": [], "videos": [], "audio": [], "files": []}
        seen = set()
        for item in _iter_output_files(entry):
            filename = item["filename"]
            key = (filename, item.get("subfolder"), item.get("type"))
            if key in seen:
                continue
            seen.add(key)
            data = _fetch_output(item)
            grouped[_bucket_for(filename)].append(_maybe_upload(job_id, filename, data))

        result = {
            "prompt_id": prompt_id,
            "images": grouped["images"],
            "videos": grouped["videos"],
            "audio": grouped["audio"],
            "files": grouped["files"],
            "inputs": {"images": saved_images, "videos": saved_videos, "files": saved_files},
        }
        if not any(grouped.values()):
            result["outputs"] = entry.get("outputs") or {}
        return result
    except JobError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": str(exc), "details": traceback.format_exc()[-4000:]}


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
