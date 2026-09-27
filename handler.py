"""RunPod serverless handler for the Wan Animate 2 distilled ComfyUI workflow."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import subprocess
import time
import traceback
import uuid
from pathlib import Path
from urllib.parse import urlparse

import requests

import workflow

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
COMFY_INPUT_DIR = Path(os.environ.get("COMFY_INPUT_DIR", "/comfyui/input"))
COMFY_OUTPUT_DIR = Path(os.environ.get("COMFY_OUTPUT_DIR", "/comfyui/output"))
COMFY_TIMEOUT_S = int(os.environ.get("COMFY_TIMEOUT_S", "3600"))
MAX_INPUT_BYTES = int(os.environ.get("MAX_INPUT_MB", "200")) * 1024 * 1024
POLL_INTERVAL_S = float(os.environ.get("COMFY_POLL_INTERVAL_S", "1.0"))


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
    base = os.path.basename(name.strip()) or fallback
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
    path = COMFY_INPUT_DIR / filename
    path.write_bytes(data)
    return filename


def _reference_bytes(job_input: dict) -> tuple[bytes, str]:
    if job_input.get("reference_image"):
        raw = job_input["reference_image"]
        name = _safe_name(str(job_input.get("reference_image_name") or "reference.png"), "reference.png")
        if raw.strip().startswith("http"):
            guessed = os.path.basename(urlparse(raw).path)
            if guessed and "." in guessed and job_input.get("reference_image_name") is None:
                name = _safe_name(guessed, "reference.png")
        return _media_bytes(raw), name
    raise JobError("reference_image is required (URL or base64)")


def _pose_bytes(job_input: dict) -> tuple[bytes, str]:
    if job_input.get("pose_video"):
        raw = job_input["pose_video"]
        name = _safe_name(str(job_input.get("pose_video_name") or "pose.mp4"), "pose.mp4")
        if raw.strip().startswith("http"):
            guessed = os.path.basename(urlparse(raw).path)
            if guessed and "." in guessed and job_input.get("pose_video_name") is None:
                name = _safe_name(guessed, "pose.mp4")
        return _media_bytes(raw), name
    raise JobError("pose_video is required (URL or base64)")


def _probe_frames(path: Path) -> int:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(completed.stdout or "{}")
        streams = payload.get("streams") or []
        raw = streams[0].get("nb_frames") if streams else None
        if raw not in (None, "N/A"):
            count = int(raw)
            if count > 0:
                return count
    except Exception:
        pass

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise JobError("Could not read a video stream from pose_video")
    return int(streams[0].get("nb_read_frames") or 0)


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


def _queue_prompt(prompt: dict) -> str:
    client_id = str(uuid.uuid4())
    response = requests.post(
        f"http://{COMFY_HOST}/prompt",
        json={"prompt": prompt, "client_id": client_id},
        timeout=60,
    )
    if response.status_code != 200:
        raise JobError(f"ComfyUI rejected the prompt ({response.status_code}): {response.text[:2000]}")
    body = response.json()
    if body.get("node_errors"):
        raise JobError(f"ComfyUI node errors: {json.dumps(body['node_errors'])[:4000]}")
    prompt_id = body.get("prompt_id")
    if not prompt_id:
        raise JobError(f"ComfyUI did not return a prompt_id: {body}")
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
    suffix = Path(filename).suffix or ".mp4"
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
        "mime": mimetypes.guess_type(filename)[0] or "video/mp4",
        "data": base64.b64encode(data).decode("utf-8"),
    }


def _options_from_input(job_input: dict, reference_name: str, pose_name: str, frames: int | None) -> dict:
    length = workflow.align_length(job_input.get("length", workflow.DEFAULT_LENGTH))
    max_chunks = int(job_input.get("max_chunks", 8))
    if "chunks" in job_input and job_input["chunks"] is not None:
        chunks = int(job_input["chunks"])
    else:
        chunks = workflow.chunk_count(
            frames,
            length,
            _as_bool(job_input.get("match_video_length"), False),
            max_chunks,
        )
    options = {
        "reference_image_name": reference_name,
        "pose_video_name": pose_name,
        "prompt": job_input.get("prompt"),
        "pose_prompt": job_input.get("pose_prompt"),
        "negative_prompt": job_input.get("negative_prompt"),
        "width": job_input.get("width", workflow.DEFAULT_WIDTH),
        "height": job_input.get("height", workflow.DEFAULT_HEIGHT),
        "length": length,
        "chunks": chunks,
        "seed": job_input.get("seed", workflow.DEFAULT_SEED),
        "steps": job_input.get("steps", workflow.DEFAULT_STEPS),
        "cfg": job_input.get("cfg", workflow.DEFAULT_CFG),
        "shift": job_input.get("shift", workflow.DEFAULT_SHIFT),
        "sampler_name": job_input.get("sampler_name", "lcm"),
        "scheduler": job_input.get("scheduler", "simple"),
        "denoise": job_input.get("denoise", 1.0),
        "pose_strength": job_input.get("pose_strength", 1.0),
        "pose_start_percent": job_input.get("pose_start_percent", 0.0),
        "pose_end_percent": job_input.get("pose_end_percent", 1.0),
        "reference_image_strength": job_input.get("reference_image_strength", 1.0),
        "video_frame_offset": job_input.get("video_frame_offset", 0),
        "enable_context_window": _as_bool(job_input.get("enable_context_window"), False),
        "trim_duplicated_frame": _as_bool(job_input.get("trim_duplicated_frame"), False),
        "cache_device": job_input.get("cache_device", "gpu"),
        "cache_dtype": job_input.get("cache_dtype", "int8"),
        "fps": job_input.get("fps"),
        "unet_name": job_input.get("unet_name", workflow.UNET_NAME),
        "clip_name": job_input.get("clip_name", workflow.CLIP_NAME),
        "clip_vision_name": job_input.get("clip_vision_name", workflow.CLIP_VISION_NAME),
        "vae_name": job_input.get("vae_name", workflow.VAE_NAME),
    }
    return options


def handler(job: dict) -> dict:
    """Process one RunPod job."""
    job_input = job.get("input") or {}
    job_id = job.get("id") or uuid.uuid4().hex
    try:
        if _as_bool(job_input.get("dry_run"), False):
            options = _options_from_input(
                {**job_input, "reference_image": "x", "pose_video": "x"},
                job_input.get("reference_image_name") or "reference.png",
                job_input.get("pose_video_name") or "pose.mp4",
                job_input.get("frame_count"),
            )
            prompt = workflow.build_prompt(options)
            return {
                "status": "dry_run",
                "chunks": options["chunks"],
                "width": workflow.align_dimension(options["width"]),
                "height": workflow.align_dimension(options["height"]),
                "length": options["length"],
                "workflow": prompt,
            }

        reference, reference_name = _reference_bytes(job_input)
        pose, pose_name = _pose_bytes(job_input)
        stamp = uuid.uuid4().hex[:8]
        reference_name = f"{stamp}_{reference_name}"
        pose_name = f"{stamp}_{pose_name}"
        _write_input(reference, reference_name)
        pose_path = COMFY_INPUT_DIR / pose_name
        _write_input(pose, pose_name)

        frames = None
        if _as_bool(job_input.get("match_video_length"), False):
            frames = _probe_frames(pose_path)

        options = _options_from_input(job_input, reference_name, pose_name, frames)
        prompt = workflow.build_prompt(options)
        _wait_for_server()
        prompt_id = _queue_prompt(prompt)
        entry = _wait_for_prompt(prompt_id)

        videos = []
        seen = set()
        for item in _iter_output_files(entry):
            filename = item["filename"]
            if not str(filename).lower().endswith((".mp4", ".webm", ".mkv", ".gif")):
                continue
            key = (filename, item.get("subfolder"), item.get("type"))
            if key in seen:
                continue
            seen.add(key)
            data = _fetch_output(item)
            videos.append(_maybe_upload(job_id, filename, data))

        if not videos:
            raise JobError(f"ComfyUI finished without a video file. Outputs: {json.dumps(entry.get('outputs'))[:2000]}")

        return {
            "videos": videos,
            "chunks": options["chunks"],
            "width": workflow.align_dimension(options["width"]),
            "height": workflow.align_dimension(options["height"]),
            "length": options["length"],
            "seed": int(options["seed"]),
            "frame_count": frames,
            "prompt_id": prompt_id,
        }
    except JobError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": str(exc), "details": traceback.format_exc()[-4000:]}


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
