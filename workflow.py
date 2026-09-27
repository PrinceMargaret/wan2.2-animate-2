"""Build the ComfyUI API prompt for the Wan Animate 2 distilled workflow.

The source graph is ``workflows/video_wan_animate2_distilled.json``. The active
path is one Motion Transfer subgraph (the second copy is bypassed), then
Create Video + Save Video. This module flattens that subgraph into the API
prompt format ComfyUI's ``/prompt`` endpoint accepts.
"""

from __future__ import annotations

import math
from typing import Any

UNET_NAME = "wan_animate_2_distill_int8_convrot.safetensors"
CLIP_NAME = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
CLIP_VISION_NAME = "clip_vision_h.safetensors"
VAE_NAME = "Wan2_1_VAE_bf16.safetensors"

# CLIP Text Encode (Negative Prompt) from the subgraph.
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

# Text (Multiline - Character Prompt) and Text (Multiline - Post Prompt).
DEFAULT_PROMPT = (
    "Character appearance description: A realistic premium collectible action figure of a male "
    "street dancer, lifelike matte vinyl and resin material with realistic skin-like paint finish, "
    "detailed fabric folds on a black hoodie, dark cargo pants, white sneakers, subtle articulated "
    "joints, premium designer toy craftsmanship, standing in a relaxed confident pose facing the "
    "camera, full body from head to toe.\n"
    "Background description: rainy neon-lit city street corner at night, wet asphalt reflecting "
    "purple-blue and red neon signs, light mist, cinematic night photography."
)
DEFAULT_POSE_PROMPT = (
    "A man doing energetic street dance with popping body waves and fast spinning moves, "
    "rain falling around him."
)

# Subgraph widget defaults (portrait driving clip in the template).
DEFAULT_WIDTH = 482
DEFAULT_HEIGHT = 854
DEFAULT_LENGTH = 81
DEFAULT_SEED = 714297762067883
DEFAULT_STEPS = 10
DEFAULT_CFG = 1.0
DEFAULT_SHIFT = 5.0
MAX_CHUNKS = 50


def align_dimension(value: int) -> int:
    """WanAnimate2ToVideo width/height step is 16."""
    value = int(value)
    if value < 16:
        return 16
    return max(16, int(round(value / 16.0)) * 16)


def align_length(value: int) -> int:
    """Frame count must be 4n+1."""
    value = int(value)
    if value < 1:
        return 1
    if (value - 1) % 4 == 0:
        return value
    return max(1, ((value - 1) // 4) * 4 + 1)


def chunk_count(frame_count: int | None, length: int, match_video_length: bool, max_chunks: int) -> int:
    """How many  length-frame windows are required to cover the pose video."""
    max_chunks = max(1, min(int(max_chunks), MAX_CHUNKS))
    if not match_video_length or not frame_count or frame_count <= 0:
        return 1
    needed = max(1, math.ceil(int(frame_count) / int(length)))
    if needed > max_chunks:
        raise ValueError(
            f"Pose video needs {needed} chunks of {length} frames, which is above max_chunks={max_chunks}. "
            "Raise max_chunks or use a shorter driving video."
        )
    return needed


def _node(class_type: str, inputs: dict[str, Any], title: str) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inputs, "_meta": {"title": title}}


def _resize(image: list[Any], width: Any, height: Any) -> dict[str, Any]:
    return {
        "input": image,
        "resize_type": {
            "resize_type": "scale dimensions",
            "width": width,
            "height": height,
            "crop": "center",
        },
        "scale_method": "area",
    }


def build_prompt(options: dict[str, Any]) -> dict[str, Any]:
    """Return a ComfyUI API prompt for one job.

    ``options`` keys mirror the handler input. Filenames must already exist in
    ComfyUI's input directory.
    """
    width = align_dimension(options.get("width", DEFAULT_WIDTH))
    height = align_dimension(options.get("height", DEFAULT_HEIGHT))
    length = align_length(options.get("length", DEFAULT_LENGTH))
    chunks = int(options.get("chunks", 1))
    if chunks < 1 or chunks > MAX_CHUNKS:
        raise ValueError(f"chunks must be between 1 and {MAX_CHUNKS}")

    seed = int(options.get("seed", DEFAULT_SEED))
    steps = int(options.get("steps", DEFAULT_STEPS))
    cfg = float(options.get("cfg", DEFAULT_CFG))
    shift = float(options.get("shift", DEFAULT_SHIFT))
    prompt = options.get("prompt") or DEFAULT_PROMPT
    pose_prompt = options.get("pose_prompt") or DEFAULT_POSE_PROMPT
    negative = options.get("negative_prompt") or NEGATIVE_PROMPT
    reference_name = options["reference_image_name"]
    pose_name = options["pose_video_name"]
    pose_strength = float(options.get("pose_strength", 1.0))
    pose_start = float(options.get("pose_start_percent", 0.0))
    pose_end = float(options.get("pose_end_percent", 1.0))
    reference_strength = float(options.get("reference_image_strength", 1.0))
    if pose_start > pose_end:
        raise ValueError("pose_start_percent must be less than or equal to pose_end_percent")

    enable_context = bool(options.get("enable_context_window", False))
    trim_first = bool(options.get("trim_duplicated_frame", False))
    cache_device = options.get("cache_device", "gpu")
    cache_dtype = options.get("cache_dtype", "int8")
    fps_override = options.get("fps")

    prompt_graph: dict[str, Any] = {}

    prompt_graph["1"] = _node(
        "UNETLoader",
        {"unet_name": options.get("unet_name", UNET_NAME), "weight_dtype": "default"},
        "Load Diffusion Model",
    )
    prompt_graph["2"] = _node(
        "CLIPLoader",
        {"clip_name": options.get("clip_name", CLIP_NAME), "type": "wan", "device": "default"},
        "Load CLIP",
    )
    prompt_graph["3"] = _node(
        "CLIPVisionLoader",
        {"clip_name": options.get("clip_vision_name", CLIP_VISION_NAME)},
        "Load CLIP Vision",
    )
    prompt_graph["4"] = _node(
        "VAELoader",
        {"vae_name": options.get("vae_name", VAE_NAME)},
        "Load VAE",
    )
    prompt_graph["5"] = _node(
        "CLIPTextEncode",
        {"text": negative, "clip": ["2", 0]},
        "CLIP Text Encode (Negative Prompt)",
    )
    prompt_graph["6"] = _node(
        "CLIPTextEncode",
        {"text": prompt, "clip": ["2", 0]},
        "CLIP Text Encode (Positive Prompt)",
    )
    prompt_graph["7"] = _node(
        "CLIPTextEncode",
        {"text": pose_prompt, "clip": ["2", 0]},
        "CLIP Text Encode (Pose Prompt)",
    )
    prompt_graph["10"] = _node("LoadImage", {"image": reference_name}, "Load Image (Reference Image)")
    prompt_graph["11"] = _node("LoadVideo", {"file": pose_name}, "Load Video (Pose Video)")
    prompt_graph["12"] = _node(
        "GetVideoComponents",
        {"video": ["11", 0]},
        "Get Video Components",
    )
    prompt_graph["13"] = _node(
        "ResizeImageMaskNode",
        _resize(["12", 0], width, height),
        "Resize Pose Video",
    )
    prompt_graph["14"] = _node("GetImageSize", {"image": ["13", 0]}, "Get Image Size")
    prompt_graph["15"] = _node(
        "ResizeImageMaskNode",
        _resize(["10", 0], ["14", 0], ["14", 1]),
        "Resize Reference Image",
    )
    prompt_graph["16"] = _node(
        "CLIPVisionEncode",
        {"clip_vision": ["3", 0], "image": ["15", 0], "crop": "none"},
        "CLIP Vision Encode (Reference)",
    )
    prompt_graph["17"] = _node(
        "ImageFromBatch",
        {"image": ["13", 0], "batch_index": 0, "length": 1},
        "Pose First Frame",
    )
    prompt_graph["18"] = _node(
        "CLIPVisionEncode",
        {"clip_vision": ["3", 0], "image": ["17", 0], "crop": "none"},
        "CLIP Vision Encode (Pose)",
    )
    prompt_graph["19"] = _node("KSamplerSelect", {"sampler_name": options.get("sampler_name", "lcm")}, "KSampler Select")

    model_for_cache = ["1", 0]
    if enable_context:
        prompt_graph["20"] = _node(
            "ContextWindowsManual",
            {
                "model": ["1", 0],
                "context_length": int(options.get("context_length", 21)),
                "context_overlap": int(options.get("context_overlap", 8)),
                "context_schedule": options.get("context_schedule", "standard_static"),
                "context_stride": int(options.get("context_stride", 1)),
                "closed_loop": bool(options.get("closed_loop", False)),
                "fuse_method": options.get("fuse_method", "pyramid"),
                "dim": int(options.get("context_dim", 2)),
                "freenoise": bool(options.get("freenoise", True)),
                "cond_retain_index_list": str(options.get("cond_retain_index_list", "0")),
                "split_conds_to_windows": bool(options.get("split_conds_to_windows", False)),
                "latent_retain_index_list": str(options.get("latent_retain_index_list", "")),
                "causal_window_fix": bool(options.get("causal_window_fix", True)),
            },
            "Context Windows (Manual)",
        )
        model_for_cache = ["20", 0]

    image_outputs: list[list[Any]] = []
    previous_motion: list[Any] | None = None
    previous_offset: list[Any] | None = None

    for index in range(chunks):
        base = 100 + index * 20
        wan_id = str(base)
        cache_id = str(base + 1)
        sampling_id = str(base + 2)
        scheduler_id = str(base + 3)
        sampler_id = str(base + 4)
        trim_id = str(base + 5)
        decode_id = str(base + 6)

        wan_inputs: dict[str, Any] = {
            "positive": ["6", 0],
            "negative": ["5", 0],
            "vae": ["4", 0],
            "width": ["14", 0],
            "height": ["14", 1],
            "length": length,
            "batch_size": 1,
            "reference_image": ["15", 0],
            "pose_video": ["13", 0],
            "clip_vision_output": ["16", 0],
            "positive_pose": ["7", 0],
            "clip_vision_output_pose": ["18", 0],
            "video_frame_offset": int(options.get("video_frame_offset", 0)) if previous_offset is None else previous_offset,
            "pose_strength": pose_strength,
            "pose_start_percent": pose_start,
            "pose_end_percent": pose_end,
            "reference_image_strength": reference_strength,
        }
        if previous_motion is not None:
            wan_inputs["continue_motion"] = previous_motion

        prompt_graph[wan_id] = _node("WanAnimate2ToVideo", wan_inputs, f"Wan Animate 2 To Video {index}")
        prompt_graph[cache_id] = _node(
            "WanAnimate2Cache",
            {"model": model_for_cache, "device": cache_device, "dtype": cache_dtype},
            f"Wan Animate 2 Cache {index}",
        )
        prompt_graph[sampling_id] = _node(
            "ModelSamplingSD3",
            {"model": [cache_id, 0], "shift": shift},
            f"Model Sampling SD3 {index}",
        )
        prompt_graph[scheduler_id] = _node(
            "BasicScheduler",
            {
                "model": [cache_id, 0],
                "scheduler": options.get("scheduler", "simple"),
                "steps": steps,
                "denoise": float(options.get("denoise", 1.0)),
            },
            f"Basic Scheduler {index}",
        )
        prompt_graph[sampler_id] = _node(
            "SamplerCustom",
            {
                "model": [sampling_id, 0],
                "add_noise": True,
                "noise_seed": seed + index,
                "cfg": cfg,
                "positive": [wan_id, 0],
                "negative": [wan_id, 1],
                "sampler": ["19", 0],
                "sigmas": [scheduler_id, 0],
                "latent_image": [wan_id, 2],
            },
            f"Sampler Custom {index}",
        )
        prompt_graph[trim_id] = _node(
            "TrimVideoLatent",
            {"samples": [sampler_id, 0], "trim_amount": [wan_id, 3]},
            f"Trim Video Latent {index}",
        )
        prompt_graph[decode_id] = _node(
            "VAEDecode",
            {"samples": [trim_id, 0], "vae": ["4", 0]},
            f"VAE Decode {index}",
        )

        frame_output: list[Any] = [decode_id, 0]
        # The template drops the overlapping first frame only when trim is enabled.
        # Later chunks always share one motion frame with the previous chunk.
        if trim_first or index > 0:
            drop_id = str(base + 7)
            prompt_graph[drop_id] = _node(
                "ImageFromBatch",
                {"image": [decode_id, 0], "batch_index": 1, "length": 4096},
                f"Trim Duplicated Frame {index}",
            )
            frame_output = [drop_id, 0]

        image_outputs.append(frame_output)
        previous_motion = [decode_id, 0]
        previous_offset = [wan_id, 5]

    if len(image_outputs) == 1:
        images_link = image_outputs[0]
    else:
        batch_inputs = {f"images.image{i}": link for i, link in enumerate(image_outputs)}
        prompt_graph["900"] = _node("BatchImagesNode", batch_inputs, "Batch Images")
        images_link = ["900", 0]

    create_inputs: dict[str, Any] = {
        "images": images_link,
        "audio": ["12", 1],
        "bit_depth": 8,
        "color_space": "sRGB",
        "codec": "none",
    }
    if fps_override is None:
        create_inputs["fps"] = ["12", 2]
    else:
        create_inputs["fps"] = float(fps_override)
    prompt_graph["910"] = _node("CreateVideo", create_inputs, "Create Video")
    prompt_graph["911"] = _node(
        "SaveVideo",
        {
            "video": ["910", 0],
            "filename_prefix": options.get("filename_prefix", "video/wan_animate"),
            "format": {
                "format": "mp4",
                "codec": {"codec": "h264", "encoding": {"encoding": "auto"}},
            },
        },
        "Save Video",
    )
    return prompt_graph
