# Wan Animate 2 Distilled — RunPod Serverless

ComfyUI worker that runs the **Wan Animate 2 distilled** character-animation workflow on [RunPod Serverless](https://docs.runpod.io/serverless/overview).

A reference image (who) and a driving video (the motion) produce a video of that character performing the motion. The graph matches `workflows/video_wan_animate2_distilled.json`:

- diffusion model `wan_animate_2_distill_int8_convrot.safetensors`
- text encoder `umt5_xxl_fp8_e4m3fn_scaled.safetensors` (`wan`)
- CLIP vision `clip_vision_h.safetensors`
- VAE `Wan2_1_VAE_bf16.safetensors`
- LCM sampler, 10 steps, CFG 1, ModelSamplingSD3 shift 5
- `WanAnimate2Cache` on GPU in int8
- one 81-frame window by default (the second subgraph in the template is bypassed)

ComfyUI **v0.37.0** is required. `WanAnimate2ToVideo` and `WanAnimate2Cache` are built in.

## Endpoint input

```json
{
  "input": {
    "reference_image": "https://example.com/character.png",
    "pose_video": "https://example.com/driving.mp4",
    "prompt": "Character appearance description: ...\nBackground description: ...",
    "pose_prompt": "A person performing the motion in the driving video.",
    "width": 482,
    "height": 854,
    "length": 81,
    "seed": 714297762067883
  }
}
```

`reference_image` and `pose_video` are required. Each value is an `http(s)` URL or a base64 string (a `data:` URI prefix is optional).

| Field | Default | Notes |
| --- | --- | --- |
| `prompt` | template character prompt | Identity and background. |
| `pose_prompt` | template motion prompt | Motion description for the pose branch. |
| `negative_prompt` | template Chinese negative prompt | |
| `width`, `height` | `482`, `854` | Rounded to a multiple of 16 (`480`×`848`). |
| `length` | `81` | Frames per chunk. Snapped to `4n+1`. |
| `seed` | `714297762067883` | Each extra chunk uses `seed + chunk_index`. |
| `steps` | `10` | |
| `cfg` | `1` | Distilled checkpoint is trained at CFG 1. |
| `shift` | `5` | ModelSamplingSD3. |
| `sampler_name` | `lcm` | |
| `scheduler` | `simple` | |
| `pose_strength` | `1` | |
| `pose_start_percent` | `0` | |
| `pose_end_percent` | `1` | Must be `>= pose_start_percent`. |
| `reference_image_strength` | `1` | |
| `video_frame_offset` | `0` | Seek into the driving video. |
| `enable_context_window` | `false` | Turns on the template Context Windows (Manual) settings. |
| `trim_duplicated_frame` | `false` | Drops the first decoded frame. Later chunks always drop it. |
| `cache_device` | `gpu` | `cpu` if the int8 cache does not fit in VRAM. |
| `cache_dtype` | `int8` | `default` or `int4`. |
| `fps` | driving video fps | Override with a number. |
| `chunks` | `1` | How many length-sized windows to chain. |
| `match_video_length` | `false` | Set `true` to cover the whole driving video. |
| `max_chunks` | `8` | Cap when `match_video_length` is set. Maximum 50. |
| `dry_run` | `false` | Return the API prompt and do not run ComfyUI. |

Videos longer than `length` frames need extra windows, the same way the template note says to duplicate the subgraph. Set `match_video_length` to `true` or pass `chunks`. Each window continues from the previous window's `continue_motion` and `video_frame_offset`.

### Output

```json
{
  "videos": [
    {
      "filename": "wan_animate_00001_.mp4",
      "type": "base64",
      "mime": "video/mp4",
      "data": "..."
    }
  ],
  "chunks": 1,
  "width": 480,
  "height": 848,
  "length": 81,
  "seed": 714297762067883
}
```

When `BUCKET_ENDPOINT_URL` (and the other RunPod S3 variables) are set, each video is uploaded and `type` is `s3_url`. Base64 responses for a full clip can exceed the `/runsync` body limit. Use `/run` and S3 for long videos.

```bash
curl -X POST \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"reference_image":"https://example.com/ref.png","pose_video":"https://example.com/drive.mp4"}}' \
  https://api.runpod.ai/v2/$ENDPOINT_ID/run
```

## Models

About **25 GB**, from [Comfy-Org/Wan-Animate-2](https://huggingface.co/Comfy-Org/Wan-Animate-2):

| File | Folder | Size |
| --- | --- | --- |
| `wan_animate_2_distill_int8_convrot.safetensors` | `models/diffusion_models` | 16.7 GB |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `models/text_encoders` | 6.7 GB |
| `clip_vision_h.safetensors` | `models/clip_vision` | 1.3 GB |
| `Wan2_1_VAE_bf16.safetensors` | `models/vae` | 0.25 GB |

The Docker build downloads them (`DOWNLOAD_MODELS=true`). On start, missing files are downloaded again. If `/runpod-volume` is mounted and writable, downloads go to `/runpod-volume/models/...` and ComfyUI reads them through `extra_model_paths.yaml`.

To keep the image small and store weights on a network volume:

```bash
docker build --build-arg DOWNLOAD_MODELS=false -t wan-animate2-distilled .
```

Put the four files in the volume using the folders above, in the same region as the endpoint.

## Deploy on RunPod

GPU: **48 GB** is the practical target (RTX A6000 / L40 / A40). The int8 diffusion weights are ~17 GB and the pose cache is large. `cache_device=cpu` is the fallback on 24 GB cards. Host CUDA must be **12.8+** (the image uses PyTorch cu128).

Container disk: **40 GB** is enough for temporary video when the weights are inside the image. The image itself is large because of those weights.

1. Push this repository.
2. In RunPod Serverless, create an endpoint **from this GitHub repo**.
3. Dockerfile path: `Dockerfile`. Branch: the branch you deploy.
4. GPU count 1, 48 GB class, CUDA 12.8 or newer.
5. Optional environment variables:
   - `HF_TOKEN` if Hugging Face rate-limits the weight download
   - `DOWNLOAD_ON_START=false` when the weights are already baked in or already on the volume
   - `COMFY_EXTRA_ARGS` extra arguments to `python main.py`
   - S3 settings from the [RunPod S3 guide](https://docs.runpod.io/serverless/workers/s3-upload) if you want URL outputs
   - `COMFY_TIMEOUT_S` (default `3600`)

FlashBoot helps after the first worker has pulled the image.

## Local check

The workflow builder does not need a GPU:

```bash
python -m unittest tests.test_workflow
python -c "import json; print(json.load(open('test_input.json'))['input']['dry_run'])"
python -c "from handler import handler; import json; print(json.dumps({k: handler(json.load(open('test_input.json')))[k] for k in ('status','width','height','length','chunks')}))"
```

`python handler.py` starts the RunPod local test runner using `test_input.json` (`dry_run: true`), so it does not launch ComfyUI.
