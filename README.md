# ComfyUI worker — RunPod Serverless

Queue-based [RunPod Serverless](https://docs.runpod.io/serverless/overview) worker that runs **whatever ComfyUI API workflow you send**. The graph is not fixed in the handler.

ComfyUI **v0.37.0** is installed. The image also includes the Wan Animate 2 distilled weights, so that workflow can run without extra downloads. Any other graph runs the same way once its models are on the image or on a network volume.

## Endpoint input

In ComfyUI choose **Workflow > Export (API)** and send that file as `input.workflow`. The editor save (the file with `nodes` and `links`) is rejected.

Put the reference image and driving video directly on the loader nodes inside `workflow`. Use an `http(s)` URL or base64 (a `data:` URI prefix is optional) in `image`, `file`, `video`, `audio`, or `mask`. The worker saves that payload and swaps in the filename before ComfyUI runs. A normal filename such as `reference.png` is left unchanged.

```json
{
  "input": {
    "workflow": {
      "189": {
        "inputs": {"image": "https://example.com/character.png"},
        "class_type": "LoadImage"
      },
      "240": {
        "inputs": {"file": "https://example.com/driving.mp4"},
        "class_type": "LoadVideo"
      }
    }
  }
}
```

| Field | Required | Notes |
| --- | --- | --- |
| `workflow` | yes | ComfyUI API prompt. A JSON object or a JSON string. A `{"prompt": {...}}` wrapper is accepted. Media URLs and base64 go on the loader inputs inside this object. |
| `comfy_org_api_key` | no | Per-request key for Comfy.org API nodes. |
| `dry_run` | no | Validate the prompt and return it with media inputs already replaced by filenames. ComfyUI is not called. |

### Output

Images, videos, and audio from Save Image / Save Video / similar nodes are returned separately. Other files land in `files`.

```json
{
  "prompt_id": "...",
  "images": [],
  "videos": [
    {
      "filename": "wan_animate_00001_.mp4",
      "type": "base64",
      "mime": "video/mp4",
      "data": "..."
    }
  ],
  "audio": [],
  "files": []
}
```

When `BUCKET_ENDPOINT_URL` (and the other RunPod S3 variables) are set, each file is uploaded and `type` is `s3_url`. Base64 responses for a full clip can exceed the `/runsync` body limit. Use `/run` and S3 for long videos.

```bash
curl -X POST \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d @request.json \
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

The image does not contain these files. On first boot the worker downloads them onto the attached network volume at `/runpod-volume/models/...`. ComfyUI reads that path through `extra_model_paths.yaml`. Later workers in the same data center reuse the volume and skip the download.

To bake the weights into the image instead:

```bash
docker build --build-arg DOWNLOAD_MODELS=true -t wan-animate2-distilled .
```

## Deploy on RunPod

GPU: **48 GB** is the practical target (RTX A6000 / L40 / A40). The int8 diffusion weights are ~17 GB and the pose cache is large. `cache_device=cpu` is the fallback on 24 GB cards. Host CUDA must be **12.8+** (the image uses PyTorch cu128).

Container disk: **20 GB** is enough for temporary video. The weights live on the network volume.

1. Create a network volume in the data center where the endpoint will run. Size at least 50 GB.
2. In RunPod Serverless, create an endpoint **from this GitHub repo** and attach that volume.
3. Dockerfile path: `Dockerfile`. Branch: `cursor/wan-animate2-runpod-baff`.
4. GPU count 1, 48 GB class, CUDA 12.8 or newer.
5. Optional environment variables:
   - `HF_TOKEN` if Hugging Face rate-limits the weight download
   - `DOWNLOAD_ON_START=false` when the weights are already baked in or already on the volume
   - `COMFY_EXTRA_ARGS` extra arguments to `python main.py`
   - S3 settings from the [RunPod S3 guide](https://docs.runpod.io/serverless/workers/s3-upload) if you want URL outputs
   - `COMFY_TIMEOUT_S` (default `3600`)

FlashBoot helps after the first worker has pulled the image.

## Local check

```bash
python -m unittest tests.test_handler tests.test_workflow
```

`python handler.py` starts the RunPod local test runner using `test_input.json` (`dry_run: true`), so it does not launch ComfyUI.
