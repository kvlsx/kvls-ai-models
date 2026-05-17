# KVLS-AI Models

On-device vision model bundled with the [KVLS-AI](https://github.com/kvlsx/KVLS-AI)
roleplay companion app. The model runs locally on the user's phone and gives
the chat LLM a textual description of any attached image without sending it
anywhere — works completely offline once downloaded.

## What's in the release

The current release ships **WD-ViT-Tagger-v3** by
[SmilingWolf](https://huggingface.co/SmilingWolf/wd-vit-tagger-v3):

| File | Size | Purpose |
|---|---|---|
| `model-int8.onnx` (release asset) | **93 MB** | **Per-channel dynamic INT8** quantization (`quantize_dynamic`, QInt8 weights). Verified against the FP32 reference: top-100 output correlation **0.81–0.96** across natural photos. **Default download for mobile clients.** |
| `model.onnx` (release asset) | 361 MB | Original FP32 weights — used during quantization and kept for re-runs |
| `selected_tags.csv` (in tree) | 301 KB | Tag-id → name / category / count vocabulary |
| `scripts/quantize.py` | tiny | Production recipe used to build the shipped INT8 |
| `scripts/quantize_static.py` | tiny | Experimental static-quant — does NOT work on this ViT, see file header |
| `scripts/fetch_calibration.sh` | tiny | Re-downloads the 100-image calibration set from Picsum (CC0); unused by the production recipe but kept for static-quant experiments |

Tag categories present in the vocabulary:

| Category | Count | Description |
|---|---|---|
| `9` | 4 | Content rating: `general`, `sensitive`, `questionable`, `explicit` |
| `0` | 8106 | General descriptive tags: people, body, clothing, pose, action, setting |
| `4` | 2751 | Anime character names (filtered out on the client for real photos) |

The model is intentionally **uncensored**: explicit tags are returned with
their actual confidence so the chat LLM can write in-character replies that
react to NSFW content during adult roleplay. License: Apache-2.0 (inherited
from the upstream model card).

## How the app uses it

1. App boots, checks for `model.onnx` in its private documents directory.
2. If missing, it streams the file from this repo's latest release
   (`v1-vision`) and shows a one-time progress bar.
3. Every time the user attaches a photo or picks a character avatar, the app
   pre-processes the image to 448×448, runs ONNX inference on-device
   (Apple Neural Engine on iOS via Core ML EP, NNAPI / Vulkan on Android),
   and converts the top-N tags above a confidence threshold into a short
   prose paragraph that gets injected into the system prompt or appended
   to the outgoing user message.

The phone-side post-processing groups tags into rough buckets
(person / body / clothing / setting / action / rating) and produces
output like:

> Photo shows: one person, long blonde hair, smiling, wearing red dress,
> sitting on a bed, indoor lighting. Explicit content: low.

This sentence is what the chat LLM sees — it never receives the raw image
unless the user is running a vision-capable model on their LM Studio server.

## Versioning

| Tag | Notes |
|---|---|
| `v1-vision` | Initial release with `wd-vit-tagger-v3` (this one) |

Future versions may swap the vision model (e.g. for a slightly larger
SwinV2 variant) or add a small captioning LLM for prose-style descriptions.
The download URL pattern stays stable:
`https://github.com/kvlsx/kvls-ai-models/releases/download/<tag>/<filename>`.

## Why a separate repo

The main app repo
([kvlsx/KVLS-AI](https://github.com/kvlsx/KVLS-AI)) stays light and
fast to clone. Model binaries live here as Release assets so they don't
inflate the git history and so the app can fetch them lazily on first
launch with a single anonymous HTTP GET — no GitHub token required.

## Attribution

- **Model**: WD-ViT-Tagger-v3 by SmilingWolf — Apache-2.0
- **Vocabulary**: selected_tags.csv — same upstream
