# Liar's Bench Whitebox Repo

## Project Overview
Mechanistic interpretability research to recover on-policy latents for the Liar's Bench deception detection dataset. We use the [lmprobe](https://alliedtoasters.github.io/lmprobe/) library to extract model activations and probe for deception-related features.

## Strategy
- Single-GPU inference using [lmprobe's](https://alliedtoasters.github.io/lmprobe/) `chunked` or `disk_offload` backends for forward-pass chunking ([large models guide](https://alliedtoasters.github.io/lmprobe/guides/large-models/))
  - `chunked`: loads full model on CPU, streams layers through GPU in chunks (for models that fit in CPU RAM)
  - `disk_offload`: loads layer weights directly from safetensors to GPU one at a time (for models exceeding CPU RAM)
- Prioritizing vanilla instruct (non-fine-tuned) datasets first due to ease of access to open model weights
- Fine-tuned datasets (Gender-Secret, Soft-Trigger) deferred until later

## Key Libraries
- `lmprobe` — activation extraction and probing
- `datasets` — HuggingFace datasets (already saved locally in `data/`)

## Data
Datasets are stored in `data/` as HuggingFace `datasets` format. See README.md for the full dataset table.

## Conventions
- Python 3.10+
- Use virtual environment in `.venv/`
