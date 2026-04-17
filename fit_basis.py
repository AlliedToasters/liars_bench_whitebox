"""Fit PCA bases for the experimental pipeline.

Extracted from the legacy `run_all_scans.py` (see git history prior to
2026-04-16 for the full file) as the only part of that script still
required going forward. Everything else (`run_holdout_scan`,
`run_cross_dataset`, `run_cross_fit_matrix`, `generate_plots`) is
superseded by `experiment_pipeline.py` under Experiment 0's blinded
protocol and has been retired.

What this script does
---------------------
For a target model, fit a PCA basis on the committed instructed-deception
train split (per `splits/instructed-deception_<model>.json`, seed=2026),
capturing `attn_delta` and `mlp_delta` at every layer, with generative
masks so PCA is fit only on assistant tokens. Writes the resulting
SampleScan to `scans/instructed-deception_<model>_holdout/`.

Note on existing bases
----------------------
The `scans/` directories currently on disk were fit by the legacy
`run_all_scans.py` using a random 400-train / 100-test split (seed=42)
before the frozen 60/20/20 protocol was adopted. Under EXPERIMENT.md §10
the bases should be fit on the committed train partition; this script
is the protocol-aligned path for any future refit. The existing bases
are retained (development-set artifacts) until a refit is run
deliberately.

Usage
-----
    python fit_basis.py --model mistral
    python fit_basis.py --model all
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

from lmprobe import SampleScan

from on_policy_affirmation import _template_to_ids, get_assistant_turn_boundaries

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


MODELS = {
    "mistral": {
        "hf_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "dataset_key": "mistral-small-3.1-24b-instruct",
        "short": "mistral-24b",
        "backend": "chunked",
    },
    "gemma": {
        "hf_id": "google/gemma-3-27b-it",
        "dataset_key": "gemma-3-27b-it",
        "short": "gemma-27b",
        "backend": "chunked",
    },
    "llama": {
        "hf_id": "meta-llama/Llama-3.3-70B-Instruct",
        "dataset_key": "llama-v3.3-70b-instruct",
        "short": "llama-70b",
        "backend": "disk_offload",
    },
    "qwen": {
        "hf_id": "Qwen/Qwen2.5-72B-Instruct",
        "dataset_key": "qwen-2.5-72b-instruct",
        "short": "qwen-72b",
        "backend": "disk_offload",
    },
}


def _load_tokenizer(model_name: str):
    """Load tokenizer, fetching chat_template.json from HF hub if not embedded.

    Mistral-Small-3.1 and some other releases ship the chat template as a
    separate file rather than in tokenizer_config.json.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        from huggingface_hub import hf_hub_download
        try:
            ct_path = hf_hub_download(model_name, "chat_template.json")
            ct_data = json.load(open(ct_path))
            if isinstance(ct_data, dict) and "chat_template" in ct_data:
                tokenizer.chat_template = ct_data["chat_template"]
        except Exception as e:
            logger.warning(f"Could not fetch chat_template.json for {model_name}: {e}")
    return tokenizer


def _load_train_indices(model_key: str) -> list[int]:
    """Load the frozen instructed-deception train split (seed=2026)."""
    split_file = Path("splits") / f"instructed-deception_{model_key}.json"
    with open(split_file) as f:
        splits = json.load(f)
    train = splits["train"]
    return list(train["deceptive"]) + list(train["truthful"])


def _build_train_corpus(model_key: str, tokenizer):
    """Build train prompts + labels + generative masks from the committed split."""
    indices = _load_train_indices(model_key)
    loaded = load_from_disk("data/instructed-deception")
    ds = loaded["test"] if isinstance(loaded, dict) else loaded
    ds_filtered = ds.filter(lambda x: x["model"] == model_key)

    prompts: list[str] = []
    labels: list[int] = []
    masks: list[np.ndarray] = []

    for idx in indices:
        row = ds_filtered[idx]
        prompts.append(tokenizer.apply_chat_template(row["messages"], tokenize=False))
        labels.append(1 if row["deceptive"] else 0)
        boundaries = get_assistant_turn_boundaries(row["messages"], tokenizer)
        ids = _template_to_ids(tokenizer, row["messages"])
        mask = np.zeros(len(ids), dtype=bool)
        for s, e in boundaries:
            mask[s:min(e, len(ids))] = True
        masks.append(mask)

    logger.info(f"  train corpus: {len(prompts)} samples "
                f"({sum(labels)} deceptive, {len(labels) - sum(labels)} truthful)")
    return prompts, labels, masks


def fit_basis(model_cfg: dict, device: str = "cuda:0", force: bool = False) -> SampleScan:
    """Fit the instructed-deception PCA basis for a model and write to disk.

    Returns the fitted SampleScan. If an existing scan is present and
    `force=False`, reuses it rather than refitting.
    """
    model_name = model_cfg["hf_id"]
    model_key = model_cfg["dataset_key"]
    short = model_cfg["short"]
    backend_type = model_cfg["backend"]

    scan_dir = Path("scans") / f"instructed-deception_{model_key}_holdout"

    if scan_dir.exists() and (scan_dir / "metadata.json").exists() and not force:
        logger.info(f"=== {short}: basis already at {scan_dir} (use --force to refit) ===")
        return SampleScan(str(scan_dir))

    logger.info(f"=== Fitting basis for {short} ({model_name}, backend={backend_type}) ===")
    tokenizer = _load_tokenizer(model_name)
    prompts, labels, masks = _build_train_corpus(model_key, tokenizer)

    scan = SampleScan.run(
        prompts=prompts,
        labels=labels,
        model_name=model_name,
        scan_dir=str(scan_dir),
        signals=["attn_delta", "mlp_delta"],
        n_components=64,
        device=device,
        dtype=torch.bfloat16,
        generative_masks=masks,
        backend=backend_type,
    )
    logger.info(f"=== {short}: basis fit. "
                f"{scan.n_layers} layers, {scan.n_samples} samples → {scan_dir} ===")
    return scan


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--model", type=str, default="all",
                        help="Model to fit (mistral/gemma/llama/qwen/all)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--force", action="store_true",
                        help="Refit even if a scan already exists")
    args = parser.parse_args()

    models = list(MODELS.keys()) if args.model == "all" else [args.model]
    for model_key in models:
        if model_key not in MODELS:
            logger.error(f"Unknown model: {model_key}")
            continue
        fit_basis(MODELS[model_key], device=args.device, force=args.force)


if __name__ == "__main__":
    main()
