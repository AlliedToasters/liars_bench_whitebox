"""Recompute separability map using assistant-turn-only token positions.

Loads the existing scan and computes separability using the last token
of the assistant turn (not the last token of the full sequence) to avoid
prompt leakage from differing system instructions.

Usage:
    python recompute_separability.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer

from lmprobe import SampleScan
from on_policy_affirmation import _template_to_ids, get_assistant_turn_boundaries

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    model_name = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    dataset_name = "instructed-deception"
    model_key = "mistral-small-3.1-24b-instruct"

    scan_dir = Path("scans") / f"{dataset_name}_{model_key}"
    logger.info(f"Loading scan from {scan_dir}")
    scan = SampleScan(str(scan_dir))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        from huggingface_hub import hf_hub_download
        ct_path = hf_hub_download(model_name, "chat_template.json")
        ct_data = json.load(open(ct_path))
        tokenizer.chat_template = ct_data["chat_template"]

    # Load dataset and get the same samples used in the scan
    ds = load_from_disk(f"data/{dataset_name}")["test"]
    ds_filtered = ds.filter(lambda x: x["model"] == model_key)

    deceptive = ds_filtered.filter(lambda x: x["deceptive"])
    truthful = ds_filtered.filter(lambda x: not x["deceptive"])
    n_per_class = scan.n_samples // 2
    logger.info(f"Reconstructing {n_per_class} deceptive + {n_per_class} truthful samples")

    # Reconstruct the same order used in brain_scan_id.py
    samples = list(deceptive.select(range(n_per_class))) + list(truthful.select(range(n_per_class)))
    labels = [1] * n_per_class + [0] * n_per_class

    # Find assistant turn boundaries for each sample
    assistant_last_tokens = []
    for row in samples:
        messages = row["messages"]
        boundaries = get_assistant_turn_boundaries(messages, tokenizer)
        if boundaries:
            # Last token of the last assistant turn
            last_assistant_end = boundaries[-1][1] - 1  # end is exclusive
            assistant_last_tokens.append(last_assistant_end)
        else:
            # Fallback to last token
            ids = _template_to_ids(tokenizer, messages)
            assistant_last_tokens.append(len(ids) - 1)

    assistant_last_tokens = np.array(assistant_last_tokens, dtype=np.int64)
    logger.info(f"Assistant last-token positions: min={assistant_last_tokens.min()}, "
                f"max={assistant_last_tokens.max()}, mean={assistant_last_tokens.mean():.0f}")

    # Recompute separability with corrected token positions
    logger.info("Computing separability map (assistant-turn-only)...")
    sep_map = scan.separability_map(
        labels=np.array(labels),
        token_positions=assistant_last_tokens,
    )
    logger.info(f"Separability map shape: {sep_map.shape}")

    for sig_idx, sig_name in enumerate(scan.signals):
        scores = sep_map[:, sig_idx]
        top_layers = np.argsort(scores)[::-1][:10]
        logger.info(f"  {sig_name}: top layers = {list(top_layers)}, "
                     f"AUROCs = {[f'{scores[l]:.3f}' for l in top_layers]}")

    # Save
    results_dir = Path("results")
    sep_path = results_dir / f"separability_assistant_only_{dataset_name}_{model_key}.npy"
    np.save(sep_path, sep_map)
    logger.info(f"Saved to {sep_path}")

    # Also save comparison summary
    old_sep = np.load(results_dir / f"separability_{dataset_name}_{model_key}.npy")
    logger.info("=== COMPARISON ===")
    for sig_idx, sig_name in enumerate(scan.signals):
        old_max = old_sep[:, sig_idx].max()
        new_max = sep_map[:, sig_idx].max()
        old_mean = old_sep[:, sig_idx].mean()
        new_mean = sep_map[:, sig_idx].mean()
        logger.info(f"  {sig_name}: max AUROC {old_max:.3f} -> {new_max:.3f}, "
                     f"mean AUROC {old_mean:.3f} -> {new_mean:.3f}")


if __name__ == "__main__":
    main()
