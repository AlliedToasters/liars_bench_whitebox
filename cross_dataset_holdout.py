"""Cross-dataset holdout: train on instructed-deception, test on other datasets.

Uses the PCA basis fit on instructed-deception (generative-only) and
projects samples from other deception datasets through it. If the signal
transfers, the deception representation is universal, not task-specific.

Usage:
    python cross_dataset_holdout.py --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

from lmprobe import SampleScan
from on_policy_affirmation import _template_to_ids, get_assistant_turn_boundaries

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
MODEL_KEY = "mistral-small-3.1-24b-instruct"

TEST_DATASETS = [
    "harm-pressure-choice",
    "harm-pressure-knowledge-report",
    "insider-trading",
]


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, fix_mistral_regex=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        from huggingface_hub import hf_hub_download
        ct_path = hf_hub_download(MODEL_NAME, "chat_template.json")
        ct_data = json.load(open(ct_path))
        tokenizer.chat_template = ct_data["chat_template"]
    return tokenizer


def evaluate_dataset(scan, tokenizer, dataset_name, n_per_class=50):
    """Batch-project samples from a dataset and compute separability."""
    loaded = load_from_disk(f"data/{dataset_name}")
    ds = loaded["test"] if isinstance(loaded, dict) else loaded
    ds_filtered = ds.filter(lambda x: x["model"] == MODEL_KEY)

    deceptive = ds_filtered.filter(lambda x: x["deceptive"])
    truthful = ds_filtered.filter(lambda x: not x["deceptive"])

    n = min(n_per_class, len(deceptive), len(truthful))
    if n < 10:
        logger.warning(f"  {dataset_name}: only {n} per class, skipping")
        return None

    logger.info(f"  {dataset_name}: {n} per class ({len(deceptive)} dec / {len(truthful)} truth available)")

    # Build test samples
    test_rows = list(deceptive.select(range(n))) + list(truthful.select(range(n)))
    test_labels = [1] * n + [0] * n
    test_prompts = [tokenizer.apply_chat_template(r["messages"], tokenize=False) for r in test_rows]

    # Get assistant last-token positions
    test_asst_last = []
    for row in test_rows:
        boundaries = get_assistant_turn_boundaries(row["messages"], tokenizer)
        if boundaries:
            test_asst_last.append(boundaries[-1][1] - 1)
        else:
            ids = _template_to_ids(tokenizer, row["messages"])
            test_asst_last.append(len(ids) - 1)

    # Batch-project through the ID-trained basis (fast!)
    n_layers = scan.n_layers
    k = scan.n_components
    n_signals = len(scan.signals)

    logger.info(f"    Batch-projecting {len(test_prompts)} prompts...")
    projections, coords, token_ids_per_sample, seq_lengths = scan.batch_project(
        test_prompts, batch_size=4,
    )

    # Reconstruct per-sample vectors at assistant last-token
    coord_sids = np.array(coords["sample_id"], dtype=np.int32)
    coord_layers = np.array(coords["layer"], dtype=np.int16)
    coord_toks = np.array(coords["token_pos"], dtype=np.int16)
    coord_sigs = np.array(coords["signal"], dtype=np.int8)

    test_vecs = np.zeros((len(test_prompts), n_layers, n_signals, k), dtype=np.float32)

    for sid in range(len(test_prompts)):
        target_tok = min(test_asst_last[sid], seq_lengths[sid] - 1)
        for sig_idx in range(n_signals):
            for layer_idx in range(n_layers):
                mask = (
                    (coord_sids == sid)
                    & (coord_layers == layer_idx)
                    & (coord_toks == target_tok)
                    & (coord_sigs == sig_idx)
                )
                rows = projections[mask]
                if len(rows) > 0:
                    test_vecs[sid, layer_idx, sig_idx, :] = rows[0, 0, :k].astype(np.float32)

    # Compute separability
    test_labels_arr = np.array(test_labels)
    sep_map = np.zeros((n_layers, n_signals), dtype=np.float64)

    for sig_idx in range(n_signals):
        for layer_idx in range(n_layers):
            vecs = test_vecs[:, layer_idx, sig_idx, :]
            max_auroc = 0.5
            for pc in range(k):
                pc_vals = vecs[:, pc]
                if np.std(pc_vals) < 1e-10:
                    continue
                try:
                    auroc = roc_auc_score(test_labels_arr, pc_vals)
                    auroc = max(auroc, 1.0 - auroc)
                    max_auroc = max(max_auroc, auroc)
                except ValueError:
                    continue
            sep_map[layer_idx, sig_idx] = max_auroc

    return {
        "dataset": dataset_name,
        "n_per_class": n,
        "sep_map": sep_map,
        "signals": scan.signals,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n-per-class", type=int, default=50)
    args = parser.parse_args()

    # Load the scan trained on instructed-deception
    scan_dir = Path("scans") / f"instructed-deception_{MODEL_KEY}_holdout"
    logger.info(f"Loading scan from {scan_dir}")
    scan = SampleScan(str(scan_dir))
    logger.info(f"Scan: {scan.n_layers} layers, {scan.n_samples} train samples, signals={scan.signals}")

    tokenizer = load_tokenizer()

    results = {}
    for ds_name in TEST_DATASETS:
        logger.info(f"Evaluating {ds_name}...")
        result = evaluate_dataset(scan, tokenizer, ds_name, n_per_class=args.n_per_class)
        if result is None:
            continue

        sep_map = result["sep_map"]
        results[ds_name] = result

        for sig_idx, sig_name in enumerate(scan.signals):
            scores = sep_map[:, sig_idx]
            top_layers = np.argsort(scores)[::-1][:5]
            logger.info(f"  {sig_name}: top={list(top_layers)}, "
                         f"AUROCs={[f'{scores[l]:.3f}' for l in top_layers]}, "
                         f"max={scores.max():.3f}, mean={scores.mean():.3f}")

    # Summary comparison
    logger.info("=" * 60)
    logger.info("CROSS-DATASET HOLDOUT SUMMARY")
    logger.info("PCA basis: instructed-deception (generative-only)")
    logger.info("=" * 60)
    logger.info(f"{'Dataset':<35} {'attn max':>10} {'mlp max':>10}")
    logger.info("-" * 60)

    # Include the within-dataset holdout result
    holdout_path = Path("results") / f"separability_holdout_instructed-deception_{MODEL_KEY}.npy"
    if holdout_path.exists():
        holdout_sep = np.load(holdout_path)
        logger.info(f"{'instructed-deception (holdout)':<35} {holdout_sep[:, 0].max():>10.3f} {holdout_sep[:, 1].max():>10.3f}")

    for ds_name, result in results.items():
        sep = result["sep_map"]
        logger.info(f"{ds_name:<35} {sep[:, 0].max():>10.3f} {sep[:, 1].max():>10.3f}")

    # Save
    results_dir = Path("results")
    for ds_name, result in results.items():
        np.save(
            results_dir / f"separability_cross_{ds_name}_{MODEL_KEY}.npy",
            result["sep_map"],
        )

    summary = {
        "train_dataset": "instructed-deception",
        "train_scan": str(scan_dir),
        "model": MODEL_NAME,
        "evaluation": "cross_dataset_holdout_assistant_only",
        "test_datasets": {
            ds_name: {
                "n_per_class": result["n_per_class"],
                "attn_delta_max_auroc": float(result["sep_map"][:, 0].max()),
                "mlp_delta_max_auroc": float(result["sep_map"][:, 1].max()),
                "attn_delta_best_layer": int(np.argmax(result["sep_map"][:, 0])),
                "mlp_delta_best_layer": int(np.argmax(result["sep_map"][:, 1])),
            }
            for ds_name, result in results.items()
        },
    }

    with open(results_dir / f"cross_dataset_holdout_{MODEL_KEY}.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved to {results_dir / f'cross_dataset_holdout_{MODEL_KEY}.json'}")


if __name__ == "__main__":
    main()
