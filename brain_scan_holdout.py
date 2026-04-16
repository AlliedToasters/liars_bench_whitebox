"""Brain scan with train/test holdout evaluation.

Fits the SampleScan PCA basis on a training split, then projects
held-out test samples and computes separability on those — no data
leakage between basis fitting and evaluation.

Usage:
    python brain_scan_holdout.py \
        --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
        --n-train 400 --n-test 100 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

from lmprobe import SampleScan
from on_policy_affirmation import _template_to_ids, get_assistant_turn_boundaries

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_KEY_TO_HF = {
    "mistral-small-3.1-24b-instruct": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "gemma-3-27b-it": "google/gemma-3-27b-it",
    "llama-v3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen-2.5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct",
}


def main():
    parser = argparse.ArgumentParser(description="Brain scan with holdout evaluation")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-Small-3.1-24B-Instruct-2503")
    parser.add_argument("--dataset", type=str, default="instructed-deception")
    parser.add_argument("--n-train", type=int, default=400,
                        help="Training samples (balanced per class)")
    parser.add_argument("--n-test", type=int, default=100,
                        help="Test samples (balanced per class)")
    parser.add_argument("--n-components", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--signals", type=str, nargs="+",
                        default=["attn_delta", "mlp_delta"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    # Resolve model
    model_name = args.model
    dataset_model_key = None
    for key, hf_id in MODEL_KEY_TO_HF.items():
        if hf_id == model_name:
            dataset_model_key = key
            break
    if dataset_model_key is None:
        dataset_model_key = model_name
    logger.info(f"Model: {model_name}")

    # Load and filter dataset
    ds = load_from_disk(f"data/{args.dataset}")["test"]
    ds_filtered = ds.filter(lambda x: x["model"] == dataset_model_key)

    deceptive_all = ds_filtered.filter(lambda x: x["deceptive"])
    truthful_all = ds_filtered.filter(lambda x: not x["deceptive"])
    logger.info(f"Available: {len(deceptive_all)} deceptive, {len(truthful_all)} truthful")

    n_train_per_class = args.n_train // 2
    n_test_per_class = args.n_test // 2

    # Shuffle indices
    dec_indices = rng.permutation(len(deceptive_all))
    truth_indices = rng.permutation(len(truthful_all))

    # Split
    dec_train_idx = dec_indices[:n_train_per_class]
    dec_test_idx = dec_indices[n_train_per_class:n_train_per_class + n_test_per_class]
    truth_train_idx = truth_indices[:n_train_per_class]
    truth_test_idx = truth_indices[n_train_per_class:n_train_per_class + n_test_per_class]

    logger.info(f"Train: {n_train_per_class} dec + {n_train_per_class} truth = {args.n_train}")
    logger.info(f"Test: {n_test_per_class} dec + {n_test_per_class} truth = {args.n_test}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        from huggingface_hub import hf_hub_download
        try:
            ct_path = hf_hub_download(model_name, "chat_template.json")
            ct_data = json.load(open(ct_path))
            if isinstance(ct_data, dict) and "chat_template" in ct_data:
                tokenizer.chat_template = ct_data["chat_template"]
            logger.info("Loaded chat template from chat_template.json")
        except Exception:
            pass

    # Build train prompts/labels + generative masks
    train_prompts, train_labels, train_masks = [], [], []
    for idx in dec_train_idx:
        row = deceptive_all[int(idx)]
        train_prompts.append(tokenizer.apply_chat_template(row["messages"], tokenize=False))
        train_labels.append(1)
        boundaries = get_assistant_turn_boundaries(row["messages"], tokenizer)
        ids = _template_to_ids(tokenizer, row["messages"])
        mask = np.zeros(len(ids), dtype=bool)
        for s, e in boundaries:
            mask[s:min(e, len(ids))] = True
        train_masks.append(mask)
    for idx in truth_train_idx:
        row = truthful_all[int(idx)]
        train_prompts.append(tokenizer.apply_chat_template(row["messages"], tokenize=False))
        train_labels.append(0)
        boundaries = get_assistant_turn_boundaries(row["messages"], tokenizer)
        ids = _template_to_ids(tokenizer, row["messages"])
        mask = np.zeros(len(ids), dtype=bool)
        for s, e in boundaries:
            mask[s:min(e, len(ids))] = True
        train_masks.append(mask)

    # Build test data
    test_rows = []
    test_labels = []
    for idx in dec_test_idx:
        test_rows.append(deceptive_all[int(idx)])
        test_labels.append(1)
    for idx in truth_test_idx:
        test_rows.append(truthful_all[int(idx)])
        test_labels.append(0)

    test_prompts = [tokenizer.apply_chat_template(r["messages"], tokenize=False) for r in test_rows]

    # Compute assistant boundaries for test set
    test_asst_last_tokens = []
    for row in test_rows:
        boundaries = get_assistant_turn_boundaries(row["messages"], tokenizer)
        if boundaries:
            test_asst_last_tokens.append(boundaries[-1][1] - 1)
        else:
            ids = _template_to_ids(tokenizer, row["messages"])
            test_asst_last_tokens.append(len(ids) - 1)

    # --- Phase 1: Fit scan on training set ---
    model_short = dataset_model_key.replace("/", "_")
    scan_dir = Path("scans") / f"{args.dataset}_{model_short}_holdout"

    logger.info("Phase 1: Fitting SampleScan on training set...")
    scan = SampleScan.run(
        prompts=train_prompts,
        labels=train_labels,
        model_name=model_name,
        scan_dir=str(scan_dir),
        signals=args.signals,
        n_components=args.n_components,
        device=args.device,
        dtype=torch.bfloat16,
        generative_masks=train_masks,
    )
    logger.info(f"Scan fit: {scan.n_layers} layers, {scan.n_samples} train samples")

    # --- Phase 2: Project test set through trained basis ---
    logger.info(f"Phase 2: Projecting {len(test_prompts)} test samples...")
    n_layers = scan.n_layers
    k = scan.n_components
    n_signals = len(scan.signals)

    # For each test sample, get the projection at the assistant last token
    test_vecs = np.zeros((len(test_prompts), n_layers, n_signals, k), dtype=np.float32)

    for i, prompt in enumerate(test_prompts):
        proj, tokens, _ = scan.project_prompt(prompt)
        # proj: [seq_len, n_layers, n_signals, k]
        tok_pos = min(test_asst_last_tokens[i], proj.shape[0] - 1)
        test_vecs[i] = proj[tok_pos]  # [n_layers, n_signals, k]

        if (i + 1) % 10 == 0:
            logger.info(f"  [{i+1}/{len(test_prompts)}] projected")

    # --- Phase 3: Compute separability on test set ---
    logger.info("Phase 3: Computing test-set separability...")
    test_labels_arr = np.array(test_labels)

    sep_map = np.zeros((n_layers, n_signals), dtype=np.float64)
    for sig_idx in range(n_signals):
        for layer_idx in range(n_layers):
            vecs = test_vecs[:, layer_idx, sig_idx, :]  # [n_test, k]
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

    # Report results
    logger.info("=" * 60)
    logger.info("HOLDOUT RESULTS (assistant-turn-only, test set)")
    logger.info("=" * 60)
    for sig_idx, sig_name in enumerate(scan.signals):
        scores = sep_map[:, sig_idx]
        top_layers = np.argsort(scores)[::-1][:10]
        logger.info(f"  {sig_name}:")
        logger.info(f"    top layers = {list(top_layers)}")
        logger.info(f"    AUROCs     = {[f'{scores[l]:.3f}' for l in top_layers]}")
        logger.info(f"    max AUROC  = {scores.max():.3f}")
        logger.info(f"    mean AUROC = {scores.mean():.3f}")

    # Save
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    sep_path = results_dir / f"separability_holdout_{args.dataset}_{model_short}.npy"
    np.save(sep_path, sep_map)

    summary = {
        "model": model_name,
        "dataset": args.dataset,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "seed": args.seed,
        "signals": args.signals,
        "n_components": args.n_components,
        "n_layers": n_layers,
        "evaluation": "holdout_assistant_only",
        "separability": {
            sig: {
                "top_layers": [int(l) for l in np.argsort(sep_map[:, i])[::-1][:10]],
                "top_aurocs": [float(sep_map[l, i]) for l in np.argsort(sep_map[:, i])[::-1][:10]],
                "max_auroc": float(sep_map[:, i].max()),
                "mean_auroc": float(sep_map[:, i].mean()),
            }
            for i, sig in enumerate(scan.signals)
        },
    }

    summary_path = results_dir / f"brain_scan_holdout_{args.dataset}_{model_short}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {summary_path}")


if __name__ == "__main__":
    main()
