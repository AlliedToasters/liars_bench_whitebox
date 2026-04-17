"""Run brain scan holdout + cross-dataset + all plots for all models.

Usage:
    python run_all_scans.py --model llama
    python run_all_scans.py --model qwen
    python run_all_scans.py --model all
"""

from __future__ import annotations

import argparse
import json
import logging
import re
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

MODELS = {
    "llama": {
        "hf_id": "meta-llama/Llama-3.3-70B-Instruct",
        "dataset_key": "llama-v3.3-70b-instruct",
        "short": "llama-3.3-70b",
        "backend": "disk_offload",
    },
    "qwen": {
        "hf_id": "Qwen/Qwen2.5-72B-Instruct",
        "dataset_key": "qwen-2.5-72b-instruct",
        "short": "qwen-2.5-72b",
        "backend": "disk_offload",
    },
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
}

DATASETS = ["instructed-deception", "harm-pressure-choice",
            "harm-pressure-knowledge-report", "insider-trading"]


def load_tokenizer(model_name):
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
        except Exception:
            pass
    return tokenizer


def build_scan_data(ds_name, dataset_key, tokenizer, n_train=400, n_test=100, seed=42):
    """Build train/test splits with generative masks."""
    rng = np.random.RandomState(seed)
    loaded = load_from_disk(f"data/{ds_name}")
    ds = loaded["test"] if isinstance(loaded, dict) else loaded
    ds_filtered = ds.filter(lambda x: x["model"] == dataset_key)

    deceptive_all = ds_filtered.filter(lambda x: x["deceptive"])
    truthful_all = ds_filtered.filter(lambda x: not x["deceptive"])
    logger.info(f"  {ds_name}: {len(deceptive_all)} dec, {len(truthful_all)} truth for {dataset_key}")

    # Cap per-class counts to what's available (80% train, 20% test)
    min_class = min(len(deceptive_all), len(truthful_all))
    n_train_per = min(n_train // 2, int(min_class * 0.8))
    n_test_per = min(n_test // 2, min_class - n_train_per)
    logger.info(f"  Using {n_train_per} train + {n_test_per} test per class")

    dec_idx = rng.permutation(len(deceptive_all))
    truth_idx = rng.permutation(len(truthful_all))

    # Train
    train_prompts, train_labels, train_masks = [], [], []
    for idx in dec_idx[:n_train_per]:
        row = deceptive_all[int(idx)]
        train_prompts.append(tokenizer.apply_chat_template(row["messages"], tokenize=False))
        train_labels.append(1)
        boundaries = get_assistant_turn_boundaries(row["messages"], tokenizer)
        ids = _template_to_ids(tokenizer, row["messages"])
        mask = np.zeros(len(ids), dtype=bool)
        for s, e in boundaries:
            mask[s:min(e, len(ids))] = True
        train_masks.append(mask)
    for idx in truth_idx[:n_train_per]:
        row = truthful_all[int(idx)]
        train_prompts.append(tokenizer.apply_chat_template(row["messages"], tokenize=False))
        train_labels.append(0)
        boundaries = get_assistant_turn_boundaries(row["messages"], tokenizer)
        ids = _template_to_ids(tokenizer, row["messages"])
        mask = np.zeros(len(ids), dtype=bool)
        for s, e in boundaries:
            mask[s:min(e, len(ids))] = True
        train_masks.append(mask)

    # Test
    test_rows, test_labels = [], []
    for idx in dec_idx[n_train_per:n_train_per + n_test_per]:
        test_rows.append(deceptive_all[int(idx)])
        test_labels.append(1)
    for idx in truth_idx[n_train_per:n_train_per + n_test_per]:
        test_rows.append(truthful_all[int(idx)])
        test_labels.append(0)

    test_prompts = [tokenizer.apply_chat_template(r["messages"], tokenize=False) for r in test_rows]

    test_asst_last = []
    for row in test_rows:
        boundaries = get_assistant_turn_boundaries(row["messages"], tokenizer)
        if boundaries:
            test_asst_last.append(boundaries[-1][1] - 1)
        else:
            ids = _template_to_ids(tokenizer, row["messages"])
            test_asst_last.append(len(ids) - 1)

    return (train_prompts, train_labels, train_masks,
            test_prompts, test_labels, test_asst_last)


def _batch_project_to_vecs(scan, prompts, asst_last_positions, n_layers, n_signals, k):
    """Batch-project prompts and extract assistant-last-token vectors.

    Uses scan.batch_project() in sample-groups to stay within memory.
    Still batched (not per-sample project_prompt), just chunked by sample count.
    """
    from transformers import AutoTokenizer
    sample_len = len(AutoTokenizer.from_pretrained(scan._metadata.model_id).encode(prompts[0]))

    # Chunk size: fewer samples per call for long sequences
    if sample_len > 1000:
        chunk = 10  # insider-trading: ~1775 tok
    elif sample_len > 500:
        chunk = 25
    else:
        chunk = 100  # short sequences: all at once

    logger.info(f"    sample_chunks={chunk}, ~{sample_len} tok/sample")

    test_vecs = np.zeros((len(prompts), n_layers, n_signals, k), dtype=np.float32)
    sig_names = scan.signals

    for start in range(0, len(prompts), chunk):
        end = min(start + chunk, len(prompts))
        chunk_prompts = prompts[start:end]
        chunk_asst = asst_last_positions[start:end]

        projections, coords, _, seq_lengths = scan.batch_project(chunk_prompts, batch_size=min(4, len(chunk_prompts)))

        coord_sample = np.array(coords["sample_id"])
        coord_layer = np.array(coords["layer"])
        coord_tok = np.array(coords["token_pos"])
        coord_sig = np.array(coords["signal"])

        for i in range(len(chunk_prompts)):
            tok_pos = min(chunk_asst[i], seq_lengths[i] - 1)
            for si, sig in enumerate(sig_names):
                mask = ((coord_sample == i) & (coord_tok == tok_pos) & (coord_sig == sig))
                rows_matched = np.where(mask)[0]
                for row_idx in rows_matched:
                    layer = coord_layer[row_idx]
                    test_vecs[start + i, layer, si, :] = projections[row_idx, 0, :]

        logger.info(f"    [{end}/{len(prompts)}] projected")

    return test_vecs


def compute_metrics(test_vecs, labels_arr, n_layers, n_signals, k):
    """Compute all evaluation metrics from cached test_vecs.

    Returns dict with separability map and per-layer/signal detailed metrics.
    Matches Liar's Bench paper evaluation: balanced accuracy at 1% FPR.
    """
    from sklearn.metrics import (balanced_accuracy_score, average_precision_score,
                                  roc_curve, recall_score, precision_score, f1_score)

    sep_map = np.zeros((n_layers, n_signals), dtype=np.float64)
    bal_acc_map = np.zeros((n_layers, n_signals), dtype=np.float64)
    bal_acc_1pct_fpr_map = np.zeros((n_layers, n_signals), dtype=np.float64)
    recall_1pct_fpr_map = np.zeros((n_layers, n_signals), dtype=np.float64)
    avg_prec_map = np.zeros((n_layers, n_signals), dtype=np.float64)
    best_pc_map = np.zeros((n_layers, n_signals), dtype=int)
    # Optimal threshold metrics
    bal_acc_optimal_map = np.zeros((n_layers, n_signals), dtype=np.float64)
    f1_optimal_map = np.zeros((n_layers, n_signals), dtype=np.float64)

    for sig_idx in range(n_signals):
        for layer_idx in range(n_layers):
            vecs = test_vecs[:, layer_idx, sig_idx, :]
            max_auroc = 0.5
            best_pc = 0
            best_flip = False
            for pc in range(k):
                pc_vals = vecs[:, pc]
                if np.std(pc_vals) < 1e-10:
                    continue
                try:
                    auroc = roc_auc_score(labels_arr, pc_vals)
                    flip = auroc < 0.5
                    auroc_adj = 1.0 - auroc if flip else auroc
                    if auroc_adj > max_auroc:
                        max_auroc = auroc_adj
                        best_pc = pc
                        best_flip = flip
                except ValueError:
                    continue
            sep_map[layer_idx, sig_idx] = max_auroc
            best_pc_map[layer_idx, sig_idx] = best_pc

            # Orient scores so positive = deceptive
            pc_vals = vecs[:, best_pc]
            if best_flip:
                pc_vals = -pc_vals

            # --- Median threshold (simple) ---
            threshold = np.median(pc_vals)
            preds = (pc_vals >= threshold).astype(int)
            bal_acc_map[layer_idx, sig_idx] = balanced_accuracy_score(labels_arr, preds)

            # --- 1% FPR threshold (paper's evaluation) ---
            try:
                fpr, tpr, thresholds = roc_curve(labels_arr, pc_vals)
                # Find threshold where FPR <= 1%
                idx_1pct = np.searchsorted(fpr, 0.01, side='right') - 1
                idx_1pct = max(0, idx_1pct)
                if idx_1pct < len(thresholds):
                    thresh_1pct = thresholds[idx_1pct]
                else:
                    thresh_1pct = thresholds[-1]
                preds_1pct = (pc_vals >= thresh_1pct).astype(int)
                bal_acc_1pct_fpr_map[layer_idx, sig_idx] = balanced_accuracy_score(labels_arr, preds_1pct)
                recall_1pct_fpr_map[layer_idx, sig_idx] = recall_score(labels_arr, preds_1pct, zero_division=0)
            except ValueError:
                bal_acc_1pct_fpr_map[layer_idx, sig_idx] = 0.5
                recall_1pct_fpr_map[layer_idx, sig_idx] = 0.0

            # --- Optimal threshold (Youden's J) ---
            try:
                fpr, tpr, thresholds = roc_curve(labels_arr, pc_vals)
                j_scores = tpr - fpr
                best_j_idx = np.argmax(j_scores)
                if best_j_idx < len(thresholds):
                    thresh_opt = thresholds[best_j_idx]
                else:
                    thresh_opt = thresholds[-1]
                preds_opt = (pc_vals >= thresh_opt).astype(int)
                bal_acc_optimal_map[layer_idx, sig_idx] = balanced_accuracy_score(labels_arr, preds_opt)
                f1_optimal_map[layer_idx, sig_idx] = f1_score(labels_arr, preds_opt, zero_division=0)
            except ValueError:
                bal_acc_optimal_map[layer_idx, sig_idx] = 0.5
                f1_optimal_map[layer_idx, sig_idx] = 0.0

            try:
                avg_prec_map[layer_idx, sig_idx] = average_precision_score(labels_arr, pc_vals)
            except ValueError:
                avg_prec_map[layer_idx, sig_idx] = 0.0

    return {
        "auroc": sep_map,
        "balanced_accuracy": bal_acc_map,
        "balanced_accuracy_1pct_fpr": bal_acc_1pct_fpr_map,
        "recall_1pct_fpr": recall_1pct_fpr_map,
        "balanced_accuracy_optimal": bal_acc_optimal_map,
        "f1_optimal": f1_optimal_map,
        "average_precision": avg_prec_map,
        "best_pc": best_pc_map,
    }


def run_holdout_scan(model_cfg, device="cuda:0", metrics_only=False):
    """Phase 1 + 2: Fit scan + evaluate on holdout test set."""
    model_name = model_cfg["hf_id"]
    dataset_key = model_cfg["dataset_key"]
    backend_type = model_cfg["backend"]
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    cache_path = results_dir / f"test_vecs_holdout_{dataset_key}.npz"

    # Check cache
    if metrics_only and cache_path.exists():
        logger.info(f"=== Holdout metrics (cached): {model_name} ===")
        cached = np.load(cache_path)
        test_vecs = cached["test_vecs"]
        test_labels_arr = cached["labels"]
        scan_dir = Path("scans") / f"instructed-deception_{dataset_key}_holdout"
        scan = SampleScan(str(scan_dir))
        n_layers, n_signals, k = scan.n_layers, len(scan.signals), scan.n_components
    else:
        logger.info(f"=== Holdout scan: {model_name} (backend={backend_type}) ===")
        tokenizer = load_tokenizer(model_name)

        (train_prompts, train_labels, train_masks,
         test_prompts, test_labels, test_asst_last) = build_scan_data(
            "instructed-deception", dataset_key, tokenizer)

        scan_dir = Path("scans") / f"instructed-deception_{dataset_key}_holdout"

        # Phase 1: Fit scan
        logger.info(f"Phase 1: Fitting scan ({len(train_prompts)} samples)...")
        scan = SampleScan.run(
            prompts=train_prompts,
            labels=train_labels,
            model_name=model_name,
            scan_dir=str(scan_dir),
            signals=["attn_delta", "mlp_delta"],
            n_components=64,
            device=device,
            dtype=torch.bfloat16,
            generative_masks=train_masks,
            backend=backend_type,
        )
        logger.info(f"Scan fit: {scan.n_layers} layers, {scan.n_samples} samples")

        # Phase 2: Evaluate holdout
        logger.info(f"Phase 2: Evaluating {len(test_prompts)} test samples...")
        n_layers = scan.n_layers
        k = scan.n_components
        n_signals = len(scan.signals)

        # Batch project — all samples in one forward pass per layer chunk
        logger.info(f"  Batch projecting {len(test_prompts)} holdout samples...")
        test_vecs = _batch_project_to_vecs(scan, test_prompts, test_asst_last,
                                            n_layers, n_signals, k)

        test_labels_arr = np.array(test_labels)

        # Cache test_vecs to disk
        np.savez(cache_path, test_vecs=test_vecs, labels=test_labels_arr)
        logger.info(f"Cached test_vecs → {cache_path}")

    # Compute all metrics from (cached) test_vecs
    metrics = compute_metrics(test_vecs, test_labels_arr, n_layers, n_signals, k)
    sep_map = metrics["auroc"]

    # Save results
    np.save(results_dir / f"separability_holdout_instructed-deception_{dataset_key}.npy", sep_map)

    summary = {
        "model": model_name,
        "dataset": "instructed-deception",
        "n_train": 400, "n_test": 100, "seed": 42,
        "signals": ["attn_delta", "mlp_delta"],
        "n_components": 64, "n_layers": n_layers,
        "evaluation": "holdout_assistant_only",
        "separability": {
            sig: {
                "top_layers": [int(l) for l in np.argsort(sep_map[:, i])[::-1][:10]],
                "top_aurocs": [float(sep_map[l, i]) for l in np.argsort(sep_map[:, i])[::-1][:10]],
                "max_auroc": float(sep_map[:, i].max()),
                "mean_auroc": float(sep_map[:, i].mean()),
                "max_balanced_accuracy": float(metrics["balanced_accuracy"][:, i].max()),
                "max_bal_acc_1pct_fpr": float(metrics["balanced_accuracy_1pct_fpr"][:, i].max()),
                "max_recall_1pct_fpr": float(metrics["recall_1pct_fpr"][:, i].max()),
                "max_bal_acc_optimal": float(metrics["balanced_accuracy_optimal"][:, i].max()),
                "max_f1_optimal": float(metrics["f1_optimal"][:, i].max()),
                "max_average_precision": float(metrics["average_precision"][:, i].max()),
            }
            for i, sig in enumerate(["attn_delta", "mlp_delta"])
        },
    }
    with open(results_dir / f"brain_scan_holdout_instructed-deception_{dataset_key}.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Holdout results:")
    for sig_idx, sig in enumerate(["attn_delta", "mlp_delta"]):
        logger.info(f"  {sig}: AUROC={sep_map[:, sig_idx].max():.3f} "
                     f"BalAcc@1%FPR={metrics['balanced_accuracy_1pct_fpr'][:, sig_idx].max():.3f} "
                     f"Recall@1%FPR={metrics['recall_1pct_fpr'][:, sig_idx].max():.3f} "
                     f"BalAccOpt={metrics['balanced_accuracy_optimal'][:, sig_idx].max():.3f} "
                     f"F1Opt={metrics['f1_optimal'][:, sig_idx].max():.3f} "
                     f"at layer {sep_map[:, sig_idx].argmax()}")

    return scan, sep_map, metrics


def run_cross_dataset(scan, model_cfg, device="cuda:0", metrics_only=False):
    """Phase 3: Cross-dataset evaluation using batch_project."""
    model_name = model_cfg["hf_id"]
    dataset_key = model_cfg["dataset_key"]

    logger.info(f"=== Cross-dataset eval: {model_name} ===")
    tokenizer = load_tokenizer(model_name)

    results_dir = Path("results")
    cross_results = {}

    for ds_name in DATASETS:
        cache_path = results_dir / f"test_vecs_cross_{ds_name}_{dataset_key}.npz"

        if metrics_only and cache_path.exists():
            logger.info(f"  {ds_name}: loading cached test_vecs")
            cached = np.load(cache_path)
            test_vecs = cached["test_vecs"]
            labels_arr = cached["labels"]
            n_layers = test_vecs.shape[1]
            n_signals = test_vecs.shape[2]
            k = test_vecs.shape[3]
        else:
            loaded = load_from_disk(f"data/{ds_name}")
            ds = loaded["test"] if isinstance(loaded, dict) else loaded
            ds_filtered = ds.filter(lambda x: x["model"] == dataset_key)

            if len(ds_filtered) == 0:
                logger.info(f"  {ds_name}: no data for {dataset_key}, skipping")
                continue

            deceptive = ds_filtered.filter(lambda x: x["deceptive"])
            truthful = ds_filtered.filter(lambda x: not x["deceptive"])
            n_per = min(50, len(deceptive), len(truthful))
            logger.info(f"  {ds_name}: {n_per} per class")

            rows = []
            labels = []
            for i in range(n_per):
                rows.append(deceptive[i])
                labels.append(1)
            for i in range(n_per):
                rows.append(truthful[i])
                labels.append(0)

            prompts = [tokenizer.apply_chat_template(r["messages"], tokenize=False) for r in rows]

            asst_last = []
            for row in rows:
                boundaries = get_assistant_turn_boundaries(row["messages"], tokenizer)
                if boundaries:
                    asst_last.append(boundaries[-1][1] - 1)
                else:
                    ids = _template_to_ids(tokenizer, row["messages"])
                    asst_last.append(len(ids) - 1)

            n_layers = scan.n_layers
            n_signals = len(scan.signals)
            k = scan.n_components

            # Batch project — all samples in one forward pass per layer chunk
            logger.info(f"  Batch projecting {len(prompts)} samples...")
            test_vecs = _batch_project_to_vecs(scan, prompts, asst_last,
                                                n_layers, n_signals, k)

            labels_arr = np.array(labels)

            # Cache test_vecs to disk
            np.savez(cache_path, test_vecs=test_vecs, labels=labels_arr)
            logger.info(f"  Cached test_vecs → {cache_path}")

        # Compute all metrics from (cached) test_vecs
        metrics = compute_metrics(test_vecs, labels_arr, n_layers, n_signals, k)
        sep = metrics["auroc"]

        np.save(results_dir / f"separability_cross_{ds_name}_{dataset_key}.npy", sep)

        cross_results[ds_name] = {
            "attn_max": float(sep[:, 0].max()),
            "mlp_max": float(sep[:, 1].max()),
            "attn_best_layer": int(sep[:, 0].argmax()),
            "mlp_best_layer": int(sep[:, 1].argmax()),
            "attn_max_bal_acc": float(metrics["balanced_accuracy"][:, 0].max()),
            "mlp_max_bal_acc": float(metrics["balanced_accuracy"][:, 1].max()),
            "attn_max_avg_prec": float(metrics["average_precision"][:, 0].max()),
            "mlp_max_avg_prec": float(metrics["average_precision"][:, 1].max()),
        }
        logger.info(f"    attn={sep[:, 0].max():.3f} mlp={sep[:, 1].max():.3f} "
                     f"bal_acc={metrics['balanced_accuracy'][:, 0].max():.3f}/{metrics['balanced_accuracy'][:, 1].max():.3f}")

    with open(results_dir / f"cross_dataset_{dataset_key}.json", "w") as f:
        json.dump(cross_results, f, indent=2)

    return cross_results


def generate_plots(model_cfg):
    """Generate all visualization plots from stored scan data."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from lmprobe.scan_plot import render_scan_figure

    dataset_key = model_cfg["dataset_key"]
    short = model_cfg["short"]
    model_name = model_cfg["hf_id"]

    scan_dir = f"scans/instructed-deception_{dataset_key}_holdout"
    scan = SampleScan(scan_dir)
    samples = scan.samples.to_pandas()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    n_layers = scan.n_layers
    n_signals = len(scan.signals)
    k = scan.n_components

    logger.info(f"=== Generating plots for {short} ===")

    # Helper to get tokens and generative mask
    def get_tokens_and_mask(prompt_text):
        encoded = tokenizer(prompt_text, return_offsets_mapping=True)
        offsets = encoded.get('offset_mapping', [])
        input_ids = encoded['input_ids']
        tokens = [tokenizer.decode(tid) for tid in input_ids]
        gen_mask = np.zeros(len(input_ids), dtype=bool)

        # Find assistant turn boundaries in text
        # Try common patterns
        model_patterns = [
            r'<start_of_turn>model\n',       # Gemma
            r'\[/INST\]',                      # Mistral/Llama
            r'<\|im_start\|>assistant\n',      # Qwen
            r'<\|start_header_id\|>assistant<\|end_header_id\|>\n\n',  # Llama 3
        ]
        end_patterns = [
            r'<end_of_turn>',
            r'</s>',
            r'<\|im_end\|>',
            r'<\|eot_id\|>',
        ]

        model_starts = []
        for pat in model_patterns:
            model_starts.extend([m.end() for m in re.finditer(pat, prompt_text)])

        model_ends = []
        for pat in end_patterns:
            model_ends.extend([m.start() for m in re.finditer(pat, prompt_text)])
        model_ends.sort()

        for ms in model_starts:
            me = len(prompt_text)
            for end in model_ends:
                if end > ms:
                    me = end
                    break
            if offsets:
                for ti, (s, e) in enumerate(offsets):
                    if s is not None and s >= ms and e <= me:
                        gen_mask[ti] = True

        return input_ids, tokens, gen_mask

    # --- 1. Cohen's d + Population Probe ---
    logger.info("  Cohen's d + population probe...")
    sep = np.load(f"results/separability_holdout_instructed-deception_{dataset_key}.npy")

    dec_vecs = []
    truth_vecs = []
    for _, row in samples.iterrows():
        sid = int(row['sample_id'])
        proj = scan.get_projections(sid)
        seq_len = int(row['seq_length'])
        last_tok = min(seq_len - 1, proj.shape[0] - 1)
        vec = proj[last_tok]
        if row['label'] == 1:
            dec_vecs.append(vec)
        else:
            truth_vecs.append(vec)
    dec_vecs = np.array(dec_vecs)
    truth_vecs = np.array(truth_vecs)

    # Cohen's d per layer (attn_delta, best PC)
    cohens_d = np.zeros(n_layers)
    best_pc_per_layer = np.zeros(n_layers, dtype=int)
    labels_arr = np.array([1]*len(dec_vecs) + [0]*len(truth_vecs))
    all_vecs = np.concatenate([dec_vecs, truth_vecs], axis=0)

    for layer in range(n_layers):
        max_auroc = 0.5
        best_pc = 0
        for pc in range(min(k, 16)):
            vals = all_vecs[:, layer, 0, pc]
            if np.std(vals) < 1e-10:
                continue
            try:
                auroc = roc_auc_score(labels_arr, vals)
                if auroc < 0.5:
                    auroc = 1 - auroc
                if auroc > max_auroc:
                    max_auroc = auroc
                    best_pc = pc
            except:
                continue
        best_pc_per_layer[layer] = best_pc
        # Compute d for best PC
        d_vals = dec_vecs[:, layer, 0, best_pc]
        t_vals = truth_vecs[:, layer, 0, best_pc]
        pooled_std = np.sqrt((np.var(d_vals) + np.var(t_vals)) / 2)
        if pooled_std > 1e-10:
            cohens_d[layer] = abs(np.mean(d_vals) - np.mean(t_vals)) / pooled_std

    # Cohen's d plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(n_layers)), y=cohens_d,
        mode='lines+markers', name="Cohen's d",
        line=dict(color='#d62728', width=2), marker=dict(size=4)))
    peak_layer = cohens_d.argmax()
    fig.add_trace(go.Scatter(x=[peak_layer], y=[cohens_d[peak_layer]],
        mode='markers+text',
        text=[f"L{peak_layer}: d={cohens_d[peak_layer]:.2f}"],
        textposition='top center',
        marker=dict(size=10, color='#d62728'), showlegend=False))
    fig.update_layout(
        title=f"{short} — Cohen's d by Layer (attn_delta, best PC)",
        xaxis_title="Layer", yaxis_title="Cohen's d",
        template='plotly_white', width=900, height=500)
    fig.write_html(f"docs/cohens_d_{short}.html")
    fig.write_image(f"docs/cohens_d_{short}.png", scale=2)

    # Population probe plot
    fig = go.Figure()
    for layer in [peak_layer]:
        pc = best_pc_per_layer[layer]
        d_vals = dec_vecs[:, layer, 0, pc]
        t_vals = truth_vecs[:, layer, 0, pc]
        fig.add_trace(go.Histogram(x=d_vals, name=f"Deceptive (L{layer})",
            opacity=0.6, marker_color='red', nbinsx=30))
        fig.add_trace(go.Histogram(x=t_vals, name=f"Truthful (L{layer})",
            opacity=0.6, marker_color='blue', nbinsx=30))
    fig.update_layout(
        title=f"{short} — Population Probe Scores (Layer {peak_layer}, PC{best_pc_per_layer[peak_layer]})",
        xaxis_title="Probe Score", yaxis_title="Count",
        barmode='overlay', template='plotly_white', width=900, height=500)
    fig.write_html(f"docs/population_probe_{short}.html")
    fig.write_image(f"docs/population_probe_{short}.png", scale=2)

    # --- 2. Separability plot ---
    logger.info("  Separability by layer...")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(n_layers)), y=sep[:, 0],
        name='attn_delta', line=dict(color='#1f77b4', width=2)))
    fig.add_trace(go.Scatter(x=list(range(n_layers)), y=sep[:, 1],
        name='mlp_delta', line=dict(color='#ff7f0e', width=2)))
    fig.add_hline(y=0.5, line_dash='dash', line_color='gray', opacity=0.5)
    attn_peak = sep[:, 0].argmax()
    mlp_peak = sep[:, 1].argmax()
    fig.add_trace(go.Scatter(x=[attn_peak], y=[sep[attn_peak, 0]],
        mode='markers+text', text=[f'L{attn_peak}: {sep[attn_peak,0]:.3f}'],
        textposition='top center', marker=dict(size=10, color='#1f77b4'), showlegend=False))
    fig.add_trace(go.Scatter(x=[mlp_peak], y=[sep[mlp_peak, 1]],
        mode='markers+text', text=[f'L{mlp_peak}: {sep[mlp_peak,1]:.3f}'],
        textposition='top center', marker=dict(size=10, color='#ff7f0e'), showlegend=False))
    fig.update_layout(
        title=f'{short} — Holdout Separability by Layer',
        xaxis_title='Layer', yaxis_title='AUROC (best PC)',
        yaxis_range=[0.45, 1.02], template='plotly_white', width=900, height=500)
    fig.write_html(f'docs/separability_{short}.html')
    fig.write_image(f'docs/separability_{short}.png', scale=2)

    # --- 3. Contrastive brain scan ---
    logger.info("  Contrastive brain scan...")
    mean_dec = dec_vecs.mean(axis=0)
    mean_truth = truth_vecs.mean(axis=0)
    diff = mean_dec - mean_truth
    n_pcs_show = 16

    sig_idx = 0
    diff_attn = diff[:, sig_idx, :n_pcs_show]
    l2_div = np.sqrt((diff[:, sig_idx, :] ** 2).sum(axis=-1))

    fig = make_subplots(rows=1, cols=2, column_widths=[0.8, 0.2],
        subplot_titles=["Mean(Dec) − Mean(Truth)", "L2 Divergence"])
    fig.add_trace(go.Heatmap(z=diff_attn,
        x=[f"PC{i}" for i in range(n_pcs_show)], y=list(range(n_layers)),
        colorscale="RdBu_r", zmid=0), row=1, col=1)
    fig.add_trace(go.Bar(y=list(range(n_layers)), x=l2_div,
        orientation='h', marker_color='steelblue', showlegend=False), row=1, col=2)
    fig.update_layout(title=f"{short} — Contrastive Brain Scan (attn_delta)",
        height=800, width=900, template='plotly_white',
        yaxis=dict(title="Layer", autorange="reversed"),
        yaxis2=dict(autorange="reversed"))
    fig.write_html(f"docs/contrastive_{short}.html")
    fig.write_image(f"docs/contrastive_{short}.png", scale=2)

    # --- 4. Brain scan hero plots ---
    logger.info("  Brain scan hero plots...")
    dec_samples = samples[samples['label'] == 1].sort_values('seq_length')
    truth_samples_df = samples[samples['label'] == 0].sort_values('seq_length')

    picks = [
        ("dec", dec_samples.iloc[len(dec_samples)//4]),
        ("dec", dec_samples.iloc[len(dec_samples)//3]),
        ("truth", truth_samples_df.iloc[len(truth_samples_df)//4]),
        ("truth", truth_samples_df.iloc[len(truth_samples_df)//3]),
    ]

    dec_i, truth_i = 0, 0
    for label, row in picks:
        sid = int(row['sample_id'])
        proj = scan.get_projections(sid)
        seq_len = int(row['seq_length'])
        ids, tokens, gen_mask = get_tokens_and_mask(row['prompt_text'])
        actual_len = min(seq_len, len(tokens), proj.shape[0])
        proj_c = proj[:actual_len]
        tokens_c = tokens[:actual_len]
        gen_mask_c = gen_mask[:actual_len]

        if gen_mask_c.any():
            gen_start = np.where(gen_mask_c)[0][0]
            proj_gen = proj_c[gen_start:]
            tokens_gen = tokens_c[gen_start:]
        else:
            proj_gen = proj_c
            tokens_gen = tokens_c

        idx = dec_i if label == "dec" else truth_i
        fig = render_scan_figure(
            projections=proj_gen, tokens=tokens_gen,
            signal_names=scan.signals,
            title=f"{short} — {'Deceptive' if label == 'dec' else 'Truthful'} #{idx+1}")
        fig.savefig(f"docs/brain_{label}_{short}_{idx}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        if label == "dec":
            dec_i += 1
        else:
            truth_i += 1

    # --- 5. Deception probe heatmaps ---
    logger.info("  Deception probe heatmaps...")
    pc_sign = np.ones(n_layers)
    best_auroc_per_layer = np.zeros(n_layers)
    for layer in range(n_layers):
        pc = best_pc_per_layer[layer]
        d_mean = dec_vecs[:, layer, 0, pc].mean()
        t_mean = truth_vecs[:, layer, 0, pc].mean()
        if d_mean < t_mean:
            pc_sign[layer] = -1
        max_auroc = 0.5
        for pc_i in range(min(k, 16)):
            vals = all_vecs[:, layer, 0, pc_i]
            if np.std(vals) < 1e-10:
                continue
            try:
                auroc = roc_auc_score(labels_arr, vals)
                auroc = max(auroc, 1 - auroc)
                max_auroc = max(max_auroc, auroc)
            except:
                continue
        best_auroc_per_layer[layer] = max_auroc

    probe_examples = [
        (dec_samples.iloc[len(dec_samples)//4], "dec", 0),
        (dec_samples.iloc[len(dec_samples)//3], "dec", 1),
        (truth_samples_df.iloc[len(truth_samples_df)//4], "truth", 0),
        (truth_samples_df.iloc[len(truth_samples_df)//3], "truth", 1),
    ]

    for row, label, idx in probe_examples:
        sid = int(row['sample_id'])
        proj = scan.get_projections(sid)
        seq_len = int(row['seq_length'])
        ids, tokens, gen_mask = get_tokens_and_mask(row['prompt_text'])
        actual_len = min(seq_len, len(tokens), proj.shape[0])

        probe_scores = np.zeros((n_layers, actual_len))
        for layer in range(n_layers):
            pc = best_pc_per_layer[layer]
            vals = proj[:actual_len, layer, 0, pc] * pc_sign[layer]
            probe_scores[layer] = vals * best_auroc_per_layer[layer]

        if gen_mask[:actual_len].any():
            gen_start = np.where(gen_mask[:actual_len])[0][0]
        else:
            gen_start = 0

        probe_gen = probe_scores[:, gen_start:]
        tokens_gen = tokens[gen_start:actual_len]

        fig = go.Figure(data=go.Heatmap(
            z=probe_gen, x=tokens_gen, y=list(range(n_layers)),
            colorscale="RdBu_r", zmid=0))
        fig.update_layout(
            title=f"{short} — Deception Probe ({'Dec' if label == 'dec' else 'Truth'} #{idx+1})",
            xaxis=dict(title="Token", tickangle=60, tickfont=dict(size=8, family="monospace")),
            yaxis=dict(title="Layer", autorange="reversed"),
            height=700, width=max(600, len(tokens_gen) * 15), template='plotly_white')
        fig.write_image(f"docs/probe_{label}_{short}_{idx}.png", scale=2)

    # --- 6. Difference plot ---
    logger.info("  Difference plot...")
    fig = make_subplots(rows=2, cols=2, column_widths=[0.8, 0.2],
        subplot_titles=["attn_delta: Δ(Dec-Truth)", "L2 (attn)",
                        "mlp_delta: Δ(Dec-Truth)", "L2 (mlp)"],
        vertical_spacing=0.12)
    for si, sig in enumerate(scan.signals):
        row_num = si + 1
        d = diff[:, si, :n_pcs_show]
        l2 = np.sqrt((diff[:, si, :] ** 2).sum(axis=-1))
        fig.add_trace(go.Heatmap(z=d,
            x=[f"PC{i}" for i in range(n_pcs_show)], y=list(range(n_layers)),
            colorscale="RdBu_r", zmid=0, showscale=(si == 0)), row=row_num, col=1)
        fig.add_trace(go.Bar(y=list(range(n_layers)), x=l2,
            orientation='h', marker_color='steelblue', showlegend=False), row=row_num, col=2)
        fig.update_yaxes(autorange="reversed", row=row_num, col=1)
        fig.update_yaxes(autorange="reversed", row=row_num, col=2)
    fig.update_layout(title=f"{short} — Dec − Truth Difference",
        height=1200, width=900, template='plotly_white')
    fig.write_html(f"docs/diff_{short}.html")
    fig.write_image(f"docs/diff_{short}.png", scale=2)

    logger.info(f"All plots saved for {short}")


def get_available_datasets(dataset_key):
    """Return list of datasets that have deceptive+truthful samples for this model."""
    available = []
    for ds_name in DATASETS:
        loaded = load_from_disk(f"data/{ds_name}")
        ds = loaded["test"] if isinstance(loaded, dict) else loaded
        ds_filtered = ds.filter(lambda x: x["model"] == dataset_key)
        dec = ds_filtered.filter(lambda x: x["deceptive"])
        truth = ds_filtered.filter(lambda x: not x["deceptive"])
        if len(dec) >= 50 and len(truth) >= 50:
            available.append(ds_name)
    return available


def fit_basis(model_cfg, fit_dataset, device="cuda:0"):
    """Fit a PCA basis on the given dataset. Returns the SampleScan object."""
    model_name = model_cfg["hf_id"]
    dataset_key = model_cfg["dataset_key"]
    backend_type = model_cfg["backend"]

    scan_dir = Path("scans") / f"{fit_dataset}_{dataset_key}_holdout"

    # Reuse existing basis if already fit
    if (scan_dir / "metadata.json").exists():
        logger.info(f"  Basis already exists: {scan_dir}")
        return SampleScan(str(scan_dir))

    logger.info(f"  Fitting basis on {fit_dataset} for {model_name}...")
    tokenizer = load_tokenizer(model_name)

    (train_prompts, train_labels, train_masks,
     _, _, _) = build_scan_data(fit_dataset, dataset_key, tokenizer)

    scan = SampleScan.run(
        prompts=train_prompts,
        labels=train_labels,
        model_name=model_name,
        scan_dir=str(scan_dir),
        signals=["attn_delta", "mlp_delta"],
        n_components=64,
        device=device,
        dtype=torch.bfloat16,
        generative_masks=train_masks,
        backend=backend_type,
    )
    logger.info(f"  Basis fit: {scan.n_layers} layers, {scan.n_samples} samples")
    return scan


def eval_with_basis(scan, model_cfg, eval_dataset, fit_dataset, metrics_only=False):
    """Evaluate a pre-fit basis on a given dataset. Returns metrics dict."""
    dataset_key = model_cfg["dataset_key"]
    model_name = model_cfg["hf_id"]
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    cache_path = results_dir / f"test_vecs_crossfit_{fit_dataset}_eval_{eval_dataset}_{dataset_key}.npz"

    n_layers = scan.n_layers
    n_signals = len(scan.signals)
    k = scan.n_components

    if metrics_only and cache_path.exists():
        logger.info(f"    Loading cached: {fit_dataset} → {eval_dataset}")
        cached = np.load(cache_path)
        test_vecs = cached["test_vecs"]
        labels_arr = cached["labels"]
    else:
        tokenizer = load_tokenizer(model_name)

        # Use holdout split for within-dataset, full test for cross-dataset
        if fit_dataset == eval_dataset:
            (_, _, _, test_prompts, test_labels, test_asst_last) = build_scan_data(
                eval_dataset, dataset_key, tokenizer)
        else:
            loaded = load_from_disk(f"data/{eval_dataset}")
            ds = loaded["test"] if isinstance(loaded, dict) else loaded
            ds_filtered = ds.filter(lambda x: x["model"] == dataset_key)
            deceptive = ds_filtered.filter(lambda x: x["deceptive"])
            truthful = ds_filtered.filter(lambda x: not x["deceptive"])
            n_per = min(50, len(deceptive), len(truthful))

            rows, test_labels = [], []
            for i in range(n_per):
                rows.append(deceptive[i])
                test_labels.append(1)
            for i in range(n_per):
                rows.append(truthful[i])
                test_labels.append(0)

            test_prompts = [tokenizer.apply_chat_template(r["messages"], tokenize=False) for r in rows]
            test_asst_last = []
            for row in rows:
                boundaries = get_assistant_turn_boundaries(row["messages"], tokenizer)
                if boundaries:
                    test_asst_last.append(boundaries[-1][1] - 1)
                else:
                    ids = _template_to_ids(tokenizer, row["messages"])
                    test_asst_last.append(len(ids) - 1)

        # Batch project — all samples in one forward pass per layer chunk
        logger.info(f"    Batch projecting {len(test_prompts)} samples...")
        test_vecs = _batch_project_to_vecs(scan, test_prompts, test_asst_last,
                                            n_layers, n_signals, k)

        labels_arr = np.array(test_labels)
        np.savez(cache_path, test_vecs=test_vecs, labels=labels_arr)
        logger.info(f"    Cached → {cache_path}")

    metrics = compute_metrics(test_vecs, labels_arr, n_layers, n_signals, k)
    return metrics


def run_cross_fit_matrix(model_cfg, device="cuda:0", metrics_only=False):
    """Full cross-fit matrix: fit basis on each dataset, eval on all others."""
    model_name = model_cfg["hf_id"]
    dataset_key = model_cfg["dataset_key"]
    short = model_cfg["short"]

    logger.info(f"=== Cross-fit matrix: {model_name} ===")
    available = get_available_datasets(dataset_key)
    logger.info(f"  Available datasets: {available}")

    results_dir = Path("results")
    matrix = {}

    for fit_ds in available:
        logger.info(f"\n  --- Fitting on: {fit_ds} ---")
        scan = fit_basis(model_cfg, fit_ds, device=device)

        matrix[fit_ds] = {}
        for eval_ds in available:
            logger.info(f"    Evaluating on: {eval_ds}")
            metrics = eval_with_basis(scan, model_cfg, eval_ds, fit_ds, metrics_only=metrics_only)
            matrix[fit_ds][eval_ds] = {
                "attn_auroc": float(metrics["auroc"][:, 0].max()),
                "mlp_auroc": float(metrics["auroc"][:, 1].max()),
                "attn_bal_acc": float(metrics["balanced_accuracy"][:, 0].max()),
                "mlp_bal_acc": float(metrics["balanced_accuracy"][:, 1].max()),
                "attn_bal_acc_1pct_fpr": float(metrics["balanced_accuracy_1pct_fpr"][:, 0].max()),
                "mlp_bal_acc_1pct_fpr": float(metrics["balanced_accuracy_1pct_fpr"][:, 1].max()),
                "attn_recall_1pct_fpr": float(metrics["recall_1pct_fpr"][:, 0].max()),
                "mlp_recall_1pct_fpr": float(metrics["recall_1pct_fpr"][:, 1].max()),
                "attn_bal_acc_optimal": float(metrics["balanced_accuracy_optimal"][:, 0].max()),
                "mlp_bal_acc_optimal": float(metrics["balanced_accuracy_optimal"][:, 1].max()),
                "attn_f1_optimal": float(metrics["f1_optimal"][:, 0].max()),
                "mlp_f1_optimal": float(metrics["f1_optimal"][:, 1].max()),
                "attn_avg_prec": float(metrics["average_precision"][:, 0].max()),
                "mlp_avg_prec": float(metrics["average_precision"][:, 1].max()),
                "attn_best_layer": int(metrics["auroc"][:, 0].argmax()),
                "mlp_best_layer": int(metrics["auroc"][:, 1].argmax()),
            }

        # Free GPU memory between basis fits
        if not metrics_only:
            torch.cuda.empty_cache()

    # Save full matrix
    with open(results_dir / f"cross_fit_matrix_{dataset_key}.json", "w") as f:
        json.dump(matrix, f, indent=2)

    # Log summary table
    logger.info(f"\n  Cross-fit matrix for {short}:")
    fit_eval = "Fit \\ Eval"
    header = f"  {fit_eval:<35s}" + "".join(f"{d[:15]:>18s}" for d in available)
    logger.info(header)
    logger.info("  AUROC (attn/mlp):")
    for fit_ds in available:
        row = f"  {fit_ds:<35s}"
        for eval_ds in available:
            cell = matrix[fit_ds][eval_ds]
            row += f"  {cell['attn_auroc']:.3f}/{cell['mlp_auroc']:.3f}"
        logger.info(row)
    logger.info("  BalAcc@1%FPR (attn/mlp):")
    for fit_ds in available:
        row = f"  {fit_ds:<35s}"
        for eval_ds in available:
            cell = matrix[fit_ds][eval_ds]
            row += f"  {cell['attn_bal_acc_1pct_fpr']:.3f}/{cell['mlp_bal_acc_1pct_fpr']:.3f}"
        logger.info(row)

    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Model to run: llama, qwen, or all")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip scan, just regenerate plots from existing data")
    parser.add_argument("--metrics-only", action="store_true",
                        help="Recompute metrics from cached test_vecs (no GPU needed)")
    parser.add_argument("--cross-fit", action="store_true",
                        help="Run full cross-fit matrix: fit on each dataset, eval on all others")
    args = parser.parse_args()

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model_key in models_to_run:
        if model_key not in MODELS:
            logger.error(f"Unknown model: {model_key}")
            continue

        cfg = MODELS[model_key]

        if args.cross_fit:
            run_cross_fit_matrix(cfg, device=args.device, metrics_only=args.metrics_only)
        elif args.metrics_only:
            scan, sep_map, metrics = run_holdout_scan(cfg, device=args.device, metrics_only=True)
            cross_results = run_cross_dataset(scan, cfg, device=args.device, metrics_only=True)
        elif not args.plots_only:
            scan, sep_map, metrics = run_holdout_scan(cfg, device=args.device)
            cross_results = run_cross_dataset(scan, cfg, device=args.device)

        if not args.cross_fit:
            generate_plots(cfg)


if __name__ == "__main__":
    main()
