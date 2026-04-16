"""Pre-registered experiment pipeline.

Two phases:
  1. GPU phase: project all splits through bases, cache everything.
  2. CPU phase: evaluate frozen probes, run ablations, compute metrics.

GPU phase runs once. CPU phase runs in <1 second per evaluation cell,
supporting rapid iteration on probe selection, ablations, and baselines.

Usage:
  # Phase 1: cache all projections (GPU, ~2-3 hours total)
  python experiment_pipeline.py cache --model mistral
  python experiment_pipeline.py cache --model all

  # Phase 2: evaluate with blinded protocol (CPU, seconds)
  python experiment_pipeline.py evaluate --model all

  # Phase 2b: run all ablations and baselines (CPU, seconds)
  python experiment_pipeline.py ablations --model all
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import (roc_auc_score, roc_curve, balanced_accuracy_score,
                              average_precision_score, recall_score, f1_score)
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS = {
    "mistral": {
        "hf_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "dataset_key": "mistral-small-3.1-24b-instruct",
        "short": "mistral-24b",
    },
    "gemma": {
        "hf_id": "google/gemma-3-27b-it",
        "dataset_key": "gemma-3-27b-it",
        "short": "gemma-27b",
    },
    "llama": {
        "hf_id": "meta-llama/Llama-3.3-70B-Instruct",
        "dataset_key": "llama-v3.3-70b-instruct",
        "short": "llama-70b",
    },
    "qwen": {
        "hf_id": "Qwen/Qwen2.5-72B-Instruct",
        "dataset_key": "qwen-2.5-72b-instruct",
        "short": "qwen-72b",
    },
}

DATASETS = ["instructed-deception", "harm-pressure-choice",
            "harm-pressure-knowledge-report", "insider-trading"]

CACHE_DIR = Path("experiment_cache")


# ============================================================
# Caching helpers
# ============================================================

def _load_split_indices(ds_name, model_key):
    """Load committed split indices for a dataset/model pair."""
    split_file = Path("splits") / f"{ds_name}_{model_key}.json"
    if not split_file.exists():
        return None
    with open(split_file) as f:
        return json.load(f)


def _get_transcripts_and_boundaries(ds_name, model_key, indices, tokenizer):
    """Load transcripts and compute assistant turn boundaries for given indices."""
    from datasets import load_from_disk
    from on_policy_affirmation import _template_to_ids, get_assistant_turn_boundaries

    loaded = load_from_disk(f"data/{ds_name}")
    ds = loaded["test"] if isinstance(loaded, dict) else loaded
    ds_filtered = ds.filter(lambda x: x["model"] == model_key)

    transcripts = []
    labels = []
    asst_boundaries = []  # list of list of (start, end) per sample

    for idx in indices:
        row = ds_filtered[idx]
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
        transcripts.append(text)
        labels.append(1 if row["deceptive"] else 0)
        boundaries = get_assistant_turn_boundaries(row["messages"], tokenizer)
        asst_boundaries.append(boundaries)

    return transcripts, np.array(labels), asst_boundaries


def _extract_aggregated_vecs(scan, transcripts, asst_boundaries, tokenizer):
    """Project transcripts and extract three aggregation variants.

    Returns dict with:
      'last_token': [n, n_layers, n_signals, k] — assistant's last token
      'mean_asst': [n, n_layers, n_signals, k] — mean over all assistant tokens
      'mean_excl5': [n, n_layers, n_signals, k] — mean over assistant tokens, last 5 excluded
    """
    n_layers = scan.n_layers
    n_signals = len(scan.signals)
    k = scan.n_components
    n = len(transcripts)

    # Adaptive batch size for long sequences
    sample_len = len(tokenizer.encode(transcripts[0]))
    if sample_len > 1000:
        chunk = 10
    elif sample_len > 500:
        chunk = 25
    else:
        chunk = 100

    last_token_vecs = np.zeros((n, n_layers, n_signals, k), dtype=np.float32)
    mean_asst_vecs = np.zeros((n, n_layers, n_signals, k), dtype=np.float32)
    mean_excl5_vecs = np.zeros((n, n_layers, n_signals, k), dtype=np.float32)

    sig_names = scan.signals

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        chunk_transcripts = transcripts[start:end]

        projections, coords, _, seq_lengths = scan.batch_project(
            chunk_transcripts, batch_size=min(4, len(chunk_transcripts))
        )

        coord_sample = np.array(coords["sample_id"])
        coord_layer = np.array(coords["layer"])
        coord_tok = np.array(coords["token_pos"])
        coord_sig = np.array(coords["signal"])

        for i in range(len(chunk_transcripts)):
            global_i = start + i
            boundaries = asst_boundaries[global_i]
            if not boundaries:
                continue

            # Get all assistant token positions from boundaries
            asst_positions = set()
            for s, e in boundaries:
                for t in range(s, min(e, seq_lengths[i])):
                    asst_positions.add(t)

            if not asst_positions:
                continue

            last_asst = max(asst_positions)
            asst_sorted = sorted(asst_positions)
            excl5_positions = asst_sorted[:-5] if len(asst_sorted) > 5 else asst_sorted

            for si, sig in enumerate(sig_names):
                sample_mask = (coord_sample == i) & (coord_sig == sig)
                sample_rows = np.where(sample_mask)[0]

                # Collect per-token projections for this sample/signal
                for row_idx in sample_rows:
                    layer = coord_layer[row_idx]
                    tok = coord_tok[row_idx]
                    vec = projections[row_idx, 0, :]

                    # Last token
                    if tok == last_asst:
                        last_token_vecs[global_i, layer, si, :] = vec

                    # Mean over assistant (accumulate, divide later)
                    if tok in asst_positions:
                        mean_asst_vecs[global_i, layer, si, :] += vec

                    # Mean excluding last 5
                    if tok in set(excl5_positions):
                        mean_excl5_vecs[global_i, layer, si, :] += vec

            # Normalize means
            n_asst = len(asst_positions)
            n_excl5 = len(excl5_positions)
            if n_asst > 0:
                mean_asst_vecs[global_i] /= n_asst
            if n_excl5 > 0:
                mean_excl5_vecs[global_i] /= n_excl5

        logger.info(f"    [{end}/{n}] projected")

    return {
        "last_token": last_token_vecs,
        "mean_asst": mean_asst_vecs,
        "mean_excl5": mean_excl5_vecs,
    }


def cache_projections(model_cfg):
    """GPU phase: project all splits through the ID basis, cache everything."""
    model_name = model_cfg["hf_id"]
    model_key = model_cfg["dataset_key"]
    short = model_cfg["short"]

    from lmprobe import SampleScan

    # Load the existing ID basis
    scan_dir = f"scans/instructed-deception_{model_key}_holdout"
    if not Path(scan_dir).exists():
        logger.error(f"No ID basis found at {scan_dir}. Run the basis fit first.")
        return
    scan = SampleScan(scan_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"=== Caching projections for {short} ===")

    for ds_name in DATASETS:
        splits = _load_split_indices(ds_name, model_key)
        if splits is None:
            logger.info(f"  {ds_name}: no split file, skipping")
            continue

        for split_name, split_data in splits.items():
            cache_path = CACHE_DIR / model_key / ds_name / f"{split_name}.npz"
            if cache_path.exists():
                logger.info(f"  {ds_name}/{split_name}: already cached")
                continue

            # Combine deceptive + truthful indices
            indices = split_data["deceptive"] + split_data["truthful"]
            if len(indices) == 0:
                continue

            logger.info(f"  {ds_name}/{split_name}: projecting {len(indices)} samples...")
            transcripts, labels, asst_boundaries = _get_transcripts_and_boundaries(
                ds_name, model_key, indices, tokenizer
            )

            vecs = _extract_aggregated_vecs(scan, transcripts, asst_boundaries, tokenizer)

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_path,
                last_token=vecs["last_token"],
                mean_asst=vecs["mean_asst"],
                mean_excl5=vecs["mean_excl5"],
                labels=labels,
            )
            logger.info(f"    Cached → {cache_path} ({vecs['last_token'].shape})")

    logger.info(f"=== Done caching {short} ===")


# ============================================================
# Probe selection (CPU only)
# ============================================================

def _load_cache(model_key, ds_name, split_name):
    """Load cached projections."""
    path = CACHE_DIR / model_key / ds_name / f"{split_name}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return {k: d[k] for k in d.files}


def select_probe(vecs, labels, aggregation="last_token"):
    """Select best (layer, signal, PC, flip) from labeled data.

    Returns a frozen probe dict.
    """
    data = vecs[aggregation]
    n_layers = data.shape[1]
    n_signals = data.shape[2]
    k = data.shape[3]

    best_auroc = 0.5
    best = {"layer": 0, "signal": 0, "pc": 0, "flip": False, "auroc": 0.5}

    for sig in range(n_signals):
        for layer in range(n_layers):
            for pc in range(k):
                vals = data[:, layer, sig, pc]
                if np.std(vals) < 1e-10:
                    continue
                try:
                    auroc = roc_auc_score(labels, vals)
                    flip = auroc < 0.5
                    auroc_adj = 1.0 - auroc if flip else auroc
                    if auroc_adj > best_auroc:
                        best_auroc = auroc_adj
                        best = {
                            "layer": int(layer),
                            "signal": int(sig),
                            "pc": int(pc),
                            "flip": bool(flip),
                            "auroc": float(auroc_adj),
                        }
                except ValueError:
                    continue

    return best


def evaluate_frozen_probe(probe, vecs, labels, aggregation="last_token"):
    """Evaluate a frozen probe on cached data. Returns metrics dict."""
    data = vecs[aggregation]
    scores = data[:, probe["layer"], probe["signal"], probe["pc"]].copy()
    if probe["flip"]:
        scores = -scores

    results = {}

    # AUROC
    try:
        results["auroc"] = float(roc_auc_score(labels, scores))
    except ValueError:
        results["auroc"] = 0.5

    # Balanced accuracy at median threshold
    threshold = np.median(scores)
    preds = (scores >= threshold).astype(int)
    results["bal_acc_median"] = float(balanced_accuracy_score(labels, preds))

    # 1% FPR threshold
    try:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        idx = max(0, np.searchsorted(fpr, 0.01, side='right') - 1)
        thresh = thresholds[min(idx, len(thresholds) - 1)]
        preds_1pct = (scores >= thresh).astype(int)
        results["bal_acc_1pct_fpr"] = float(balanced_accuracy_score(labels, preds_1pct))
        results["recall_1pct_fpr"] = float(recall_score(labels, preds_1pct, zero_division=0))
    except ValueError:
        results["bal_acc_1pct_fpr"] = 0.5
        results["recall_1pct_fpr"] = 0.0

    # Optimal threshold (Youden's J)
    try:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        j_idx = np.argmax(tpr - fpr)
        thresh_opt = thresholds[min(j_idx, len(thresholds) - 1)]
        preds_opt = (scores >= thresh_opt).astype(int)
        results["bal_acc_optimal"] = float(balanced_accuracy_score(labels, preds_opt))
        results["f1_optimal"] = float(f1_score(labels, preds_opt, zero_division=0))
    except ValueError:
        results["bal_acc_optimal"] = 0.5
        results["f1_optimal"] = 0.0

    # Average precision
    try:
        results["avg_precision"] = float(average_precision_score(labels, scores))
    except ValueError:
        results["avg_precision"] = 0.0

    return results


# ============================================================
# Baselines (CPU only, from cached vecs)
# ============================================================

def compute_dom_direction(vecs, labels, aggregation="last_token"):
    """Difference-of-means baseline: one direction per (layer, signal)."""
    data = vecs[aggregation]
    dec_mask = labels == 1
    truth_mask = labels == 0
    dec_mean = data[dec_mask].mean(axis=0)   # [n_layers, n_signals, k]
    truth_mean = data[truth_mask].mean(axis=0)
    diff = dec_mean - truth_mean  # [n_layers, n_signals, k]
    # Normalize per (layer, signal)
    norms = np.linalg.norm(diff, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return diff / norms


def evaluate_dom(dom_direction, vecs, labels, aggregation="last_token"):
    """Evaluate difference-of-means: dot product with direction, pick best layer."""
    data = vecs[aggregation]
    n_layers = data.shape[1]
    n_signals = data.shape[2]

    # Score = dot(activation, direction) per (layer, signal)
    # Pick best (layer, signal) on the data provided
    best_auroc = 0.5
    best = {"layer": 0, "signal": 0}

    for sig in range(n_signals):
        for layer in range(n_layers):
            scores = (data[:, layer, sig, :] * dom_direction[layer, sig, :]).sum(axis=-1)
            try:
                auroc = roc_auc_score(labels, scores)
                auroc = max(auroc, 1 - auroc)
                if auroc > best_auroc:
                    best_auroc = auroc
                    best = {"layer": layer, "signal": sig}
            except ValueError:
                continue

    # Evaluate at best (layer, signal)
    scores = (data[:, best["layer"], best["signal"], :] *
              dom_direction[best["layer"], best["signal"], :]).sum(axis=-1)
    if roc_auc_score(labels, scores) < 0.5:
        scores = -scores

    return evaluate_frozen_probe(
        {"layer": best["layer"], "signal": best["signal"], "pc": 0, "flip": False},
        # Hack: create a single-PC view for the evaluate function
        {"last_token": scores.reshape(-1, 1, 1, 1)},  # won't work directly
        labels
    ), best


def evaluate_dom_frozen(dom_direction, frozen_layer, frozen_signal, vecs, labels,
                         aggregation="last_token"):
    """Evaluate DoM with a frozen (layer, signal) choice."""
    data = vecs[aggregation]
    d = dom_direction[frozen_layer, frozen_signal, :]
    scores = (data[:, frozen_layer, frozen_signal, :] * d).sum(axis=-1)

    # Orient using training data sign (caller must ensure this)
    results = {}
    try:
        auroc = roc_auc_score(labels, scores)
        if auroc < 0.5:
            scores = -scores
            auroc = 1 - auroc
        results["auroc"] = float(auroc)
    except ValueError:
        results["auroc"] = 0.5

    threshold = np.median(scores)
    preds = (scores >= threshold).astype(int)
    results["bal_acc_median"] = float(balanced_accuracy_score(labels, preds))

    try:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        idx = max(0, np.searchsorted(fpr, 0.01, side='right') - 1)
        thresh = thresholds[min(idx, len(thresholds) - 1)]
        preds_1pct = (scores >= thresh).astype(int)
        results["bal_acc_1pct_fpr"] = float(balanced_accuracy_score(labels, preds_1pct))
        results["recall_1pct_fpr"] = float(recall_score(labels, preds_1pct, zero_division=0))
    except ValueError:
        results["bal_acc_1pct_fpr"] = 0.5
        results["recall_1pct_fpr"] = 0.0

    return results


# ============================================================
# Main evaluation flow
# ============================================================

def run_evaluate(model_cfg):
    """Blinded evaluation: freeze probe on ID train/dev, evaluate everywhere."""
    model_key = model_cfg["dataset_key"]
    short = model_cfg["short"]

    logger.info(f"=== Blinded evaluation: {short} ===")

    # Load ID train and dev caches
    id_train = _load_cache(model_key, "instructed-deception", "train")
    id_dev = _load_cache(model_key, "instructed-deception", "dev")

    if id_train is None or id_dev is None:
        logger.error(f"  Missing ID train/dev cache for {model_key}. Run 'cache' first.")
        return

    # Freeze probe on dev split (basis was fit on train, select direction on dev)
    probe = select_probe(id_dev, id_dev["labels"], aggregation="last_token")
    logger.info(f"  Frozen probe: layer={probe['layer']}, signal={probe['signal']}, "
                f"pc={probe['pc']}, flip={probe['flip']}, dev_auroc={probe['auroc']:.3f}")

    # Also compute DoM direction on train split
    dom_dir = compute_dom_direction(id_train, id_train["labels"], aggregation="last_token")

    # Evaluate on all datasets and splits
    results = {}
    for ds_name in DATASETS:
        for split_name in ["dev", "held_out"]:
            cache = _load_cache(model_key, ds_name, split_name)
            if cache is None:
                continue

            key = f"{ds_name}/{split_name}"
            labels = cache["labels"]

            # Our method (frozen probe)
            for agg in ["last_token", "mean_asst", "mean_excl5"]:
                metrics = evaluate_frozen_probe(probe, cache, labels, aggregation=agg)
                results[f"{key}/ours/{agg}"] = metrics

            # DoM baseline (frozen layer/signal from probe)
            dom_metrics = evaluate_dom_frozen(
                dom_dir, probe["layer"], probe["signal"],
                cache, labels, aggregation="last_token"
            )
            results[f"{key}/dom/last_token"] = dom_metrics

        logger.info(f"  {ds_name}: done")

    # Save results
    out_path = Path("results") / f"blinded_eval_{model_key}.json"
    with open(out_path, "w") as f:
        json.dump({"probe": probe, "results": results}, f, indent=2)
    logger.info(f"  Saved → {out_path}")

    # Print summary table
    logger.info(f"\n  === {short} Blinded Results (last_token) ===")
    logger.info(f"  {'Dataset/Split':<45s} {'AUROC':>7s} {'BA@1%':>7s} {'Recall':>7s}")
    for ds_name in DATASETS:
        for split_name in ["held_out"]:
            key = f"{ds_name}/{split_name}/ours/last_token"
            if key in results:
                r = results[key]
                logger.info(f"  {ds_name}/{split_name:<25s} {r['auroc']:7.3f} "
                            f"{r['bal_acc_1pct_fpr']:7.3f} {r['recall_1pct_fpr']:7.3f}")

    logger.info(f"\n  === {short} DoM Baseline (last_token) ===")
    logger.info(f"  {'Dataset/Split':<45s} {'AUROC':>7s} {'BA@1%':>7s} {'Recall':>7s}")
    for ds_name in DATASETS:
        for split_name in ["held_out"]:
            key = f"{ds_name}/{split_name}/dom/last_token"
            if key in results:
                r = results[key]
                logger.info(f"  {ds_name}/{split_name:<25s} {r['auroc']:7.3f} "
                            f"{r['bal_acc_1pct_fpr']:7.3f} {r['recall_1pct_fpr']:7.3f}")

    # Aggregation ablation
    logger.info(f"\n  === {short} Aggregation Ablation (ID held_out) ===")
    for agg in ["last_token", "mean_asst", "mean_excl5"]:
        key = f"instructed-deception/held_out/ours/{agg}"
        if key in results:
            r = results[key]
            logger.info(f"  {agg:<20s} AUROC={r['auroc']:.3f} BA@1%={r['bal_acc_1pct_fpr']:.3f}")

    return results


# ============================================================
# Scrambled-label sanity check (Experiment 5)
# ============================================================

def run_sanity_check(model_cfg, n_permutations=100):
    """Scrambled-label control: what AUROC do we get with random labels?"""
    model_key = model_cfg["dataset_key"]
    short = model_cfg["short"]

    id_dev = _load_cache(model_key, "instructed-deception", "dev")
    if id_dev is None:
        logger.error(f"  Missing ID dev cache for {model_key}")
        return

    rng = np.random.RandomState(42)
    null_aurocs = []

    for _ in range(n_permutations):
        shuffled = rng.permutation(id_dev["labels"])
        probe = select_probe(id_dev, shuffled, aggregation="last_token")
        null_aurocs.append(probe["auroc"])

    null_aurocs = np.array(null_aurocs)
    logger.info(f"  {short} scrambled-label AUROC: "
                f"mean={null_aurocs.mean():.3f} std={null_aurocs.std():.3f} "
                f"max={null_aurocs.max():.3f} (should be ~0.5-0.65 from multiple testing)")

    return null_aurocs


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["cache", "evaluate", "sanity", "ablations"])
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model_key in models_to_run:
        if model_key not in MODELS:
            logger.error(f"Unknown model: {model_key}")
            continue
        cfg = MODELS[model_key]

        if args.command == "cache":
            cache_projections(cfg)
        elif args.command == "evaluate":
            run_evaluate(cfg)
        elif args.command == "sanity":
            run_sanity_check(cfg)
        elif args.command == "ablations":
            # TODO: layer-fixed-at-20th-percentile, different aggregations, etc.
            run_evaluate(cfg)


if __name__ == "__main__":
    main()
