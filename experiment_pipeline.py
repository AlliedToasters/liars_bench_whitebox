"""Pre-registered experiment pipeline (Alpaca-basis methodology).

Two phases:
  1. GPU phase: fit Alpaca-train basis once per model (fit_basis.py), then
     project all LB splits + Alpaca-val through the basis in one fused
     sweep per model (this file, cache_all).
  2. CPU phase: fit linear probe on LB-ID train PC vectors, select
     (layer, signal, reducer) on LB-ID dev, calibrate 1% FPR threshold
     on Alpaca-val, evaluate on held-out splits.

GPU phase runs once per model. CPU phase runs in seconds and supports
rapid iteration on probe variants, ablations, and baselines — all on
cached PC vectors.

Usage:
  # Prereq: fit the Alpaca basis first.
  python fit_basis.py --model mistral

  # Phase 1: fused projection of every LB split + Alpaca-val.
  python experiment_pipeline.py cache_all --model mistral

  # Phase 2: fit probe on ID train, select on ID dev, calibrate on
  # Alpaca-val, evaluate on held-out. Prints a summary table.
  python experiment_pipeline.py evaluate --model mistral
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                              average_precision_score, recall_score)

from util import safe_load_tokenizer

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
ALPACA_FPR = 0.01  # Target FPR for threshold calibration (paper: 1% on Alpaca)


def _load_tokenizer(model_name):
    """Load tokenizer, fetching chat_template.json from HF hub if not embedded.

    Mistral-Small-3.1 and some other releases ship the chat template as a
    separate file rather than in tokenizer_config.json.
    """
    tokenizer = safe_load_tokenizer(model_name)
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
    from on_policy_affirmation import get_assistant_turn_boundaries

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


def _build_asst_masks_flat(transcripts_flat, boundaries_flat, tokenizer):
    """Build one `[seq_len_i]` bool mask per transcript, ``True`` at
    assistant-token positions.

    Masks must match lmprobe's tokenization exactly — the reducer
    contract is that ``len(mask[i])`` equals the number of real tokens
    in the microbatch — so we tokenize with the same call pattern
    lmprobe uses (``padding=True`` on the full flat list) and read real
    lens off the attention mask.
    """
    enc = tokenizer(transcripts_flat, return_tensors="pt", padding=True)
    attn = enc["attention_mask"]
    seq_lens = [int(row.sum()) for row in attn]
    del enc, attn

    masks = []
    for seq_len, boundaries in zip(seq_lens, boundaries_flat):
        m = np.zeros(seq_len, dtype=bool)
        for s, e in boundaries:
            m[s:min(e, seq_len)] = True
        masks.append(m)
    return masks


def _pick_inner_bs(max_len):
    """Pick microbatch size by longest sample. SDPA (lmprobe default) avoids
    the T×T softmax tensor, so inner_bs can ride higher than with eager."""
    if max_len > 1500:
        return 1
    if max_len > 1000:
        return 2
    if max_len > 500:
        return 2
    return 4


def _fused_sweep_and_cache(scan, prompt_groups, boundaries_groups, labels_groups,
                            cache_paths, tokenizer):
    """Run one fused layer-streaming sweep over `prompt_groups`, applying
    per-sample reducers in-chunk, and write per-group .npz caches.

    Uses `SampleScan.batch_project_reduced` (lmprobe spec 003): each
    microbatch's PCA projection is dispatched to LastTokenReducer /
    MeanReducer / MeanExclLastNReducer(n=5) as soon as it's produced on
    GPU, then freed. Per-token projections never accumulate across
    chunks — cross-chunk memory stays bounded by (residuals +
    ~O(N_total × L × n_sig × k) reducer outputs), independent of
    sequence length or chunk count.

    Returns total samples processed.
    """
    from lmprobe.reducers import (
        LastTokenReducer,
        MeanExclLastNReducer,
        MeanReducer,
    )

    if not prompt_groups:
        return 0

    # Preserve iteration order so reducer masks align with lmprobe's flat
    # union ordering of all prompts.
    keys = list(prompt_groups.keys())
    transcripts_flat: list[str] = []
    boundaries_flat: list = []
    for key in keys:
        transcripts_flat.extend(prompt_groups[key])
        boundaries_flat.extend(boundaries_groups[key])

    probe = transcripts_flat[:min(len(transcripts_flat), 16)]
    max_len = max(len(tokenizer.encode(t)) for t in probe)
    inner_bs = _pick_inner_bs(max_len)

    total = len(transcripts_flat)
    logger.info(f"    sweep: {total} samples across {len(prompt_groups)} groups "
                f"{keys} (max_len≈{max_len}, inner_bs={inner_bs})")

    masks_flat = _build_asst_masks_flat(
        transcripts_flat, boundaries_flat, tokenizer,
    )

    reducers = {
        "last_token": LastTokenReducer(masks_flat),
        "mean_asst":  MeanReducer(masks_flat),
        "mean_excl5": MeanExclLastNReducer(masks_flat, n=5),
    }

    results = scan.batch_project_reduced(
        prompt_groups, reducers=reducers, batch_size=inner_bs,
    )

    # Cache on disk as float32 to preserve the historical cache dtype
    # (reducers emit fp16; downstream probe code handles either, but
    # fp32 matches existing .npz files bit-equivalently in shape/dtype).
    for key in keys:
        per_reducer = results[key]
        cache_path = cache_paths[key]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            last_token=per_reducer["last_token"].astype(np.float32),
            mean_asst=per_reducer["mean_asst"].astype(np.float32),
            mean_excl5=per_reducer["mean_excl5"].astype(np.float32),
            labels=np.asarray(labels_groups[key]),
        )
        logger.info(
            f"    cached {per_reducer['last_token'].shape[0]} → {cache_path}",
        )

    return total


def _load_alpaca_val_corpus(model_key, tokenizer):
    """Load the committed Alpaca-val split (seed=2026, 1000 samples/model).

    Returns (transcripts, labels, boundaries). Labels are all zero — Alpaca
    is the non-deceptive calibration corpus.
    """
    from datasets import load_from_disk
    from on_policy_affirmation import get_assistant_turn_boundaries

    split_file = Path("splits") / f"alpaca_{model_key}.json"
    if not split_file.exists():
        logger.error(f"No Alpaca splits at {split_file}. Run create_alpaca_splits.py first.")
        return None
    with open(split_file) as f:
        splits = json.load(f)
    val_indices = splits["val"]

    loaded = load_from_disk("data/alpaca")
    ds = loaded["test"] if hasattr(loaded, "keys") else loaded
    ds_filtered = ds.filter(lambda x: x["model"] == model_key)

    transcripts, labels, boundaries = [], [], []
    for idx in val_indices:
        row = ds_filtered[idx]
        transcripts.append(
            tokenizer.apply_chat_template(row["messages"], tokenize=False)
        )
        labels.append(0)
        boundaries.append(
            get_assistant_turn_boundaries(row["messages"], tokenizer)
        )
    return transcripts, np.array(labels), boundaries


def _length_bucket_key(max_len):
    """Bucket label by rough max sequence length. Same bucket → one sweep."""
    if max_len > 1500:
        return "long"
    return "short"


def cache_all(model_cfg, backend: str = "auto"):
    """Fused GPU phase: project all LB splits + Alpaca-val through the
    Alpaca basis in at most two length-bucketed sweeps per model.

    Flow:
      1. Load the Alpaca basis at `scans/alpaca_{model}`.
      2. Collect all uncached (dataset, split) groups: every LB split per
         dataset plus Alpaca-val. Skip anything already cached.
      3. Length-bucket groups by max sample length: short (<=1500 tokens)
         and long (>1500). Each bucket runs one `batch_project_reduced`
         sweep — one 80-layer streaming pass amortized over all samples
         in that bucket.
      4. Per-group PC vectors are dispatched to LastTokenReducer /
         MeanReducer / MeanExclLastNReducer(n=5) on GPU and written to
         `experiment_cache/{model}/{dataset}/{split}.npz` as fp32.

    The length-bucketed fusion is a compromise: the single-sweep ideal
    pins inner_bs=1 whenever any sample is >1500 tokens, which wrecks
    throughput on short datasets (ID, Alpaca). Two buckets keeps the
    short-sample sweep at inner_bs=4 while still amortizing the basis
    streaming cost over all short-dataset samples at once.
    """
    model_name = model_cfg["hf_id"]
    model_key = model_cfg["dataset_key"]
    short = model_cfg["short"]

    from lmprobe import SampleScan

    scan_dir = f"scans/alpaca_{model_key}"
    if not Path(scan_dir).exists():
        logger.error(
            f"No Alpaca basis at {scan_dir}. Run "
            f"`python fit_basis.py --model {model_key}` first."
        )
        return
    scan = SampleScan(scan_dir, backend=backend)
    tokenizer = _load_tokenizer(model_name)

    logger.info(f"=== {short}: fused projection through Alpaca basis (backend={backend}) ===")

    prompt_groups: dict = {}
    boundaries_groups: dict = {}
    labels_groups: dict = {}
    cache_paths: dict = {}

    # --- Collect uncached LB splits ---
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
            indices = split_data["deceptive"] + split_data["truthful"]
            if not indices:
                continue
            transcripts, labels, boundaries = _get_transcripts_and_boundaries(
                ds_name, model_key, indices, tokenizer
            )
            key = (ds_name, split_name)
            prompt_groups[key] = transcripts
            boundaries_groups[key] = boundaries
            labels_groups[key] = labels
            cache_paths[key] = cache_path

    # --- Add Alpaca-val (1% FPR calibration corpus) ---
    alpaca_val_path = CACHE_DIR / model_key / "alpaca" / "val.npz"
    if alpaca_val_path.exists():
        logger.info("  alpaca/val: already cached")
    else:
        loaded = _load_alpaca_val_corpus(model_key, tokenizer)
        if loaded is not None:
            a_transcripts, a_labels, a_boundaries = loaded
            key = ("alpaca", "val")
            prompt_groups[key] = a_transcripts
            boundaries_groups[key] = a_boundaries
            labels_groups[key] = a_labels
            cache_paths[key] = alpaca_val_path

    if not prompt_groups:
        logger.info(f"  {short}: nothing to project — all splits already cached.")
        return

    # --- Bucket by max length, one fused sweep per bucket ---
    group_max_len: dict = {}
    for key, transcripts in prompt_groups.items():
        probe = transcripts[:min(len(transcripts), 16)]
        group_max_len[key] = max(len(tokenizer.encode(t)) for t in probe)

    buckets: dict = {"short": [], "long": []}
    for key in prompt_groups:
        buckets[_length_bucket_key(group_max_len[key])].append(key)

    total_cached = 0
    for bucket_name, keys in buckets.items():
        if not keys:
            continue
        bucket_groups = {k: prompt_groups[k] for k in keys}
        bucket_boundaries = {k: boundaries_groups[k] for k in keys}
        bucket_labels = {k: labels_groups[k] for k in keys}
        bucket_cache_paths = {k: cache_paths[k] for k in keys}
        logger.info(
            f"  bucket={bucket_name}: sweeping {len(keys)} group(s) "
            f"max_len={max(group_max_len[k] for k in keys)}"
        )
        total_cached += _fused_sweep_and_cache(
            scan, bucket_groups, bucket_boundaries,
            bucket_labels, bucket_cache_paths, tokenizer,
        )

    logger.info(f"=== {short}: done. {total_cached} samples projected ===")


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


REDUCERS = ["last_token", "mean_asst", "mean_excl5"]


def _load_alpaca_val(model_key):
    """Load the Alpaca-val cache (1000/model, non-deceptive, threshold calib)."""
    path = CACHE_DIR / model_key / "alpaca" / "val.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return {k: d[k] for k in d.files}


def fit_linear_probes(train_vecs, train_labels):
    """Fit L2 logistic regression on the 64-dim PC vector at each
    (reducer, layer, signal). Returns dict[reducer] → (w [L, S, k], b [L, S]).
    """
    from sklearn.linear_model import LogisticRegression

    probes = {}
    for reducer in REDUCERS:
        if reducer not in train_vecs:
            continue
        data = train_vecs[reducer]  # [n, L, S, k]
        _, L, S, k = data.shape
        w = np.zeros((L, S, k), dtype=np.float32)
        b = np.zeros((L, S), dtype=np.float32)
        for layer in range(L):
            for sig in range(S):
                X = data[:, layer, sig, :]
                if np.std(X) < 1e-10:
                    continue
                clf = LogisticRegression(
                    max_iter=1000, C=1.0, solver="lbfgs",
                ).fit(X, train_labels)
                w[layer, sig, :] = clf.coef_[0]
                b[layer, sig] = clf.intercept_[0]
        probes[reducer] = (w, b)
    return probes


def compute_dom_probes(train_vecs, train_labels):
    """Difference-of-means baseline on PC vectors. Same (w, b) shape as
    logreg so the rest of the pipeline is reducer-agnostic. Orientation:
    w = mean_dec − mean_truth, b = −<w, midpoint>. Higher score = more
    deceptive by construction.
    """
    probes = {}
    dec_mask = train_labels == 1
    truth_mask = train_labels == 0
    for reducer in REDUCERS:
        if reducer not in train_vecs:
            continue
        data = train_vecs[reducer]
        dec_mean = data[dec_mask].mean(axis=0)
        truth_mean = data[truth_mask].mean(axis=0)
        w = (dec_mean - truth_mean).astype(np.float32)
        midpoint = 0.5 * (dec_mean + truth_mean)
        b = (-np.einsum("lsk,lsk->ls", w, midpoint)).astype(np.float32)
        probes[reducer] = (w, b)
    return probes


def score_probe(probes, reducer, layer, signal, vecs):
    """Return per-sample scalar scores for a frozen (reducer, layer, signal).
    Higher = more deceptive by logreg / DoM convention."""
    w, b = probes[reducer]
    X = vecs[reducer][:, layer, signal, :]
    return X @ w[layer, signal, :] + b[layer, signal]


def select_best_probe(probes, dev_vecs, dev_labels):
    """Pick the (reducer, layer, signal) with max dev AUROC."""
    best = {"auroc": 0.5, "reducer": REDUCERS[0], "layer": 0, "signal": 0}
    for reducer in REDUCERS:
        if reducer not in probes or reducer not in dev_vecs:
            continue
        w, b = probes[reducer]
        data = dev_vecs[reducer]
        _, L, S, _ = data.shape
        for layer in range(L):
            for sig in range(S):
                scores = data[:, layer, sig, :] @ w[layer, sig, :] + b[layer, sig]
                if np.std(scores) < 1e-10:
                    continue
                try:
                    auroc = float(roc_auc_score(dev_labels, scores))
                except ValueError:
                    continue
                if auroc > best["auroc"]:
                    best = {"auroc": auroc, "reducer": reducer,
                             "layer": int(layer), "signal": int(sig)}
    return best


def calibrate_threshold(alpaca_val_scores, target_fpr=ALPACA_FPR):
    """Threshold = (1 − target_fpr) quantile of Alpaca-val probe scores.
    Alpaca-val is all non-deceptive, so this pins FPR ≤ target on the
    calibration corpus (paper §5.1)."""
    return float(np.quantile(alpaca_val_scores, 1.0 - target_fpr))


def evaluate_at_threshold(scores, labels, threshold):
    """Compute AUROC + BalAcc/Recall at the frozen threshold."""
    results = {}
    try:
        results["auroc"] = float(roc_auc_score(labels, scores))
    except ValueError:
        results["auroc"] = 0.5
    preds = (scores >= threshold).astype(int)
    results["bal_acc"] = float(balanced_accuracy_score(labels, preds))
    results["recall"] = float(recall_score(labels, preds, zero_division=0))
    try:
        results["avg_precision"] = float(average_precision_score(labels, scores))
    except ValueError:
        results["avg_precision"] = 0.0
    results["threshold"] = float(threshold)
    return results


def run_evaluate(model_cfg):
    """Blinded evaluation:
      1. Fit logreg probes on LB-ID train (per reducer, layer, signal).
      2. Select (reducer, layer, signal) on LB-ID dev.
      3. Calibrate 1% FPR threshold on Alpaca-val.
      4. Evaluate on every LB held-out split (ID, HP-C, HP-KR, IT, …).
    DoM baseline follows the same fit/select/calibrate/evaluate flow in
    parallel — everything after the probe-fit step is shared.
    """
    model_key = model_cfg["dataset_key"]
    short = model_cfg["short"]

    logger.info(f"=== Blinded evaluation: {short} ===")

    id_train = _load_cache(model_key, "instructed-deception", "train")
    id_dev = _load_cache(model_key, "instructed-deception", "dev")
    alpaca_val = _load_alpaca_val(model_key)

    if id_train is None or id_dev is None:
        logger.error(
            f"  Missing ID train/dev cache. Run `cache_all --model {model_key}` first."
        )
        return
    if alpaca_val is None:
        logger.error(
            f"  Missing Alpaca-val cache at experiment_cache/{model_key}/alpaca/val.npz. "
            f"Run `cache_all --model {model_key}` first."
        )
        return

    # --- Our method ---
    logger.info("  Fitting logreg probes on ID-train...")
    probes = fit_linear_probes(id_train, id_train["labels"])
    best = select_best_probe(probes, id_dev, id_dev["labels"])
    logger.info(
        f"  Frozen probe: reducer={best['reducer']} layer={best['layer']} "
        f"signal={best['signal']} dev_auroc={best['auroc']:.3f}"
    )
    probe_alpaca_scores = score_probe(
        probes, best["reducer"], best["layer"], best["signal"], alpaca_val,
    )
    probe_threshold = calibrate_threshold(probe_alpaca_scores, target_fpr=ALPACA_FPR)
    logger.info(
        f"  Alpaca-val threshold (1% FPR, n={len(probe_alpaca_scores)}): "
        f"τ={probe_threshold:.4f}"
    )

    # --- DoM baseline ---
    dom_probes = compute_dom_probes(id_train, id_train["labels"])
    dom_best = select_best_probe(dom_probes, id_dev, id_dev["labels"])
    logger.info(
        f"  DoM baseline: reducer={dom_best['reducer']} layer={dom_best['layer']} "
        f"signal={dom_best['signal']} dev_auroc={dom_best['auroc']:.3f}"
    )
    dom_alpaca_scores = score_probe(
        dom_probes, dom_best["reducer"], dom_best["layer"], dom_best["signal"],
        alpaca_val,
    )
    dom_threshold = calibrate_threshold(dom_alpaca_scores, target_fpr=ALPACA_FPR)

    # --- Evaluate on LB splits ---
    results = {}
    for ds_name in DATASETS:
        for split_name in ["dev", "held_out"]:
            cache = _load_cache(model_key, ds_name, split_name)
            if cache is None:
                continue
            labels = cache["labels"]
            key = f"{ds_name}/{split_name}"

            ours_scores = score_probe(
                probes, best["reducer"], best["layer"], best["signal"], cache,
            )
            results[f"{key}/ours"] = evaluate_at_threshold(
                ours_scores, labels, probe_threshold,
            )

            dom_scores = score_probe(
                dom_probes, dom_best["reducer"], dom_best["layer"], dom_best["signal"],
                cache,
            )
            results[f"{key}/dom"] = evaluate_at_threshold(
                dom_scores, labels, dom_threshold,
            )

    out_path = Path("results") / f"blinded_eval_{model_key}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "probe": {
                "reducer": best["reducer"], "layer": best["layer"],
                "signal": best["signal"], "dev_auroc": best["auroc"],
                "threshold": probe_threshold,
            },
            "dom": {
                "reducer": dom_best["reducer"], "layer": dom_best["layer"],
                "signal": dom_best["signal"], "dev_auroc": dom_best["auroc"],
                "threshold": dom_threshold,
            },
            "results": results,
        }, f, indent=2)
    logger.info(f"  Saved → {out_path}")

    logger.info(
        f"\n  === {short} Blinded Results (probe reducer={best['reducer']}) ==="
    )
    logger.info(f"  {'Dataset/Split':<45s} {'AUROC':>7s} {'BA@1%':>7s} {'Recall':>7s}")
    for ds_name in DATASETS:
        key = f"{ds_name}/held_out/ours"
        if key in results:
            r = results[key]
            logger.info(
                f"  {ds_name + '/held_out':<45s} {r['auroc']:7.3f} "
                f"{r['bal_acc']:7.3f} {r['recall']:7.3f}"
            )

    logger.info(
        f"\n  === {short} DoM Baseline (reducer={dom_best['reducer']}) ==="
    )
    for ds_name in DATASETS:
        key = f"{ds_name}/held_out/dom"
        if key in results:
            r = results[key]
            logger.info(
                f"  {ds_name + '/held_out':<45s} {r['auroc']:7.3f} "
                f"{r['bal_acc']:7.3f} {r['recall']:7.3f}"
            )
    return results


def run_sanity_check(model_cfg, n_permutations=50):
    """Scrambled-label control: shuffle ID-train labels, refit probe, select
    on dev. Expected AUROC ≈ 0.5 + multiple-testing drift. Materially higher
    indicates a pipeline bug."""
    model_key = model_cfg["dataset_key"]
    short = model_cfg["short"]

    id_train = _load_cache(model_key, "instructed-deception", "train")
    id_dev = _load_cache(model_key, "instructed-deception", "dev")
    if id_train is None or id_dev is None:
        logger.error(f"  Missing ID train/dev cache for {model_key}")
        return

    rng = np.random.RandomState(42)
    null_aurocs = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(id_train["labels"])
        probes = fit_linear_probes(id_train, shuffled)
        best = select_best_probe(probes, id_dev, id_dev["labels"])
        null_aurocs.append(best["auroc"])

    null_aurocs = np.array(null_aurocs)
    logger.info(
        f"  {short} scrambled-label AUROC: "
        f"mean={null_aurocs.mean():.3f} std={null_aurocs.std():.3f} "
        f"max={null_aurocs.max():.3f}"
    )
    return null_aurocs


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["cache_all", "evaluate", "sanity"])
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "chunked", "disk_offload"],
                        help="Projection backend (cache_all only).")
    args = parser.parse_args()

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]
    for model_key in models_to_run:
        if model_key not in MODELS:
            logger.error(f"Unknown model: {model_key}")
            continue
        cfg = MODELS[model_key]
        if args.command == "cache_all":
            cache_all(cfg, backend=args.backend)
        elif args.command == "evaluate":
            run_evaluate(cfg)
        elif args.command == "sanity":
            run_sanity_check(cfg)


if __name__ == "__main__":
    main()
