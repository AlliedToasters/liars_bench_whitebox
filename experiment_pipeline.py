"""Pre-registered experiment pipeline.

Two phases:
  1. GPU phase: project all splits through bases, cache everything.
  2. CPU phase: evaluate frozen probes, run ablations, compute metrics.

GPU phase runs once. CPU phase runs in <1 second per evaluation cell,
supporting rapid iteration on probe selection, ablations, and baselines.

Usage:
  # Phase 1 (fused, recommended): one pass caches every LB split + Alpaca.
  # Amortizes layer streaming across all datasets. Use for new models.
  python experiment_pipeline.py cache_all --model mistral
  python experiment_pipeline.py cache_all --model all

  # Phase 1 (legacy, per-dataset): preserved for partial reruns.
  python experiment_pipeline.py cache --model mistral
  python experiment_pipeline.py alpaca --model mistral

  # Phase 2: evaluate with blinded protocol + Alpaca-calibrated threshold (CPU)
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
ALPACA_FPR = 0.01  # Target FPR for threshold calibration (paper: 1% on Alpaca)


def _load_tokenizer(model_name):
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

    # Adaptive chunk and inner batch size. Outer `chunk` = samples per
    # batch_project call = amortization factor for layer-streaming cost
    # (dominant with chunked backend). Inner `inner_bs` = microbatch size
    # for the forward pass = memory pressure for activations. With SDPA
    # (lmprobe default as of 2026-04-16) inner_bs can go higher on long
    # sequences since the T×T softmax tensor isn't materialized. Length
    # probed from the LONGEST sample, not the first, to avoid undersizing
    # chunks when a short sample happens to sit at index 0.
    sample_len = max(len(tokenizer.encode(t)) for t in transcripts[:min(len(transcripts), 32)])
    if sample_len > 1500:
        chunk, inner_bs = 300, 1
    elif sample_len > 1000:
        chunk, inner_bs = 400, 2
    elif sample_len > 500:
        chunk, inner_bs = 600, 2
    else:
        chunk, inner_bs = 1000, 4

    last_token_vecs = np.zeros((n, n_layers, n_signals, k), dtype=np.float32)
    mean_asst_vecs = np.zeros((n, n_layers, n_signals, k), dtype=np.float32)
    mean_excl5_vecs = np.zeros((n, n_layers, n_signals, k), dtype=np.float32)

    sig_names = scan.signals

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        chunk_transcripts = transcripts[start:end]

        projections, coords, _, seq_lengths = scan.batch_project(
            chunk_transcripts, batch_size=min(inner_bs, len(chunk_transcripts))
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


def _aggregate_group_projections(
    projections, coords, seq_lengths, asst_boundaries,
    n_layers, n_signals, k, sig_names,
):
    """Aggregate per-group projections from a single `batch_project_grouped`
    call into the three per-sample variants (last_token, mean_asst, mean_excl5).

    Pure numpy — no forward passes. Called once per group after the single
    fused sweep. `projections`/`coords` come from the rebased per-group
    tuple returned by `scan.batch_project_grouped`.
    """
    n = len(seq_lengths)
    last_token_vecs = np.zeros((n, n_layers, n_signals, k), dtype=np.float32)
    mean_asst_vecs = np.zeros((n, n_layers, n_signals, k), dtype=np.float32)
    mean_excl5_vecs = np.zeros((n, n_layers, n_signals, k), dtype=np.float32)

    coord_sample = np.asarray(coords["sample_id"])
    coord_layer = np.asarray(coords["layer"])
    coord_tok = np.asarray(coords["token_pos"])
    coord_sig = np.asarray(coords["signal"])

    for i in range(n):
        boundaries = asst_boundaries[i]
        if not boundaries:
            continue

        asst_positions = set()
        for s, e in boundaries:
            for t in range(s, min(e, seq_lengths[i])):
                asst_positions.add(t)
        if not asst_positions:
            continue

        last_asst = max(asst_positions)
        asst_sorted = sorted(asst_positions)
        excl5_set = set(asst_sorted[:-5] if len(asst_sorted) > 5 else asst_sorted)

        for si, sig in enumerate(sig_names):
            sample_mask = (coord_sample == i) & (coord_sig == sig)
            sample_rows = np.where(sample_mask)[0]

            for row_idx in sample_rows:
                layer = coord_layer[row_idx]
                tok = coord_tok[row_idx]
                vec = projections[row_idx, 0, :]

                if tok == last_asst:
                    last_token_vecs[i, layer, si, :] = vec
                if tok in asst_positions:
                    mean_asst_vecs[i, layer, si, :] += vec
                if tok in excl5_set:
                    mean_excl5_vecs[i, layer, si, :] += vec

        n_asst = len(asst_positions)
        n_excl5 = len(excl5_set)
        if n_asst > 0:
            mean_asst_vecs[i] /= n_asst
        if n_excl5 > 0:
            mean_excl5_vecs[i] /= n_excl5

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
    tokenizer = _load_tokenizer(model_name)

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


def cache_alpaca(model_cfg):
    """GPU phase: project the Alpaca control set through the ID basis.

    Alpaca is the threshold-calibration control (paper §5.1): for each model,
    2000 model-generated responses to Alpaca prompts. We project them through
    the same ID basis used for the main probe, cache the three aggregation
    variants, and later use scores here to set the 1% FPR decision threshold.
    """
    model_name = model_cfg["hf_id"]
    model_key = model_cfg["dataset_key"]
    short = model_cfg["short"]

    from datasets import load_from_disk
    from lmprobe import SampleScan
    from on_policy_affirmation import get_assistant_turn_boundaries

    scan_dir = f"scans/instructed-deception_{model_key}_holdout"
    if not Path(scan_dir).exists():
        logger.error(f"No ID basis found at {scan_dir}. Run the basis fit first.")
        return
    scan = SampleScan(scan_dir)
    tokenizer = _load_tokenizer(model_name)

    cache_path = CACHE_DIR / model_key / "alpaca" / "control.npz"
    if cache_path.exists():
        logger.info(f"=== {short} Alpaca already cached at {cache_path} ===")
        return

    logger.info(f"=== Caching Alpaca control for {short} ===")
    loaded = load_from_disk("data/alpaca")
    ds = loaded["test"] if isinstance(loaded, dict) else loaded
    ds_filtered = ds.filter(lambda x: x["model"] == model_key)
    n = len(ds_filtered)
    if n == 0:
        logger.error(f"  No Alpaca rows for model_key={model_key}")
        return
    logger.info(f"  {n} Alpaca responses for {model_key}")

    transcripts = []
    labels = []  # all False for Alpaca (control set), kept for shape consistency
    asst_boundaries = []
    for row in ds_filtered:
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
        transcripts.append(text)
        labels.append(1 if row["deceptive"] else 0)
        boundaries = get_assistant_turn_boundaries(row["messages"], tokenizer)
        asst_boundaries.append(boundaries)

    vecs = _extract_aggregated_vecs(scan, transcripts, asst_boundaries, tokenizer)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        last_token=vecs["last_token"],
        mean_asst=vecs["mean_asst"],
        mean_excl5=vecs["mean_excl5"],
        labels=np.array(labels),
    )
    logger.info(f"  Cached → {cache_path} ({vecs['last_token'].shape})")
    logger.info(f"=== Done caching Alpaca for {short} ===")


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
    """Run one fused layer-streaming sweep over `prompt_groups`, then
    aggregate and write per-group .npz caches. Returns total samples
    processed."""
    if not prompt_groups:
        return 0

    max_len = 0
    for transcripts in prompt_groups.values():
        probe = transcripts[:min(len(transcripts), 16)]
        max_len = max(max_len, max(len(tokenizer.encode(t)) for t in probe))
    inner_bs = _pick_inner_bs(max_len)

    total = sum(len(v) for v in prompt_groups.values())
    keys = sorted(prompt_groups.keys())
    logger.info(f"    sweep: {total} samples across {len(prompt_groups)} groups "
                f"{keys} (max_len≈{max_len}, inner_bs={inner_bs})")

    results = scan.batch_project_grouped(prompt_groups, batch_size=inner_bs)

    n_layers = scan.n_layers
    n_signals = len(scan.signals)
    k = scan.n_components
    sig_names = scan.signals

    for key, (projections, coords, _tokens, seq_lengths) in results.items():
        vecs = _aggregate_group_projections(
            projections, coords, seq_lengths,
            boundaries_groups[key],
            n_layers, n_signals, k, sig_names,
        )
        cache_path = cache_paths[key]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            last_token=vecs["last_token"],
            mean_asst=vecs["mean_asst"],
            mean_excl5=vecs["mean_excl5"],
            labels=np.asarray(labels_groups[key]),
        )
        logger.info(f"    cached {len(seq_lengths)} → {cache_path}")

    return total


def cache_all(model_cfg, backend: str = "auto"):
    """Fused GPU phase: project all LB splits + Alpaca via per-dataset sweeps.

    Uses `SampleScan.batch_project_grouped` (lmprobe spec 001) once per
    dataset so that all splits within a dataset pass through each loaded
    layer-chunk before the next chunk loads. Same amortization pattern as
    DiskOffloadBackend's `extract_all` — every model layer moves CPU→GPU
    exactly once per dataset-level sweep, not once per split.

    One sweep per dataset (rather than one giant sweep over everything) is
    the memory-safe granularity. The full-union version would need to hold
    the projection tensor + CPU-resident model weights + per-sample
    intermediate residuals simultaneously; on a 128 GB box with IT-length
    sequences this exceeds RAM and OOM-kills the process. Per-dataset
    sweeps cap peak RAM at ~one dataset worth of projections while still
    giving most of the amortization benefit relative to the old
    per-split approach (e.g. ~5 sweeps instead of 12 for a fresh model).

    Per-group results are sliced by lmprobe and aggregated here into the
    three variants (last_token, mean_asst, mean_excl5), then written to
    the same cache paths as the legacy `cache` + `alpaca` commands.
    """
    model_name = model_cfg["hf_id"]
    model_key = model_cfg["dataset_key"]
    short = model_cfg["short"]

    from datasets import load_from_disk
    from lmprobe import SampleScan
    from on_policy_affirmation import get_assistant_turn_boundaries

    scan_dir = f"scans/instructed-deception_{model_key}_holdout"
    if not Path(scan_dir).exists():
        logger.error(f"No ID basis found at {scan_dir}. Run the basis fit first.")
        return
    scan = SampleScan(scan_dir, backend=backend)
    tokenizer = _load_tokenizer(model_name)

    logger.info(f"=== {short}: per-dataset fused sweeps (backend={backend}) ===")
    total_cached = 0

    # --- One fused sweep per dataset (fuses that dataset's splits) ---
    for ds_name in DATASETS:
        splits = _load_split_indices(ds_name, model_key)
        if splits is None:
            logger.info(f"  {ds_name}: no split file, skipping")
            continue

        prompt_groups: dict = {}
        boundaries_groups: dict = {}
        labels_groups: dict = {}
        cache_paths: dict = {}

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

        if prompt_groups:
            logger.info(f"  {ds_name}: sweeping {len(prompt_groups)} split(s)")
            total_cached += _fused_sweep_and_cache(
                scan, prompt_groups, boundaries_groups,
                labels_groups, cache_paths, tokenizer,
            )

    # --- Alpaca in its own sweep (bounded-size control set) ---
    alpaca_path = CACHE_DIR / model_key / "alpaca" / "control.npz"
    if alpaca_path.exists():
        logger.info("  alpaca/control: already cached")
    else:
        loaded = load_from_disk("data/alpaca")
        ds = loaded["test"] if isinstance(loaded, dict) else loaded
        ds_filtered = ds.filter(lambda x: x["model"] == model_key)
        if len(ds_filtered) > 0:
            a_transcripts, a_labels, a_boundaries = [], [], []
            for row in ds_filtered:
                a_transcripts.append(
                    tokenizer.apply_chat_template(row["messages"], tokenize=False)
                )
                a_labels.append(1 if row["deceptive"] else 0)
                a_boundaries.append(
                    get_assistant_turn_boundaries(row["messages"], tokenizer)
                )
            key = ("alpaca", "control")
            logger.info("  alpaca: sweeping control set")
            total_cached += _fused_sweep_and_cache(
                scan,
                {key: a_transcripts},
                {key: a_boundaries},
                {key: np.array(a_labels)},
                {key: alpaca_path},
                tokenizer,
            )
        else:
            logger.warning(f"  alpaca/control: no rows for {model_key}")

    if total_cached == 0:
        logger.info(f"=== {short}: everything already cached, nothing to do ===")
    else:
        logger.info(f"=== Done fused caching {short}: {total_cached} samples ===")


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


def _score_probe(probe, vecs, aggregation="last_token"):
    """Return per-sample scalar scores for a frozen probe (oriented so
    higher = more deceptive)."""
    data = vecs[aggregation]
    scores = data[:, probe["layer"], probe["signal"], probe["pc"]].copy()
    if probe["flip"]:
        scores = -scores
    return scores


def calibrate_threshold_alpaca(alpaca_scores, target_fpr=ALPACA_FPR):
    """Compute a decision threshold such that at most `target_fpr` of Alpaca
    scores exceed it. Paper §5.1: 1% FPR on the per-model Alpaca control."""
    # All Alpaca examples are negatives (control). Threshold = (1 - target_fpr)
    # quantile of Alpaca scores. A higher score = more deceptive.
    return float(np.quantile(alpaca_scores, 1.0 - target_fpr))


def evaluate_frozen_probe(probe, vecs, labels, aggregation="last_token",
                          alpaca_threshold=None):
    """Evaluate a frozen probe on cached data. Returns metrics dict.

    If `alpaca_threshold` is given, `bal_acc_1pct_fpr` and `recall_1pct_fpr`
    use that threshold (paper-matching). Otherwise they use the in-sample
    ROC-derived threshold (diagnostic only — NOT comparable to the paper).
    """
    scores = _score_probe(probe, vecs, aggregation=aggregation)
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

    # 1% FPR threshold — paper-matching when alpaca_threshold is given
    if alpaca_threshold is not None:
        preds_1pct = (scores >= alpaca_threshold).astype(int)
        results["bal_acc_1pct_fpr"] = float(balanced_accuracy_score(labels, preds_1pct))
        results["recall_1pct_fpr"] = float(recall_score(labels, preds_1pct, zero_division=0))
        results["threshold_1pct_fpr"] = float(alpaca_threshold)
        results["threshold_source"] = "alpaca"

    # In-sample 1% FPR (diagnostic only; NOT comparable to paper)
    try:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        idx = max(0, np.searchsorted(fpr, 0.01, side='right') - 1)
        thresh = thresholds[min(idx, len(thresholds) - 1)]
        preds_1pct_in = (scores >= thresh).astype(int)
        results["bal_acc_1pct_fpr_insample"] = float(balanced_accuracy_score(labels, preds_1pct_in))
        results["recall_1pct_fpr_insample"] = float(recall_score(labels, preds_1pct_in, zero_division=0))
        if alpaca_threshold is None:
            # Fallback: expose insample numbers under the canonical key with a source tag
            results["bal_acc_1pct_fpr"] = results["bal_acc_1pct_fpr_insample"]
            results["recall_1pct_fpr"] = results["recall_1pct_fpr_insample"]
            results["threshold_1pct_fpr"] = float(thresh)
            results["threshold_source"] = "insample"
    except ValueError:
        results["bal_acc_1pct_fpr_insample"] = 0.5
        results["recall_1pct_fpr_insample"] = 0.0
        if alpaca_threshold is None:
            results["bal_acc_1pct_fpr"] = 0.5
            results["recall_1pct_fpr"] = 0.0
            results["threshold_1pct_fpr"] = 0.0
            results["threshold_source"] = "insample"

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


def _score_dom(dom_direction, frozen_layer, frozen_signal, vecs, aggregation="last_token"):
    """Return per-sample DoM scalar scores (dot product with direction).

    Orientation: direction is already dec_mean - truth_mean, so higher scores
    are by construction more deceptive. No test-set flip.
    """
    data = vecs[aggregation]
    d = dom_direction[frozen_layer, frozen_signal, :]
    return (data[:, frozen_layer, frozen_signal, :] * d).sum(axis=-1)


def evaluate_dom_frozen(dom_direction, frozen_layer, frozen_signal, vecs, labels,
                         aggregation="last_token", alpaca_threshold=None):
    """Evaluate DoM with a frozen (layer, signal) choice.

    Orientation comes from `compute_dom_direction` (dec_mean - truth_mean), so
    scores are already signed correctly — no test-set AUROC flip. If
    `alpaca_threshold` is given, 1% FPR metrics use it; otherwise in-sample.
    """
    scores = _score_dom(dom_direction, frozen_layer, frozen_signal, vecs, aggregation=aggregation)

    results = {}
    try:
        results["auroc"] = float(roc_auc_score(labels, scores))
    except ValueError:
        results["auroc"] = 0.5

    threshold = np.median(scores)
    preds = (scores >= threshold).astype(int)
    results["bal_acc_median"] = float(balanced_accuracy_score(labels, preds))

    if alpaca_threshold is not None:
        preds_1pct = (scores >= alpaca_threshold).astype(int)
        results["bal_acc_1pct_fpr"] = float(balanced_accuracy_score(labels, preds_1pct))
        results["recall_1pct_fpr"] = float(recall_score(labels, preds_1pct, zero_division=0))
        results["threshold_1pct_fpr"] = float(alpaca_threshold)
        results["threshold_source"] = "alpaca"

    try:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        idx = max(0, np.searchsorted(fpr, 0.01, side='right') - 1)
        thresh = thresholds[min(idx, len(thresholds) - 1)]
        preds_1pct_in = (scores >= thresh).astype(int)
        results["bal_acc_1pct_fpr_insample"] = float(balanced_accuracy_score(labels, preds_1pct_in))
        results["recall_1pct_fpr_insample"] = float(recall_score(labels, preds_1pct_in, zero_division=0))
        if alpaca_threshold is None:
            results["bal_acc_1pct_fpr"] = results["bal_acc_1pct_fpr_insample"]
            results["recall_1pct_fpr"] = results["recall_1pct_fpr_insample"]
            results["threshold_1pct_fpr"] = float(thresh)
            results["threshold_source"] = "insample"
    except ValueError:
        results["bal_acc_1pct_fpr_insample"] = 0.5
        results["recall_1pct_fpr_insample"] = 0.0
        if alpaca_threshold is None:
            results["bal_acc_1pct_fpr"] = 0.5
            results["recall_1pct_fpr"] = 0.0
            results["threshold_1pct_fpr"] = 0.0
            results["threshold_source"] = "insample"

    return results


# ============================================================
# Main evaluation flow
# ============================================================

def _load_alpaca_cache(model_key):
    """Load the Alpaca control cache for a model, or None if not cached."""
    path = CACHE_DIR / model_key / "alpaca" / "control.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return {k: d[k] for k in d.files}


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

    # --- Alpaca threshold calibration (paper §5.1: 1% FPR on model-generated Alpaca) ---
    alpaca = _load_alpaca_cache(model_key)
    probe_alpaca_thresh = None
    dom_alpaca_thresh = None
    if alpaca is None:
        logger.warning(f"  No Alpaca cache for {model_key} at "
                       f"{CACHE_DIR}/{model_key}/alpaca/control.npz — "
                       f"falling back to in-sample 1% FPR. "
                       f"Run `experiment_pipeline.py alpaca --model {model_key}` first "
                       f"for paper-matching numbers.")
    else:
        probe_alpaca_scores = _score_probe(probe, alpaca, aggregation="last_token")
        probe_alpaca_thresh = calibrate_threshold_alpaca(probe_alpaca_scores, target_fpr=ALPACA_FPR)
        dom_alpaca_scores = _score_dom(
            dom_dir, probe["layer"], probe["signal"], alpaca, aggregation="last_token"
        )
        dom_alpaca_thresh = calibrate_threshold_alpaca(dom_alpaca_scores, target_fpr=ALPACA_FPR)
        logger.info(f"  Alpaca thresholds (1% FPR, n={len(probe_alpaca_scores)}): "
                    f"probe={probe_alpaca_thresh:.4f}, dom={dom_alpaca_thresh:.4f}")

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
                metrics = evaluate_frozen_probe(
                    probe, cache, labels, aggregation=agg,
                    alpaca_threshold=probe_alpaca_thresh,
                )
                results[f"{key}/ours/{agg}"] = metrics

            # DoM baseline (frozen layer/signal from probe)
            dom_metrics = evaluate_dom_frozen(
                dom_dir, probe["layer"], probe["signal"],
                cache, labels, aggregation="last_token",
                alpaca_threshold=dom_alpaca_thresh,
            )
            results[f"{key}/dom/last_token"] = dom_metrics

        logger.info(f"  {ds_name}: done")

    # Save results
    out_path = Path("results") / f"blinded_eval_{model_key}.json"
    with open(out_path, "w") as f:
        json.dump({
            "probe": probe,
            "alpaca_threshold_probe": probe_alpaca_thresh,
            "alpaca_threshold_dom": dom_alpaca_thresh,
            "threshold_source": "alpaca" if probe_alpaca_thresh is not None else "insample",
            "results": results,
        }, f, indent=2)
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
    parser.add_argument("command", choices=["cache", "alpaca", "cache_all",
                                             "evaluate", "sanity", "ablations"])
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "chunked", "disk_offload"],
                        help="Projection backend (cache_all only). 'auto' picks "
                             "via lmprobe heuristic; 'disk_offload' is safer for "
                             "large sample counts × long sequences.")
    args = parser.parse_args()

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model_key in models_to_run:
        if model_key not in MODELS:
            logger.error(f"Unknown model: {model_key}")
            continue
        cfg = MODELS[model_key]

        if args.command == "cache":
            cache_projections(cfg)
        elif args.command == "alpaca":
            cache_alpaca(cfg)
        elif args.command == "cache_all":
            cache_all(cfg, backend=args.backend)
        elif args.command == "evaluate":
            run_evaluate(cfg)
        elif args.command == "sanity":
            run_sanity_check(cfg)
        elif args.command == "ablations":
            # TODO: layer-fixed-at-20th-percentile, different aggregations, etc.
            run_evaluate(cfg)


if __name__ == "__main__":
    main()
