"""Brain scan experiment: Instructed-Deception with Mistral.

Runs a SampleScan on the instructed-deception dataset to build a compressed
activation index, then computes a separability map to identify which layers
and signals best distinguish deceptive from truthful trajectories.

Usage:
    python brain_scan_id.py \
        --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
        --n 500 \
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
from transformers import AutoTokenizer

from lmprobe import SampleScan

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_KEY_TO_HF = {
    "mistral-small-3.1-24b-instruct": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "gemma-3-27b-it": "google/gemma-3-27b-it",
    "llama-v3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen-2.5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct",
}


def main():
    parser = argparse.ArgumentParser(description="Brain scan: Instructed-Deception")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-Small-3.1-24B-Instruct-2503")
    parser.add_argument("--dataset", type=str, default="instructed-deception")
    parser.add_argument("--n", type=int, default=500,
                        help="Number of examples (balanced between classes)")
    parser.add_argument("--n-components", type=int, default=64,
                        help="PCA components per (layer, signal)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--signals", type=str, nargs="+",
                        default=["attn_delta", "mlp_delta"],
                        help="Signals to capture")
    args = parser.parse_args()

    # Resolve model name
    model_name = args.model
    if model_name in MODEL_KEY_TO_HF:
        model_name = MODEL_KEY_TO_HF[model_name]
    logger.info(f"Model: {model_name}")

    # Load dataset and filter to this model
    data_path = Path(__file__).parent / "data" / args.dataset
    ds = load_from_disk(str(data_path))["test"]
    logger.info(f"Dataset: {args.dataset} ({len(ds)} total rows)")

    # Find the model key
    dataset_model_key = None
    model_keys_in_data = set(ds["model"])
    for key, hf_id in MODEL_KEY_TO_HF.items():
        if hf_id == model_name and key in model_keys_in_data:
            dataset_model_key = key
            break
    if dataset_model_key is None:
        if model_name in model_keys_in_data:
            dataset_model_key = model_name
        else:
            logger.error(f"Model {model_name} not found. Available: {model_keys_in_data}")
            return

    ds_filtered = ds.filter(lambda x: x["model"] == dataset_model_key)
    logger.info(f"Filtered to '{dataset_model_key}': {len(ds_filtered)} rows")

    # Balanced subsample
    deceptive = ds_filtered.filter(lambda x: x["deceptive"])
    truthful = ds_filtered.filter(lambda x: not x["deceptive"])
    n_per_class = min(args.n // 2, len(deceptive), len(truthful))
    logger.info(f"Balanced sample: {n_per_class} deceptive + {n_per_class} truthful = {2 * n_per_class}")

    deceptive_sub = deceptive.select(range(n_per_class))
    truthful_sub = truthful.select(range(n_per_class))

    # Load tokenizer and apply chat template
    tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load chat template if needed
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

    # Build prompts and labels
    prompts = []
    labels = []

    for row in deceptive_sub:
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
        prompts.append(text)
        labels.append(1)  # deceptive = 1

    for row in truthful_sub:
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
        prompts.append(text)
        labels.append(0)  # truthful = 0

    logger.info(f"Prepared {len(prompts)} prompts")

    # Run the scan
    model_short = dataset_model_key.replace("/", "_")
    scan_dir = Path(__file__).parent / "scans" / f"{args.dataset}_{model_short}"

    logger.info(f"Running SampleScan...")
    logger.info(f"  Signals: {args.signals}")
    logger.info(f"  Components: {args.n_components}")
    logger.info(f"  Scan dir: {scan_dir}")

    scan = SampleScan.run(
        prompts=prompts,
        labels=labels,
        model_name=model_name,
        scan_dir=str(scan_dir),
        signals=args.signals,
        n_components=args.n_components,
        device=args.device,
        dtype=torch.bfloat16,
    )

    logger.info(f"Scan complete: {scan.n_layers} layers, {scan.n_samples} samples")

    # Compute separability map
    logger.info("Computing separability map...")
    sep_map = scan.separability_map(labels=np.array(labels))
    logger.info(f"Separability map shape: {sep_map.shape}")

    # Report top-separating layers per signal
    for sig_idx, sig_name in enumerate(scan.signals):
        scores = sep_map[:, sig_idx]
        top_layers = np.argsort(scores)[::-1][:5]
        logger.info(f"  {sig_name}: top layers = {list(top_layers)}, "
                     f"AUROCs = {[f'{scores[l]:.3f}' for l in top_layers]}")

    # Save separability map
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    sep_path = results_dir / f"separability_{args.dataset}_{model_short}.npy"
    np.save(sep_path, sep_map)
    logger.info(f"Separability map saved to {sep_path}")

    # Save summary
    summary = {
        "model": model_name,
        "dataset": args.dataset,
        "n_samples": len(prompts),
        "n_per_class": n_per_class,
        "signals": args.signals,
        "n_components": args.n_components,
        "n_layers": scan.n_layers,
        "scan_dir": str(scan_dir),
        "separability": {
            sig: {
                "top_layers": [int(l) for l in np.argsort(sep_map[:, i])[::-1][:10]],
                "top_aurocs": [float(sep_map[l, i]) for l in np.argsort(sep_map[:, i])[::-1][:10]],
                "mean_auroc": float(sep_map[:, i].mean()),
                "max_auroc": float(sep_map[:, i].max()),
            }
            for i, sig in enumerate(scan.signals)
        },
    }

    summary_path = results_dir / f"brain_scan_{args.dataset}_{model_short}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
