"""On-policy affirmation experiment.

Validates that reconstructed forward passes faithfully reproduce the
generative rollout by computing per-token log-likelihood of assistant
responses in the ground truth transcripts.

High likelihood (low perplexity) = correct template/tokenization.
Low likelihood = something is wrong with our reconstruction.

Usage:
    python on_policy_affirmation.py \
        --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
        --dataset alpaca \
        --n 50 \
        --backend chunked \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer

from lmprobe.backends import resolve_backend
from trajectory_cache import compute_trajectory, save_trajectories

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Map short model keys (as they appear in the dataset) to HuggingFace model IDs
MODEL_KEY_TO_HF = {
    "mistral-small-3.1-24b-instruct": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "gemma-3-27b-it": "google/gemma-3-27b-it",
    "llama-v3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen-2.5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct",
}


def _template_to_ids(tokenizer, messages, **kwargs) -> list[int]:
    """apply_chat_template that always returns a plain list of ints."""
    out = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    if hasattr(out, "input_ids"):
        out = out["input_ids"]
    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
        out = out[0]
    return out


def get_turn_boundaries(
    messages: list[dict],
    tokenizer: AutoTokenizer,
) -> list[dict]:
    """Find token-level boundaries for every turn in the conversation.

    Uses the chat template to identify where each message's content begins
    and ends by tokenizing progressively longer prefixes.

    Returns a list of dicts with keys: role, start, end.
    """
    turns = []

    # Walk through messages, accumulating as prefix
    prefix_messages = []
    for msg in messages:
        if not prefix_messages:
            # First message — starts at token 0
            start = 0
        elif msg["role"] == "assistant":
            # For assistant, use add_generation_prompt to find the start
            prefix_ids = _template_to_ids(
                tokenizer, prefix_messages, add_generation_prompt=True,
            )
            start = len(prefix_ids)
        else:
            # For system/user, the start is right after the previous prefix
            prefix_ids = _template_to_ids(tokenizer, prefix_messages)
            start = len(prefix_ids)

        prefix_plus = prefix_messages + [msg]
        prefix_plus_ids = _template_to_ids(tokenizer, prefix_plus)
        end = len(prefix_plus_ids)

        turns.append({"role": msg["role"], "start": start, "end": end})
        prefix_messages.append(msg)

    return turns


def get_assistant_turn_boundaries(
    messages: list[dict],
    tokenizer: AutoTokenizer,
) -> list[tuple[int, int]]:
    """Find token-level (start, end) for each assistant turn."""
    turns = get_turn_boundaries(messages, tokenizer)
    return [(t["start"], t["end"]) for t in turns if t["role"] == "assistant"]


def compute_log_likelihood(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    assistant_boundaries: list[tuple[int, int]],
) -> dict:
    """Compute per-token and aggregate log-likelihood for assistant tokens.

    Parameters
    ----------
    logits : (seq_len, vocab_size)
    input_ids : (seq_len,)
    assistant_boundaries : list of (start, end) pairs

    Returns
    -------
    dict with keys: mean_ll, mean_ppl, per_token_ll, n_tokens
    """
    # Shift: logits[t] predicts input_ids[t+1]
    shift_logits = logits[:-1]
    shift_labels = input_ids[1:]

    # Log-softmax over vocab dimension
    log_probs = torch.nn.functional.log_softmax(shift_logits.float(), dim=-1)

    # Gather the log-prob of the actual next token
    token_log_probs = log_probs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)

    # Build mask for assistant tokens (in the shifted space, a token at
    # position t in the original sequence corresponds to position t-1)
    mask = torch.zeros_like(token_log_probs, dtype=torch.bool)
    for start, end in assistant_boundaries:
        # In shifted space: predicting token at position `start` uses
        # logits at position `start - 1`. The range of predictions for
        # assistant tokens [start, end) maps to shifted indices [start-1, end-1).
        shifted_start = max(start - 1, 0)
        shifted_end = end - 1
        mask[shifted_start:shifted_end] = True

    assistant_ll = token_log_probs[mask]

    if len(assistant_ll) == 0:
        return {
            "mean_ll": 0.0,
            "mean_ppl": 1.0,
            "per_token_ll": [],
            "n_tokens": 0,
        }

    mean_ll = assistant_ll.mean().item()
    mean_ppl = float(np.exp(-mean_ll))

    return {
        "mean_ll": mean_ll,
        "mean_ppl": mean_ppl,
        "per_token_ll": assistant_ll.tolist(),
        "n_tokens": len(assistant_ll),
    }


def main():
    parser = argparse.ArgumentParser(description="On-policy affirmation experiment")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model ID or short key from dataset")
    parser.add_argument("--dataset", type=str, default="alpaca",
                        help="Dataset name (subdirectory of data/)")
    parser.add_argument("--n", type=int, default=50,
                        help="Number of examples to process")
    parser.add_argument("--backend", type=str, default="chunked",
                        choices=["chunked", "local", "disk_offload"],
                        help="lmprobe backend")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for extraction (lower = less VRAM)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: results/<dataset>_<model_key>.json)")
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

    # Find the model key used in the dataset
    model_keys_in_data = set(ds["model"])
    logger.info(f"Models in dataset: {model_keys_in_data}")

    # Match the requested model to a dataset key
    dataset_model_key = None
    for key, hf_id in MODEL_KEY_TO_HF.items():
        if hf_id == model_name and key in model_keys_in_data:
            dataset_model_key = key
            break

    if dataset_model_key is None:
        # Try direct match
        if model_name in model_keys_in_data:
            dataset_model_key = model_name
        else:
            logger.error(
                f"Model {model_name} not found in dataset. "
                f"Available: {model_keys_in_data}"
            )
            return

    ds_filtered = ds.filter(lambda x: x["model"] == dataset_model_key)
    logger.info(f"Filtered to model '{dataset_model_key}': {len(ds_filtered)} rows")

    # Subsample
    n = min(args.n, len(ds_filtered))
    ds_subset = ds_filtered.select(range(n))
    logger.info(f"Using {n} examples")

    # Load tokenizer and apply chat template
    tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Some models (e.g. Mistral) ship chat_template.json separately
    if tokenizer.chat_template is None:
        from huggingface_hub import hf_hub_download
        try:
            ct_path = hf_hub_download(model_name, "chat_template.json")
            ct_data = json.load(open(ct_path))
            if isinstance(ct_data, dict) and "chat_template" in ct_data:
                tokenizer.chat_template = ct_data["chat_template"]
            elif isinstance(ct_data, str):
                tokenizer.chat_template = ct_data
            logger.info("Loaded chat template from chat_template.json")
        except Exception:
            logger.warning("No chat template found — apply_chat_template may fail")

    templated_prompts = []
    all_boundaries = []
    all_turn_boundaries = []
    all_input_ids = []

    for row in ds_subset:
        messages = row["messages"]

        # Full templated string (what the model saw)
        full_str = tokenizer.apply_chat_template(
            messages, tokenize=False,
        )
        templated_prompts.append(full_str)

        # Token IDs for log-likelihood computation
        full_ids = _template_to_ids(tokenizer, messages)
        all_input_ids.append(torch.tensor(full_ids, dtype=torch.long))

        # Turn boundaries (all roles) and assistant-only subset
        all_turns = get_turn_boundaries(messages, tokenizer)
        boundaries = [(t["start"], t["end"]) for t in all_turns if t["role"] == "assistant"]
        all_boundaries.append(boundaries)
        all_turn_boundaries.append(all_turns)

    logger.info(f"Templated {len(templated_prompts)} prompts")
    logger.info(
        f"Token lengths: min={min(len(ids) for ids in all_input_ids)}, "
        f"max={max(len(ids) for ids in all_input_ids)}, "
        f"mean={np.mean([len(ids) for ids in all_input_ids]):.0f}"
    )

    # Extract logits via lmprobe
    logger.info(f"Initializing {args.backend} backend...")
    backend = resolve_backend(
        backend=args.backend,
        model_name=model_name,
        device=args.device,
        dtype=torch.bfloat16,
    )

    logger.info("Extracting logits...")
    # We only need logits, no hidden layer activations
    _, attention_mask, logits_batch, _ = backend.extract_batch_with_logits(
        prompts=templated_prompts,
        layer_indices=[],
    )
    logger.info(f"Logits shape: {logits_batch.shape}")

    # Compute per-example log-likelihood and cache trajectories
    results = []
    trajectories = []
    for i in range(n):
        # Get the unpadded length for this example
        mask = attention_mask[i].bool()
        seq_len = mask.sum().item()

        # Extract logits and input_ids for this example (unpadded)
        # Note: lmprobe may left-pad or right-pad. We use the mask to
        # find valid positions.
        example_logits = logits_batch[i][mask]  # (seq_len, vocab)
        example_ids = all_input_ids[i]

        # Handle BOS duplication: lmprobe's tokenizer may prepend an extra
        # BOS token when the chat template already includes one. If
        # extracted is exactly 1 longer and starts with BOS, strip it.
        if seq_len == len(example_ids) + 1:
            # Strip the leading duplicate BOS from logits
            example_logits = example_logits[1:]
            seq_len -= 1
            logger.debug(f"Example {i}: stripped duplicate BOS (extracted had +1 token)")
        elif len(example_ids) != seq_len:
            logger.warning(
                f"Example {i}: token count mismatch: "
                f"template={len(example_ids)} vs extracted={seq_len}. "
                f"Skipping."
            )
            continue

        ll_result = compute_log_likelihood(
            example_logits, example_ids, all_boundaries[i],
        )
        ll_result["index"] = int(ds_subset[i]["index"])
        ll_result["deceptive"] = bool(ds_subset[i]["deceptive"])
        results.append(ll_result)

        # Cache trajectory
        traj = compute_trajectory(
            example_logits, example_ids, tokenizer,
            metadata={
                "index": int(ds_subset[i]["index"]),
                "deceptive": bool(ds_subset[i]["deceptive"]),
                "assistant_boundaries": all_boundaries[i],
                "turn_boundaries": all_turn_boundaries[i],
            },
        )
        trajectories.append(traj)

        if (i + 1) % 10 == 0 or i == 0:
            logger.info(
                f"  [{i+1}/{n}] mean_ll={ll_result['mean_ll']:.3f} "
                f"ppl={ll_result['mean_ppl']:.2f} "
                f"n_tokens={ll_result['n_tokens']}"
            )

    # Aggregate statistics
    if results:
        all_ppl = [r["mean_ppl"] for r in results]
        all_ll = [r["mean_ll"] for r in results]
        all_n = [r["n_tokens"] for r in results]

        summary = {
            "model": model_name,
            "dataset": args.dataset,
            "dataset_model_key": dataset_model_key,
            "n_examples": len(results),
            "backend": args.backend,
            "ppl_mean": float(np.mean(all_ppl)),
            "ppl_median": float(np.median(all_ppl)),
            "ppl_std": float(np.std(all_ppl)),
            "ppl_min": float(np.min(all_ppl)),
            "ppl_max": float(np.max(all_ppl)),
            "ll_mean": float(np.mean(all_ll)),
            "ll_std": float(np.std(all_ll)),
            "tokens_per_example_mean": float(np.mean(all_n)),
            "per_example": results,
        }

        logger.info("=" * 60)
        logger.info(f"RESULTS: {args.dataset} / {dataset_model_key}")
        logger.info(f"  Perplexity: {summary['ppl_mean']:.2f} mean, "
                     f"{summary['ppl_median']:.2f} median "
                     f"(std={summary['ppl_std']:.2f})")
        logger.info(f"  Log-likelihood: {summary['ll_mean']:.3f} mean "
                     f"(std={summary['ll_std']:.3f})")
        logger.info(f"  Tokens/example: {summary['tokens_per_example_mean']:.0f} mean")
        logger.info("=" * 60)

        # Save results
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)
        if args.output:
            output_path = Path(args.output)
        else:
            model_short = dataset_model_key.replace("/", "_")
            output_path = output_dir / f"affirmation_{args.dataset}_{model_short}.json"

        # Strip per-token lists for the saved file (they're large)
        save_results = []
        for r in results:
            save_r = {k: v for k, v in r.items() if k != "per_token_ll"}
            save_results.append(save_r)
        summary["per_example"] = save_results

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Results saved to {output_path}")

        # Save trajectory cache
        traj_path = output_path.with_suffix(".trajectories.pt")
        save_trajectories(trajectories, traj_path)
        logger.info(f"Trajectories cached to {traj_path} ({len(trajectories)} examples)")
    else:
        logger.error("No results computed!")


if __name__ == "__main__":
    main()
