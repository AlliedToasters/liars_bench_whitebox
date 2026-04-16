"""Lightweight per-token log-prob trajectory cache.

Stores the compressed signal from every forward pass: per-token log-probs,
token IDs, and decoded token text. ~1 KB per trajectory, trivial to store
for entire datasets.

Usage (save):
    from trajectory_cache import compute_trajectory, save_trajectories
    traj = compute_trajectory(logits, token_ids, tokenizer, metadata={...})
    save_trajectories([traj, ...], "results/trajectories_alpaca_mistral.pt")

Usage (load):
    from trajectory_cache import load_trajectories
    trajs = load_trajectories("results/trajectories_alpaca_mistral.pt")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def compute_trajectory(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    tokenizer,
    metadata: dict | None = None,
) -> dict:
    """Compute per-token log-prob trajectory from a single forward pass.

    Parameters
    ----------
    logits : (seq_len, vocab_size)
    token_ids : (seq_len,)
    tokenizer : PreTrainedTokenizer
    metadata : optional dict of extra fields to store (e.g. index, deceptive)

    Returns
    -------
    dict with keys: token_ids, per_token_lp, token_texts, joint_ll, mean_ll,
    min_ll, perplexity, n_tokens, plus any metadata fields.
    """
    shift_logits = logits[:-1].float()
    shift_labels = token_ids[1:]

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    per_token_lp = log_probs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1),
    ).squeeze(-1).cpu().numpy().astype(np.float32)

    n_tokens = len(per_token_lp)
    joint_ll = float(per_token_lp.sum())
    mean_ll = float(per_token_lp.mean())
    min_ll = float(per_token_lp.min())
    perplexity = float(np.exp(-mean_ll))

    token_texts = [tokenizer.decode([tid]) for tid in token_ids.cpu().tolist()]

    traj = {
        "token_ids": token_ids.cpu().numpy().astype(np.int32),
        "per_token_lp": per_token_lp,
        "token_texts": token_texts,
        "joint_ll": joint_ll,
        "mean_ll": mean_ll,
        "min_ll": min_ll,
        "perplexity": perplexity,
        "n_tokens": n_tokens,
    }
    if metadata:
        traj.update(metadata)
    return traj


def save_trajectories(trajectories: list[dict], path: str | Path) -> None:
    """Save a list of trajectory dicts to a .pt file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trajectories, path)


def load_trajectories(path: str | Path) -> list[dict]:
    """Load trajectories from a .pt file."""
    return torch.load(path, weights_only=False)
