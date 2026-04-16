"""Per-token log-probability trajectory visualization.

Produces a diagnostic plot of per-token log-probabilities across a model's
forward pass on a transcript, for assessing whether a trajectory is on-policy.

Usage (from logits):
    from plot_logprobs import plot_trajectory_logprobs
    plot_trajectory_logprobs(logits, token_ids, tokenizer, "out/my_plot")

Usage (from cached trajectory):
    from plot_logprobs import plot_trajectory_from_cache
    from trajectory_cache import load_trajectories
    trajs = load_trajectories("results/affirmation_alpaca_mistral.trajectories.pt")
    plot_trajectory_from_cache(trajs[0], "out/my_plot")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch


def _build_figure(
    per_token_lp: np.ndarray,
    token_texts: list[str],
    title: str,
    rolling_window: int,
    alert_threshold: float,
    assistant_boundaries: list[tuple[int, int]] | None = None,
    turn_boundaries: list[dict] | None = None,
) -> tuple[go.Figure, dict]:
    """Shared plotting logic for both entry points."""
    n_tokens = len(per_token_lp)
    positions = np.arange(n_tokens)

    joint_ll = float(per_token_lp.sum())
    mean_ll = float(per_token_lp.mean())
    min_ll = float(per_token_lp.min())
    perplexity = float(np.exp(-mean_ll))

    stats = {
        "joint_ll": joint_ll,
        "mean_ll": mean_ll,
        "min_ll": min_ll,
        "perplexity": perplexity,
        "n_tokens": n_tokens,
    }

    # Rolling average
    kernel = np.ones(rolling_window) / rolling_window
    padded = np.pad(per_token_lp, (rolling_window // 2, rolling_window // 2), mode="edge")
    rolling_avg = np.convolve(padded, kernel, mode="valid")[:n_tokens]

    # Hover texts — use predicted token (position i predicts token i+1,
    # so token_texts[i+1] is the predicted token at position i)
    hover_texts = [
        f"pos {i}<br>token: {repr(token_texts[i + 1] if i + 1 < len(token_texts) else '?')}<br>log P: {lp:.3f}"
        for i, lp in enumerate(per_token_lp)
    ]

    # --- Resolve turn boundaries ---
    all_turns = turn_boundaries or (
        [{"role": "assistant", "start": s, "end": e} for s, e in assistant_boundaries]
        if assistant_boundaries else None
    )

    # --- Build generative (assistant) mask for alerts ---
    generative_mask = np.zeros(n_tokens, dtype=bool)
    if all_turns:
        for turn in all_turns:
            if turn["role"] == "assistant":
                s = max(turn["start"] - 1, 0)
                e = min(turn["end"] - 1, n_tokens)
                generative_mask[s:e] = True

    # Alerts only within assistant turns
    alert_mask = (per_token_lp < alert_threshold) & generative_mask
    alert_positions = positions[alert_mask]
    alert_lps = per_token_lp[alert_mask]
    alert_hovers = [hover_texts[i] for i in alert_positions]

    fig = go.Figure()

    # --- Toggleable role shading (scatter polygons for legend control) ---
    role_style = {
        "system": "rgba(200, 200, 200, 0.25)",
        "user": "rgba(173, 216, 230, 0.25)",
        "assistant": "rgba(144, 238, 144, 0.25)",
    }
    if all_turns:
        y_lo = float(per_token_lp.min()) - 2
        y_hi = 2.0
        roles_seen: set[str] = set()
        for turn in all_turns:
            role = turn["role"]
            fill_color = role_style.get(role, "rgba(255, 200, 100, 0.2)")
            s = max(turn["start"] - 1, 0)
            e = min(turn["end"] - 1, n_tokens)
            show_legend = role not in roles_seen
            roles_seen.add(role)
            fig.add_trace(go.Scatter(
                x=[s, s, e, e, s],
                y=[y_lo, y_hi, y_hi, y_lo, y_lo],
                fill="toself",
                fillcolor=fill_color,
                line=dict(width=0),
                name=role,
                legendgroup=role,
                showlegend=show_legend,
                hoverinfo="skip",
            ))

    # --- Data traces ---
    fig.add_trace(go.Scatter(
        x=positions, y=rolling_avg,
        mode="lines",
        line=dict(color="rgba(65, 105, 225, 0.7)", width=2),
        name=f"Rolling avg (w={rolling_window})",
    ))

    fig.add_trace(go.Scatter(
        x=positions, y=per_token_lp,
        mode="markers",
        marker=dict(size=3, color="rgba(100, 100, 100, 0.4)"),
        text=hover_texts,
        hoverinfo="text",
        name="Per-token log P",
    ))

    fig.add_hline(
        y=mean_ll,
        line=dict(color="green", width=1.5, dash="dash"),
        annotation_text=f"mean = {mean_ll:.3f}",
        annotation_position="top left",
    )

    if len(alert_positions) > 0:
        fig.add_trace(go.Scatter(
            x=alert_positions, y=alert_lps,
            mode="markers",
            marker=dict(size=6, color="red", symbol="x"),
            text=alert_hovers,
            hoverinfo="text",
            name=f"Alerts (< {alert_threshold})",
        ))

    subtitle = (
        f"joint LL = {joint_ll:.1f} | "
        f"mean log P = {mean_ll:.3f} | "
        f"min log P = {min_ll:.3f} | "
        f"PPL = {perplexity:.2f} | "
        f"n = {n_tokens}"
    )

    fig.update_layout(
        title=dict(text=f"{title}<br><sup>{subtitle}</sup>"),
        xaxis_title="Token position",
        yaxis_title="log P(t_i | t_{<i})",
        template="plotly_white",
        hovermode="closest",
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        width=1200,
        height=500,
    )

    return fig, stats


def plot_trajectory_logprobs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    tokenizer,
    save_path: str | Path,
    title: str = "Per-Token Log-Probability Trajectory",
    rolling_window: int = 50,
    alert_threshold: float = -10.0,
    assistant_boundaries: list[tuple[int, int]] | None = None,
    turn_boundaries: list[dict] | None = None,
) -> dict:
    """Plot per-token log-probabilities from raw logits. Saves HTML + PNG.

    Parameters
    ----------
    logits : (seq_len, vocab_size)
    token_ids : (seq_len,)
    tokenizer : PreTrainedTokenizer
    save_path : base path (no extension)
    title, rolling_window, alert_threshold : plot options
    assistant_boundaries : list of (start, end) token positions for agent turns
    turn_boundaries : list of dicts with role/start/end for all conversation turns

    Returns
    -------
    dict with summary statistics.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    shift_logits = logits[:-1].float()
    shift_labels = token_ids[1:]
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    per_token_lp = log_probs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1),
    ).squeeze(-1).cpu().numpy()

    token_texts = [tokenizer.decode([tid]) for tid in token_ids.cpu().tolist()]

    fig, stats = _build_figure(
        per_token_lp, token_texts, title, rolling_window, alert_threshold,
        assistant_boundaries=assistant_boundaries,
        turn_boundaries=turn_boundaries,
    )

    fig.write_html(str(save_path.with_suffix(".html")))
    fig.write_image(str(save_path.with_suffix(".png")), scale=2)

    return stats


def plot_trajectory_from_cache(
    trajectory: dict,
    save_path: str | Path,
    title: str | None = None,
    rolling_window: int = 50,
    alert_threshold: float = -10.0,
) -> dict:
    """Plot from a cached trajectory dict (no model needed).

    Parameters
    ----------
    trajectory : dict from trajectory_cache.compute_trajectory()
    save_path : base path (no extension)
    title : plot title (auto-generated from metadata if None)
    rolling_window, alert_threshold : plot options

    Returns
    -------
    dict with summary statistics.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    per_token_lp = trajectory["per_token_lp"]
    token_texts = trajectory["token_texts"]

    if title is None:
        idx = trajectory.get("index", "?")
        dec = trajectory.get("deceptive", "?")
        title = f"Trajectory (index={idx}, deceptive={dec})"

    boundaries = trajectory.get("assistant_boundaries")
    turns = trajectory.get("turn_boundaries")

    fig, stats = _build_figure(
        per_token_lp, token_texts, title, rolling_window, alert_threshold,
        assistant_boundaries=boundaries,
        turn_boundaries=turns,
    )

    fig.write_html(str(save_path.with_suffix(".html")))
    fig.write_image(str(save_path.with_suffix(".png")), scale=2)

    return stats
