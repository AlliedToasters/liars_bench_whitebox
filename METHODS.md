# Methods Reference

Core pipeline for whitebox deception detection using lmprobe's SampleScan.

## Overview

Two operations:

1. **Fit**: Given labeled prompts `(X, Y)`, run a forward pass capturing activation deltas at every `(token, layer)`, fit PCA on the generative tokens to get a reusable basis.
2. **Evaluate**: Given a fitted basis and new labeled prompts `(X', Y')`, project through the basis and compute classification metrics at the assistant's last token.

The basis is fit **once** on a training corpus and applied to any number of test prompts with no retraining. This is the core claim: a single unsupervised PCA basis transfers across deception scenarios.

## Step 1: Fit a Basis

### Inputs
- `prompts: list[str]` — tokenized chat transcripts (system + user + assistant turns)
- `labels: list[int]` — 1 = deceptive, 0 = truthful
- `generative_masks: list[np.ndarray]` — boolean mask per prompt, True for assistant-turn tokens

### Why generative masks matter
Without masking, PCA captures the dominant variance in the full sequence — which includes the system prompt. If deceptive and truthful conditions use different system prompts (as in instructed-deception), the basis trivially encodes the instruction text, not a deception representation. Generative-only PCA forces the basis to capture variance in the model's own output activations.

### Code
```python
from lmprobe import SampleScan
import torch

scan = SampleScan.run(
    prompts=train_prompts,        # list of chat-templated strings
    labels=train_labels,          # list of 0/1
    model_name="meta-llama/Llama-3.3-70B-Instruct",
    scan_dir="scans/my_basis",    # where to store the basis
    signals=["attn_delta", "mlp_delta"],
    n_components=64,              # PCA components per layer per signal
    device="cuda:0",
    dtype=torch.bfloat16,
    generative_masks=train_masks, # only fit PCA on assistant tokens
    backend="disk_offload",       # or "chunked" for models that fit in CPU RAM
)
```

### What gets stored
```
scans/my_basis/
  metadata.json                           # model info, n_layers, n_components
  channels/0_global/basis_attn_delta.npy  # [n_layers, hidden_dim, n_components]
  channels/0_global/basis_mlp_delta.npy   # [n_layers, hidden_dim, n_components]
  projections/values.npy                  # training set projections (sparse)
  projections/coords.parquet              # (sample_id, layer, token_pos, signal)
  samples/samples.parquet                 # prompt text, labels, seq lengths
```

The basis files (`basis_*.npy`) are the reusable artifact. Everything else is for analysis of the training set.

## Step 2: Evaluate on a Test Set

### Inputs
- A fitted `SampleScan` object (or loaded from disk)
- `test_prompts: list[str]` — new chat-templated strings
- `test_labels: list[int]` — ground truth
- `asst_last_positions: list[int]` — token index of the last assistant token per prompt

### Projection
```python
scan = SampleScan("scans/my_basis")
projections, coords, token_ids, seq_lengths = scan.batch_project(
    test_prompts, batch_size=4
)
```

`batch_project` runs a single forward pass per layer chunk for ALL test samples, projecting activations onto the stored PCA basis. Returns sparse projections at every `(sample, layer, token, signal)`.

### Extract evaluation vectors
We evaluate at the **assistant's last token** — the position where the model has processed its full response and the deception signal is strongest.

```python
# Extract [n_samples, n_layers, n_signals, n_components] array
test_vecs = np.zeros((n_test, n_layers, n_signals, k), dtype=np.float32)

coord_sample = np.array(coords["sample_id"])
coord_layer = np.array(coords["layer"])
coord_tok = np.array(coords["token_pos"])
coord_sig = np.array(coords["signal"])

for i in range(n_test):
    tok_pos = min(asst_last[i], seq_lengths[i] - 1)
    for si, sig in enumerate(scan.signals):
        mask = (coord_sample == i) & (coord_tok == tok_pos) & (coord_sig == sig)
        for row_idx in np.where(mask)[0]:
            test_vecs[i, coord_layer[row_idx], si, :] = projections[row_idx, 0, :]
```

### Compute metrics
For each `(layer, signal)` pair, find the best PC direction (max AUROC), then compute:

```python
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score

# Best PC selection
auroc = roc_auc_score(labels, pc_vals)
auroc = max(auroc, 1 - auroc)  # orient direction

# Balanced accuracy at 1% FPR (paper's evaluation protocol)
fpr, tpr, thresholds = roc_curve(labels, oriented_scores)
idx = np.searchsorted(fpr, 0.01, side='right') - 1
preds = (oriented_scores >= thresholds[idx]).astype(int)
bal_acc_1pct = balanced_accuracy_score(labels, preds)
```

## Experimental Design

### Within-dataset holdout
- **Train**: 400 samples (200 dec + 200 truth) — fit PCA basis
- **Test**: 100 samples (50 + 50) — project through basis, compute metrics
- Same dataset, disjoint samples, fixed seed=42

### Cross-dataset transfer
- **Basis**: fit on dataset A (e.g. instructed-deception)
- **Test**: project dataset B samples (e.g. insider-trading) through A's basis
- No retraining. Tests whether the deception direction generalizes.

### Cross-fit matrix
- For each available dataset, fit a basis. Evaluate every basis on every dataset.
- Produces an N x N transfer matrix per model.
- Diagonal = within-dataset holdout. Off-diagonal = transfer.

## Key Design Decisions

1. **PCA, not a trained classifier.** The basis is unsupervised — it captures the directions of maximum variance in generative activations. Separability is measured post-hoc. This avoids overfitting a probe to the training distribution.

2. **Activation deltas, not raw activations.** We capture `attn_delta` (residual change from attention) and `mlp_delta` (residual change from MLP) rather than the full hidden state. These isolate what each sublayer *contributes*, not what it receives.

3. **Assistant-last-token evaluation.** The model has processed its full response at this position. Earlier tokens may not yet reflect the deception strategy.

4. **Generative-only PCA fitting.** Prevents the basis from encoding system prompt differences between conditions.

## Comparison with Liar's Bench Paper

The paper evaluates linear probes at **1% FPR** and reports **balanced accuracy** and **AUROC**.

Their results:
- Mean probe: ~59% balanced accuracy on ID, near-chance on cross-dataset
- Upper-bound probe (trained with privileged info on test set): 0.73 balanced accuracy, 0.91 AUROC

Our approach differs in that:
- We use internal activations (whitebox), not behavioral features
- Our basis is unsupervised (PCA) and transfers zero-shot
- We evaluate at the same 1% FPR threshold for direct comparison
