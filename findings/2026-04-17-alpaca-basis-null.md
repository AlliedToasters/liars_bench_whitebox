# Finding 2026-04-17: Alpaca-basis does not fix cross-dataset overfitting

**Model:** Mistral-Small-3.1-24B-Instruct
**Status:** Conclusive null result on methodology; conclusive positive result on infrastructure.

## TL;DR

We pivoted to an **unsupervised PCA basis fit on Alpaca-train**, hoping to remove dataset-specific contamination from the basis and thus recover cross-dataset transfer. End-to-end on Mistral shows this did **not** work: within-domain is near-perfect, cross-dataset is at or below the DoM baseline, and on insider-trading our probe is literally anti-correlated. The bottleneck is downstream of the basis — it is in the **probe fit**, not in the feature space.

As a side effect, the pivot forced us to build a GPU-accelerated, memory-conscious index-and-project pipeline. That pipeline works, is tractable on a single 32 GB GPU with chunked/disk-offload backends, and will let us iterate on probe recipes without refitting anything expensive.

## What we tried

Per the pre-registered 2026-04-17 amendment to `EXPERIMENT.md` and `METHODS.md`:

1. **Basis fit.** PCA on `attn_delta` and `mlp_delta` at every layer, fit only on assistant tokens over 1000 Alpaca-train prompts (seed=2026). `n_components=64` per (layer, signal). Fit on GPU via cuml (jacobi SVD) with chunked layer streaming.
2. **Cache all PCs.** Project the entire LB-ID train + LB-\* dev + LB-\* held-out pools into the frozen Alpaca basis, for 3 reducers × 40 layers × 2 signals = 240 feature slots per sample.
3. **Probe fit.** L2 logistic regression (C=1.0) per (reducer, layer, signal) on LB-ID train.
4. **Probe selection.** Pick the single (reducer, layer, signal) with the highest dev AUROC on LB-ID.
5. **Threshold calibration.** 1% FPR on Alpaca-val (non-deceptive).
6. **Eval.** Held-out of all four LB-\* datasets at the frozen τ.

## Results

Selected probe: `reducer=last_token, layer=16, signal=attn_delta`, dev AUROC 0.9864, τ=2.555.
DoM baseline winner: same layer/signal, `reducer=mean_asst`, dev AUROC 0.9623.

| Dataset                            | Split     | Ours (AUROC) | DoM (AUROC) | Ours (BA@1%FPR) | DoM (BA@1%FPR) |
|------------------------------------|-----------|-------------:|------------:|----------------:|---------------:|
| instructed-deception               | held_out  |   **0.990**  |       0.959 |       **0.809** |          0.698 |
| harm-pressure-choice               | held_out  |       0.586  |   **0.608** |           0.500 |          0.500 |
| harm-pressure-knowledge-report     | held_out  |   **0.698**  |       0.652 |           0.500 |          0.488 |
| insider-trading                    | held_out  |       0.416  |   **0.660** |           0.500 |          0.508 |
| **mean (held_out)**                |           |       0.673  |   **0.720** |           0.577 |          0.548 |

(Probe bal-acc at the Alpaca-val 1% FPR threshold collapses to 0.5 on out-of-domain splits because the probe score distribution shifts; almost no held-out sample crosses τ. That is a calibration issue on top of the ranking issue.)

Paper upper-bound reference: best reported cross-dataset AUROC in the Liar's Bench paper is ~0.91. We are a long way from that.

## Why this is a null result, not a tuning problem

Three signals say the miss is structural, not incidental:

1. **Insider-trading is anti-correlated (AUROC 0.416).** A 0.5 score means "no signal"; a 0.416 means the probe direction is literally backwards on this dataset. The LR fit on LB-ID PCs found a direction that is *inversely* associated with deception on IT. An unsupervised basis can't fix this — whatever LR latches onto inside those 64 PCs is an ID-specific feature that happens to anti-correlate with the IT deception direction.

2. **DoM beats LR on three of four held-out datasets.** DoM is a single direction fit on the same data; LR is a 64-d linear classifier fit on the same data. If the basis were the problem, DoM should suffer the same collapse. It doesn't. That points at probe capacity / fit regularization, not at the feature space.

3. **Within-domain is perfect, out-of-domain is chance.** ID held-out AUROC 0.990, three others mean ~0.57. Classic overfitting pattern — but now the basis is unsupervised, so we've removed one candidate explanation for it.

## What the pivot did give us

The methodology pivot forced a pipeline rebuild, and that rebuild is valuable independent of the probe result:

- **`fit_basis.py`** — unsupervised, reusable PCA basis per model, written to `scans/alpaca_<model>/`. Fit time on Mistral: **8 min 29 s** (10 chunks × ~50 s) on a single GPU.
- **`experiment_pipeline.py` `cache_all`** — projects any dataset into the frozen basis, writes compact `(N, 240 slots, 64)` PC tensors. Uses memory-aware sub-bucketing so large-sequence datasets (insider-trading, max_len 2782) don't OOM. Total cache time for LB-ID + 4 LB-\* splits on Mistral: **~54 min**.
- **Evaluate** — probe fit + selection + calibration + held-out scoring from cached PCs runs in **~15 s** on CPU.

Most importantly: the cached PC index for Mistral now lives on disk as `results/cached_vecs_<dataset>_<split>_<model>.npz`. Anything we want to try on the probe side — regularization sweeps, L1, per-layer averaging, cross-dataset training pools, nonlinear probes, union-of-directions, CCS-style methods — is now a CPU-only experiment that runs in seconds against the cached tensors. The expensive GPU step only re-runs if we change the basis or the sample pool.

## Infra notes worth keeping

Things that were hard-won and should not be forgotten:

- **cuml + jacobi SVD for PCA at this scale is the right call.** sklearn PCA on the flattened (1000 × 40 layers × 2 signals × ~300 tokens × 5120 hidden) corpus was prohibitive on CPU. cuml with `svd_solver="jacobi"` fits in seconds per (layer, signal).
- **Cross-chunk memory constancy requires pre-allocated output arrays,** not Python lists that grow through the run. Also: progressive capture release (free `per_layer_captures[layer][signal]` immediately after its PCA fit) is necessary to avoid monotonic growth across chunks.
- **The `batch_hidden_states: list[torch.Tensor]` in lmprobe is the current CPU ceiling** for projection sweeps. For Mistral/IT (max_len 2782, 9260 samples) this exceeded CPU alongside the 48 GB model. Current workaround: byte-budget sub-bucketing in `cache_all` with `MAX_HIDDEN_GB=15.0`, oversized groups split into `.split.part{p}of{N}.npz` and stitched at the end. Long-term fix is a memmap refactor in lmprobe (issue #272, PR #271 in review).
- **Disk-offload path does not support multimodal configs (Mistral-3).** `AutoModelForCausalLM.from_config(Mistral3Config)` fails. For Mistral we must use chunked backend; for Llama-70B / Qwen-72B disk_offload remains the plan.
- **glibc fragmentation matters at this scale.** `MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=131072` materially reduced RSS after sub-bucketing.

## Open questions for the brainstorm

If the basis isn't the problem and the probe is, the design space narrows:

1. **Regularization.** Is LR with stronger L2 (C ≪ 1) or L1 enough to kill the ID-specific direction that's hurting IT? Cheap experiment, minutes on cached PCs.
2. **Training pool.** We train only on LB-ID. What if we train on a union of LB datasets, holding out one at a time (leave-one-domain-out)? Are there still-unused datasets (e.g. any of the seven in the paper we haven't touched) that could widen the pool without leakage into our held-outs?
3. **Single-direction methods.** DoM generalizes better than our LR. Does PCA-on-class-means, CCS, or a mean-of-means-across-datasets direction beat DoM? We have the cached PCs to try any of these in seconds.
4. **Per-layer ensembling vs single-layer selection.** We pick one (reducer, layer, signal). Paper upper-bound likely averages or votes across layers. Does our 0.673 rise toward 0.91 if we combine?
5. **Is this Mistral-specific?** The pipeline is cheap enough now to rerun on Gemma/Llama/Qwen. If the IT-anti-correlation pattern is Mistral-only, that reshapes the story.

## Where the pre-registered plan stands

`EXPERIMENT.md` still commits to Alpaca-basis + LR + 1% FPR. This finding is within the freeze: Mistral was the smoke-test target, the probe recipe didn't work, and we now decide what to change before running Gemma/Llama/Qwen. Any change to the probe recipe should be documented as another amendment with a dated timestamp before we run on more models, so we don't back-fit the methodology to the held-out numbers.
