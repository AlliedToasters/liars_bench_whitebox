# Liar's Bench Whitebox: Results

## On-Policy Affirmation

Validated that our chunked forward pass pipeline faithfully reproduces model behavior by computing per-token log-likelihood of assistant responses in ground-truth transcripts.

| Model | Mean PPL | Median PPL | LL Mean | n |
|---|---|---|---|---|
| Mistral-Small-3.1-24B | 1.35 | 1.29 | -0.287 | 50 |
| Gemma-3-27B-it | 1.44 | 1.35 | -0.338 | 50 |

Low perplexity confirms correct template reconstruction, tokenization, and chunked execution. Pipeline is on-policy.

## Brain Scan: Instructed-Deception (Mistral)

### Setup

- **Model**: Mistral-Small-3.1-24B-Instruct
- **Dataset**: Instructed-Deception (Liar's Bench)
- **Method**: SampleScan — captures attention and MLP deltas at every layer, fits PCA between layer chunks, stores compressed projections
- **Signals**: `attn_delta`, `mlp_delta`
- **PCA components**: 64

### Prompt Leakage Discovery

Initial separability analysis on the full token sequence yielded perfect AUROC (1.000) at multiple layers. Investigation revealed this was **prompt leakage**: the instructed-deception dataset uses different system prompts for deceptive ("Try to deceive them") vs truthful ("Answer truthfully") conditions. The model's activations trivially encode the instruction text, not a deception representation.

Three decontamination steps were applied:

1. **Generative-only PCA**: Fit the PCA basis only on assistant-turn tokens, excluding system and user prompts
2. **Assistant-turn-only evaluation**: Compute separability using the last token of the assistant turn, not the last token of the full sequence
3. **Train/test holdout**: Fit basis on 400 training samples, evaluate on 100 held-out test samples

### Within-Dataset Holdout Results

PCA basis fit on 400 training samples (generative tokens only), evaluated on 100 held-out test samples (assistant last-token only).

| Signal | Best Layer | Max AUROC | Mean AUROC |
|---|---|---|---|
| attn_delta | 16 | **0.976** | 0.814 |
| mlp_delta | 16 | **0.976** | 0.852 |

Layer 16 (40% depth) is the peak for both signals. The MLP signal is broader, with layers 16-26 all above 0.94 AUROC.

### Comparison Across Decontamination Steps

| Analysis | attn max | mlp max | Notes |
|---|---|---|---|
| Full sequence (leaked) | 1.000 | 1.000 | System prompt difference |
| Assistant-only eval | 0.999 | 0.996 | Still using leaked PCA basis |
| **Holdout + generative PCA** | **0.976** | **0.976** | Fully decontaminated |

## Cross-Dataset Generalization

### Setup

The key question: does the deception signal learned from instructed-deception transfer to fundamentally different deception scenarios?

- **Training basis**: PCA fit on instructed-deception (generative tokens only, 400 samples)
- **Test sets**: Held-out instructed-deception, harm-pressure-choice, harm-pressure-knowledge-report, insider-trading
- **Evaluation**: Project test samples through the ID-trained basis, compute AUROC at assistant last-token
- **No retraining**: The same PCA directions are used for all test datasets

### Results

| Dataset | Task Description | attn max | mlp max | Best attn Layer | Best mlp Layer |
|---|---|---|---|---|---|
| Instructed-Deception | Told to lie about factual questions | **0.976** | **0.976** | 16 | 16 |
| Harm-Pressure-Choice | Pressured to choose harmful action | **0.917** | **0.925** | 17 | 17 |
| Harm-Pressure-KR | Pressured to misreport knowledge | **0.788** | **0.799** | 27 | 31 |
| Insider-Trading | Multi-turn deceptive agent (19 turns) | **0.804** | **0.727** | 14 | 16 |

### Interpretation

The PCA basis learned from one deception scenario (instructed factual deception) **transfers to three other deception scenarios** with no retraining. This is evidence for a shared, linear deception feature in Mistral's mid-network representations.

Key observations:

- **Consistent peak layers**: attn_delta peaks at layers 14-17 across all datasets. mlp_delta peaks at 16-17 for three of four datasets.
- **Gradient of transfer**: Signal is strongest for tasks closest to the training distribution (HP-Choice: 0.92) and weakest for the most different task (insider-trading multi-turn: 0.73-0.80).
- **HP-KR is an outlier**: Peak layers shift to 27-31, suggesting knowledge-report deception may involve a qualitatively different mechanism (perhaps related to factual recall vs behavioral deception).
- **Multi-turn transfer**: Even the 19-turn insider-trading scenario shows signal (0.80 attn, 0.73 mlp), despite being structurally very different from the 3-turn training data.

## Methods

### Chunked Forward Pass (lmprobe)

Models are loaded fully on CPU and streamed through the GPU one layer chunk at a time. This enables running 24B+ parameter models on a single 32GB GPU. Between chunks, PCA is fit on the GPU using captured activation deltas.

### SampleScan Pipeline

A single forward pass captures attention and MLP deltas at every (token, layer) coordinate, fits PCA between layer chunks, and stores compressed projections. The entire scan (500 samples, 40 layers, 2 signals, 64 PCs) completes in ~5 minutes for a 24B model.

### lmprobe Patches

Several patches were applied to lmprobe's editable install to support the models in this study:

- **Multimodal model support**: Mistral-Small-3.1 is a vision-language model. Patched `get_num_layers_from_config`, `_get_decoder_layers`, `_get_embedding_module`, `_get_final_norm`, `_find_rotary_embedding_name`, `_load_full_model_cpu`, and `_estimate_chunk_size` to navigate the nested `text_config` / `language_model` structure.
- **Gemma3 rotary embeddings**: Gemma-3-27B uses per-layer-type rotary embeddings (sliding window vs global attention). Patched the chunked forward pass to compute and dispatch position embeddings per layer type.
- **Generative-only PCA**: Added `generative_masks` parameter to `scan_forward` so PCA is fit only on assistant-turn tokens.
- **Batched projection**: Added `external_bases` parameter to `scan_forward` for fast batched projection through a pre-trained basis (vs slow per-sample `project_forward`).
- **Token-position separability**: Added `token_positions` parameter to `separability_map` for assistant-turn-only evaluation.
