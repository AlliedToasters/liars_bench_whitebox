# Liar's Bench Whitebox Repo
This is research project attempting to recover the on-policy latents for the liar's bench dataset. [results](https://alliedtoasters.github.io/liars_bench_whitebox/)

## Datasets

| Dataset | Abbreviation | Fine-tuned? |
|---|---|---|
| Alpaca (control) | Alpaca | No |
| Convincing-Game | CG | No |
| Harm-Pressure-Choice | HP-C | No |
| Harm-Pressure-Knowledge-Report | HP-KR | No |
| Insider-Trading | IT | No |
| Instructed-Deception | ID | No |
| Gender-Secret | GS | Yes |
| Soft-Trigger | ST | Yes |

> **Note:** We are initially prioritizing the non-fine-tuned (vanilla instruct) datasets, as the open weights for these models are more readily accessible.

## Strategy

We use the [lmprobe](https://alliedtoasters.github.io/lmprobe/) library to extract on-policy activations from the models on a single GPU. This is achievable via forward-pass chunking, where model layers are streamed through the GPU rather than loaded all at once. See the [lmprobe large models guide](https://alliedtoasters.github.io/lmprobe/guides/large-models/) for details on the chunked and disk offload backends.

## Liar's Bench Citation

If you use this work, please also cite the [original Liars' Bench paper](https://arxiv.org/pdf/2511.16035):

```bibtex
@article{kretschmar2025liarsbench,
  title={Liars' Bench: Evaluating Lie Detectors for Language Models},
  author={Kretschmar, Kieron and Laurito, Walter and Maiya, Sharan and Marks, Samuel},
  journal={arXiv preprint arXiv:2511.16035},
  year={2025}
}
```