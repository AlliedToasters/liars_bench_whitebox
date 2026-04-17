"""Create and commit Alpaca train/val splits per target model.

Run once. Commit the output. Never modify.

Split strategy:
- 1000 train / 1000 val per model (seed=2026).
- train: used for PCA basis fit (scans/alpaca_{model}).
- val: used for 1% FPR threshold calibration.
- Disjoint by construction.

Alpaca is all non-deceptive, so no label stratification is needed.
"""

import json
from pathlib import Path

import numpy as np
from datasets import load_from_disk

SEED = 2026
MODELS = [
    "mistral-small-3.1-24b-instruct",
    "gemma-3-27b-it",
    "llama-v3.3-70b-instruct",
    "qwen-2.5-72b-instruct",
]
N_TRAIN = 1000
N_VAL = 1000

output_dir = Path("splits")
output_dir.mkdir(exist_ok=True)

loaded = load_from_disk("data/alpaca")
ds = loaded["test"] if hasattr(loaded, "keys") else loaded

rng = np.random.RandomState(SEED)
manifest = {"seed": SEED, "n_train": N_TRAIN, "n_val": N_VAL, "models": {}}

for model_key in MODELS:
    ds_filtered = ds.filter(lambda x, m=model_key: x["model"] == m)
    n = len(ds_filtered)
    if n < N_TRAIN + N_VAL:
        raise RuntimeError(
            f"{model_key}: {n} Alpaca samples, need at least {N_TRAIN + N_VAL}"
        )

    perm = rng.permutation(n).tolist()
    split = {
        "train": perm[:N_TRAIN],
        "val": perm[N_TRAIN:N_TRAIN + N_VAL],
    }
    manifest["models"][model_key] = {"train": N_TRAIN, "val": N_VAL, "pool": n}

    split_file = output_dir / f"alpaca_{model_key}.json"
    with open(split_file, "w") as f:
        json.dump(split, f, indent=2)

    print(f"alpaca/{model_key}: {N_TRAIN} train + {N_VAL} val (pool={n})")

with open(output_dir / "alpaca_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nAlpaca splits saved to {output_dir}/")
print(f"Seed: {SEED}")
