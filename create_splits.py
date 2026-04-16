"""Create and commit train/dev/held-out splits for all LB datasets.

Run once. Commit the output. Never modify.

Split strategy:
- ID: 60/20/20 train/dev/held-out (probe fitting / layer+variant selection / headline eval)
- HP-C, HP-KR, IT: 50/50 dev/held-out (cross-dataset eval sets, no training)
- GS, CG: 100% held-out (never touched, single frozen-probe evaluation)

All splits are stratified by label, per-model.
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
DATASETS = {
    "instructed-deception": {"split": "60/20/20", "role": "training"},
    "harm-pressure-choice": {"split": "50/50", "role": "cross-dataset-eval"},
    "harm-pressure-knowledge-report": {"split": "50/50", "role": "cross-dataset-eval"},
    "insider-trading": {"split": "50/50", "role": "cross-dataset-eval"},
    # GS and CG: entire dataset is held-out
}

output_dir = Path("splits")
output_dir.mkdir(exist_ok=True)

rng = np.random.RandomState(SEED)
manifest = {"seed": SEED, "datasets": {}}

for ds_name, config in DATASETS.items():
    loaded = load_from_disk(f"data/{ds_name}")
    ds = loaded["test"] if isinstance(loaded, dict) else loaded

    manifest["datasets"][ds_name] = {
        "split_ratio": config["split"],
        "role": config["role"],
        "models": {},
    }

    for model_key in MODELS:
        ds_filtered = ds.filter(lambda x, m=model_key: x["model"] == m)
        dec_indices = [i for i, x in enumerate(ds_filtered) if x["deceptive"]]
        truth_indices = [i for i, x in enumerate(ds_filtered) if not x["deceptive"]]

        if len(dec_indices) < 10 or len(truth_indices) < 10:
            continue

        dec_perm = rng.permutation(dec_indices).tolist()
        truth_perm = rng.permutation(truth_indices).tolist()

        if config["split"] == "60/20/20":
            n_dec = len(dec_perm)
            n_truth = len(truth_perm)
            d_train = int(n_dec * 0.6)
            d_dev = int(n_dec * 0.2)
            t_train = int(n_truth * 0.6)
            t_dev = int(n_truth * 0.2)

            split = {
                "train": {
                    "deceptive": dec_perm[:d_train],
                    "truthful": truth_perm[:t_train],
                },
                "dev": {
                    "deceptive": dec_perm[d_train:d_train + d_dev],
                    "truthful": truth_perm[t_train:t_train + t_dev],
                },
                "held_out": {
                    "deceptive": dec_perm[d_train + d_dev:],
                    "truthful": truth_perm[t_train + t_dev:],
                },
            }
        else:  # 50/50
            d_half = len(dec_perm) // 2
            t_half = len(truth_perm) // 2

            split = {
                "dev": {
                    "deceptive": dec_perm[:d_half],
                    "truthful": truth_perm[:t_half],
                },
                "held_out": {
                    "deceptive": dec_perm[d_half:],
                    "truthful": truth_perm[t_half:],
                },
            }

        counts = {k: {kk: len(vv) for kk, vv in v.items()} for k, v in split.items()}
        manifest["datasets"][ds_name]["models"][model_key] = counts

        # Save split indices
        split_file = output_dir / f"{ds_name}_{model_key}.json"
        with open(split_file, "w") as f:
            json.dump(split, f, indent=2)

        print(f"{ds_name:40s} {model_key:40s} {counts}")

# Save manifest
with open(output_dir / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nAll splits saved to {output_dir}/")
print(f"Seed: {SEED}")
print("GS and CG are fully held-out (no split file needed).")
