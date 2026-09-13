"""Sample synthetic client profiles, balanced over the willingness x capacity grid.

Why the grid and not the blended tier: the study is about what happens when a
client's willingness fights their capacity, and balancing a weighted SUM cannot
balance its parts. In the June pilot, which balanced the blended tier, each conflict
corner held ~3.5% of profiles -- 93% of the data sat on cases that do not test the
question. Filling the grid puts an equal share in every cell, conflict corners
included.

Usage: python src/profiles.py [--config config.yaml] [--out data/profiles.jsonl]
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import yaml

from rubric import BINS, factor_cell, sample_profile


def generate_cell_balanced_profiles(rubric, n_total, seed, bins=BINS):
    """Reject-sample until every (willingness_bin, capacity_bin) cell is equally full.

    Returns (profiles, n_drawn). Cells are ~6-20% likely each under the natural field
    distribution, so this converges in a few tens of thousands of draws.
    """
    cells = [(w, c) for w in bins for c in bins]
    per_cell = n_total // len(cells)
    rng = random.Random(seed)
    counts, profiles, drawn = Counter(), [], 0
    while any(counts[c] < per_cell for c in cells):
        profile = sample_profile(rubric, rng, profile_id=f"p{drawn:05d}")
        drawn += 1
        cell = factor_cell(profile, rubric)
        if counts[cell] < per_cell:
            counts[cell] += 1
            profiles.append(profile)
    return profiles, drawn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="data/profiles.jsonl")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    rubric = config["rubric"]
    profiles, drawn = generate_cell_balanced_profiles(
        rubric, config["data"]["n_profiles"], config["seed"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for profile in profiles:
            f.write(json.dumps(profile) + "\n")

    print(f"Wrote {len(profiles)} profiles to {out_path} ({drawn} sampled)")
    cells = Counter(p["factor_cell"] for p in profiles)
    print("  willingness/capacity cells:")
    for cell in sorted(cells):
        print(f"    {cell:12s} {cells[cell]}")
    tiers = Counter(p["tier"] for p in profiles)
    print(f"  derived tier mix: {dict(tiers)}")
    print(f"  conflicted: {sum(p['conflicted'] for p in profiles)} "
          f"({sum(p['conflicted'] for p in profiles)/len(profiles):.1%})")


if __name__ == "__main__":
    main()
