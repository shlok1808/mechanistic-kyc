"""S1B: targeted EDGE-CASE profile generator for stress-testing S2 (and later S3).

Unlike S1 (a balanced random draw), this builds ~60 profiles that deliberately hit the
hard corners of the rubric:

  - borderline tiers   : risk_score within +/-2 of a cutoff (38-42, 68-72)
  - contradictions     : stated goal vs drawdown reaction point opposite ways
  - extreme demographics: age in {23,75} x horizon in {1,35}
  - pure anchors       : fully unambiguous conservative / aggressive (score ~0 / ~100)
  - full decoy coverage: every occupation/city/hobby appears (decoy-leakage check)

Output: data/profiles_edge.jsonl  (same schema as S1's profiles.jsonl + an `edge_bucket`).

Run S2 on it (separate output dir, no synthetic pairs):
    python src/vignettes.py --profiles data/profiles_edge.jsonl \
        --out-dir data/vignettes_edge --n-pairs 0
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import yaml

from rubric import (
    DECOY_POOLS, NAMES, is_contradictory, risk_score, risk_tier, sample_rubric_fields,
)

# Fully unambiguous extremes (score 0 / 100 under the finalized rubric).
ALL_CONSERVATIVE = {
    "past_drawdown_reaction": "sold_everything", "horizon_years": 1,
    "investing_experience": "none", "income_stability": "precarious",
    "stated_goal": "capital_preservation", "emergency_fund_months": 0,
    "dependents": 4, "age": 75,
}
ALL_AGGRESSIVE = {
    "past_drawdown_reaction": "bought_more", "horizon_years": 35,
    "investing_experience": "extensive", "income_stability": "stable",
    "stated_goal": "maximum_growth", "emergency_fund_months": 24,
    "dependents": 0, "age": 23,
}
CONTRA_COMBOS = [
    {"stated_goal": "maximum_growth", "past_drawdown_reaction": "sold_everything"},
    {"stated_goal": "capital_preservation", "past_drawdown_reaction": "bought_more"},
]


def _borderline(rubric, rng, windows, n):
    """Reject-sample full random draws whose score lands in a +/-2 cutoff window."""
    out = []
    while len(out) < n:
        fields = sample_rubric_fields(rubric, rng)
        s = risk_score(fields, rubric)
        if any(lo <= s <= hi for lo, hi in windows):
            out.append((fields, "borderline"))
    return out


def build_edge_rubric_fields(rubric, rng):
    """Return a list of (rubric_fields, bucket) covering all edge buckets (~60)."""
    items = []

    # Borderline tiers: 10 near 40, 10 near 70.
    items += _borderline(rubric, rng, [(38, 42)], 10)
    items += _borderline(rubric, rng, [(68, 72)], 10)

    # Deliberate contradictions: 6 per combo, other fields random.
    for combo in CONTRA_COMBOS:
        for _ in range(6):
            fields = sample_rubric_fields(rubric, rng)
            fields.update(combo)
            items.append((fields, "contradiction"))

    # Extreme demographics: each age x horizon corner, 2 draws.
    for age in (23, 75):
        for horizon in (1, 35):
            for _ in range(2):
                fields = sample_rubric_fields(rubric, rng)
                fields.update({"age": age, "horizon_years": horizon})
                items.append((fields, "extreme_demo"))

    # Pure anchors: 10 each, rubric fixed (decoys will vary).
    items += [(dict(ALL_CONSERVATIVE), "anchor_conservative") for _ in range(10)]
    items += [(dict(ALL_AGGRESSIVE), "anchor_aggressive") for _ in range(10)]

    return items


def assign_decoys(items, rng):
    """Round-robin decoys so every occupation/city/hobby is covered; finalize profiles."""
    occ, city, hob = DECOY_POOLS["occupation"], DECOY_POOLS["city"], DECOY_POOLS["hobbies"]
    nw = DECOY_POOLS["net_worth_band"]
    order = list(range(len(items)))
    rng.shuffle(order)  # decorrelate bucket from decoy index
    profiles = []
    for pos, idx in enumerate(order):
        fields, bucket = items[idx]
        name, gender = NAMES[pos % len(NAMES)]
        profile = {"profile_id": f"e{pos:05d}", **fields,
                   "net_worth_band": nw[pos % len(nw)],
                   "occupation": occ[pos % len(occ)],
                   "hobbies": hob[pos % len(hob)],
                   "city": city[pos % len(city)],
                   "name": name, "gender_hint": gender,
                   "edge_bucket": bucket}
        profiles.append(profile)
    return profiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="data/profiles_edge.jsonl")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rubric = cfg["rubric"]
    rng = random.Random(cfg["seed"])

    items = build_edge_rubric_fields(rubric, rng)
    profiles = assign_decoys(items, rng)
    for p in profiles:
        s = risk_score(p, rubric)
        p["risk_score"] = round(s, 2)
        p["tier"] = risk_tier(s, rubric)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for p in profiles:
            f.write(json.dumps(p) + "\n")

    buckets = Counter(p["edge_bucket"] for p in profiles)
    tiers = Counter(p["tier"] for p in profiles)
    contra = sum(is_contradictory(p, rubric) for p in profiles)
    decoy_cov = {k: len({p[k] for p in profiles}) for k in ("occupation", "city", "hobbies")}
    print(f"Wrote {len(profiles)} edge profiles to {out_path}")
    print(f"  buckets: {dict(buckets)}")
    print(f"  tiers:   {dict(tiers)}")
    print(f"  contradictory (rubric-detected): {contra}")
    print(f"  decoy coverage (distinct / 10):  {decoy_cov}")


if __name__ == "__main__":
    main()
