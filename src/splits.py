"""Export the probe train / dev / held-out split as three dataset files.

The split itself is `probes.make_splits` (seed from config, 70/10/20 by profile_id), the
same one every probe, baseline and steering script uses. This script only writes it out.

Course naming vs. project naming:
  training               = train  (70%)  fit probes / directions
  test                   = val    (10%)  dev set: picks C, layer, position
  validation (unseen)    = test   (20%)  headline eval, scored once at the chosen cell

Splitting by profile keeps a client's explicit and implicit renderings on the same side.
Counterfactual pairs (pair_id set) are not in any of the three files: they are synthetic
one-field edits used only for patching, and the probes never train or evaluate on them.

Usage: python src/splits.py   ->  data/splits/{training,test,validation,pairs}.jsonl + summary.json
"""
import json
from collections import Counter
from pathlib import Path

import yaml

from probes import make_splits

COURSE_NAME = {"train": "training", "val": "test", "test": "validation"}


def load_vignettes(vdir):
    rows = []
    for name in ("explicit", "implicit", "pairs"):
        with open(Path(vdir) / f"{name}.jsonl") as f:
            rows += [json.loads(line) for line in f]
    return rows


def check_no_leak(by_split):
    seen = {}
    for s, rows in by_split.items():
        for r in rows:
            other = seen.setdefault(r["profile_id"], s)
            if other != s:
                raise AssertionError(f"profile {r['profile_id']} in both {other} and {s}")


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    rows = load_vignettes(cfg["paths"]["vignettes_dir"])
    # same profile_id set as the activation-cache labels, so the permutation matches
    split = make_splits([r["profile_id"] for r in rows], seed=cfg["seed"])

    out = {v: [] for v in COURSE_NAME.values()}
    pairs = []
    for r in rows:
        if r.get("pair_id") is not None:
            pairs.append(r)
        else:
            out[COURSE_NAME[split[r["profile_id"]]]].append({**r, "split": COURSE_NAME[split[r["profile_id"]]]})
    check_no_leak(out)

    odir = Path("data/splits")
    odir.mkdir(parents=True, exist_ok=True)
    for name, rs in {**out, "pairs": pairs}.items():
        with open(odir / f"{name}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")

    n_prof = {k: len({r["profile_id"] for r in v}) for k, v in out.items()}
    total = sum(n_prof.values())
    summary = {
        "seed": cfg["seed"],
        "unit": "profile_id",
        "profiles": n_prof,
        "profile_share": {k: round(v / total, 3) for k, v in n_prof.items()},
        "vignettes": {k: dict(Counter(r["vignette_type"] for r in v)) for k, v in out.items()},
        "tier": {k: dict(Counter(r["tier"] for r in v if r["vignette_type"] == "implicit"))
                 for k, v in out.items()},
        "factor_cell": {k: dict(sorted(Counter(r["factor_cell"] for r in v
                                               if r["vignette_type"] == "implicit").items()))
                        for k, v in out.items()},
        "pairs_excluded": len(pairs),
        "leak_check": "passed: no profile_id in two splits",
    }
    json.dump(summary, open(odir / "summary.json", "w"), indent=2)
    print(json.dumps({k: summary[k] for k in ("profiles", "profile_share", "vignettes", "tier",
                                               "pairs_excluded", "leak_check")}, indent=2))


if __name__ == "__main__":
    main()
