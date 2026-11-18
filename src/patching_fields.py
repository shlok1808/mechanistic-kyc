"""RQ2a follow-up (CPU): patching flip rate broken down by the one rubric field a pair differs in.

patching.py reports one pooled flip rate. Eligibility (|donor - base| >= min_behavior_gap)
is dominated by stated_goal (150 of the 452 eligible pairs per direction), so a pooled 1.0
could be carried by one field. This slices every (layer, site) cell by field, both
directions, with bootstrap 95% CIs over pairs, and re-pools with stated_goal removed.

Reads results/<model>/patching/patching*.jsonl (the shards), reuses patching.flip_summary's
definitions (eligible / flip) from config `patching:`. Writes patching/per_field.json.

Usage:
    PYTHONPATH=src python src/patching_fields.py [--config config.yaml]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from patching import is_dev_pair, transfer_effect
from utils.paths import run_dir

DIRECTIONS = ("lo_to_hi", "hi_to_lo")


def load_rows(out_dir):
    rows = []
    for p in sorted(Path(out_dir).glob("patching.shard*.jsonl")):
        rows += [json.loads(line) for line in open(p)]
    return rows


def flip_table(rows, site, min_gap, flip_min):
    """Per-row table for one site: pair id, direction, eligible, flipped, effect.

    Rows whose patched value is missing at this site (suffix tokenised differently) are
    dropped, as in patching.summarize.
    """
    out = []
    for r in rows:
        patched = r["patched"].get(site)
        if patched is None:
            continue
        elig = abs(r["donor"] - r["base"]) >= min_gap
        eff = transfer_effect(r["base"], r["donor"], patched) if elig else float("nan")
        out.append((r["pair_id"], r["direction"], bool(elig), bool(elig and eff >= flip_min), eff))
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(max(0.0, c - h)), 4), round(float(min(1.0, c + h)), 4)]


def summarize(table, direction=None, n_boot=2000, seed=42):
    """Flip rate over eligible rows, bootstrap-resampling PAIRS (so both directions of a pair
    move together in the pooled version). direction=None pools both directions."""
    sub = [t for t in table if t[2] and (direction is None or t[1] == direction)]
    n = len(sub)
    if n == 0:
        return {"n_eligible": 0, "n_flips": 0, "flip_rate": None, "ci": None, "wilson": None,
                "mean_effect": None}
    pairs = sorted({t[0] for t in sub})
    by_pair = {p: [t for t in sub if t[0] == p] for p in pairs}
    flips = np.array([sum(t[3] for t in by_pair[p]) for p in pairs], float)
    sizes = np.array([len(by_pair[p]) for p in pairs], float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(pairs), size=(n_boot, len(pairs)))
    boot = flips[draws].sum(1) / sizes[draws].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    k = int(flips.sum())
    return {"n_eligible": n, "n_flips": k, "flip_rate": round(k / n, 4),
            "ci": [round(float(lo), 4), round(float(hi), 4)], "wilson": wilson(k, n),
            "mean_effect": round(float(np.mean([t[4] for t in sub])), 4)}


def cell(table):
    return {**{d: summarize(table, d) for d in DIRECTIONS}, "pooled": summarize(table)}


def run(cfg):
    pc = cfg["patching"]
    out_dir = run_dir(cfg, cfg["model"]["primary"]) / "patching"
    rows = load_rows(out_dir)
    sites = sorted(rows[0]["patched"], key=lambda s: (int(s.split(":")[0][1:]), s))
    summary = json.loads((out_dir / "patching.json").read_text())
    best = summary["best_site_selected_on_dev"]
    fields = sorted({r["field"] for r in rows})
    gap, fmin = pc["min_behavior_gap"], pc["flip_effect_min"]

    result = {"n_rows": len(rows), "min_behavior_gap": gap, "flip_effect_min": fmin,
              "headline_site": best, "sites": sites,
              "ci": "bootstrap 95% percentile over pairs (2000 draws); wilson = Wilson 95% score interval "
                    "(non-degenerate when the rate is exactly 0 or 1)",
              "by_field": {}}
    tables = {s: flip_table(rows, s, gap, fmin) for s in sites}
    field_of = {r["pair_id"]: r["field"] for r in rows}
    for field in fields:
        sub_rows = [r for r in rows if r["field"] == field]
        gaps = {d: [abs(r["donor"] - r["base"]) for r in sub_rows if r["direction"] == d] for d in DIRECTIONS}
        result["by_field"][field] = {
            "n_pairs": len({r["pair_id"] for r in sub_rows}),
            "n_eligible_per_direction": {d: int(sum(g >= gap for g in gaps[d])) for d in DIRECTIONS},
            "mean_behavior_gap": round(float(np.mean(gaps["lo_to_hi"] + gaps["hi_to_lo"])), 4),
            "sites": {s: cell([t for t in tables[s] if field_of[t[0]] == field]) for s in sites}}

    def pooled_over(keep, only_report=False):
        return {s: cell([t for t in tables[s] if keep(field_of[t[0]])
                         and (not only_report or not is_dev_pair(t[0]))]) for s in sites}

    result["all_fields_all_pairs"] = pooled_over(lambda f: True)
    result["all_fields_report_pairs_odd"] = pooled_over(lambda f: True, only_report=True)
    result["without_stated_goal_all_pairs"] = pooled_over(lambda f: f != "stated_goal")
    result["without_stated_goal_report_pairs_odd"] = pooled_over(lambda f: f != "stated_goal", only_report=True)
    result["stated_goal_only_all_pairs"] = pooled_over(lambda f: f == "stated_goal")

    # how close to the donor is the patched advice at the headline site? (ceiling check)
    close = [abs(r["patched"][best] - r["donor"]) for r in rows if r["patched"].get(best) is not None
             and abs(r["donor"] - r["base"]) >= gap]
    result["headline_site_ceiling_check"] = {
        "n": len(close), "median_abs_patched_minus_donor": round(float(np.median(close)), 5),
        "share_within_0.01_of_donor": round(float(np.mean(np.array(close) < 0.01)), 4)}

    # three raw rows: the headline site on a non-stated_goal field, plus one of each regime
    def pick(field, d, min_gap=gap):
        for r in rows:
            if r["field"] == field and r["direction"] == d and abs(r["donor"] - r["base"]) >= min_gap:
                return r
    ex = [pick("stated_goal", "lo_to_hi"), pick("horizon_years", "hi_to_lo"), pick("age", "lo_to_hi")]
    result["example_rows"] = [{k: r[k] for k in ("pair_id", "direction", "field", "factor", "base", "donor")}
                              | {"patched_at_headline": r["patched"][best],
                                 "patched_all_sites": r["patched"]} for r in ex if r]
    dest = out_dir / "per_field.json"
    dest.write_text(json.dumps(result, indent=2))

    def fmt(c):
        return (f"{c['flip_rate']:.3f} [{c['ci'][0]:.3f},{c['ci'][1]:.3f}] n={c['n_eligible']}"
                if c["n_eligible"] else "n=0")
    print(f"headline site {best}")
    for field in fields:
        c = result["by_field"][field]["sites"][best]
        print(f"  {field:24s} lo->hi {fmt(c['lo_to_hi'])}   hi->lo {fmt(c['hi_to_lo'])}")
    for key in ("all_fields_all_pairs", "without_stated_goal_all_pairs", "without_stated_goal_report_pairs_odd"):
        c = result[key][best]
        print(f"  {key:40s} lo->hi {fmt(c['lo_to_hi'])}   hi->lo {fmt(c['hi_to_lo'])}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    run(yaml.safe_load(open(args.config)))
