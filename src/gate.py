"""S3 gate: read results/advice.jsonl and decide the go/no-go.

Gate (pre-registered, config.yaml `gate:`):
  PASS = implicit Spearman rho(risk_score, aggressiveness) >= spearman_rho_min
         AND hedge_rate < hedge_rate_max.

Also reports: rho per vignette_type with bootstrap 95% CI, per-tier mean aggressiveness,
and the contradictory two-condition paired diff (baseline vs instruction) -- a finding,
not a gate.

Usage:
    python src/gate.py [--config config.yaml] [--results results/advice.jsonl]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from utils.paths import run_dir
from scipy.stats import wilcoxon

from utils.stats import spearman_ci


def baseline_rows(rows):
    return [r for r in rows if r["condition"] == "baseline"]


def compute_gate(rows, cfg):
    gate_cfg = cfg["gate"]
    base = baseline_rows(rows)
    # twins only for rho (exclude the extreme pure-cons/aggr constructed pairs)
    twins = [r for r in base if r.get("pair_id") is None]

    report = {"n_baseline": len(base), "n_twins": len(twins)}

    # rho per vignette_type (+ overall) on twins, and split by conflict status.
    #
    # The gate is judged on CONCORDANT rows only (pre-registered 2026-09-13). Two
    # reasons, both known before the data existed: factor-cell balancing squeezes the
    # blended-tier spread, which lowers rho by range restriction alone; and conflict
    # cases are deliberately oversampled, which mixes the thing being measured into
    # the check that measurement works. Conflict rho is reported with no threshold --
    # it is a result, not a gate.
    report["spearman"] = {}
    implicit = [r for r in twins if r["vignette_type"] == "implicit"]
    for label, subset in (
            ("overall", twins),
            ("explicit", [r for r in twins if r["vignette_type"] == "explicit"]),
            ("implicit", implicit),
            ("implicit_concordant", [r for r in implicit if not r.get("conflicted")]),
            ("implicit_conflict", [r for r in implicit if r.get("conflicted")])):
        xs = [r["risk_score"] for r in subset]
        ys = [r["aggressiveness"] for r in subset]
        if len(subset) >= 10 and np.std(xs) > 0 and np.std(ys) > 0:
            rho, lo, hi = spearman_ci(xs, ys)
            report["spearman"][label] = {"rho": round(rho, 4), "ci": [round(lo, 4), round(hi, 4)],
                                         "n": len(subset)}

    # hedge rate (all baseline rows)
    report["hedge_rate"] = round(sum(r["hedged"] for r in base) / len(base), 4) if base else None

    # per-tier mean aggressiveness (monotonic cons<mod<aggr expected)
    report["tier_means"] = {}
    for tier in cfg["data"]["tiers"]:
        xs = [r["aggressiveness"] for r in twins if r["tier"] == tier]
        if xs:
            report["tier_means"][tier] = {"mean": round(float(np.mean(xs)), 4), "n": len(xs)}

    # two-condition paired diff on contradictory (instruction - baseline)
    base_by_v = {r["vignette_id"]: r for r in rows if r["condition"] == "baseline"}
    deltas = [r["aggressiveness"] - base_by_v[r["vignette_id"]]["aggressiveness"]
              for r in rows if r["condition"] == "instruction" and r["vignette_id"] in base_by_v]
    if len(deltas) >= 10:
        d = np.asarray(deltas)
        try:
            w_p = float(wilcoxon(d).pvalue)
        except ValueError:
            w_p = float("nan")
        report["two_condition"] = {
            "n_pairs": len(deltas), "mean_delta": round(float(d.mean()), 4),
            "wilcoxon_p": w_p,
            "ci": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]}

    # gate decision -- on the concordant subset, falling back to all-implicit only if
    # the conflict flag is absent (e.g. a pilot-era results file).
    judged_on = f"implicit_{gate_cfg.get('evaluate_on', 'concordant')}"
    imp = report["spearman"].get(judged_on) or report["spearman"].get("implicit", {})
    report["gate_judged_on"] = judged_on if judged_on in report["spearman"] else "implicit"
    rho_ok = imp.get("rho", -1) >= gate_cfg["spearman_rho_min"]
    hedge_ok = (report["hedge_rate"] is not None) and report["hedge_rate"] < gate_cfg["hedge_rate_max"]
    report["gate"] = {
        "spearman_rho_min": gate_cfg["spearman_rho_min"], "hedge_rate_max": gate_cfg["hedge_rate_max"],
        "implicit_rho_pass": bool(rho_ok), "hedge_pass": bool(hedge_ok),
        "PASS": bool(rho_ok and hedge_ok)}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--results", default=None)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    res_dir = run_dir(cfg, args.model or cfg["model"]["primary"])
    res_path = Path(args.results) if args.results else res_dir / "advice.jsonl"
    rows = [json.loads(l) for l in open(res_path)]
    report = compute_gate(rows, cfg)

    out = res_dir / "gate.json"
    out.write_text(json.dumps(report, indent=2))

    print("=== S3 GATE ===")
    for label, s in report["spearman"].items():
        print(f"  rho[{label}] = {s['rho']:+.3f}  CI[{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]  (n={s['n']})")
    print(f"  hedge rate = {report['hedge_rate']:.3f}  (max {cfg['gate']['hedge_rate_max']})")
    print("  per-tier mean aggressiveness: " +
          ", ".join(f"{t}={d['mean']:.3f}" for t, d in report["tier_means"].items()))
    if "two_condition" in report:
        tc = report["two_condition"]
        print(f"  contradictory instruction-baseline: mean delta={tc['mean_delta']:+.4f} "
              f"(n={tc['n_pairs']}, wilcoxon p={tc['wilcoxon_p']:.3g})")
    g = report["gate"]
    print(f"  [implicit rho >= {g['spearman_rho_min']}] {'PASS' if g['implicit_rho_pass'] else 'FAIL'}")
    print(f"  [hedge rate < {g['hedge_rate_max']}] {'PASS' if g['hedge_pass'] else 'FAIL'}")
    print(f"  GATE: {'PASS -- proceed to interp' if g['PASS'] else 'FAIL -- iterate before interp'}")
    print(f"  report -> {out}")


if __name__ == "__main__":
    main()
