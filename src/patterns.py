"""S3b: behavioral pattern analysis over the S3 results -- the paper-number source for the
cold-read findings (2026-07-01) and the behavioral rungs of the audit ladder.

What it computes (all from results/advice.jsonl + data/profiles.jsonl, CPU only):

  1. Subgroup Spearman rhos: explicit/implicit x contradictory/not, with bootstrap CIs.
     The headline gate rho (0.64 implicit) averages over an easy subpopulation (signals
     agree) and a failing one (contradictory clients, rho ~ 0).
  2. Field-weight decomposition: standardized OLS of advice aggressiveness on the eight
     rubric-normalized fields, relative |beta| shares vs the rubric's own weights. This is
     the "heuristic collapse" measurement (stated_goal dominance).
  3. Whose-side-on-conflicts: on contradictory rows, partial Spearman of advice vs
     stated_goal controlling risk_score, and vice versa.
  4. Menu use per tier/type (option-probability means + argmax shares) and the
     explicit->implicit twin gap by tier (the asymmetric-caution pattern).
  5. Two-condition instruction deltas by tier/type (directionally corrective, tiny).
  6. Audit-ladder behavioral rungs, protocol-matched to the S5 probe: a decoder is fit on
     EXPLICIT-TRAIN behavior (same profile-id splits, same seed) and evaluated as
     macro-OvR AUROC on the IMPLICIT-TEST rows -- directly comparable to the probe's
     implicit held-out AUROC. Both the 1-D aggressiveness scalar and the full 4-option
     distribution versions are reported (the 4-D one is the fair "black-box with training
     data" baseline). Zero-supervision reference: implicit-test Spearman rho.

Output: results/patterns.json + printed report. S5b reads this JSON to assemble the
full audit ladder and to compare behavioral vs probe field weights.

Usage:
    python src/patterns.py [--config config.yaml] [--n-boot 2000]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from utils.paths import run_dir

from rubric import field_subscore, load_rubric
from probes import config_sha, git_commit, make_splits, split_indices
from utils.stats import (auroc_ci, macro_ovr_auroc, partial_spearman, spearman_ci,
                         standardized_rel_weights)


# --------------------------------------------------------------------------------------
# Pure logic -- unit-tested in tests/test_s3b.py
# --------------------------------------------------------------------------------------
def rubric_field_names(rubric):
    """Rubric field names in config order (the field-weight table's row order)."""
    return list(rubric["fields"].keys())


def fields_matrix(profiles, rubric, field_names=None):
    """[n, k] matrix of rubric-normalized field values a_i in [0,1] for the given profiles.

    Uses rubric.field_subscore so the normalization is THE ground-truth one by construction
    (levels evenly spaced, ranges min-maxed, inversions applied).
    """
    names = field_names or rubric_field_names(rubric)
    return np.array([[field_subscore(rubric["fields"][f], p[f]) for f in names]
                     for p in profiles], dtype=float)


def twin_gap_by_tier(rows_explicit, rows_implicit, tiers):
    """Per-tier mean(explicit aggr - implicit aggr) over profiles present in both renders."""
    imp_by_pid = {r["profile_id"]: r["aggressiveness"] for r in rows_implicit}
    out = {}
    for tier in tiers:
        diffs = [r["aggressiveness"] - imp_by_pid[r["profile_id"]]
                 for r in rows_explicit if r["tier"] == tier and r["profile_id"] in imp_by_pid]
        out[tier] = {"mean_gap": round(float(np.mean(diffs)), 4) if diffs else None,
                     "n": len(diffs)}
    return out


def menu_use(rows, tiers):
    """Per-tier mean option probabilities and argmax shares (how much of the menu is used)."""
    out = {}
    for tier in tiers:
        P = np.array([r["p_alloc"] for r in rows if r["tier"] == tier], dtype=float)
        if not len(P):
            continue
        am = np.bincount(P.argmax(1), minlength=P.shape[1]) / len(P)
        out[tier] = {"mean_p": [round(float(x), 4) for x in P.mean(0)],
                     "argmax_share": [round(float(x), 4) for x in am],
                     "mean_aggressiveness": round(float(np.mean([r["aggressiveness"] for r in rows
                                                                 if r["tier"] == tier])), 4),
                     "n": int(len(P))}
    return out


def instruction_deltas(base_rows, instr_rows):
    """Per-vignette instruction-baseline aggressiveness deltas, keyed by vignette_id."""
    base = {r["vignette_id"]: r for r in base_rows}
    out = []
    for r in instr_rows:
        b = base.get(r["vignette_id"])
        if b is not None:
            out.append({"vignette_id": r["vignette_id"], "tier": b["tier"],
                        "vignette_type": b["vignette_type"],
                        "baseline": b["aggressiveness"],
                        "delta": r["aggressiveness"] - b["aggressiveness"]})
    return out


def summarize_deltas(deltas, tiers):
    """Mean delta / share-positive overall, by tier, by vignette type, + slope on baseline."""
    d = np.array([x["delta"] for x in deltas], dtype=float)
    b = np.array([x["baseline"] for x in deltas], dtype=float)
    slope = float(np.polyfit(b, d, 1)[0]) if len(d) > 1 else float("nan")
    rec = {"n": int(len(d)), "mean_delta": round(float(d.mean()), 4),
           "median_delta": round(float(np.median(d)), 4), "sd": round(float(d.std()), 4),
           "share_positive": round(float((d > 0).mean()), 4),
           "share_abs_gt_0.1": round(float((np.abs(d) > 0.1).mean()), 4),
           "slope_on_baseline": round(slope, 4), "by_tier": {}, "by_type": {}}
    for tier in tiers:
        dd = np.array([x["delta"] for x in deltas if x["tier"] == tier])
        if len(dd):
            rec["by_tier"][tier] = {"n": int(len(dd)), "mean_delta": round(float(dd.mean()), 4),
                                    "share_positive": round(float((dd > 0).mean()), 4)}
    for vt in ("explicit", "implicit"):
        dd = np.array([x["delta"] for x in deltas if x["vignette_type"] == vt])
        if len(dd):
            rec["by_type"][vt] = {"n": int(len(dd)), "mean_delta": round(float(dd.mean()), 4),
                                  "share_positive": round(float((dd > 0).mean()), 4)}
    return rec


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def run(cfg, args):
    res_dir = run_dir(cfg, args.model or cfg["model"]["primary"])
    rubric = cfg["rubric"]
    tiers = cfg["data"]["tiers"]
    fnames = rubric_field_names(rubric)

    s3_rows = load_jsonl(res_dir / "advice.jsonl")
    profiles = {p["profile_id"]: p for p in load_jsonl(Path(cfg["paths"]["data_dir"]) / "profiles.jsonl")}

    base = [r for r in s3_rows if r["condition"] == "baseline"]
    twins = [r for r in base if r.get("pair_id") is None]
    exp_rows = [r for r in twins if r["vignette_type"] == "explicit"]
    imp_rows = [r for r in twins if r["vignette_type"] == "implicit"]
    print(f"S3b: {len(twins)} baseline twin rows ({len(exp_rows)} explicit / {len(imp_rows)} implicit)")

    report = {"meta": {"config_sha": config_sha(args.config), "git_commit": git_commit(),
                       "n_twins": len(twins), "seed": cfg["seed"], "n_boot": args.n_boot}}

    # ---- 1. subgroup rhos ----
    rhos = {}
    for vt, rows in (("explicit", exp_rows), ("implicit", imp_rows)):
        for contra in (False, True):
            rr = [r for r in rows if r["contradictory"] == contra]
            rho, lo, hi = spearman_ci([r["risk_score"] for r in rr],
                                      [r["aggressiveness"] for r in rr],
                                      n_boot=args.n_boot, seed=cfg["seed"])
            rhos[f"{vt}_{'contradictory' if contra else 'concordant'}"] = {
                "rho": round(rho, 4), "ci": [round(lo, 4), round(hi, 4)], "n": len(rr)}
    report["subgroup_spearman"] = rhos

    # ---- 2. field weights (behavioral heuristic-collapse measurement) ----
    fw = {}
    for vt, rows in (("explicit", exp_rows), ("implicit", imp_rows)):
        X = fields_matrix([profiles[r["profile_id"]] for r in rows], rubric, fnames)
        y = np.array([r["aggressiveness"] for r in rows])
        betas, rel, r2 = standardized_rel_weights(X, y)
        fw[vt] = {"r2": round(r2, 4),
                  "fields": {f: {"beta": round(float(b), 4), "rel_weight": round(float(w), 4),
                                 "rubric_weight": rubric["fields"][f]["weight"],
                                 "ratio_vs_rubric": round(float(w) / rubric["fields"][f]["weight"], 2)}
                             for f, b, w in zip(fnames, betas, rel)}}
    report["field_weights"] = fw

    # ---- 3. whose side on conflicts (partial Spearman) ----
    sides = {}
    for vt, rows in (("explicit", exp_rows), ("implicit", imp_rows)):
        rr = [r for r in rows if r["contradictory"]]
        aggr = [r["aggressiveness"] for r in rr]
        goal = [field_subscore(rubric["fields"]["stated_goal"], profiles[r["profile_id"]]["stated_goal"])
                for r in rr]
        score = [r["risk_score"] for r in rr]
        sides[vt] = {"n": len(rr),
                     "advice_vs_goal_given_score": round(partial_spearman(aggr, goal, score), 4),
                     "advice_vs_score_given_goal": round(partial_spearman(aggr, score, goal), 4)}
    report["conflict_sides"] = sides

    # ---- 3b. graded collapse: rho by conflict magnitude |a_goal - a_drawdown| ----
    # The contradictory flag is binary; the underlying conflict is graded (|diff| takes the
    # discrete values 0, 1/3, 2/3, 1). Behavioral integration degrades as a dose-response
    # in conflict magnitude -- the strongest version of the collapse finding.
    graded = {}
    for vt, rows in (("explicit", exp_rows), ("implicit", imp_rows)):
        by_mag = {}
        for r in rows:
            p = profiles[r["profile_id"]]
            mag = round(abs(field_subscore(rubric["fields"]["stated_goal"], p["stated_goal"])
                            - field_subscore(rubric["fields"]["past_drawdown_reaction"],
                                             p["past_drawdown_reaction"])) * 3)
            by_mag.setdefault(mag, []).append(r)
        graded[vt] = {}
        for mag in sorted(by_mag):
            rr = by_mag[mag]
            rho, lo, hi = spearman_ci([r["risk_score"] for r in rr],
                                      [r["aggressiveness"] for r in rr],
                                      n_boot=args.n_boot, seed=cfg["seed"])
            graded[vt][f"conflict_{mag}_of_3"] = {"rho": round(rho, 4),
                                                  "ci": [round(lo, 4), round(hi, 4)], "n": len(rr)}
    report["graded_collapse"] = graded

    # ---- 3c. decoy-field leakage (fairness/robustness checks; decoys are non-rubric) ----
    nw_order = ["under_100k", "100k_to_500k", "500k_to_2m", "over_2m"]
    leak = {}
    for vt, rows in (("explicit", exp_rows), ("implicit", imp_rows)):
        ag = np.array([r["aggressiveness"] for r in rows])
        rs = np.array([r["risk_score"] for r in rows])
        resid = ag - np.poly1d(np.polyfit(rs, ag, 1))(rs)
        gender = np.array([profiles[r["profile_id"]]["gender_hint"] for r in rows])
        nw = np.array([nw_order.index(profiles[r["profile_id"]]["net_worth_band"]) for r in rows])
        leak[vt] = {
            "net_worth_partial_spearman": round(partial_spearman(ag, nw, rs), 4),
            "gender_resid_means": {g: round(float(resid[gender == g].mean()), 4)
                                   for g in ("male", "female", "neutral")}}
    report["decoy_leakage"] = leak

    # ---- 4. menu use + twin gap ----
    report["menu_use"] = {"explicit": menu_use(exp_rows, tiers), "implicit": menu_use(imp_rows, tiers)}
    report["twin_gap_by_tier"] = twin_gap_by_tier(exp_rows, imp_rows, tiers)

    # ---- 5. two-condition instruction deltas (contradictory twins) ----
    instr = [r for r in s3_rows if r["condition"] == "instruction" and r.get("pair_id") is None]
    deltas = instruction_deltas([r for r in twins if r["contradictory"]], instr)
    report["instruction_effect"] = summarize_deltas(deltas, tiers)

    # ---- 6. audit-ladder behavioral rungs (protocol-matched to the S5 probe) ----
    # Same profile-id splits as S5 (same seed, same fracs), decoder fit on EXPLICIT-TRAIN
    # behavior, evaluated on IMPLICIT-TEST -- the probe's exact transfer protocol.
    # S5 built its split map over the profile_ids of the FULL cache (twins + counterfactual
    # pairs; pairs are then excluded row-wise), so the map must be built the same way here
    # or the test membership diverges (make_splits permutes the sorted unique-id list).
    labels = [{"profile_id": r["profile_id"], "pair_id": r.get("pair_id"),
               "vignette_type": r["vignette_type"], "tier": r["tier"],
               "risk_score": r["risk_score"]} for r in twins]
    split = make_splits([r["profile_id"] for r in base], seed=cfg["seed"])
    idx = split_indices(labels, split)
    tier_arr = np.array([lab["tier"] for lab in labels])
    twins_arr = twins  # index-aligned with labels

    from sklearn.linear_model import LogisticRegression
    ladder = {}
    for name, feats in (("behavior_1d_aggressiveness", lambda r: [r["aggressiveness"]]),
                        ("behavior_4d_option_dist", lambda r: list(r["p_alloc"]))):
        Xtr = np.array([feats(twins_arr[i]) for i in idx["train"]])
        Xit = np.array([feats(twins_arr[i]) for i in idx["implicit_test"]])
        clf = LogisticRegression(max_iter=1000, random_state=cfg["seed"]).fit(Xtr, tier_arr[idx["train"]])
        proba = clf.predict_proba(Xit)
        point, lo, hi = auroc_ci(tier_arr[idx["implicit_test"]], proba, list(clf.classes_),
                                 n_boot=args.n_boot, seed=cfg["seed"])
        ladder[name] = {"implicit_test_auroc": round(point, 4), "ci": [round(lo, 4), round(hi, 4)],
                        "n": int(len(Xit)), "protocol": "fit explicit-train -> eval implicit-test"}
    it_rows = [twins_arr[i] for i in idx["implicit_test"]]
    rho, lo, hi = spearman_ci([r["risk_score"] for r in it_rows],
                              [r["aggressiveness"] for r in it_rows],
                              n_boot=args.n_boot, seed=cfg["seed"])
    ladder["behavior_zero_supervision_spearman"] = {"rho": round(rho, 4),
                                                    "ci": [round(lo, 4), round(hi, 4)],
                                                    "n": len(it_rows)}
    report["audit_ladder_behavioral"] = ladder

    out = res_dir / "patterns.json"
    out.write_text(json.dumps(report, indent=2))
    _print_summary(report)
    print(f"  report -> {out}")
    return report


def _print_summary(r):
    print("=== S3b BEHAVIORAL PATTERNS ===")
    print("  subgroup rhos:")
    for k, v in r["subgroup_spearman"].items():
        print(f"    {k:26s} rho={v['rho']:+.3f} CI[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] n={v['n']}")
    for vt in ("explicit", "implicit"):
        fws = r["field_weights"][vt]["fields"]
        top = sorted(fws.items(), key=lambda kv: -kv[1]["rel_weight"])[:3]
        print(f"  {vt} top fields: " + ", ".join(
            f"{f} rel={v['rel_weight']:.3f} ({v['ratio_vs_rubric']}x rubric)" for f, v in top))
    for vt, v in r["conflict_sides"].items():
        print(f"  conflicts[{vt}]: advice~goal|score={v['advice_vs_goal_given_score']:+.3f}  "
              f"advice~score|goal={v['advice_vs_score_given_goal']:+.3f}  (n={v['n']})")
    for vt, g in r["graded_collapse"].items():
        curve = " ".join(f"{k.split('_')[1]}/3:{v['rho']:+.2f}(n={v['n']})" for k, v in g.items())
        print(f"  graded collapse [{vt}]: {curve}")
    for vt, v in r["decoy_leakage"].items():
        print(f"  decoys[{vt}]: net_worth partial={v['net_worth_partial_spearman']:+.3f}  "
              f"gender resid means={v['gender_resid_means']}")
    ie = r["instruction_effect"]
    print(f"  instruction: mean_delta={ie['mean_delta']:+.4f} pos%={ie['share_positive']:.2f} "
          f"by_tier=" + " ".join(f"{t}:{v['mean_delta']:+.4f}" for t, v in ie["by_tier"].items()))
    print("  audit ladder (behavioral rungs, implicit test):")
    for k, v in r["audit_ladder_behavioral"].items():
        val = v.get("implicit_test_auroc", v.get("rho"))
        print(f"    {k:36s} {val:+.4f}  CI[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]  n={v['n']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg, args)


if __name__ == "__main__":
    main()
