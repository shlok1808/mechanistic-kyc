"""S5b: where does heuristic collapse happen -- encoding or readout? Plus the audit ladder.

The cold-read of S3 (see patterns.py) found that advice on contradictory clients is
uncorrelated with ground truth (implicit rho ~ -0.1) and that advice overweights
stated_goal ~5x. Two mechanistic stories are consistent with that:

  READOUT collapse (the paper's dream): the internal client representation stays intact on
    contradictory clients (probe AUROC holds) while the advice pathway applies the
    stated-goal heuristic -- integration exists internally, behavior discards it.
  ENCODING collapse: the representation itself is goal-dominated on conflicts (probe AUROC
    drops toward chance there) -- the model never integrates conflicting signals.

This script decides between them using the FROZEN S5 probes (no refitting -- the probe was
trained before the contradictory question was asked, so this is pure evaluation):

  1. Probe macro-AUROC on the implicit test split: overall / contradictory / concordant,
     with bootstrap CIs; every saved probe (best + per-position) is evaluated for robustness.
  2. Probe field-weight decomposition: the probe's continuous readout
     (p_aggressive - p_conservative) regressed on the rubric-normalized fields, compared
     against the behavioral weights (s3b) and the rubric's own weights.
  3. Whose-side-on-conflicts, probe edition: partial Spearman of the probe readout vs
     stated_goal controlling risk_score and vice versa (mirrors s3b's behavioral stat).
  4. The AUDIT LADDER, all rungs on the SAME implicit-test rows and metric (macro-OvR
     AUROC): zero/low-supervision behavior, supervised behavior decoders (from s3b),
     TF-IDF transcript baseline (from the S5 report), verbal elicitation (from s3c, if
     run), and the probe. This is the apples-to-apples version of "0.89 vs 0.64".

Verdicts are pre-registered in config `dissociation:` and `elicitation:` (added 2026-07-02
BEFORE any probe-side contradictory number was computed; the behavioral side was observed
2026-07-01).

Needs the S4 activation cache (results/activations/<tag>/) -- Lambda box, not local.

Usage (Lambda):
    python src/dissociation.py [--config config.yaml]
        [--activations results/activations/gemma-2-9b-it] [--probe path/to/probe.npz]
        [--n-boot 2000]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from utils.paths import run_dir

from rubric import field_subscore
from patterns import fields_matrix, rubric_field_names
from activations import model_tag
from probes import config_sha, consolidate, git_commit, load_plane, make_splits, split_indices
from utils.stats import auroc_ci, macro_ovr_auroc, partial_spearman, standardized_rel_weights


# --------------------------------------------------------------------------------------
# Pure logic -- unit-tested in tests/test_s5b.py
# --------------------------------------------------------------------------------------
def load_probe_npz(path):
    """Saved probe npz -> plain dict of numpy arrays / scalars (see s5 probe_to_npz)."""
    z = np.load(path, allow_pickle=True)
    return {"classes": [str(c) for c in z["classes"]],
            "coef": np.asarray(z["coef"], dtype=np.float64),
            "intercept": np.asarray(z["intercept"], dtype=np.float64),
            "scaler_mean": np.asarray(z["scaler_mean"], dtype=np.float64),
            "scaler_scale": np.asarray(z["scaler_scale"], dtype=np.float64),
            "layer": int(z["layer"]), "position": str(z["position"])}


def probe_predict_proba(probe, X):
    """Multinomial predict_proba from the saved linear params (standardize -> softmax).

    Matches sklearn LogisticRegression(multi_class multinomial, the lbfgs default for 3
    classes): logits = X_std @ coef.T + intercept, softmax rows. Verified against sklearn
    in tests/test_s5b.py.
    """
    Xs = (np.asarray(X, dtype=np.float64) - probe["scaler_mean"]) / probe["scaler_scale"]
    logits = Xs @ probe["coef"].T + probe["intercept"]
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def continuous_readout(proba, classes):
    """Scalar probe readout = p(aggressive) - p(conservative), the rank-statistic analogue
    of S3's aggressiveness score."""
    ia, ic = classes.index("aggressive"), classes.index("conservative")
    return proba[:, ia] - proba[:, ic]


def implicit_rows_for_splits(labels, split, which=("val", "test")):
    """Row indices of implicit twin rows whose profile fell in the given splits.

    split_indices only exposes implicit_test/implicit_train; the dissociation analysis also
    wants the larger val+test pool (contradictory rows are ~12%, so the test split alone
    holds only ~145 of them).
    """
    out = []
    for i, lab in enumerate(labels):
        if lab.get("pair_id") is not None or lab["vignette_type"] != "implicit":
            continue
        if split.get(lab["profile_id"]) in which:
            out.append(i)
    return np.asarray(out, dtype=int)


def dissociation_verdict(contra_auroc, concord_auroc, bands):
    """Pre-registered call: readout_collapse / encoding_collapse / partial."""
    if (contra_auroc >= bands["readout_collapse_auroc_min"]
            and concord_auroc - contra_auroc <= bands["max_drop_vs_noncontradictory"]):
        return "readout_collapse"
    if contra_auroc < bands["encoding_collapse_auroc_max"]:
        return "encoding_collapse"
    return "partial"


def elicitation_verdict(probe_auroc, elicit_auroc, bands):
    """Pre-registered call: latent_knowledge / promptable_knowledge / partial."""
    margin = probe_auroc - elicit_auroc
    if margin >= bands["latent_margin_min"]:
        return "latent_knowledge"
    if margin <= bands["promptable_margin_max"]:
        return "promptable_knowledge"
    return "partial"


def eval_probe_on(proba, tier, rows_idx, classes, n_boot, seed):
    """Macro-OvR AUROC + CI + accuracy on a row subset (proba precomputed for all rows)."""
    y = tier[rows_idx]
    p = proba[rows_idx]
    point, lo, hi = auroc_ci(y, p, classes, n_boot=n_boot, seed=seed)
    acc = float((np.asarray(classes)[p.argmax(1)] == y).mean())
    return {"auroc": round(point, 4), "ci": [round(lo, 4), round(hi, 4)],
            "acc": round(acc, 4), "n": int(len(rows_idx))}


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def _latest_cache(res_dir):
    """Newest fingerprinted activation cache under results/<model>/activations/.

    Caches are now keyed by a fingerprint of model + prompt + data + layers + dtype,
    so there can be several. Pass --activations to pin one explicitly.
    """
    caches = sorted((res_dir / "activations").glob("*/meta.json"),
                    key=lambda p: p.stat().st_mtime, reverse=True)
    if not caches:
        raise FileNotFoundError(
            f"no activation cache under {res_dir / 'activations'} -- run activations.py first")
    return caches[0].parent


def run(cfg, args):
    model_id = args.model or cfg["model"]["primary"]
    res_dir = run_dir(cfg, model_id)
    tag = model_tag(cfg["model"]["primary"])
    act_dir = Path(args.activations) if args.activations else _latest_cache(res_dir)

    try:
        acts, labels, meta = consolidate(act_dir, cfg["paths"]["vignettes_dir"])
    except FileNotFoundError:
        print(f"S5b: no activation shards in {act_dir}.\n"
              f"  This analysis needs the S4 cache (Lambda box). Runbook:\n"
              f"    tmux new -s s5b\n"
              f"    python src/activations.py            # only if the cache is gone\n"
              f"    python src/elicit.py                      # elicitation rung (GPU)\n"
              f"    python src/dissociation.py                # this script (CPU, needs cache)\n"
              f"  (detach with Ctrl+B D)")
        sys.exit(2)

    probe_dir = res_dir / "weights"
    if args.probe:
        best_path = Path(args.probe)
    else:
        cands = sorted(probe_dir.glob("probe_best_*.npz"))
        if not cands:
            print(f"S5b: no probe_best_*.npz in {probe_dir} -- run S5 first.")
            sys.exit(2)
        best_path = cands[0]
    probe = load_probe_npz(best_path)
    l_idx = meta["layers"].index(probe["layer"])
    p_idx = meta["positions"].index(probe["position"])
    print(f"S5b: probe {best_path.name} (L{probe['layer']} @ {probe['position']})  cache {acts.shape}")

    split = make_splits([lab["profile_id"] for lab in labels], seed=cfg["seed"])
    idx = split_indices(labels, split)
    tier = np.array([lab["tier"] for lab in labels])
    score = np.array([lab["risk_score"] for lab in labels], dtype=float)
    contra = np.array([bool(lab["contradictory"]) for lab in labels])

    plane = load_plane(acts, l_idx, p_idx)
    proba = probe_predict_proba(probe, plane)
    classes = probe["classes"]

    report = {"meta": {"model": meta["model"], "probe": best_path.name,
                       "layer": probe["layer"], "position": probe["position"],
                       "config_sha": config_sha(args.config), "git_commit": git_commit(),
                       "seed": cfg["seed"], "n_boot": args.n_boot}}

    # ---- 1. probe on the implicit test split: overall / contradictory / concordant ----
    it = idx["implicit_test"]
    it_contra, it_concord = it[contra[it]], it[~contra[it]]
    heldout = implicit_rows_for_splits(labels, split, ("val", "test"))
    ho_contra, ho_concord = heldout[contra[heldout]], heldout[~contra[heldout]]
    sets = {"implicit_test": it, "implicit_test_contradictory": it_contra,
            "implicit_test_concordant": it_concord,
            "implicit_heldout_valtest": heldout,
            "implicit_heldout_contradictory": ho_contra,
            "implicit_heldout_concordant": ho_concord}
    report["probe_eval"] = {name: eval_probe_on(proba, tier, rows, classes, args.n_boot, cfg["seed"])
                            for name, rows in sets.items()}

    bands = cfg["dissociation"]
    v_test = dissociation_verdict(report["probe_eval"]["implicit_test_contradictory"]["auroc"],
                                  report["probe_eval"]["implicit_test_concordant"]["auroc"], bands)
    v_held = dissociation_verdict(report["probe_eval"]["implicit_heldout_contradictory"]["auroc"],
                                  report["probe_eval"]["implicit_heldout_concordant"]["auroc"], bands)
    report["dissociation"] = {"bands": bands, "verdict_test": v_test, "verdict_heldout": v_held}

    # robustness: every saved probe on the heldout contradictory subset
    others = {}
    for p in sorted(probe_dir.glob("probe_*.npz")):
        if p == best_path:
            continue
        pr = load_probe_npz(p)
        pl = load_plane(acts, meta["layers"].index(pr["layer"]), meta["positions"].index(pr["position"]))
        pb = probe_predict_proba(pr, pl)
        others[p.name] = {
            "contradictory": eval_probe_on(pb, tier, ho_contra, pr["classes"], args.n_boot, cfg["seed"]),
            "concordant": eval_probe_on(pb, tier, ho_concord, pr["classes"], args.n_boot, cfg["seed"])}
    report["probe_robustness_heldout"] = others

    # ---- 2 + 3. probe field weights and conflict sides (implicit heldout pool) ----
    profiles = {p["profile_id"]: p for p in load_jsonl(Path(cfg["paths"]["data_dir"]) / "profiles.jsonl")}
    rubric = cfg["rubric"]
    fnames = rubric_field_names(rubric)
    ho_profiles = [profiles[labels[i]["profile_id"]] for i in heldout]
    X = fields_matrix(ho_profiles, rubric, fnames)
    readout = continuous_readout(proba[heldout], classes)
    betas, rel, r2 = standardized_rel_weights(X, readout)
    report["probe_field_weights"] = {
        "r2": round(r2, 4),
        "fields": {f: {"beta": round(float(b), 4), "rel_weight": round(float(w), 4),
                       "rubric_weight": rubric["fields"][f]["weight"],
                       "ratio_vs_rubric": round(float(w) / rubric["fields"][f]["weight"], 2)}
                   for f, b, w in zip(fnames, betas, rel)}}

    goal = np.array([field_subscore(rubric["fields"]["stated_goal"], p["stated_goal"])
                     for p in ho_profiles])
    mask = contra[heldout]
    report["probe_conflict_sides"] = {
        "n": int(mask.sum()),
        "readout_vs_goal_given_score": round(partial_spearman(readout[mask], goal[mask],
                                                              score[heldout][mask]), 4),
        "readout_vs_score_given_goal": round(partial_spearman(readout[mask], score[heldout][mask],
                                                              goal[mask]), 4)}

    # ---- 4. the audit ladder (same rows, same metric) ----
    ladder = {"probe_internal": {"auroc": report["probe_eval"]["implicit_test"]["auroc"],
                                 "ci": report["probe_eval"]["implicit_test"]["ci"],
                                 "n": report["probe_eval"]["implicit_test"]["n"]}}
    s3b_path = res_dir / "patterns.json"
    if s3b_path.exists():
        b = json.loads(s3b_path.read_text())["audit_ladder_behavioral"]
        ladder["behavior_1d"] = {"auroc": b["behavior_1d_aggressiveness"]["implicit_test_auroc"],
                                 "ci": b["behavior_1d_aggressiveness"]["ci"]}
        ladder["behavior_4d"] = {"auroc": b["behavior_4d_option_dist"]["implicit_test_auroc"],
                                 "ci": b["behavior_4d_option_dist"]["ci"]}
    else:
        ladder["behavior_1d"] = ladder["behavior_4d"] = "pending (run patterns.py)"
    s5_path = res_dir / "probes.json"
    if s5_path.exists():
        ladder["tfidf_transcript"] = {"auroc": json.loads(s5_path.read_text())
                                      ["baselines"]["tfidf"]["implicit_test"]}
    elic_path = res_dir / "elicit.jsonl"
    if elic_path.exists():
        erows = {r["vignette_id"]: r for r in load_jsonl(elic_path)}
        it_vids = [labels[i]["vignette_id"] for i in it]
        pairs = [(labels[i]["tier"], erows[v]["p_tier"]) for i, v in zip(it, it_vids) if v in erows]
        if pairs:
            y = [t for t, _ in pairs]
            p = np.array([pt for _, pt in pairs])
            point, lo, hi = auroc_ci(y, p, cfg["data"]["tiers"], n_boot=args.n_boot, seed=cfg["seed"])
            ladder["verbal_elicitation"] = {"auroc": round(point, 4),
                                            "ci": [round(lo, 4), round(hi, 4)], "n": len(pairs)}
            report["elicitation"] = {
                "bands": cfg["elicitation"],
                "verdict": elicitation_verdict(ladder["probe_internal"]["auroc"], point,
                                               cfg["elicitation"])}
    else:
        ladder["verbal_elicitation"] = "pending (run elicit.py)"
    report["audit_ladder"] = ladder

    out = res_dir / "dissociation.json"
    out.write_text(json.dumps(report, indent=2))
    _print_summary(report)
    print(f"  report -> {out}")
    return report


def _print_summary(r):
    print("=== S5b DISSOCIATION (encoding vs readout) ===")
    for name, v in r["probe_eval"].items():
        print(f"  {name:34s} AUROC={v['auroc']:.4f} CI[{v['ci'][0]:.4f},{v['ci'][1]:.4f}] "
              f"acc={v['acc']:.3f} n={v['n']}")
    d = r["dissociation"]
    print(f"  VERDICT: test={d['verdict_test']}  heldout={d['verdict_heldout']}")
    pf = r["probe_field_weights"]["fields"]
    top = sorted(pf.items(), key=lambda kv: -kv[1]["rel_weight"])[:3]
    print("  probe top fields: " + ", ".join(
        f"{f} rel={v['rel_weight']:.3f} ({v['ratio_vs_rubric']}x rubric)" for f, v in top))
    cs = r["probe_conflict_sides"]
    print(f"  probe conflicts: readout~goal|score={cs['readout_vs_goal_given_score']:+.3f}  "
          f"readout~score|goal={cs['readout_vs_score_given_goal']:+.3f}  (n={cs['n']})")
    print("  AUDIT LADDER (implicit test, macro-OvR AUROC):")
    for k, v in r["audit_ladder"].items():
        print(f"    {k:22s} {v if isinstance(v, str) else v['auroc']}")
    if "elicitation" in r:
        print(f"  elicitation verdict: {r['elicitation']['verdict']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--activations", default=None, help="S4 cache dir (results/activations/<tag>)")
    ap.add_argument("--probe", default=None, help="probe npz override (default: probe_best_*.npz)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--model", default=None, help="override config model.primary")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg, args)


if __name__ == "__main__":
    main()
