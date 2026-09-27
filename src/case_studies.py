"""Case studies: individual clients chosen by the pre-registered rule in config
`case_studies:`, each written up with its population context and the one-field
counterfactual evidence that bears on it.

Layers per case (see the 2026-09-27 plan):
  1. behavior        -- profile, vignette text, option probabilities, expected vs actual
  2. counterfactual  -- the one-field pair effects for the fields this case turns on
  3. internal        -- probe readout of the client; PENDING until probes.py has run
  4. population      -- where the client sits in the residual distribution

Cases are picked by rule, never by eye, and every number is read from results files.

Outputs (results/<model>/case_studies/):
  counterfactual_effects.png   mean advice shift per one-field flip, with 95% CI
  pair_effects.json            the numbers behind that figure
  case_<archetype>.md          one write-up per archetype
  case_<archetype>.png         option probabilities: this client vs similar-risk clients
  README.md                    index of the cases and the selection rule

Usage:
    python src/case_studies.py [--config config.yaml] [--model ID]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from utils.paths import run_dir

RUBRIC_FIELDS = ["past_drawdown_reaction", "horizon_years", "investing_experience",
                 "income_stability", "stated_goal", "emergency_fund_months", "dependents", "age"]


# --------------------------------------------------------------------------------------
# Pure logic (no plotting) -- unit-tested in tests/test_case_studies.py
# --------------------------------------------------------------------------------------
def fit_residuals(rows):
    """Residual of aggressiveness on a linear fit of risk_score. Returns (residuals, (slope, b))."""
    x = np.array([r["risk_score"] for r in rows], float)
    y = np.array([r["aggressiveness"] for r in rows], float)
    slope, intercept = np.polyfit(x, y, 1)
    return y - (slope * x + intercept), (float(slope), float(intercept))


def matches(row, profile, flt):
    """A filter key is looked up on the profile first, then on the advice row."""
    for k, v in (flt or {}).items():
        have = profile.get(k, row.get(k)) if profile else row.get(k)
        if have != v:
            return False
    return True


def select_case(rows, profiles, resid, spec):
    """Index into `rows` chosen by one archetype spec. Deterministic: ties -> vignette_id."""
    cand = [i for i, r in enumerate(rows) if matches(r, profiles.get(r["profile_id"]), spec.get("filter"))]
    band = spec.get("min_abs_risk_from_50")
    if band is not None:
        cand = [i for i in cand if abs(rows[i]["risk_score"] - 50) >= band]
    if not cand:
        raise ValueError(f"no candidates for {spec}")
    pick = spec["pick"]
    if pick == "closest_to_group_mean":
        m = np.mean([rows[i]["aggressiveness"] for i in cand])
        key = lambda i: (abs(rows[i]["aggressiveness"] - m), rows[i]["vignette_id"])
    elif pick == "min_abs_residual":
        key = lambda i: (abs(resid[i]), rows[i]["vignette_id"])
    elif pick == "median_residual":
        med = float(np.median(resid[cand]))
        key = lambda i: (abs(resid[i] - med), rows[i]["vignette_id"])
    elif pick == "max_residual":
        key = lambda i: (-resid[i], rows[i]["vignette_id"])
    elif pick == "min_residual":
        key = lambda i: (resid[i], rows[i]["vignette_id"])
    else:
        raise ValueError(f"unknown pick {pick!r}")
    return min(cand, key=key)


def pair_shifts(pair_rows):
    """{pair_id: {"field", "factor", "lo", "hi", "shift"}} for complete lo/hi pairs.
    shift = hi aggressiveness - lo aggressiveness (hi always means 'supports more risk')."""
    by = defaultdict(dict)
    for r in pair_rows:
        side = r["vignette_id"].rsplit("_", 1)[-1]
        by[r["pair_id"]][side] = r
    out = {}
    for pid, d in sorted(by.items()):
        if "lo" in d and "hi" in d:
            out[pid] = {"field": d["lo"]["pair_field"], "factor": d["lo"].get("pair_factor"),
                        "lo": d["lo"], "hi": d["hi"],
                        "shift": d["hi"]["aggressiveness"] - d["lo"]["aggressiveness"],
                        "d_risk": d["hi"]["risk_score"] - d["lo"]["risk_score"]}
    return out


def bootstrap_mean_ci(x, n_boot=2000, seed=42):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def field_effects(shifts, n_boot=2000, seed=42):
    """Per-field summary of one-field pair shifts."""
    per = defaultdict(list)
    for p in shifts.values():
        per[p["field"]].append(p)
    out = {}
    for f, ps in per.items():
        s = [p["shift"] for p in ps]
        mean, lo, hi = bootstrap_mean_ci(s, n_boot, seed)
        out[f] = {"n": len(s), "mean_shift": round(mean, 4), "ci": [round(lo, 4), round(hi, 4)],
                  "mean_d_risk": round(float(np.mean([p["d_risk"] for p in ps])), 2),
                  "share_wrong_direction": round(float(np.mean(np.array(s) < 0)), 4),
                  "factor": ps[0]["factor"]}
    return out


def percentile_of(value, population):
    return float(100 * np.mean(np.asarray(population) <= value))


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------
NICE_FIELD = {"past_drawdown_reaction": "Past crash behavior", "horizon_years": "Time horizon",
              "investing_experience": "Investing experience", "income_stability": "Income stability",
              "stated_goal": "Stated goal", "emergency_fund_months": "Emergency fund (months)",
              "dependents": "Dependents", "age": "Age"}

WHY = {
    "say_vs_did_panic_seller": "Says they want maximum growth, but sold everything in the last crash. "
        "Their behavior is strong evidence of low willingness to bear losses.",
    "say_vs_did_cautious_buyer": "Says they want to protect capital, but bought more during the last crash. "
        "Their behavior suggests more risk tolerance than they state.",
    "best_clear_cut": "An unambiguous client (no say-vs-did or want-vs-afford conflict, risk far from the middle) "
        "whose advice is closest to what their risk implies. What the model gets right.",
    "median": "The client whose error sits at the middle of the distribution. The typical case.",
    "worst_over_aggressive": "The client the model over-recommends risk to by the largest margin.",
    "worst_over_conservative": "The client the model under-recommends risk to by the largest margin.",
}
RELEVANT = {"say_vs_did_panic_seller": ["stated_goal", "past_drawdown_reaction"],
            "say_vs_did_cautious_buyer": ["stated_goal", "past_drawdown_reaction"]}


def fmt_val(field, v):
    return str(v).replace("_", " ")


def option_labels(cfg):
    return cfg["advice_options"]["portfolio_choice"]


def case_markdown(name, spec, row, profile, text, resid_i, pred_i, resid_all, pool, profiles,
                  effects, similar, twin_row, cfg, shifts_note=None):
    labels = option_labels(cfg)
    L = [f"# Case: {name.replace('_', ' ')}", "",
         f"**Why this case:** {WHY.get(name, '')}", "",
         f"**Selection rule (pre-registered):** `{json.dumps(spec)}`  ",
         f"**Vignette:** `{row['vignette_id']}` (implicit)  ", ""]
    # layer 1: behavior
    L += ["## 1. The client", "", "| Detail | Value |", "|---|---|"]
    for f in RUBRIC_FIELDS:
        L.append(f"| {NICE_FIELD[f]} | {fmt_val(f, profile[f])} |")
    L += [f"| Risk score (rubric, 0-100) | {row['risk_score']:.1f} ({row['tier']}) |",
          f"| Willingness / capacity / goals | {row['willingness']:.2f} / {row['capacity']:.2f} / {row['goals']:.2f} ({row['factor_cell']}) |",
          f"| Say-vs-did contradiction | {'yes' if row['contradictory'] else 'no'} |",
          f"| Want-vs-afford conflict | {'yes' if row['conflicted'] else 'no'} |", "",
          "**What the model read:**", "", "> " + text.replace("\n", "\n> "), ""]
    L += ["## 2. What the model advised", "", "| Portfolio | This client | Similar-risk clients (±{}) |".format(cfg["case_studies"]["similar_risk_band"]), "|---|---|---|"]
    sim_p = np.mean([r["p_alloc"] for r in similar], axis=0)
    for k, lab in enumerate(labels):
        L.append(f"| {lab} | {row['p_alloc'][k]:.2f} | {sim_p[k]:.2f} |")
    sim_mean = float(np.mean([r["aggressiveness"] for r in similar]))
    L += ["", f"- **Advice riskiness (share in stocks): {row['aggressiveness']:.3f}**. "
              f"Similar-risk clients average {sim_mean:.3f} (n={len(similar)}).",
          f"- Expected from their risk score: {pred_i:.3f}. Residual **{resid_i:+.3f}** "
          f"({'more aggressive' if resid_i > 0 else 'more conservative'} than their risk implies).",
          f"- Residual percentile in the pool: **{percentile_of(resid_i, resid_all):.1f}** "
          f"(pool n={len(resid_all)}; residual range {resid_all.min():+.2f} to {resid_all.max():+.2f})."]
    if twin_row is not None:
        L.append(f"- Same client written with plain facts (explicit twin): advice {twin_row['aggressiveness']:.3f}.")
    L.append("")
    # layer 2: population/group context for say-vs-did
    if name.startswith("say_vs_did"):
        goal = profile["stated_goal"]
        L += ["## 3. Everyone who says the same thing", "",
              f"Mean advice for all implicit clients whose stated goal is **{fmt_val('', goal)}**, by what they did in the last crash:", "",
              "| Crash behavior | Mean advice | n |", "|---|---|---|"]
        for d in ["sold_everything", "reduced", "held", "bought_more"]:
            g = [r["aggressiveness"] for r in pool if profiles[r["profile_id"]]["stated_goal"] == goal
                 and profiles[r["profile_id"]]["past_drawdown_reaction"] == d]
            if g:
                L.append(f"| {fmt_val('', d)} | {np.mean(g):.3f} | {len(g)} |")
        other = "capital_preservation" if goal == "maximum_growth" else "maximum_growth"
        g = [r["aggressiveness"] for r in pool if profiles[r["profile_id"]]["stated_goal"] == other
             and profiles[r["profile_id"]]["past_drawdown_reaction"] == profile["past_drawdown_reaction"]]
        L += ["", f"Clients who did the **same** thing in the crash ({fmt_val('', profile['past_drawdown_reaction'])}) "
                  f"but say **{fmt_val('', other)}** average {np.mean(g):.3f} (n={len(g)}).", ""]
    # counterfactual layer
    fields = RELEVANT.get(name, RUBRIC_FIELDS)
    L += ["## 4. Counterfactual evidence (one-field pairs)", "",
          "Each pair is two clients identical in every detail except one. Shift = advice(higher-risk value) - advice(lower-risk value).", "",
          "| Field flipped | Mean advice shift [95% CI] | Rubric Δrisk | Pairs moving the wrong way | n |", "|---|---|---|---|---|"]
    for f in sorted(fields, key=lambda f: -effects[f]["mean_shift"]):
        e = effects[f]
        L.append(f"| {NICE_FIELD[f]} | {e['mean_shift']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}] | "
                 f"{e['mean_d_risk']:.0f} | {100 * e['share_wrong_direction']:.0f}% | {e['n']} |")
    L.append("")
    if shifts_note:
        L += [shifts_note, ""]
    L += ["## 5. Internal readout (probe)", "",
          "_Pending: filled in after `probes.py` and `dissociation.py` run. Question: at the best layer, "
          "does the probe decode this client's willingness and capacity correctly, even where the advice ignores them?_", ""]
    return "\n".join(L)


def case_figure(path, name, row, similar, cfg):
    import matplotlib.pyplot as plt
    import behavior_figures as bf                           # shared plot style
    labels = [l.replace(", ", "\n") for l in option_labels(cfg)]
    sim_p = np.mean([r["p_alloc"] for r in similar], axis=0)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, row["p_alloc"], w, color=bf.BLUE, label="This client")
    ax.bar(x + w / 2 + 0.01, sim_p, w, color=bf.MUTED, label=f"Similar-risk clients (±{cfg['case_studies']['similar_risk_band']}, n={len(similar)})")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability of recommending")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(name.replace("_", " ").capitalize(), pad=22)
    bf.subtitle(ax, f"Advice {row['aggressiveness']:.2f} vs {np.mean([r['aggressiveness'] for r in similar]):.2f} "
                    f"for clients with the same risk score")
    bf.save(fig, path)


def effects_figure(path, effects):
    import matplotlib.pyplot as plt
    import behavior_figures as bf
    fields = sorted(effects, key=lambda f: effects[f]["mean_shift"])
    y = np.arange(len(fields))
    m = np.array([effects[f]["mean_shift"] for f in fields])
    lo = m - np.array([effects[f]["ci"][0] for f in fields])
    hi = np.array([effects[f]["ci"][1] for f in fields]) - m
    colors = [bf.ORANGE if f == "stated_goal" else bf.BLUE for f in fields]
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.barh(y, m, color=colors, height=0.6, xerr=[lo, hi], error_kw={"elinewidth": 1.2, "ecolor": bf.INK2})
    for i, f in enumerate(fields):
        ax.text(m[i] + hi[i] + 0.01, i, f"{m[i]:+.2f}", va="center", fontsize=9.5, color=bf.INK)
    ax.set_yticks(y, [f"{NICE_FIELD[f]}  (rubric Δ{effects[f]['mean_d_risk']:.0f})" for f in fields])
    ax.axvline(0, color=bf.INK2, lw=1)
    ax.set_xlim(min(-0.05, (m - lo).min() - 0.02), (m + hi).max() + 0.08)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Change in advice when ONLY this detail flips (low → high risk)")
    ax.set_title("Only the stated goal really moves the advice", pad=22)
    bf.subtitle(ax, f"One-field pairs, n={effects[fields[0]]['n']} per field. 95% bootstrap CI.")
    bf.save(fig, path)


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def run(cfg, args):
    cs = cfg["case_studies"]
    res = run_dir(cfg, args.model or cfg["model"]["primary"])
    out = res / "case_studies"
    out.mkdir(exist_ok=True)

    rows = load_jsonl(res / "advice.jsonl")
    profiles = {p["profile_id"]: p for p in load_jsonl(Path(cfg["paths"]["data_dir"]) / "profiles.jsonl")}
    texts = {}
    for name in ("implicit", "explicit", "pairs"):
        for v in load_jsonl(Path(cfg["paths"]["vignettes_dir"]) / f"{name}.jsonl"):
            texts[v["vignette_id"]] = v["text"]

    base = [r for r in rows if r["condition"] == "baseline"]
    pool = sorted([r for r in base if r.get("pair_id") is None and r["vignette_type"] == "implicit"],
                  key=lambda r: r["vignette_id"])
    explicit = {r["profile_id"]: r for r in base if r.get("pair_id") is None and r["vignette_type"] == "explicit"}
    resid, (slope, intercept) = fit_residuals(pool)

    shifts = pair_shifts([r for r in base if r.get("pair_id")])
    effects = field_effects(shifts, cs["n_boot"], cfg["seed"])
    (out / "pair_effects.json").write_text(json.dumps(effects, indent=2, sort_keys=True))
    effects_figure(out / "counterfactual_effects.png", effects)

    band = cs["similar_risk_band"]
    index = []
    for name, spec in cs["archetypes"].items():
        i = select_case(pool, profiles, resid, spec)
        row = pool[i]
        similar = [r for r in pool if abs(r["risk_score"] - row["risk_score"]) <= band]
        md = case_markdown(name, spec, row, profiles[row["profile_id"]], texts[row["vignette_id"]],
                           float(resid[i]), float(slope * row["risk_score"] + intercept), resid, pool,
                           profiles, effects, similar, explicit.get(row["profile_id"]), cfg)
        (out / f"case_{name}.md").write_text(md)
        case_figure(out / f"case_{name}.png", name, row, similar, cfg)
        index.append((name, row["vignette_id"], row["aggressiveness"], float(resid[i])))

    # capacity-ignored: the pair where more capacity most LOWERED advice
    ci = cs["capacity_ignored"]
    cand = [p for p in shifts.values() if p["field"] in ci["fields"]]
    worst = min(cand, key=lambda p: (p["shift"], p["lo"]["pair_id"]))
    from vignettes import field_extremes
    lo_v, hi_v = field_extremes(cfg["rubric"]["fields"][worst["field"]]) if "rubric" in cfg else (None, None)
    L = ["# Case: capacity ignored", "",
         "**Why this case:** a one-field pair where giving the client MORE capacity to bear risk "
         "made the advice LESS aggressive, the largest such reversal among the capacity fields.", "",
         f"**Selection rule (pre-registered):** `{json.dumps(ci)}`  ",
         f"**Pair:** `{worst['lo']['pair_id']}`, field flipped: **{NICE_FIELD[worst['field']]}**"
         + (f" ({fmt_val('', lo_v)} → {fmt_val('', hi_v)})" if lo_v is not None else ""), "",
         "| | Lower-capacity side | Higher-capacity side |", "|---|---|---|",
         f"| Risk score | {worst['lo']['risk_score']:.1f} | {worst['hi']['risk_score']:.1f} |",
         f"| Advice riskiness | {worst['lo']['aggressiveness']:.3f} | {worst['hi']['aggressiveness']:.3f} |",
         "", f"Shift: **{worst['shift']:+.3f}** (should be ≥ 0).", "",
         "**Lower-capacity side:**", "", "> " + texts[worst["lo"]["vignette_id"]], "",
         "**Higher-capacity side:**", "", "> " + texts[worst["hi"]["vignette_id"]], "",
         "## Population context", "", "| Field | Mean shift [95% CI] | Pairs moving the wrong way |", "|---|---|---|"]
    for f in ci["fields"]:
        e = effects[f]
        L.append(f"| {NICE_FIELD[f]} | {e['mean_shift']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}] | {100 * e['share_wrong_direction']:.0f}% |")
    (out / "case_capacity_ignored.md").write_text("\n".join(L) + "\n")
    index.append(("capacity_ignored", worst["lo"]["pair_id"], worst["hi"]["aggressiveness"], worst["shift"]))

    readme = ["# Case studies", "",
              "Chosen by the pre-registered rule in `config.yaml` (`case_studies:`), never by eye. "
              f"Pool: {len(pool)} implicit clients. Residual = advice − ({slope:.4f}·risk_score + {intercept:.3f}).", "",
              "| Case | Vignette / pair | Advice | Residual (or pair shift) |", "|---|---|---|---|"]
    readme += [f"| [{n}](case_{n}.md) | `{v}` | {a:.3f} | {r:+.3f} |" for n, v, a, r in index]
    readme += ["", "Headline counterfactual figure: `counterfactual_effects.png` (numbers in `pair_effects.json`)."]
    (out / "README.md").write_text("\n".join(readme) + "\n")
    print(f"wrote {len(index)} cases -> {out}")
    for n, v, a, r in index:
        print(f"  {n:28s} {v:22s} advice {a:.3f}  resid/shift {r:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None, help="override config model.primary")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg, args)


if __name__ == "__main__":
    main()
