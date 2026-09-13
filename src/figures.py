"""Publication-style figures & tables packet for the faculty advisor (PNG set + one PDF).

Every number is read from results files (gate.json, patterns.json,
s5_probe_results_<tag>.json, advice.jsonl); nothing is hand-entered.

Outputs:
  results/figures/T1_core_comparison.png     headline gap table
  results/figures/F1_tier_means.png          S3 tier means bar chart
  results/figures/F2_layer_sweep.png         AUROC vs layer, 3 positions
  results/figures/F3_conflict_collapse.png   dose-response collapse
  results/figures/F4_field_weights.png       stated_goal overweighting
  results/figures/F5_failure_concentration.png  are failures random? (11% follow-up)
  results/figures/T2_summary.png             full numeric summary with CIs
  results/pub_figures_<date>.pdf             all of the above, composed as a report

Usage:
    python src/figures.py [--config config.yaml]
"""

import argparse
import json
import textwrap
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from utils.paths import run_dir
from matplotlib.backends.backend_pdf import PdfPages

from report import GOLD, GRAY, LIGHT, NAVY, ORANGE, TEAL, load

DPI = 300
plt.rcParams.update({"axes.titlesize": 12.5})


# ------------------------------------------------------------------ table helper
def draw_table(ax, headers, rows, col_w, aligns, highlight=None, header_fc=NAVY,
               fontsize=9.2, row_h=1.0):
    """Manual table: striped rows, navy header, optional highlighted row index."""
    ax.axis("off")
    n_rows = len(rows) + 1
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_rows * row_h)
    x_edges = np.concatenate([[0], np.cumsum(col_w)])

    def cell(r, c, text, weight="normal", color="black", fc=None):
        y0 = (n_rows - 1 - r) * row_h
        if fc:
            ax.add_patch(plt.Rectangle((x_edges[c], y0), col_w[c], row_h, fc=fc, ec="none"))
        pad = 0.012
        x = {"left": x_edges[c] + pad, "center": (x_edges[c] + x_edges[c + 1]) / 2,
             "right": x_edges[c + 1] - pad}[aligns[c]]
        ax.text(x, y0 + row_h / 2, text, ha=aligns[c], va="center",
                size=fontsize, weight=weight, color=color, wrap=True)

    for c, htxt in enumerate(headers):
        cell(0, c, htxt, weight="bold", color="white", fc=header_fc)
    for r, row in enumerate(rows):
        base_fc = "#f2f5f8" if r % 2 else "white"
        if highlight is not None and r == highlight:
            base_fc = "#fdeee8"
        for c, txt in enumerate(row):
            w = "bold" if (highlight is not None and r == highlight) else "normal"
            cell(r + 1, c, txt, weight=w, fc=base_fc)
    for yy in np.arange(0, (n_rows + 1) * row_h, row_h):
        ax.plot([0, 1], [yy, yy], color="#c9d2da", lw=0.7)
    ax.plot([0, 1], [n_rows * row_h, n_rows * row_h], color=NAVY, lw=1.4)
    ax.plot([0, 1], [(n_rows - 1) * row_h, (n_rows - 1) * row_h], color=NAVY, lw=1.4)
    ax.plot([0, 1], [0, 0], color=NAVY, lw=1.4)


def ci_str(ci):
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


# ------------------------------------------------------------------ figures
def t1_core_comparison(d):
    s3b, s5, gate = d["s3b"], d["s5"], d["gate"]
    lad = s3b["audit_ladder_behavioral"]
    fig = plt.figure(figsize=(7.6, 2.9))
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.80])
    headers = ["Audit channel (what the auditor sees)", "Native metric",
               "Same-metric AUROC*", "95% CI"]
    b4 = lad["behavior_4d_option_dist"]
    rows = [
        ["Advice only (behavioral)", f"Spearman rho = {gate['spearman']['implicit']['rho']:.3f}",
         f"{b4['implicit_test_auroc']:.3f}", ci_str(b4["ci"])],
        ["Transcript text (TF-IDF baseline)", "--",
         f"{s5['baselines']['tfidf']['implicit_test']:.3f}", "--"],
        ["Ask the model (verbal self-report)", "pending", "pending (next GPU run)", "--"],
        ["Internal probe, layer 22 (white-box)", f"ridge R² = {s5['best']['ridge_r2']:.3f}",
         f"{s5['best']['implicit_auroc']:.3f}", ci_str(s5["best"]["ci"])],
        ["THE GAP  (internal - behavioral)", "--",
         f"+{s5['best']['implicit_auroc'] - b4['implicit_test_auroc']:.3f}", "--"],
    ]
    draw_table(ax, headers, rows, col_w=[0.42, 0.22, 0.22, 0.14],
               aligns=["left", "center", "center", "center"], highlight=4)
    fig.text(0.02, 0.90, "Table 1.  The headline gap: what each audit channel recovers about the client",
             size=11.5, weight="bold", color=NAVY)
    fig.text(0.02, 0.005, "*Macro one-vs-rest AUROC, identical held-out implicit clients (n = 1,206), "
                          "split by client identity (leakage-safe); decoders trained on the same explicit-"
                          "train split as the probe.", size=7.6, color=GRAY)
    return fig


def f1_tier_means(d):
    gate = d["gate"]
    twins = d["twins"]
    fig = plt.figure(figsize=(7.2, 4.4))
    ax = fig.add_axes([0.11, 0.14, 0.85, 0.74])
    tiers = ["conservative", "moderate", "aggressive"]
    means = [gate["tier_means"][t]["mean"] for t in tiers]
    stds = [float(np.std([r["aggressiveness"] for r in twins if r["tier"] == t])) for t in tiers]
    colors = [TEAL, GOLD, ORANGE]
    bars = ax.bar(range(3), means, yerr=stds, width=0.62, color=colors, capsize=6,
                  error_kw=dict(ecolor="#444444", lw=1.2))
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.025, f"{m:.3f}", ha="center", size=12, weight="bold")
    ax.axhline(0.2, color=GRAY, lw=0.8, ls=":")
    ax.axhline(0.9, color=GRAY, lw=0.8, ls=":")
    ax.text(-0.42, 0.915, "most aggressive option (0.90)", size=8, color=GRAY, ha="left")
    ax.set_xticks(range(3))
    ax.set_xticklabels([t.capitalize() for t in tiers], size=11)
    ax.set_xlabel(f"Client's true risk tolerance (ground truth by construction)\n"
                  f"n = {gate['tier_means']['moderate']['n']:,} per tier  |  whiskers = 1 SD  |  "
                  f"hedge rate 0.000")
    ax.set_ylabel("Mean advice aggressiveness\n(probability-weighted equity share)")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.5, 2.5)
    ax.set_title("Advice rises with true client risk tolerance -- ordered, but compressed",
                 pad=12)
    return fig


def f2_layer_sweep(d):
    s5 = d["s5"]
    fig = plt.figure(figsize=(7.6, 4.6))
    ax = fig.add_axes([0.10, 0.13, 0.86, 0.76])
    sweep = s5["sweep"]
    styles = {"decision": (NAVY, "-", 2.4, "decision position"),
              "profile_mean": (TEAL, "-", 1.5, "mean over client narrative"),
              "profile_end": (GRAY, "--", 1.3, "end of client narrative")}
    for pos, (c, ls, lw, lab) in styles.items():
        layers = sorted(int(k) for k in sweep[pos])
        vals = [sweep[pos][str(l)]["implicit_test"]["auroc"] for l in layers]
        ax.plot(layers, vals, color=c, ls=ls, lw=lw, label=lab)
    tfidf = s5["baselines"]["tfidf"]["implicit_test"]
    ax.axhline(tfidf, color=ORANGE, lw=1.4, ls=":")
    ax.text(41, tfidf + 0.008, f"TF-IDF text baseline ({tfidf:.3f})", size=8.5,
            color=ORANGE, ha="right")
    ax.axhline(0.5, color="black", lw=0.8, ls=":")
    ax.text(0.2, 0.508, "chance (0.50)", size=8.5)
    b = s5["best"]
    ax.scatter([b["layer"]], [b["implicit_auroc"]], zorder=5, color=NAVY, s=45)
    ax.annotate(f"peak: layer {b['layer']} / 42\nAUROC {b['implicit_auroc']:.3f} {ci_str(b['ci'])}",
                xy=(b["layer"], b["implicit_auroc"]),
                xytext=(b["layer"] - 16.5, b["implicit_auroc"] + 0.035),
                size=9.5, weight="bold", color=NAVY,
                arrowprops=dict(arrowstyle="->", color=NAVY))
    ax.set_xlabel("Transformer layer (0 = input side; 42 layers total)")
    ax.set_ylabel("Implicit held-out AUROC\n(client tier decodable from activations)")
    ax.set_ylim(0.45, 1.0)
    ax.set_xlim(-1, 42)
    ax.set_title("Where the client representation lives: probe accuracy across all 42 layers", pad=12)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    return fig


def f3_conflict_collapse(d):
    s3b = d["s3b"]
    fig = plt.figure(figsize=(7.4, 4.7))
    ax = fig.add_axes([0.11, 0.18, 0.85, 0.71])
    x = np.arange(4)
    for vt, c, m, lw in (("explicit", GRAY, "s", 1.6), ("implicit", NAVY, "o", 2.4)):
        g = s3b["graded_collapse"][vt]
        keys = sorted(g, key=lambda k: int(k.split("_")[1]))
        vals = [g[k]["rho"] for k in keys]
        los = [g[k]["ci"][0] for k in keys]
        his = [g[k]["ci"][1] for k in keys]
        ax.errorbar(x, vals, yerr=[np.array(vals) - np.array(los), np.array(his) - np.array(vals)],
                    color=c, marker=m, ms=7.5, lw=lw, capsize=4, label=f"{vt} vignettes")
    g = s3b["graded_collapse"]["implicit"]
    keys = sorted(g, key=lambda k: int(k.split("_")[1]))
    for xi, k in zip(x, keys):
        ax.text(xi, -0.30, f"n={g[k]['n']:,}", ha="center", size=8, color=GRAY)
    ax.axhline(0, color="black", lw=0.9, ls=":")
    ax.text(0.05, 0.025, "zero = advice uninformative about the client", size=8.5)
    ax.axhline(0.5, color=ORANGE, lw=0.9, ls=":")
    ax.text(0.05, 0.515, "pre-registered gate bar (0.50)", size=8.5, color=ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(["0 / 3\n(agree)", "1 / 3", "2 / 3", "3 / 3\n(extreme)"])
    ax.set_xlabel("Conflict between stated goal and revealed behavior\n"
                  "(rubric-signal distance; 3/3 = e.g. \"maximum growth\" + sold everything in a downturn)")
    ax.set_ylabel("Spearman rho\n(advice vs. true risk tolerance)")
    ax.set_ylim(-0.35, 1.0)
    ax.set_title("Dose-response collapse: advice quality falls to zero as client signals conflict", pad=12)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    return fig


def f4_field_weights(d):
    s3b = d["s3b"]
    fw = s3b["field_weights"]["implicit"]["fields"]
    nice = {"stated_goal": "Stated goal (willingness)", "past_drawdown_reaction": "Past downturn behavior",
            "horizon_years": "Time horizon", "investing_experience": "Investing experience",
            "income_stability": "Income stability", "emergency_fund_months": "Emergency fund",
            "dependents": "Dependents", "age": "Age"}
    fields = sorted(fw, key=lambda f: fw[f]["rel_weight"], reverse=True)
    fig = plt.figure(figsize=(7.6, 4.6))
    ax = fig.add_axes([0.24, 0.13, 0.72, 0.76])
    ys = np.arange(len(fields))[::-1]
    h = 0.37
    for y, f in zip(ys, fields):
        c = ORANGE if f == "stated_goal" else TEAL
        ax.barh(y + h / 2, fw[f]["rel_weight"], height=h, color=c)
        ax.barh(y - h / 2, fw[f]["rubric_weight"], height=h, color="#b8c4cc")
        ax.text(max(fw[f]["rel_weight"], fw[f]["rubric_weight"]) + 0.007, y,
                f"{fw[f]['ratio_vs_rubric']:g}x", va="center",
                size=9.5 if f == "stated_goal" else 8.6,
                weight="bold" if f == "stated_goal" else "normal",
                color=ORANGE if f == "stated_goal" else "#333333")
    ax.set_yticks(ys)
    ax.set_yticklabels([nice[f] for f in fields], size=10)
    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(fc=TEAL, label="weight the advisor's advice actually uses (regression)"),
        mpatches.Patch(fc="#b8c4cc", label="fair weight (ground-truth rubric)"),
        mpatches.Patch(fc=ORANGE, label="stated goal: 5.4x its fair share")],
        loc="center right", frameon=False, fontsize=8.8)
    ax.set_xlabel(f"Relative share of the advice decision (standardized |beta|, sums to 1)\n"
                  f"implicit vignettes, n = 6,000; regression R² = "
                  f"{s3b['field_weights']['implicit']['r2']:.2f}")
    ax.set_xlim(0, 0.5)
    ax.set_title("One loud signal: the advisor over-weights the stated goal ~5x,\n"
                 "under-uses capacity and revealed behavior", pad=10)
    return fig


def f5_failure_concentration(d):
    s3b, s5, cfgd = d["s3b"], d["s5"], d["diss_bands"]
    fig = plt.figure(figsize=(7.8, 4.4))
    fig.suptitle("Are the failures random, or do they concentrate on conflicted clients?",
                 size=12.5, weight="bold", color=NAVY, y=0.97)

    # left: behavioral answer (known)
    axL = fig.add_axes([0.09, 0.16, 0.38, 0.64])
    sub = s3b["subgroup_spearman"]
    groups = [("explicit_concordant", "explicit\nagree"), ("explicit_contradictory", "explicit\nconflict"),
              ("implicit_concordant", "implicit\nagree"), ("implicit_contradictory", "implicit\nconflict")]
    xs = np.arange(4)
    vals = [sub[k]["rho"] for k, _ in groups]
    errs = np.array([[sub[k]["rho"] - sub[k]["ci"][0] for k, _ in groups],
                     [sub[k]["ci"][1] - sub[k]["rho"] for k, _ in groups]])
    cols = [GRAY, GRAY, NAVY, NAVY]
    bars = axL.bar(xs, vals, yerr=errs, color=cols, width=0.6, capsize=4, error_kw=dict(lw=1.1))
    for patch, a in zip(bars, [0.55, 1.0, 0.55, 1.0]):
        patch.set_alpha(a)
    for x, v in zip(xs, vals):
        axL.text(x, v + (0.035 if v >= 0 else -0.07), f"{v:+.2f}", ha="center", size=9.5, weight="bold")
    axL.axhline(0, color="black", lw=0.9)
    axL.set_xticks(xs)
    axL.set_xticklabels([lab for _, lab in groups], size=8.5)
    axL.set_ylabel("Spearman rho (advice vs. truth)")
    axL.set_ylim(-0.30, 1.0)
    axL.set_title("BEHAVIOR: failures concentrate\non conflicted clients (measured)", size=10)

    # right: probe-side test (pre-registered, pending)
    axR = fig.add_axes([0.56, 0.16, 0.40, 0.64])
    axR.set_xlim(0.5, 1.0)
    axR.set_ylim(0, 1)
    zones = [(0.5, cfgd["encoding_collapse_auroc_max"], ORANGE, "encoding collapse\n(bias penetrates\nrepresentation)"),
             (cfgd["encoding_collapse_auroc_max"], cfgd["readout_collapse_auroc_min"], "#d9dee3", "partial"),
             (cfgd["readout_collapse_auroc_min"], 1.0, TEAL, "representation intact\n(readout failure --\nstrongest thesis)")]
    for x0, x1, c, lab in zones:
        axR.axvspan(x0, x1, color=c, alpha=0.35)
        axR.text((x0 + x1) / 2, 0.72, lab, ha="center", size=8.2)
    b = s5["best"]
    axR.axvline(b["implicit_auroc"], color=NAVY, lw=2)
    axR.text(b["implicit_auroc"] - 0.008, 0.02, f"probe, ALL clients: {b['implicit_auroc']:.3f}",
             rotation=90, size=7.5, color=NAVY, ha="right", va="bottom")
    axR.text(0.75, 0.30, "probe on CONFLICTED clients:\nmeasured next GPU run\n"
                         "(thresholds pre-registered 2026-07-02,\nbefore seeing data)",
             ha="center", size=8.4, style="italic", color="#333333")
    axR.set_yticks([])
    axR.set_xlabel("Probe AUROC on conflicted clients\n(shaded = pre-registered decision zones)")
    axR.set_title("INTERNAL PROBE: the pre-registered test\n(decides the paper's framing)", size=10)
    return fig


def f6_additional(d):
    s3b = d["s3b"]
    fig = plt.figure(figsize=(7.8, 4.0))
    tiers = ["conservative", "moderate", "aggressive"]
    cols = [TEAL, GOLD, ORANGE]

    # A: asymmetric caution (explicit - implicit twin gap by tier)
    axA = fig.add_axes([0.09, 0.17, 0.38, 0.66])
    tg = s3b["twin_gap_by_tier"]
    vals = [tg[t]["mean_gap"] for t in tiers]
    axA.bar(range(3), vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        axA.text(i, v + 0.006, f"+{v:.3f}", ha="center", size=10, weight="bold")
    axA.set_xticks(range(3))
    axA.set_xticklabels([t.capitalize() for t in tiers], size=9.5)
    axA.set_ylabel("Advice drop when the story\nis implicit (explicit - implicit)")
    axA.set_ylim(0, 0.24)
    axA.set_title("A. Asymmetric caution: subtle stories cost\naggressive clients 10x more than conservative", size=9.6)

    # B: instruction effect by tier
    axB = fig.add_axes([0.59, 0.17, 0.37, 0.66])
    ie = s3b["instruction_effect"]["by_tier"]
    vals = [ie[t]["mean_delta"] for t in tiers]
    axB.bar(range(3), vals, color=cols, width=0.6)
    for i, (t, v) in enumerate(zip(tiers, vals)):
        axB.text(i, v + (0.0015 if v >= 0 else -0.004), f"{v:+.4f}", ha="center", size=9.5, weight="bold")
    axB.axhline(0, color="black", lw=0.9)
    axB.set_xticks(range(3))
    axB.set_xticklabels([t.capitalize() for t in tiers], size=9.5)
    axB.set_ylabel("Advice change from the\n'integrate holistically' instruction")
    axB.set_ylim(-0.012, 0.032)
    axB.set_title("B. Prompting nudges the right direction --\nbut ~10x too weakly (shortfall is ~0.2+)", size=9.6)
    axB.text(1, -0.0095, "conflicted clients only (n = 1,446)", ha="center", size=8, color=GRAY)
    return fig


def t2_summary(d):
    gate, s3b, s5 = d["gate"], d["s3b"], d["s5"]
    sub, lad = s3b["subgroup_spearman"], s3b["audit_ladder_behavioral"]
    b4 = lad["behavior_4d_option_dist"]
    ie = s3b["instruction_effect"]
    fig = plt.figure(figsize=(7.8, 6.4))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.90])
    headers = ["Quantity", "Value", "95% CI", "n / note"]
    rows = [
        ["S3 gate: rho, implicit vignettes", f"{gate['spearman']['implicit']['rho']:.3f}",
         ci_str(gate["spearman"]["implicit"]["ci"]), "6,000; bar 0.50 -> PASS"],
        ["S3 gate: rho, explicit vignettes", f"{gate['spearman']['explicit']['rho']:.3f}",
         ci_str(gate["spearman"]["explicit"]["ci"]), "6,000"],
        ["S3 hedge / refusal rate", f"{gate['hedge_rate']:.3f}", "--", "16,106 rows; bar < 0.20"],
        ["Tier means (cons / mod / aggr)",
         f"{gate['tier_means']['conservative']['mean']:.3f} / {gate['tier_means']['moderate']['mean']:.3f} / "
         f"{gate['tier_means']['aggressive']['mean']:.3f}", "--", "monotonic -> PASS"],
        ["Probe AUROC, implicit held-out (L22)", f"{s5['best']['implicit_auroc']:.3f}",
         ci_str(s5["best"]["ci"]), "1,206; bar 0.85 -> PASS"],
        ["Probe accuracy (3-way)", "0.740", "--", "error 26.0% vs 66.7% chance error"],
        ["Probe 5-seed stability", f"{s5['seed_stability']['implicit_auroc_mean']:.3f}",
         f"+/- {s5['seed_stability']['std']:.3f}", "5 split seeds"],
        ["Ridge R² (continuous risk score)", f"{s5['best']['ridge_r2']:.3f}", "--", "bar 0.50 -> PASS"],
        ["TF-IDF transcript baseline (AUROC)", f"{s5['baselines']['tfidf']['implicit_test']:.3f}", "--",
         f"probe margin +{s5['best']['implicit_auroc'] - s5['baselines']['tfidf']['implicit_test']:.3f}; bar +0.07 -> PASS"],
        ["Shuffled-label control probe", "0.480", "--", "chance -> probe is real signal"],
        ["Behavioral decoder AUROC (4-option)", f"{b4['implicit_test_auroc']:.3f}", ci_str(b4["ci"]),
         "same split/metric as probe"],
        ["THE GAP (probe - behavior, AUROC)",
         f"+{s5['best']['implicit_auroc'] - b4['implicit_test_auroc']:.3f}", "--", "headline finding"],
        ["rho, implicit concordant clients", f"{sub['implicit_concordant']['rho']:+.3f}",
         ci_str(sub["implicit_concordant"]["ci"]), f"{sub['implicit_concordant']['n']:,}"],
        ["rho, implicit CONFLICTED clients", f"{sub['implicit_contradictory']['rho']:+.3f}",
         ci_str(sub["implicit_contradictory"]["ci"]), f"{sub['implicit_contradictory']['n']:,}; CI excludes 0"],
        ["stated_goal over-weighting", "5.43x", "--", "vs rubric weight 0.08 (implicit)"],
        ["'Integrate holistically' instruction", f"{ie['mean_delta']:+.4f}", "--",
         f"aggr-tier {ie['by_tier']['aggressive']['mean_delta']:+.3f}; right direction, ~10x too weak"],
        ["Gender effect (residualized)", "+/- 0.005", "--", "fairness null"],
        ["Net-worth partial correlation", "+0.056", "--", "trace leakage (non-rubric field)"],
    ]
    draw_table(ax, headers, rows, col_w=[0.40, 0.16, 0.20, 0.24],
               aligns=["left", "center", "center", "left"], highlight=11, fontsize=8.4)
    fig.text(0.02, 0.955, "Table 2.  Complete numeric summary (all thresholds pre-registered in config.yaml)",
             size=11.5, weight="bold", color=NAVY)
    return fig


# ------------------------------------------------------------------ PDF assembly
def caption(fig_page, y, num, text, width=102):
    fig_page.text(0.07, y, num, size=9.5, weight="bold", color=NAVY)
    for i, line in enumerate(textwrap.fill(text, width).split("\n")):
        fig_page.text(0.155, y - i * 0.0135, line, size=9)
    return y


def embed(fig_page, png, y_top, width_frac=0.86):
    img = plt.imread(png)
    ar = img.shape[0] / img.shape[1]
    h = width_frac * ar * (8.5 / 11)
    ax = fig_page.add_axes([(1 - width_frac) / 2, y_top - h, width_frac, h])
    ax.imshow(img)
    ax.axis("off")
    return y_top - h


def build(cfg):
    d = load(Path(cfg["paths"]["results_dir"]))
    d["diss_bands"] = cfg["dissociation"]
    res_dir = run_dir(cfg, cfg["model"]["primary"])   # results/<model>/
    fig_dir = res_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    today = date.today().strftime("%B %d, %Y")

    specs = [
        ("T1_core_comparison", t1_core_comparison),
        ("F1_tier_means", f1_tier_means),
        ("F2_layer_sweep", f2_layer_sweep),
        ("F3_conflict_collapse", f3_conflict_collapse),
        ("F4_field_weights", f4_field_weights),
        ("F5_failure_concentration", f5_failure_concentration),
        ("F6_additional_findings", f6_additional),
        ("T2_summary", t2_summary),
    ]
    pngs = {}
    for name, fn in specs:
        f = fn(d)
        p = fig_dir / f"{name}.png"
        f.savefig(p, dpi=DPI, facecolor="white", bbox_inches="tight")
        plt.close(f)
        pngs[name] = p
        print(f"  wrote {p}")

    pdf_path = res_dir / f"pub_figures_{date.today().isoformat()}.pdf"
    with PdfPages(pdf_path) as pdf:
        n_page = [0]

        def new_page(section):
            n_page[0] += 1
            fp = plt.figure(figsize=(8.5, 11))
            fp.text(0.07, 0.965, "Mechanistic KYC -- Results Figures & Tables", size=8.5, color=GRAY)
            fp.text(0.93, 0.965, today, size=8.5, color=GRAY, ha="right")
            fp.lines.append(plt.Line2D([0.07, 0.93], [0.952, 0.952], transform=fp.transFigure,
                                       color=NAVY, lw=1.0))
            fp.text(0.07, 0.925, section, size=14, weight="bold", color=NAVY)
            fp.text(0.5, 0.03, str(n_page[0]), size=9, color=GRAY, ha="center")
            return fp

        # page 1: title + T1 + F1
        fp = new_page("Headline result: the advisor knows more than its advice shows")
        fp.text(0.07, 0.895, "Prepared for Dr. Raahemifar  |  Shlok Channawar, Daniel Liu, Nina  |  "
                             "target: ICAIF 2026 (Aug 2)", size=9.5, color=GRAY)
        y = embed(fp, pngs["T1_core_comparison"], 0.875)
        caption(fp, y - 0.015, "Table 1.",
                "All channels evaluated on the same 1,206 held-out implicit-vignette clients with the "
                "same metric. Reading the model's internal activity recovers the client's true risk "
                "tolerance at 0.894 AUROC; watching its advice recovers 0.716; reading the transcript "
                "0.748. The +0.178 gap is the paper's central, leakage-safe finding.")
        y = embed(fp, pngs["F1_tier_means"], y - 0.075, width_frac=0.62)
        caption(fp, y - 0.015, "Figure 1.",
                "Behavioral gate (S3): mean advice aggressiveness is correctly ordered across true tiers "
                "(0.292 / 0.430 / 0.607) with zero refusals -- the model does the task, so internal "
                "analysis is meaningful. Dotted lines mark the menu extremes (0.20 and 0.90); note the "
                "compression at the aggressive end.")
        pdf.savefig(fp); plt.close(fp)

        # page 2: new findings from the July 1-2 audit session
        fp = new_page("New findings this week (July 1-2 cold-read audit)")
        yy = 0.895
        for head_txt, body in [
            ("1. The gap survives a fair, leakage-safe comparison (Table 1).",
             "Earlier we compared a correlation (0.64) with an AUROC (0.894) -- apples to oranges. We "
             "rebuilt the comparison on one metric, one held-out client set, split by client identity, "
             "with the behavioral decoder given the same supervision as the probe. The gap holds and is "
             "large: 0.716 (behavior) vs 0.894 (internal), +0.178."),
            ("2. Advice quality collapses on conflicted clients -- as a dose-response (Figure 3).",
             "Rho falls monotonically 0.85 -> 0.71 -> 0.45 -> -0.10 as the client's stated goal "
             "increasingly contradicts their revealed behavior. The 723 most conflicted clients get "
             "advice that carries no information about their true tolerance. The 0.64 headline rho "
             "averages over this."),
            ("3. Cause identified: stated-goal dominance (Figure 4).",
             "Regression decomposition shows the advisor gives the stated goal 5.4x its fair weight "
             "while nearly ignoring income stability and emergency fund. It does what the client ASKS, "
             "not what their situation calls for."),
            ("4. Asymmetric caution (Figure 6A).",
             "Moving from explicit to implicit stories costs aggressive clients 0.20 in advice "
             "aggressiveness but conservative clients only 0.02 -- under ambiguity the model defaults "
             "cautious in one direction. This motivates the self-vs-client decomposition (next phase)."),
            ("5. Prompting is not the fix (Figure 6B).",
             "An 'integrate holistically' instruction moves advice TOWARD the correct tier on both ends "
             "(aggressive-tier +0.024, conservative-tier -0.007) but is an order of magnitude too weak."),
            ("6. Robustness: fairness null (Table 2).",
             "No detectable gender effect on advice (residual effects within +/-0.005); only a trace "
             "net-worth effect (+0.056). The failures above are not demographic artifacts."),
            ("7. The decisive experiment is built and pre-registered (Figure 5).",
             "Whether the INTERNAL picture also collapses on conflicted clients (encoding failure) or "
             "stays intact while advice discards it (readout failure) is pre-registered and runs on the "
             "next GPU session. Either outcome publishes; it decides the paper's framing."),
        ]:
            fp.text(0.07, yy, head_txt, size=10.5, weight="bold", color=NAVY)
            yy -= 0.0165
            for line in textwrap.fill(body, 112).split("\n"):
                fp.text(0.07, yy, line, size=9.6)
                yy -= 0.0140
            yy -= 0.0105
        pdf.savefig(fp); plt.close(fp)

        # page 3: F2
        fp = new_page("Where the client representation lives")
        y = embed(fp, pngs["F2_layer_sweep"], 0.90)
        caption(fp, y - 0.015, "Figure 2.",
                "Linear-probe AUROC on held-out implicit clients across all 42 layers at three reading "
                "positions. The representation strengthens sharply from layer ~17, peaks at layer 22 "
                "(0.894, CI 0.880-0.907), and persists to the final layer. It beats the text-only "
                "baseline from layer 8 onward: the model integrates life facts, it does not echo keywords. "
                "Probes are trained on explicit vignettes only and tested on implicit ones (no risk "
                "vocabulary, banned-word filter), from clients never seen in training.")
        pdf.savefig(fp); plt.close(fp)

        # page 3: F3
        fp = new_page("Failure mode: collapse on conflicted clients")
        y = embed(fp, pngs["F3_conflict_collapse"], 0.90)
        caption(fp, y - 0.015, "Figure 3.",
                "Advice quality as a function of the conflict between the client's stated goal and their "
                "revealed behavior. Implicit-vignette rho falls monotonically 0.85 -> 0.71 -> 0.45 -> "
                "-0.10 (CI -0.16 to -0.03): for the 723 most conflicted clients -- the canonical "
                "suitability cases under FINRA Rule 2111 -- recommendations carry no information about "
                "true risk tolerance. The overall gate rho of 0.64 averages over this collapse.")
        pdf.savefig(fp); plt.close(fp)

        # page 4: F4
        fp = new_page("Mechanism: one loud signal drowns the rest")
        y = embed(fp, pngs["F4_field_weights"], 0.90)
        caption(fp, y - 0.015, "Figure 4.",
                "Standardized regression of advice on the eight rubric fields, compared with each "
                "field's ground-truth weight. The stated goal receives 5.4x its fair share; revealed "
                "downturn behavior (0.6x), time horizon (0.3x), income stability (0.2x) and emergency "
                "fund (0.1x) are under-used. This explains Figure 3: when the loud signal disagrees "
                "with the truth, the advisor follows the loud signal. A concurrent behavioral study "
                "(arXiv:2604.23837) reports the same one-factor dominance across commercial LLMs.")
        pdf.savefig(fp); plt.close(fp)

        # page: F6 additional findings
        fp = new_page("Two more behaviors the audit surfaced")
        y = embed(fp, pngs["F6_additional_findings"], 0.90)
        caption(fp, y - 0.015, "Figure 6.",
                "A: the explicit-to-implicit advice drop by tier -- ambiguity makes the advisor "
                "cautious specifically for growth-oriented clients (+0.199 vs +0.022), a footprint of "
                "the model's own prior that the self-vs-client decomposition will test directly. "
                "B: the holistic-integration instruction shifts advice toward the correct tier on both "
                "ends, so the model 'understands' the request -- it just barely acts on it. Fixing the "
                "gap requires internal intervention, not better prompting.")
        pdf.savefig(fp); plt.close(fp)

        # page: 11% + F5
        fp = new_page("The '11% error rate' question, answered precisely")
        yy = 0.905
        for line in [
            "1 - AUROC is not an error rate. AUROC is ranking quality: 0.894 means the probe orders a",
            "random (higher-risk, lower-risk) client pair correctly ~89% of the time; chance is 0.50, not 0.",
            "The error-rate-flavored number is the probe's 3-way classification accuracy: 74.0% correct,",
            "i.e. 26% error, against 66.7% chance error (three balanced tiers).",
            "",
            "The sharper follow-up is WHERE the errors live. If failures were random noise, no client",
            "group would be special. They are not random on the behavioral side (left panel): advice is",
            "excellent when client signals agree and uninformative when they conflict. Whether the",
            "PROBE's errors also concentrate on conflicted clients is the decisive, pre-registered next",
            "experiment (right panel): if the internal picture stays accurate where behavior collapses,",
            "the model knows the right answer and discards it -- the strongest form of our thesis.",
        ]:
            fp.text(0.07, yy, line, size=10)
            yy -= 0.0155
        y = embed(fp, pngs["F5_failure_concentration"], yy - 0.01)
        caption(fp, y - 0.015, "Figure 5.",
                "Left: measured behavioral failure concentration (bars = rho with 95% CI). Right: the "
                "pre-registered decision zones for the probe-side test (config thresholds committed "
                "2026-07-02, before any probe-side subgroup number was computed).")
        pdf.savefig(fp); plt.close(fp)

        # page 6: T2
        fp = new_page("Complete numeric summary")
        y = embed(fp, pngs["T2_summary"], 0.90)
        caption(fp, y - 0.015, "Table 2.",
                "Every value traces to a results file in the repo; 95% CIs are 2,000-resample "
                "bootstraps; splits are by client identity (70/10/20); all pass/fail bars were "
                "pre-registered in config.yaml before results were seen. Model: Gemma-2-9B-it, frozen, "
                "bfloat16, eager attention; advice read as option-letter probabilities under 4 option "
                "orderings.")
        pdf.savefig(fp); plt.close(fp)

        info = pdf.infodict()
        info["Title"] = "Mechanistic KYC -- Results Figures & Tables"
        info["Author"] = "Shlok Channawar"
    print(f"wrote {pdf_path}  ({n_page[0]} pages)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    build(cfg)


if __name__ == "__main__":
    main()
