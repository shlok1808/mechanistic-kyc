"""Generate the faculty-advisor results report (clean multi-page PDF, plain language).

Inputs (all already on disk; nothing is recomputed from the model):
  results/advice.jsonl                     raw advice readouts (tier distributions)
  results/gate.json                         behavioral gate numbers
  results/s5_probe_results_<tag>.json          probe sweep + baselines
  results/patterns.json                    cold-read pattern numbers (ladder, collapse,
                                               field weights) -- derived from advice.jsonl

Output: results/advisor_report_<date>.pdf

Audience: Dr. Raahemifar (financial-research background, not deep-ML). Every chart gets a
plain-language takeaway; the probe is framed as the measuring instrument, the GAP as the
finding. Regenerate after S5b/S3c land to fill the pending audit-ladder rung.

Usage:
    python src/report.py [--config config.yaml] [--out results/advisor_report.pdf]
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

NAVY, TEAL, ORANGE, GOLD, GRAY = "#1f3b57", "#2a9d8f", "#e76f51", "#e9c46a", "#8d99ae"
LIGHT = "#f4f4f4"
TIER_COLORS = {"conservative": TEAL, "moderate": GOLD, "aggressive": ORANGE}
PAGE = (8.5, 11)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10.5,
})


# ---------------------------------------------------------------- page helpers
class Report:
    def __init__(self, pdf, title, date_str):
        self.pdf, self.title, self.date_str, self.n = pdf, title, date_str, 0

    def page(self):
        self.n += 1
        fig = plt.figure(figsize=PAGE)
        fig.text(0.07, 0.965, self.title, size=8.5, color=GRAY)
        fig.text(0.93, 0.965, self.date_str, size=8.5, color=GRAY, ha="right")
        fig.text(0.5, 0.03, str(self.n), size=9, color=GRAY, ha="center")
        return fig

    def close_page(self, fig):
        self.pdf.savefig(fig)
        plt.close(fig)


def head(fig, y, text, size=15, color=NAVY):
    fig.text(0.07, y, text, size=size, weight="bold", color=color)
    return y - 0.028


def para(fig, y, text, size=10.5, width=104, dy=0.0148, color="black", x=0.07, weight="normal"):
    for block in text.split("\n"):
        wrapped = textwrap.fill(block.strip(), width=width) if block.strip() else ""
        for line in wrapped.split("\n"):
            fig.text(x, y, line, size=size, color=color, weight=weight)
            y -= dy
        y -= 0.004
    return y


def bullet(fig, y, items, size=10.5, width=98, dy=0.0148, x=0.09):
    for it in items:
        first = True
        for line in textwrap.fill(it, width=width).split("\n"):
            fig.text(x - 0.017 if first else x, y, ("•  " if first else "") + line, size=size)
            first = False
            y -= dy
        y -= 0.005
    return y


def takeaway(fig, y, text, width=100):
    wrapped = textwrap.fill(text, width=width)
    n_lines = wrapped.count("\n") + 1
    h = 0.018 * n_lines + 0.022
    fig.patches.append(FancyBboxPatch((0.06, y - h + 0.002), 0.88, h,
                                      boxstyle="round,pad=0.008", transform=fig.transFigure,
                                      fc="#eef3f7", ec=NAVY, lw=1.2))
    fig.text(0.075, y - 0.010, "Takeaway:", size=10.5, weight="bold", color=NAVY)
    para(fig, y - 0.029, wrapped, size=10.5, x=0.075, width=width)
    return y - h - 0.016


# ---------------------------------------------------------------- data loading
def load(res_dir):
    d = {}
    d["gate"] = json.loads((res_dir / "gate.json").read_text())
    d["s3b"] = json.loads((res_dir / "patterns.json").read_text())
    s5p = res_dir / "probes.json"
    if not s5p.exists():
        s5p = sorted(res_dir.glob("s5_probe_results_*.json"))[0]
    d["s5"] = json.loads(s5p.read_text())
    rows = [json.loads(l) for l in open(res_dir / "advice.jsonl")]
    d["twins"] = [r for r in rows if r["condition"] == "baseline" and r.get("pair_id") is None]
    return d


# ---------------------------------------------------------------- figures
def fig_pipeline(fig, y0):
    ax = fig.add_axes([0.06, y0 - 0.20, 0.88, 0.19])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [(0.2, 1.6, 1.9, "6,000 synthetic clients\n(risk tolerance KNOWN\nby construction)", LIGHT, NAVY),
             (2.5, 1.6, 1.9, "Client stories\nexplicit & implicit\n(no risk words)", LIGHT, NAVY),
             (4.8, 1.6, 1.9, "Frozen AI advisor\n(Gemma-2-9B)", "#dde7ee", NAVY),
             (7.1, 1.6, 2.4, "Advice: one of 4 portfolios\n(80/20 bonds ... 10/90)\nread as probabilities", LIGHT, NAVY)]
    for x, y, w, txt, fc, ec in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, 1.1, boxstyle="round,pad=0.06", fc=fc, ec=ec, lw=1.3))
        ax.text(x + w / 2, y + 0.55, txt, ha="center", va="center", size=8.6)
    for x0, x1 in [(2.1, 2.5), (4.4, 4.8), (6.7, 7.1)]:
        ax.add_patch(FancyArrowPatch((x0, 2.15), (x1, 2.15), arrowstyle="-|>",
                                     mutation_scale=14, color=NAVY))
    ax.add_patch(FancyBboxPatch((4.55, 0.1), 2.45, 0.85, boxstyle="round,pad=0.06",
                                fc="#fdf1ec", ec=ORANGE, lw=1.3))
    ax.text(5.78, 0.52, "Linear probe = our instrument:\nreads the advisor's INTERNAL\nestimate of the client", ha="center",
            va="center", size=8.6, color="#b3502f")
    ax.add_patch(FancyArrowPatch((5.78, 1.6), (5.78, 0.98), arrowstyle="-|>",
                                 mutation_scale=13, color=ORANGE, linestyle=(0, (4, 2))))
    return y0 - 0.225


def fig_tier_boxes(fig, y0, twins):
    ax = fig.add_axes([0.10, y0 - 0.30, 0.80, 0.29])
    tiers = ["conservative", "moderate", "aggressive"]
    data, positions, colors = [], [], []
    for i, vt in enumerate(("explicit", "implicit")):
        for j, t in enumerate(tiers):
            data.append([r["aggressiveness"] for r in twins if r["vignette_type"] == vt and r["tier"] == t])
            positions.append(j + i * 0.32 - 0.16)
            colors.append(TIER_COLORS[t])
    bp = ax.boxplot(data, positions=positions, widths=0.26, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black"))
    for patch, c, i in zip(bp["boxes"], colors, range(6)):
        patch.set_facecolor(c)
        patch.set_alpha(0.9 if i < 3 else 0.45)
    ax.set_xticks(range(3))
    ax.set_xticklabels([t.capitalize() for t in tiers])
    ax.set_xlabel("Client's TRUE risk tolerance (known by construction)")
    ax.set_ylabel("Advice aggressiveness (share of stocks recommended)")
    ax.set_title("The advisor's recommendations do respond to client type")
    ax.axhline(0.2, color=GRAY, lw=0.7, ls=":")
    ax.axhline(0.9, color=GRAY, lw=0.7, ls=":")
    ax.set_ylim(0.13, 0.99)
    ax.text(2.45, 0.155, "most cautious option (20% stocks)", size=8, color=GRAY, ha="right")
    ax.text(2.45, 0.925, "most aggressive option (90% stocks)", size=8, color=GRAY, ha="right")
    import matplotlib.patches as mpatches
    ax.legend(handles=[mpatches.Patch(fc="gray", alpha=0.9, label="explicit story (risk words allowed)"),
                       mpatches.Patch(fc="gray", alpha=0.45, label="implicit story (no risk words)")],
              loc="upper left", frameon=False, fontsize=9)
    return y0 - 0.375


def fig_layer_sweep(fig, y0, s5):
    ax = fig.add_axes([0.10, y0 - 0.27, 0.80, 0.26])
    sweep = s5["sweep"]
    styles = {"decision": (NAVY, "-", 2.2), "profile_mean": (TEAL, "-", 1.4), "profile_end": (GRAY, "--", 1.2)}
    nice = {"decision": "at the decision point", "profile_mean": "averaged over the story",
            "profile_end": "end of the story"}
    for pos, (c, ls, lw) in styles.items():
        layers = sorted(int(k) for k in sweep[pos])
        vals = [sweep[pos][str(l)]["implicit_test"]["auroc"] for l in layers]
        ax.plot(layers, vals, color=c, ls=ls, lw=lw, label=f"probe {nice[pos]}")
    tfidf = s5["baselines"]["tfidf"]["implicit_test"]
    ax.axhline(tfidf, color=ORANGE, lw=1.4, ls=":")
    ax.text(0.5, tfidf + 0.006, f"best text-only baseline ({tfidf:.2f}) -- keyword reading", size=8.5, color=ORANGE)
    ax.axhline(0.5, color="black", lw=0.8, ls=":")
    ax.text(0.5, 0.507, "coin flip (0.50)", size=8.5)
    b = s5["best"]
    ax.annotate(f"peak: layer {b['layer']}, {b['implicit_auroc']:.3f}",
                xy=(b["layer"], b["implicit_auroc"]), xytext=(b["layer"] + 4, b["implicit_auroc"] + 0.028),
                size=9.5, weight="bold", color=NAVY,
                arrowprops=dict(arrowstyle="->", color=NAVY))
    ax.set_xlabel("Layer of the network (processing depth, 0 = input side)")
    ax.set_ylabel("Client-type read-out quality (AUROC)")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("Where the model's internal picture of the client lives")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    return y0 - 0.35


def fig_ladder(fig, y0, s3b, s5):
    ax = fig.add_axes([0.30, y0 - 0.26, 0.62, 0.25])
    lad = s3b["audit_ladder_behavioral"]
    probe = s5["best"]
    rows = [
        ("Internal probe\n(read the model's mind)", probe["implicit_auroc"], probe["ci"], NAVY),
        ("Ask-the-model baseline\n(verbal self-report)", None, None, GRAY),
        ("Transcript baseline\n(text keywords, TF-IDF)", s5["baselines"]["tfidf"]["implicit_test"], None, GOLD),
        ("Watch its advice\n(all 4 option probabilities)", lad["behavior_4d_option_dist"]["implicit_test_auroc"],
         lad["behavior_4d_option_dist"]["ci"], TEAL),
        ("Watch its advice\n(single riskiness score)", lad["behavior_1d_aggressiveness"]["implicit_test_auroc"],
         lad["behavior_1d_aggressiveness"]["ci"], TEAL),
    ]
    ys = np.arange(len(rows))[::-1]
    for y, (label, val, ci, c) in zip(ys, rows):
        if val is None:
            ax.barh(y, 0.001, left=0.5, color="none")
            ax.text(0.505, y, "pending (next GPU run)", va="center", size=9, color=GRAY, style="italic")
        else:
            err = None
            if ci:
                err = np.array([[val - ci[0]], [ci[1] - val]])
            ax.barh(y, val - 0.5, left=0.5, color=c, height=0.62,
                    xerr=err, error_kw=dict(ecolor="black", capsize=3, lw=1))
            ax.text(val + 0.008, y, f"{val:.3f}", va="center", size=10, weight="bold", color=c)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], size=9.5)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("How well each audit method recovers the client's true risk tolerance\n(AUROC; 0.5 = coin flip, 1.0 = perfect)  --  same 1,206 held-out clients, same yardstick")
    ax.axvline(0.5, color="black", lw=0.8)
    ax.set_title("The audit ladder: what you can learn from outside vs. inside", pad=14)
    y_probe, y_beh = ys[0], ys[3]
    ax.annotate("", xy=(0.894, (y_probe + y_beh) / 2 + 0.9), xytext=(0.716, (y_probe + y_beh) / 2 + 0.9),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.6))
    ax.text(0.803, (y_probe + y_beh) / 2 + 1.12, "the gap: +0.18", ha="center",
            size=10.5, weight="bold", color=ORANGE)
    return y0 - 0.365


def fig_auroc_vs_accuracy(fig, y0, s5):
    ax = fig.add_axes([0.34, y0 - 0.17, 0.56, 0.16])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 1.6)
    for y, (label, val, chance, cl) in enumerate([
            ("AUROC (ranking quality)", s5["best"]["implicit_auroc"], 0.5, NAVY),
            ("Accuracy (3-way classification)", 0.740, 1 / 3, TEAL)][::-1]):
        ax.barh(y, 1, color=LIGHT, height=0.45, ec=GRAY, lw=0.6)
        ax.barh(y, val, color=cl, height=0.45)
        ax.plot([chance, chance], [y - 0.32, y + 0.32], color=ORANGE, lw=2)
        ax.text(chance, y + 0.38, f"chance = {chance:.2f}", size=8.5, color=ORANGE, ha="center")
        ax.text(val + 0.012, y, f"{val:.3f}", va="center", size=10.5, weight="bold", color=cl)
        ax.text(-0.015, y, label, va="center", ha="right", size=10)
    ax.axis("off")
    ax.set_title("Same probe, two different yardsticks (different chance levels)", size=11.5, pad=10)
    return y0 - 0.20


def fig_graded_collapse(fig, y0, s3b):
    ax = fig.add_axes([0.10, y0 - 0.27, 0.80, 0.26])
    labels = ["none", "mild", "strong", "extreme"]
    x = np.arange(4)
    for vt, c, m in (("explicit", GRAY, "s"), ("implicit", NAVY, "o")):
        g = s3b["graded_collapse"][vt]
        keys = sorted(g, key=lambda k: int(k.split("_")[1]))
        vals = [g[k]["rho"] for k in keys]
        los = [g[k]["ci"][0] for k in keys]
        his = [g[k]["ci"][1] for k in keys]
        ns = [g[k]["n"] for k in keys]
        ax.errorbar(x, vals, yerr=[np.array(vals) - np.array(los), np.array(his) - np.array(vals)],
                    color=c, marker=m, ms=7, lw=2, capsize=4,
                    label=f"{vt} client stories")
        if vt == "implicit":
            for xi, v, n in zip(x, vals, ns):
                ax.text(xi + 0.06, v - 0.055, f"n={n:,}", size=8, color=c)
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.text(2.95, 0.03, "zero = advice carries no information", size=8.5, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\nconflict" for l in labels])
    ax.set_xlabel('Conflict between what the client SAYS they want and how they BEHAVED\n(e.g., "maximum growth" but sold everything in the last downturn = extreme)')
    ax.set_ylabel("Advice quality (correlation with\ntrue risk tolerance)")
    ax.set_ylim(-0.25, 1.0)
    ax.set_title("Advice quality collapses as client signals conflict")
    ax.legend(loc="lower left", frameon=False, fontsize=9.5)
    return y0 - 0.36


def fig_field_weights(fig, y0, s3b):
    ax = fig.add_axes([0.33, y0 - 0.30, 0.59, 0.29])
    fw = s3b["field_weights"]["implicit"]["fields"]
    nice = {"stated_goal": "Stated goal (what they SAY)",
            "past_drawdown_reaction": "Behavior in past downturn",
            "horizon_years": "Investment time horizon",
            "investing_experience": "Investing experience",
            "income_stability": "Income stability",
            "emergency_fund_months": "Emergency fund",
            "dependents": "Number of dependents", "age": "Age"}
    fields = sorted(fw, key=lambda f: fw[f]["rel_weight"], reverse=True)
    ys = np.arange(len(fields))[::-1]
    h = 0.36
    for y, f in zip(ys, fields):
        c = ORANGE if f == "stated_goal" else TEAL
        ax.barh(y + h / 2, fw[f]["rel_weight"], height=h, color=c, label=None)
        ax.barh(y - h / 2, fw[f]["rubric_weight"], height=h, color=GRAY, alpha=0.55)
        ax.text(max(fw[f]["rel_weight"], fw[f]["rubric_weight"]) + 0.006, y,
                f"{fw[f]['ratio_vs_rubric']:g}x", va="center", size=9,
                weight="bold" if f == "stated_goal" else "normal",
                color=ORANGE if f == "stated_goal" else "black")
    ax.set_yticks(ys)
    ax.set_yticklabels([nice[f] for f in fields], size=9.5)
    import matplotlib.patches as mpatches
    ax.legend(handles=[mpatches.Patch(fc=TEAL, label="weight the ADVISOR actually uses (implicit stories)"),
                       mpatches.Patch(fc=GRAY, alpha=0.55, label="weight it SHOULD use (ground-truth rubric)"),
                       mpatches.Patch(fc=ORANGE, label="stated goal: 5.4x over-weighted")],
              loc="lower right", frameon=False, fontsize=8.8)
    ax.set_xlabel("Share of the advice decision each client fact accounts for")
    ax.set_title("What the advisor listens to vs. what it should listen to", pad=10)
    return y0 - 0.345


# ---------------------------------------------------------------- pages
def build(cfg, out_path):
    res_dir = run_dir(cfg, cfg["model"]["primary"])   # results/<model>/
    d = load(res_dir)
    gate, s3b, s5 = d["gate"], d["s3b"], d["s5"]
    today = date.today().strftime("%B %d, %Y")

    with PdfPages(out_path) as pdf:
        rep = Report(pdf, "Mechanistic KYC -- Interim Results Report", today)

        # ---- Page 1: title + executive summary ----
        fig = rep.page()
        fig.text(0.07, 0.90, "Does an AI Financial Advisor Really", size=21, weight="bold", color=NAVY)
        fig.text(0.07, 0.872, "Know Its Customer?", size=21, weight="bold", color=NAVY)
        fig.text(0.07, 0.845, "Interim results report -- prepared for Dr. Raahemifar", size=12, color=GRAY)
        fig.text(0.07, 0.828, "Shlok Channawar, with Daniel Liu & Nina  |  target venue: ICAIF 2026 (submission Aug 2)", size=10, color=GRAY)
        y = head(fig, 0.775, "Executive summary", size=14)
        y = bullet(fig, y - 0.005, [
            "We test a frozen open-weight AI model acting as a financial advisor for 6,000 synthetic clients whose "
            "true risk tolerance we constructed, so we can grade every recommendation against ground truth.",
            "The advisor's recommendations DO track client type overall (rank correlation 0.64-0.75; zero refusals). "
            "That cleared our pre-registered go/no-go gate.",
            "Inside the model, however, a simple linear read-out of its internal activity identifies the client's "
            "true risk tolerance far better (0.894 AUROC) than anything visible from outside -- its advice (0.716) "
            "or the transcript text (0.748) -- on the same held-out clients, same metric. The advisor KNOWS more "
            "about the client than its advice USES. This gap is the paper's central finding.",
            "New this week: advice quality collapses to zero exactly on CONFLICTED clients (say one thing, did "
            "another) -- the very cases suitability rules like FINRA 2111 are hardest on. Cause: the advisor "
            "over-weights the client's stated goal ~5x and under-uses horizon, income stability, and emergency fund.",
            "A gentle instruction to 'integrate holistically' nudges advice in the right direction but is ~10x too "
            "weak to fix it. This motivates the next phase: intervening on and monitoring the model's internal "
            "client picture directly.",
        ], size=10.8, width=101)
        y = takeaway(fig, y - 0.012,
                     "The probe is our measuring instrument -- like a thermometer. The contribution is not the "
                     "thermometer; it is what the reading reveals: the advisor holds an accurate internal picture "
                     "of the client that its advice only partially uses, and the shortfall concentrates on exactly "
                     "the clients regulators care most about.")
        para(fig, y - 0.02, "Section 5 answers the '11% error rate' question from our last meeting. "
                            "Section 6 covers the new audit findings. Section 7 lists next steps to the August 2 deadline.",
             size=9.5, color=GRAY)
        rep.close_page(fig)

        # ---- Page 2: what we built ----
        fig = rep.page()
        y = head(fig, 0.93, "1. What we built (and why synthetic clients)")
        y = para(fig, y, "Each synthetic client is generated from a transparent scoring rubric of the kind "
                         "robo-advisors use at onboarding (age, horizon, past behavior in downturns, income stability, "
                         "goal, emergency fund, dependents, experience). The rubric gives every client a known risk "
                         "tolerance: conservative, moderate, or aggressive. Because WE constructed the ground truth, "
                         "we can grade the AI's advice exactly -- something impossible with real client data. Each "
                         "client is written up as a short story in two styles: an explicit one that may use risk "
                         "vocabulary, and an implicit one where risk words are banned by filter, so the model must "
                         "infer the client's tolerance from life facts alone.")
        y = fig_pipeline(fig, y)
        y = para(fig, y - 0.01,
                 "The advisor model is frozen -- we never train or modify it. It reads one client story and must "
                 "recommend exactly one of four portfolios, from most cautious (80% bonds / 20% stocks) to most "
                 "aggressive (10% bonds / 90% stocks). We read the probability it assigns to each option (averaged "
                 "over 4 option orderings to cancel positional habits), giving a continuous 'advice aggressiveness' "
                 "score per client. Scale: 14,400 client stories, 16,106 scored recommendations, zero refusals.")
        y = takeaway(fig, y - 0.005,
                     "Think of it as a driving test where we secretly know every correct answer in advance, "
                     "administered 14,400 times to the same driver.")
        rep.close_page(fig)

        # ---- Page 3: gate ----
        fig = rep.page()
        y = head(fig, 0.93, "2. First result: the advisor does respond to client type")
        sp = gate["spearman"]
        y = para(fig, y, f"Before any internal analysis, we verified the advisor is doing the task at all (our "
                         f"pre-registered go/no-go gate). It is: rank correlation between true client risk tolerance and "
                         f"advice aggressiveness is {sp['explicit']['rho']:.2f} on explicit stories and "
                         f"{sp['implicit']['rho']:.2f} (95% CI {sp['implicit']['ci'][0]:.2f}-{sp['implicit']['ci'][1]:.2f}) "
                         f"on implicit ones -- both clear our bar of 0.50. Average recommendations rise in the right "
                         f"order across the three true tiers, and the model never refused or hedged (0 of 16,106).")
        y = fig_tier_boxes(fig, y, d["twins"])
        y = para(fig, y - 0.005,
                 "Note two things visible already in this chart, which Section 6 quantifies: the boxes overlap "
                 "substantially (advice is far from a clean separation), and the implicit boxes sit lower for "
                 "aggressive clients -- when the story is subtle, the advisor drifts cautious for growth-oriented "
                 "clients specifically.")
        y = takeaway(fig, y,
                     "The advisor passes the basic test -- clients who should get bolder portfolios do, on average, "
                     "get bolder portfolios. The interesting question is what it does NOT use.")
        rep.close_page(fig)

        # ---- Page 4: probe ----
        fig = rep.page()
        y = head(fig, 0.93, "3. Reading the advisor's mind: the internal client picture")
        y = para(fig, y, "A linear probe is the simplest possible read-out: a weighted sum over the model's internal "
                         "activity at one processing layer, trained to predict the client's true tier. It cannot add "
                         "intelligence of its own -- if it succeeds, the information was already sitting there. We train "
                         "probes only on EXPLICIT stories and test them on IMPLICIT stories from clients the probe has "
                         "never seen, so it cannot succeed by spotting risk vocabulary (there is none) or by memorizing "
                         "clients (they are held out).")
        y = fig_layer_sweep(fig, y, s5)
        b = s5["best"]
        y = para(fig, y - 0.005,
                 f"The internal picture peaks at layer {b['layer']} of 42 (about mid-network) at {b['implicit_auroc']:.3f} "
                 f"AUROC (95% CI {b['ci'][0]:.3f}-{b['ci'][1]:.3f}; stable across 5 re-runs at 0.889 +/- 0.007). "
                 f"Crucially, it beats the best text-only baseline ({s5['baselines']['tfidf']['implicit_test']:.3f}) by "
                 f"+{b['implicit_auroc'] - s5['baselines']['tfidf']['implicit_test']:.2f}: the model is not echoing "
                 f"keywords, it is INTEGRATING life facts into a genuine assessment. A shuffled-label control probe "
                 f"scores 0.48 (chance), confirming the read-out is real signal, not statistical artifact.")
        y = takeaway(fig, y,
                     "By the middle of its processing, the advisor has quietly formed an accurate judgment of what "
                     "kind of investor this client truly is -- even when the client never says a single risk-related word.")
        rep.close_page(fig)

        # ---- Page 5: the gap ----
        fig = rep.page()
        y = head(fig, 0.93, "4. The central finding: it knows more than its advice shows")
        y = para(fig, y, "Is 0.894 'inside' actually better than what an auditor could get from outside? To compare "
                         "fairly, we put every audit method on the SAME 1,206 held-out implicit clients and the SAME "
                         "yardstick (AUROC), with the same training data where applicable. This includes a fix to an "
                         "earlier apples-to-oranges comparison (a correlation vs. an AUROC) and a leakage-safe split: "
                         "clients are split by identity, so no client ever appears on both the training and test side "
                         "in any rung.")
        y = fig_ladder(fig, y, s3b, s5)
        y = para(fig, y - 0.005,
                 "Watching the advisor's recommendations -- even using its full probability distribution over options "
                 "and training a decoder on it -- recovers the client at 0.716. Reading the transcript text directly: "
                 "0.748. Reading the model's internal activity: 0.894. The +0.18 gap means a substantial share of what "
                 "the advisor knows about the client never reaches its advice.")
        y = takeaway(fig, y,
                     "For AI-advisor oversight this is the headline: an auditor who can only observe recommendations "
                     "will systematically UNDERESTIMATE what the model knew about the client. 'Know your customer' "
                     "compliance of AI advisors is, in part, only checkable from inside the model.")
        rep.close_page(fig)

        # ---- Page 6: the 11% question ----
        fig = rep.page()
        y = head(fig, 0.93, "5. Your question: is there an '11% error rate'?")
        y = para(fig, y, "Short answer: no -- the 11% figure comes from reading our headline number 0.894 as an "
                         "accuracy and subtracting it from 1. But 0.894 is an AUROC, which is a different kind of "
                         "quantity, with a different chance level, and '1 - AUROC' is not an error rate.")
        y = para(fig, y, "AUROC answers a ranking question: take any two clients with genuinely different risk "
                         "tolerance -- how often does the probe's internal read-out rank the truly-riskier client "
                         "higher? A coin flip scores 0.50; perfection scores 1.00. Our 0.894 means the internal "
                         "picture orders client pairs correctly about 9 times in 10.", width=104)
        y = fig_auroc_vs_accuracy(fig, y, s5)
        y = para(fig, y - 0.01,
                 "If you want an error-rate-flavored number, the right one is the probe's 3-way classification "
                 "accuracy on held-out implicit clients: 74.0% correct, i.e. a 26% error rate -- against a 33.3% "
                 "chance baseline (three balanced tiers), not 50%. Both numbers describe the same probe; they are "
                 "just different yardsticks with different zero points. We report AUROC as the headline because it "
                 "uses the full probability output and is standard in the auditing literature we compare against.")
        y = takeaway(fig, y,
                     "0.894 AUROC = 'ranks client pairs correctly ~89% of the time' (chance 50%). The accuracy view "
                     "of the same probe: 74% correct three-way, vs. 33% chance. There is no hidden 11% error rate.")
        rep.close_page(fig)

        # ---- Page 7: collapse ----
        fig = rep.page()
        y = head(fig, 0.93, "6a. New finding: advice collapses on conflicted clients")
        y = para(fig, y, "The overall correlation of 0.64 hides two very different populations. When a client's "
                         "signals all point the same way, the advisor is excellent. But real suitability cases are "
                         "rarely like that: the canonical hard client SAYS 'maximum growth' yet SOLD everything in the "
                         "last downturn. We graded advice quality as a function of how strongly the client's stated "
                         "goal conflicts with their revealed behavior:")
        y = fig_graded_collapse(fig, y, s3b)
        g = s3b["graded_collapse"]["implicit"]
        y = para(fig, y - 0.005,
                 f"On implicit stories, advice quality falls from 0.85 (no conflict) to {g['conflict_3_of_3']['rho']:.2f} "
                 f"(95% CI {g['conflict_3_of_3']['ci'][0]:.2f} to {g['conflict_3_of_3']['ci'][1]:.2f}) at extreme "
                 f"conflict -- statistically indistinguishable from, or slightly below, zero. For the 723 most "
                 f"conflicted clients, the advisor's recommendations carry essentially no information about their true "
                 f"risk tolerance.")
        y = takeaway(fig, y,
                     "The advisor fails precisely where the suitability obligation (FINRA Rule 2111 / Reg BI) is "
                     "hardest and matters most: clients whose stated wishes and actual circumstances disagree. "
                     "An average-case audit would never see this.")
        rep.close_page(fig)

        # ---- Page 8: field weights ----
        fig = rep.page()
        y = head(fig, 0.93, "6b. Why it collapses: one loud signal drowns the rest")
        y = para(fig, y, "We decomposed the advice statistically: how much does each client fact actually move "
                         "the recommendation, versus how much SHOULD it (per the ground-truth rubric)?", width=98)
        y = fig_field_weights(fig, y, s3b)
        y = para(fig, y - 0.005,
                 "The advisor gives the client's stated goal about 5x its proper weight, while the strongest true "
                 "signals -- revealed behavior in downturns and time horizon -- are under-used (0.6x and 0.3x), and "
                 "financial-capacity facts (income stability, emergency fund) are nearly ignored (0.2x). This is why "
                 "conflicted clients break it: when the loud signal and the truth disagree, the advisor follows the "
                 "loud signal. Consistent with this, it is also asymmetrically cautious: on subtle (implicit) stories "
                 "it almost never recommends the boldest portfolio (5% of truly aggressive clients, vs. 41% on "
                 "explicit stories). A concurrent behavioral study (arXiv:2604.23837, Apr 2026) reports the same "
                 "one-factor dominance across commercial LLMs -- but cannot say what the model knew. Our internal "
                 "probe can, which is the next experiment (Section 7). Robustness checks: advice shows no detectable "
                 "gender effect (a clean fairness null) and only a trace effect of net worth.")
        y = takeaway(fig, y,
                     "The advisor does what the client ASKS, not what the client's situation calls for -- the "
                     "opposite of the 'know your customer' duty. And a polite instruction to consider everything "
                     "holistically moves advice in the right direction but ~10x too weakly to matter.")
        rep.close_page(fig)

        # ---- Page 9: next steps ----
        fig = rep.page()
        y = head(fig, 0.93, "7. What happens next (to the August 2 submission)")
        y = bullet(fig, y - 0.005, [
            "DECIDING EXPERIMENT (built, pre-registered, runs on the GPU box next): does the internal picture stay "
            "accurate on conflicted clients while the advice collapses? If yes ('readout failure'), the model knows "
            "the right answer and discards it -- the strongest version of our thesis. If the internal picture "
            "collapses too ('encoding failure'), the story becomes 'the bias penetrates the representation'. Either "
            "way it publishes; the framing changes.",
            "ASK-THE-MODEL BASELINE (built): simply asking the advisor to classify the client fills the pending rung "
            "of the audit ladder and answers 'why not just prompt it?'.",
            "CAUSAL TESTS (week of Jul 7): steering the internal client picture up and down and checking that advice "
            "follows dose-dependently, plus activation patching between matched client pairs. This upgrades our "
            "claims from correlation to causation.",
            "SELF-VS-CLIENT DECOMPOSITION (novelty core, week of Jul 14): is the model's own risk attitude a "
            "separate internal direction from its estimate of the client's -- and does the model's own conservatism "
            "explain the cautious drift we see on subtle stories?",
            "Replication on a second model family (Llama-3.1-8B), then writing. Internal deadline Aug 1.",
        ], size=10.6, width=99)
        y = para(fig, y - 0.01, "One-sentence version of the paper as it now stands:", weight="bold")
        y = takeaway(fig, y + 0.005,
                     "\"An LLM financial advisor forms an accurate internal model of its client that its advice "
                     "only partially uses; the shortfall concentrates on conflicted clients, is driven by "
                     "over-weighting the client's stated goal, and is invisible to purely behavioral audits -- "
                     "making internal (mechanistic) auditing a necessary complement for AI suitability compliance.\"")
        y = para(fig, y - 0.01,
                 "Methods fine print for reference: Gemma-2-9B-it, frozen, bfloat16; advice read as option-letter "
                 "probabilities under 4 option orderings; splits by client identity (70/10/20); probes are L2 "
                 "logistic regressions chosen on a validation split; all thresholds pre-registered in the repo "
                 "config before results were seen; 95% CIs are 2,000-resample bootstraps. Code and all numbers: "
                 "github.com/shlok1808/mechanistic-kyc (analysis scripts s3b/s5b), full numeric log in the team "
                 "research log.", size=9, color=GRAY)
        rep.close_page(fig)

        info = pdf.infodict()
        info["Title"] = "Mechanistic KYC -- Interim Results Report"
        info["Author"] = "Shlok Channawar"

    print(f"wrote {out_path}  ({rep.n} pages)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out = args.out or str(Path(cfg["paths"]["results_dir"]) /
                          f"advisor_report_{date.today().isoformat()}.pdf")
    build(cfg, out)


if __name__ == "__main__":
    main()
