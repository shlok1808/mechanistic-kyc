"""Behavior-side figures for the re-run (advice + gate + patterns), readable without the paper.

Every number is read from results/<model>/{advice.jsonl, gate.json, patterns.json} and
data/profiles.jsonl; nothing is hand-entered.

Outputs (results/<model>/figures/):
  said_vs_did.png       advice by stated goal x past crash behavior (the headline)
  field_weights.png     how much each client detail moves the advice vs the rubric
  graded_collapse.png   rho as say-vs-did conflict grows
  want_vs_afford.png    willingness x capacity grid of mean advice
  gate_summary.png      rho by client subset against the 0.50 gate

Usage:
    python src/behavior_figures.py [--config config.yaml] [--model ID]
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from utils.paths import run_dir

DPI = 200
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
SURFACE = "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"                    # categorical slots 1-2
BLUE_RAMP = ["#b7d3f5", "#7aaeea", "#2a78d6", "#0f4a94"]  # ordinal: light -> dark

GOALS = ["capital_preservation", "income", "balanced_growth", "maximum_growth"]
DRAWDOWN = ["sold_everything", "reduced", "held", "bought_more"]
NICE = {"capital_preservation": "Protect my capital", "income": "Steady income",
        "balanced_growth": "Balanced growth", "maximum_growth": "Maximum growth",
        "sold_everything": "Sold\neverything", "reduced": "Sold\nsome", "held": "Held",
        "bought_more": "Bought\nmore",
        "past_drawdown_reaction": "Past crash behavior", "horizon_years": "Time horizon",
        "investing_experience": "Investing experience", "income_stability": "Income stability",
        "stated_goal": "Stated goal", "emergency_fund_months": "Emergency fund",
        "dependents": "Dependents", "age": "Age"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {path}")


def implicit_twins(rows):
    return [r for r in rows if r["condition"] == "baseline" and r.get("pair_id") is None
            and r["vignette_type"] == "implicit"]


def said_vs_did(rows, profiles, out):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(DRAWDOWN))
    ends = []
    for goal, color in zip(GOALS, BLUE_RAMP):
        ys = []
        for d in DRAWDOWN:
            v = [r["aggressiveness"] for r in rows
                 if profiles[r["profile_id"]]["stated_goal"] == goal
                 and profiles[r["profile_id"]]["past_drawdown_reaction"] == d]
            ys.append(np.mean(v))
        ax.plot(x, ys, color=color, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3)
        ends.append([ys[-1], NICE[goal]])
    ends.sort()
    for i in range(1, len(ends)):                      # keep labels >= 0.035 apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.035)
    for y, name in ends:
        ax.text(x[-1] + 0.12, y, name, color=INK, va="center", fontsize=9.5)
    ax.set_xticks(x, [NICE[d] for d in DRAWDOWN])
    ax.set_xlim(-0.3, len(DRAWDOWN) - 1 + 1.25)
    ax.set_ylim(0.15, 0.85)
    ax.set_ylabel("Advice riskiness (share in stocks)")
    ax.set_xlabel("What the client DID in the last market crash")
    ax.set_title("The model follows what clients say, not what they did", pad=22)
    subtitle(ax, "Lines = what the client SAYS they want. Far apart lines + flat lines = "
                 "stated goal dominates.")
    save(fig, out / "said_vs_did.png")


def field_weights(pat, out):
    f = pat["field_weights"]["implicit"]["fields"]
    names = sorted(f, key=lambda k: f[k]["rel_weight"])
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    h = 0.36
    ax.barh(y + h / 2 + 0.01, [f[k]["rubric_weight"] for k in names], height=h, color=MUTED,
            label="Our rubric's weight (ground truth)")
    ax.barh(y - h / 2 - 0.01, [f[k]["rel_weight"] for k in names], height=h, color=BLUE,
            label="Model's actual weight")
    for i, k in enumerate(names):
        if f[k]["ratio_vs_rubric"] >= 2:
            ax.text(f[k]["rel_weight"] + 0.01, i - h / 2, f"{f[k]['ratio_vs_rubric']:.0f}x",
                    va="center", fontsize=9.5, color=INK, fontweight="bold")
    ax.set_yticks(y, [NICE.get(k, k) for k in names])
    ax.set_xlabel("Share of influence on the advice")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("One detail drives most of the advice", pad=22)
    subtitle(ax, f"Implicit stories. Regression of advice on the 8 details "
                 f"(R² = {pat['field_weights']['implicit']['r2']:.2f}).")
    save(fig, out / "field_weights.png")


def graded_collapse(pat, out):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels = ["None", "Small", "Large", "Opposite"]
    x = np.arange(4)
    for vt, color, dx in (("explicit", BLUE, -0.06), ("implicit", ORANGE, 0.06)):
        g = pat["graded_collapse"][vt]
        keys = [f"conflict_{m}_of_3" for m in range(4)]
        rho = np.array([g[k]["rho"] for k in keys])
        lo = rho - np.array([g[k]["ci"][0] for k in keys])
        hi = np.array([g[k]["ci"][1] for k in keys]) - rho
        ax.errorbar(x + dx, rho, yerr=[lo, hi], color=color, lw=2, marker="o", ms=8,
                    mec=SURFACE, mew=2, capsize=0, elinewidth=1.5, zorder=3)
        name = "Explicit stories (plain facts)" if vt == "explicit" else "Implicit stories (hints)"
        ax.text(x[-1] + 0.15, rho[-1], name, color=INK, va="center", fontsize=9.5)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.3, 3 + 1.9)
    ax.set_xlabel("Gap between what the client SAYS (goal) and what they DID (crash behavior)")
    ax.set_ylabel("Spearman rho: advice vs client risk")
    ax.set_title("The more a client contradicts themself, the worse the advice fits", pad=22)
    subtitle(ax, "Below 0 = advice ranks clients backwards. Bars = 95% bootstrap CI.")
    save(fig, out / "graded_collapse.png")


def want_vs_afford(rows, out):
    levels = ["low", "mid", "high"]
    grid = np.zeros((3, 3))
    for i, w in enumerate(levels):
        for j, c in enumerate(levels):
            v = [r["aggressiveness"] for r in rows if r["factor_cell"] == f"{w}/{c}"]
            grid[i, j] = np.mean(v)
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(grid, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "blues", ["#e8f1fc", "#0f4a94"]), origin="lower", vmin=0.25, vmax=0.5)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=11,
                    color="white" if grid[i, j] > 0.42 else INK)
    ax.set_xticks(range(3), ["Low", "Mid", "High"])
    ax.set_yticks(range(3), ["Low", "Mid", "High"])
    ax.set_xlabel("CAPACITY: how much risk they can afford")
    ax.set_ylabel("WILLINGNESS: how much risk they want")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Mean advice riskiness")
    cb.outline.set_visible(False)
    ax.set_title("Want vs afford: both push advice up", pad=22)
    subtitle(ax, "Implicit stories, 666 clients per cell. Darker = riskier advice.")
    save(fig, out / "want_vs_afford.png")


def gate_summary(gate, pat, out):
    items = [
        ("Gate: implicit, want & afford agree", gate["spearman"]["implicit_concordant"], BLUE),
        ("Implicit, want & afford conflict", gate["spearman"]["implicit_conflict"], MUTED),
        ("Explicit, all clients", gate["spearman"]["explicit"], MUTED),
        ("Implicit, says vs did agree", pat["subgroup_spearman"]["implicit_concordant"], MUTED),
        ("Implicit, says vs did contradict", pat["subgroup_spearman"]["implicit_contradictory"],
         ORANGE),
    ][::-1]
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    for i, (label, s, color) in enumerate(items):
        ax.plot(s["ci"], [i, i], color=color, lw=2.5, solid_capstyle="round")
        ax.plot(s["rho"], i, "o", color=color, ms=9, mec=SURFACE, mew=2, zorder=3)
        ax.text(s["ci"][1] + 0.03, i, f"{s['rho']:+.2f}  (n={s['n']:,})", va="center",
                fontsize=9, color=INK2)
    ax.axvline(0.5, color=BLUE, lw=1, ls="--")
    ax.text(0.5, len(items) - 0.35, " gate bar 0.50", color=BLUE, fontsize=9, va="bottom")
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(items)), [l for l, _, _ in items])
    ax.set_xlim(-0.4, 1.2)
    ax.set_ylim(-0.6, len(items) - 0.1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Spearman rho: advice vs client risk (dot) with 95% CI (line)")
    ax.set_title("Gate passed; the failure is say vs did", pad=22)
    subtitle(ax, "1 = perfect ranking, 0 = none, below 0 = backwards.")
    save(fig, out / "gate_summary.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    res = run_dir(cfg, args.model or cfg["model"]["primary"])
    out = res / "figures"
    out.mkdir(exist_ok=True)

    rows = implicit_twins([json.loads(l) for l in open(res / "advice.jsonl")])
    profiles = {}
    for l in open(cfg["paths"]["profiles"] if "profiles" in cfg["paths"] else "data/profiles.jsonl"):
        p = json.loads(l)
        profiles[p["profile_id"]] = p
    gate = json.load(open(res / "gate.json"))
    pat = json.load(open(res / "patterns.json"))

    said_vs_did(rows, profiles, out)
    field_weights(pat, out)
    graded_collapse(pat, out)
    want_vs_afford(rows, out)
    gate_summary(gate, pat, out)


if __name__ == "__main__":
    main()
