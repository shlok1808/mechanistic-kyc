"""F3: patching flip rate vs layer, both directions, bootstrap CI bands (CPU).

Reads results/<model>/patching/per_field.json (written by patching_fields.py), which holds
the all-pairs flip rate and bootstrap CI for every (layer, site) cell.

Output: results/<model>/figures/patching_layer_curve.png

Usage:
    PYTHONPATH=src python src/patching_figure.py [--config config.yaml]
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from behavior_figures import BLUE, INK2, MUTED, ORANGE, save, subtitle
from utils.paths import run_dir

SITE_STYLE = {"post_profile": ("-", "o", "all task tokens"),
              "profile_end": ("--", "s", "last narrative token")}
DIR_STYLE = {"lo_to_hi": (BLUE, "conservative -> aggressive donor"),
             "hi_to_lo": (ORANGE, "aggressive -> conservative donor")}


def run(cfg):
    base = run_dir(cfg, cfg["model"]["primary"])
    data = json.loads((base / "patching" / "per_field.json").read_text())
    cells = data["all_fields_all_pairs"]
    layers = sorted({int(s.split(":")[0][1:]) for s in cells})
    n = cells[f"L{layers[0]}:post_profile"]["lo_to_hi"]["n_eligible"]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for site, (ls, marker, site_label) in SITE_STYLE.items():
        for d, (color, d_label) in DIR_STYLE.items():
            dx = -0.35 if d == "lo_to_hi" else 0.35          # nudge so overlapping lines stay visible
            rate = [cells[f"L{l}:{site}"][d]["flip_rate"] for l in layers]
            lo = [cells[f"L{l}:{site}"][d]["ci"][0] for l in layers]
            hi = [cells[f"L{l}:{site}"][d]["ci"][1] for l in layers]
            xs = [l + dx for l in layers]
            ax.fill_between(xs, lo, hi, color=color, alpha=0.22, lw=0)
            ax.plot(xs, rate, color=color, ls=ls, marker=marker, ms=5, lw=1.8,
                    label=f"{site_label} to {d_label}" if site == "post_profile" else None)
    ax.plot([], [], color=INK2, ls="--", marker="s", ms=5, lw=1.8, label="last narrative token (either direction)")
    ax.axhline(data.get("flip_bar", 0.70), color=MUTED, lw=1, ls=":")
    ax.text(layers[0], 0.715, "pre-registered bar (0.70)", fontsize=8.5, color=INK2)
    ax.set_xticks(layers)
    ax.set_xlabel("Layer whose residual stream is overwritten (42 layers total)")
    ax.set_ylabel("Flip rate (share of eligible pairs\nmoved at least half way to the donor)")
    ax.set_ylim(-0.04, 1.06)
    ax.set_title("Patching flips advice only from the task tokens, and only from mid depth on", pad=22)
    subtitle(ax, f"{n} eligible one-field pairs per direction; bands are bootstrap 95% CIs over pairs "
                 f"(zero width at 0 and 1)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.1), frameon=False, fontsize=8.5)
    out = base / "figures" / "patching_layer_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    save(fig, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    run(yaml.safe_load(open(args.config)))
