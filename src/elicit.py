"""S3c: verbal-elicitation baseline -- ask the frozen advisor to CLASSIFY the client's
risk tolerance directly, with the same constrained-MCQ machinery as S3.

Why this exists (the audit ladder's missing rung): the S5 probe is supervised while the
S3 advice readout is zero-shot, so "probe 0.89 > behavior 0.72" invites the objection
"just ask the model". This script asks. Three outcomes, pre-registered in config
`elicitation:` bands (vs the probe on the same implicit test split, same macro-OvR AUROC):
  - elicit ~ probe  -> knowledge is promptable; the paper's claim is a UTILIZATION gap
    (the model can say the tier but the advice doesn't use it);
  - elicit ~ behavior -> true ELICITATION gap; internals beat any black-box access
    (strongest version of the monitor argument);
  - in between -> report both framings honestly.

Mechanics mirror advice.py: chat template + assistant prefill "(", letter-token logit
readout, cyclic option-order permutations (3 tiers -> 3 rotations) so letter/position bias
cancels, probability mass outside the letters = hedging. One condition ("elicit"), twins
only by default (pairs are S6 material).

Output rows (results/elicit.jsonl):
  {vignette_id, profile_id, pair_id, vignette_type, tier, risk_score, contradictory,
   p_tier: [p_conservative, p_moderate, p_aggressive]  (config data.tiers order),
   pred_tier, elicit_score, p_letters, hedged}

The official ladder AUROC (exact S5 implicit-test rows) is assembled by dissociation.py;
this script prints a convenience preview over all scored rows.

Usage (Lambda, inside tmux):
    python src/elicit.py [--config config.yaml] [--dry-run] [--model ID]
        [--device cuda] [--include-pairs]
"""

import argparse
import json
from pathlib import Path

import yaml

from utils.paths import run_dir

from advice import (LETTERS, aggregate_allocation_probs, batched, cyclic_permutations,
                       is_hedged, load_model, load_vignettes, make_prompt,
                       resolve_letter_token_ids, score_prompts, slot_options,
                       stratified_sample)


# --------------------------------------------------------------------------------------
# Pure logic (no torch) -- unit-tested in tests/test_s3c.py
# --------------------------------------------------------------------------------------
def tier_display(tier):
    """Config tier name -> the label shown in the prompt ('conservative' -> 'Conservative')."""
    return tier.capitalize()


def build_elicit_message(framing, narrative, tiers_in_slots):
    """Assemble the classification user turn; `tiers_in_slots` are tier names in slot order."""
    parts = [framing.strip()]
    parts.append(f"\nClient description:\n{narrative}")
    parts.append("\nRisk tolerance categories:")
    parts += [f"{LETTERS[i]}) {tier_display(t)}" for i, t in enumerate(tiers_in_slots)]
    parts.append("\nClassify this client as exactly one category. Respond with only the letter.")
    return "\n".join(parts)


def elicit_score(p_tier, tier_values=(0.0, 0.5, 1.0)):
    """Continuous readout = probability-weighted tier value (conservative=0 .. aggressive=1),
    the elicitation analogue of S3's aggressiveness score (for rank statistics)."""
    return float(sum(p * v for p, v in zip(p_tier, tier_values)))


def build_units(vignettes, framing, tiers, permutations):
    """One unit per (vignette, perm): prompt text + bookkeeping. Single 'elicit' condition."""
    units = []
    for v in vignettes:
        for pi, perm in enumerate(permutations):
            msg = build_elicit_message(framing, v["text"], slot_options(tiers, perm))
            units.append({"vid": v["vignette_id"], "perm_idx": pi, "user_msg": msg, "v": v})
    return units


# --------------------------------------------------------------------------------------
# Driver (lazy torch via advice.load_model)
# --------------------------------------------------------------------------------------
def run(cfg, args):
    el = cfg["elicitation"]
    s3 = cfg["s3"]
    tiers = cfg["data"]["tiers"]
    n = len(tiers)
    permutations = cyclic_permutations(n, el.get("n_permutations", n))

    vignettes = load_vignettes(cfg["paths"]["vignettes_dir"])
    if not args.include_pairs:
        vignettes = [v for v in vignettes if v.get("pair_id") is None]
    if args.dry_run:
        vignettes = stratified_sample(vignettes, s3["dry_run_n"], cfg["seed"])
    print(f"S3c: {len(vignettes)} vignettes x {len(permutations)} perms  (elicitation)")

    model_id = args.model or cfg["model"]["primary"]
    tok, model = load_model(model_id, cfg["model"]["dtype"], s3["attn_implementation"], args.device)

    units = build_units(vignettes, el["prompts"]["framing"], tiers, permutations)
    prompts = [make_prompt(tok, u["user_msg"]) for u in units]
    letter_ids = resolve_letter_token_ids(tok, prompts[0], n)
    print(f"model={model_id}  letter token ids={letter_ids}")

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    results = [None] * len(prompts)
    for batch_idx in batched(order, s3["batch_size"]):
        bp = [prompts[i] for i in batch_idx]
        lp, pl, aml = score_prompts(tok, model, bp, letter_ids)
        for j, i in enumerate(batch_idx):
            results[i] = (lp[j], pl[j], aml[j])
        print(f"  scored {sum(r is not None for r in results)}/{len(prompts)}", end="\r")
    print()

    groups = {}
    for u, res in zip(units, results):
        groups.setdefault(u["vid"], {"v": u["v"], "perm": [None] * len(permutations)})
        groups[u["vid"]]["perm"][u["perm_idx"]] = res

    out_rows = []
    for vid, g in groups.items():
        v = g["v"]
        letter_probs = [g["perm"][k][0] for k in range(len(permutations))]
        p_letters = [g["perm"][k][1] for k in range(len(permutations))]
        aml = [g["perm"][k][2] for k in range(len(permutations))]
        p_tier = aggregate_allocation_probs(letter_probs, permutations)
        p_letters_mean = sum(p_letters) / len(p_letters)
        hedged = is_hedged(p_letters_mean, sum(aml) / len(aml), s3["hedge_p_letters_min"])
        out_rows.append({
            "vignette_id": vid, "profile_id": v["profile_id"], "pair_id": v.get("pair_id"),
            "vignette_type": v["vignette_type"], "tier": v["tier"],
            "risk_score": v["risk_score"], "contradictory": v["contradictory"],
            "p_tier": [round(p, 5) for p in p_tier],
            "pred_tier": tiers[max(range(n), key=lambda i: p_tier[i])],
            "elicit_score": round(elicit_score(p_tier), 5),
            "p_letters": round(p_letters_mean, 5), "hedged": bool(hedged),
        })

    res_dir = run_dir(cfg, model_id)
    res_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_dryrun" if args.dry_run else ""
    out_path = res_dir / f"elicit{suffix}.jsonl"
    with open(out_path, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    (res_dir / f"elicit_meta{suffix}.json").write_text(json.dumps({
        "model": model_id, "n_permutations": len(permutations), "tiers": tiers,
        "attn_implementation": s3["attn_implementation"], "letter_token_ids": letter_ids,
        "seed": cfg["seed"], "framing": el["prompts"]["framing"],
        "include_pairs": bool(args.include_pairs),
    }, indent=2))

    hedge_rate = sum(r["hedged"] for r in out_rows) / len(out_rows)
    print(f"Wrote {len(out_rows)} rows to {out_path}  |  hedge rate: {hedge_rate:.3f}")

    # Convenience preview (official test-split ladder number comes from dissociation.py).
    try:
        from utils.stats import macro_ovr_auroc
        for vt in ("explicit", "implicit"):
            rows = [r for r in out_rows if r["vignette_type"] == vt]
            if rows:
                auc = macro_ovr_auroc([r["tier"] for r in rows],
                                      [r["p_tier"] for r in rows], tiers)
                acc = sum(r["pred_tier"] == r["tier"] for r in rows) / len(rows)
                print(f"  preview {vt}: macro-AUROC={auc:.4f}  acc={acc:.4f}  (n={len(rows)}, ALL rows"
                      f" -- not the held-out split)")
    except Exception as exc:
        print(f"  (preview AUROC skipped: {exc})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default=None, help="override config model.primary")
    ap.add_argument("--device", default="cuda", help="cuda | cpu")
    ap.add_argument("--include-pairs", action="store_true",
                    help="also elicit on counterfactual pairs (default: twins only)")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg, args)


if __name__ == "__main__":
    main()
