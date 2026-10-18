"""S6 / RQ2a: activation patching on one-field counterfactual pairs.

Each pair is two implicit vignettes that differ in exactly ONE rubric field. Run the
"donor" member, cache its residual stream, then run the "base" member with the residual at
one (layer, site) overwritten by the donor's. If the advice moves to the donor's advice,
that site carries the information the advice used. Both directions (lo -> hi, hi -> lo).

Sites (positions are identical text in both members, so they align exactly):
  profile_end   the last narrative token -- one token summarising the client
  post_profile  every token AFTER the narrative (options, instruction, prefill), i.e. what
                the task tokens have already read out of the profile by this layer

Effect on a pair = (patched - base) / (donor - base) on aggressiveness, so 1 = the full
behavioural difference transferred and 0 = none. A pair is ELIGIBLE only if the members'
advice differs by at least `min_behavior_gap` (otherwise there is nothing to flip); a FLIP is
effect >= `flip_effect_min`. The headline (layer, site) is picked on even-numbered pairs and
reported on odd-numbered pairs, so the reported flip rate carries no selection bias.

Usage:
    python src/patching.py run [--batch-size 32] [--shard i/N] [--dry-run]
    python src/patching.py summarize
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

SITES = ("profile_end", "post_profile")


# --------------------------------------------------------------------------------------
# Pure logic -- unit-tested in tests/test_patching.py
# --------------------------------------------------------------------------------------
def pair_members(rows):
    """{pair_id: {"lo": row, "hi": row}} from pairs.jsonl rows; incomplete pairs dropped."""
    out = {}
    for r in rows:
        out.setdefault(r["pair_id"], {})[r["pair_side"]] = r
    return {k: v for k, v in sorted(out.items()) if {"lo", "hi"} <= set(v)}


def site_positions(site, span_end, unpadded_len):
    """Unpadded token positions patched for a site."""
    if site == "profile_end":
        return [span_end]
    return list(range(span_end + 1, unpadded_len))


def transfer_effect(base, donor, patched):
    gap = donor - base
    return (patched - base) / gap if gap != 0 else float("nan")


def is_dev_pair(pair_id):
    """Even-numbered pairs select the site, odd-numbered pairs report it."""
    return int(pair_id.split("_")[-1]) % 2 == 0


def flip_summary(rows, min_gap, flip_min):
    """Flip rate over eligible rows (|donor - base| >= min_gap)."""
    elig = [r for r in rows if abs(r["donor"] - r["base"]) >= min_gap]
    if not elig:
        return {"n_eligible": 0, "flip_rate": float("nan"), "mean_effect": float("nan")}
    eff = [transfer_effect(r["base"], r["donor"], r["patched"]) for r in elig]
    return {"n_eligible": len(elig),
            "flip_rate": round(float(np.mean([e >= flip_min for e in eff])), 4),
            "mean_effect": round(float(np.mean(eff)), 4)}


# --------------------------------------------------------------------------------------
# GPU stage
# --------------------------------------------------------------------------------------
def run(cfg, args):
    import torch
    from activations import narrative_span_tokens
    from advice import (aggregate_allocation_probs, aggressiveness, build_user_message,
                        cyclic_permutations, load_model, make_prompt, parse_shard,
                        resolve_letter_token_ids, score_prompts, slot_options)
    from utils.hooks import patch, residual_cache
    from utils.paths import run_dir
    pc, s3 = cfg["patching"], cfg["s3"]
    model_id = args.model or cfg["model"]["primary"]
    out_dir = run_dir(cfg, model_id) / "patching"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = pair_members([json.loads(l) for l in open(Path(cfg["paths"]["vignettes_dir"]) / "pairs.jsonl")])
    pids = list(pairs)
    if args.dry_run:
        pids = pids[:4]
    shard_i, shard_n = parse_shard(args.shard)
    pids = pids[shard_i::shard_n]
    suffix = "_dryrun" if args.dry_run else ""
    tag = f".shard{shard_i}of{shard_n}" if shard_n > 1 else ""
    out_path = out_dir / f"patching{suffix}{tag}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["pair_id"] for l in open(out_path)}
    pids = [p for p in pids if p not in done]

    texts = cfg["advice_options"]["portfolio_choice"]
    equity = cfg["advice_options"]["equity_fractions"]
    perms = cyclic_permutations(len(texts), s3["n_permutations"])
    layers = pc["layers"]
    tok, model = load_model(model_id, cfg["model"]["dtype"], s3["attn_implementation"], args.device)
    framing = s3["prompts"]["framing"]

    def prep(v, perm):
        prompt = make_prompt(tok, build_user_message(framing, v["text"], slot_options(texts, perm)))
        s, e, n = narrative_span_tokens(tok, prompt, v["text"])
        return {"prompt": prompt, "span_end": e, "len": n}

    letter_ids = resolve_letter_token_ids(tok, prep(pairs[pids[0]]["lo"], perms[0])["prompt"], len(texts))
    print(f"patching: {len(pids)} pairs x 2 directions x {len(layers)} layers x {len(SITES)} sites "
          f"x {len(perms)} perms  (shard {shard_i}/{shard_n})", flush=True)

    bs = args.batch_size
    with open(out_path, "a") as f:
        for start in range(0, len(pids), bs // 2):
            chunk = pids[start:start + bs // 2]
            # units: (pair, direction, perm) with base/donor prepared prompts
            units = []
            for pid in chunk:
                for direction, (b, d) in {"lo_to_hi": ("lo", "hi"), "hi_to_lo": ("hi", "lo")}.items():
                    for pi, perm in enumerate(perms):
                        units.append({"pid": pid, "dir": direction, "pi": pi,
                                      "base": prep(pairs[pid][b], perm), "donor": prep(pairs[pid][d], perm)})
            res = {}          # (pid, dir, pi, key) -> letter probs
            for u0 in range(0, len(units), bs):
                ub = units[u0:u0 + bs]
                # clean donor pass, caching every sweep layer
                enc = tok([u["donor"]["prompt"] for u in ub], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(model.device)
                with residual_cache(model, layers) as store, torch.inference_mode():
                    try:            # only the last position's logits are read
                        out = model(**enc, logits_to_keep=1)
                    except TypeError:
                        out = model(**enc, num_logits_to_keep=1)
                logits = out.logits[:, -1, :].float()
                cols = torch.softmax(logits, -1)[:, letter_ids]
                lp = (cols / cols.sum(-1, keepdim=True)).cpu().tolist()
                pad_d = (enc["attention_mask"] == 0).sum(1).tolist()
                for j, u in enumerate(ub):
                    res[(u["pid"], u["dir"], u["pi"], "donor")] = lp[j]
                # clean base pass
                bp, _, _ = score_prompts(tok, model, [u["base"]["prompt"] for u in ub], letter_ids)
                for j, u in enumerate(ub):
                    res[(u["pid"], u["dir"], u["pi"], "base")] = bp[j]
                enc_b = tok([u["base"]["prompt"] for u in ub], return_tensors="pt", padding=True,
                            add_special_tokens=False)
                pad_b = (enc_b["attention_mask"] == 0).sum(1).tolist()
                for layer in layers:
                    for site in SITES:
                        rows_, pos_, donor_ = [], [], []
                        for j, u in enumerate(ub):
                            pb = site_positions(site, u["base"]["span_end"], u["base"]["len"])
                            pd = site_positions(site, u["donor"]["span_end"], u["donor"]["len"])
                            if len(pb) != len(pd):          # suffix tokenised differently: skip
                                continue
                            rows_ += [j] * len(pb)
                            pos_ += [pad_b[j] + p for p in pb]
                            donor_.append(store[layer][j, [pad_d[j] + p for p in pd]])
                        donor_t = torch.cat(donor_) if donor_ else torch.zeros(0)
                        with patch(model, layer, rows_, pos_, donor_t):
                            pp, _, _ = score_prompts(tok, model, [u["base"]["prompt"] for u in ub], letter_ids)
                        ok = set(rows_)
                        for j, u in enumerate(ub):
                            if j in ok:
                                res[(u["pid"], u["dir"], u["pi"], f"L{layer}:{site}")] = pp[j]
                del store
            # aggregate over perms -> one row per (pair, direction)
            for pid in chunk:
                for direction in ("lo_to_hi", "hi_to_lo"):
                    def agg(key):
                        per = [res.get((pid, direction, pi, key)) for pi in range(len(perms))]
                        if any(p is None for p in per):
                            return None
                        return round(aggressiveness(aggregate_allocation_probs(per, perms), equity), 5)
                    row = {"pair_id": pid, "direction": direction,
                           "field": pairs[pid]["lo"]["pair_field"], "factor": pairs[pid]["lo"]["pair_factor"],
                           "base": agg("base"), "donor": agg("donor"),
                           "patched": {f"L{l}:{s}": agg(f"L{l}:{s}") for l in layers for s in SITES}}
                    f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"  pairs done: {min(start + bs // 2, len(pids))}/{len(pids)}", flush=True)


# --------------------------------------------------------------------------------------
# Summary (CPU)
# --------------------------------------------------------------------------------------
def summarize(cfg, args):
    from utils.paths import run_dir
    pc = cfg["patching"]
    model_id = args.model or cfg["model"]["primary"]
    out_dir = run_dir(cfg, model_id) / "patching"
    suffix = "_dryrun" if args.dry_run else ""
    rows = []
    for p in sorted(out_dir.glob(f"patching{suffix}*.jsonl")):
        rows += [json.loads(l) for l in open(p)]
    sites = sorted(rows[0]["patched"])

    def flat(sub, site):
        return [{"base": r["base"], "donor": r["donor"], "patched": r["patched"][site]}
                for r in sub if r["patched"].get(site) is not None]

    grid = {}
    for site in sites:
        grid[site] = {d: flip_summary(flat([r for r in rows if r["direction"] == d], site),
                                      pc["min_behavior_gap"], pc["flip_effect_min"])
                      for d in ("lo_to_hi", "hi_to_lo")}
    dev = [r for r in rows if is_dev_pair(r["pair_id"])]
    rep = [r for r in rows if not is_dev_pair(r["pair_id"])]

    def both_dirs(sub, site):
        return min(flip_summary(flat([r for r in sub if r["direction"] == d], site),
                                pc["min_behavior_gap"], pc["flip_effect_min"])["flip_rate"]
                   for d in ("lo_to_hi", "hi_to_lo"))
    best = max(sites, key=lambda s: np.nan_to_num(both_dirs(dev, s), nan=-1))
    headline = {d: flip_summary(flat([r for r in rep if r["direction"] == d], best),
                                pc["min_behavior_gap"], pc["flip_effect_min"])
                for d in ("lo_to_hi", "hi_to_lo")}
    per_field = {}
    for field in sorted({r["field"] for r in rows}):
        sub = [r for r in rows if r["field"] == field]
        gaps = [abs(r["donor"] - r["base"]) for r in sub]
        per_field[field] = {"mean_behavior_gap": round(float(np.mean(gaps)), 4),
                            "at_best_site": {d: flip_summary(flat([r for r in sub if r["direction"] == d], best),
                                                             pc["min_behavior_gap"], pc["flip_effect_min"])
                                             for d in ("lo_to_hi", "hi_to_lo")}}
    passed = all(np.isfinite(v["flip_rate"]) and v["flip_rate"] >= pc["flip_rate_min"] for v in headline.values())
    report = {"n_rows": len(rows), "best_site_selected_on_dev": best, "headline_on_report_pairs": headline,
              "RQ2a_pass": bool(passed), "flip_rate_min": pc["flip_rate_min"],
              "per_field": per_field, "grid": grid}
    (out_dir / f"patching{suffix}.json").write_text(json.dumps(report, indent=2))
    print("=== S6 PATCHING (RQ2a) ===")
    for site in sites:
        g = grid[site]
        print(f"  {site:22s} lo->hi {g['lo_to_hi']['flip_rate']:.3f} (n={g['lo_to_hi']['n_eligible']})  "
              f"hi->lo {g['hi_to_lo']['flip_rate']:.3f}  effect {g['lo_to_hi']['mean_effect']:+.2f}/{g['hi_to_lo']['mean_effect']:+.2f}")
    print(f"  best (dev): {best}  report: {headline}")
    print(f"  RQ2a pass (>= {pc['flip_rate_min']} both directions): {'YES' if passed else 'NO'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "summarize"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    {"run": run, "summarize": summarize}[args.stage](cfg, args)


if __name__ == "__main__":
    main()
