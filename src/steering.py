"""S7 / RQ2b: steering. Push the model's internal client-risk direction and check that the
advice follows the push, in proportion to its size, while a random push of the same size
does not.

This is the causal half of the readout-collapse story. The probe shows the model REPRESENTS
the client's risk; steering asks whether that representation is a lever on the advice. If
it is, the knowledge the advice ignores is still usable from the inside.

Directions are diff-in-means (high-class mean minus low-class mean) at the probe's
dev-selected layer and position, fit on train-split profiles only (explicit + implicit
renderings, pairs excluded). Doses are multiples of each direction's OWN class-mean gap: a
dose of +1 moves a representation by one full low-to-high gap. Random controls are unit
directions rescaled to the tier gap's length.

Stages, because the cache (7.5 GB) lives on the Mac and the steered passes need a GPU:
  direction     CPU, next to the cache   -> results/<model>/steering/directions.npz (small;
                                            copy it to the GPU box with the code)
  run           GPU, 4-permutation advice readout per (direction, dose), checkpointed and
                shardable (--shard i/N, one process per GPU)  -> steering.jsonl
  summarize     CPU, dose-response + the pre-registered verdict -> steering.json + figure
  side-effects  GPU, MMLU / perplexity / refusals along the tier direction -> side_effects.json

Usage:
    python src/steering.py direction [--activations DIR] [--layer 22 --position profile_mean]
    python src/steering.py run [--model ID] [--batch-size 64] [--shard i/N] [--dry-run]
    python src/steering.py summarize
    python src/steering.py side-effects [--batch-size 16]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

DIRECTION_CLASSES = {          # name -> (label key, high class, low class)
    "tier": ("tier", "aggressive", "conservative"),
    "willingness": ("willingness", "high", "low"),
    "capacity": ("capacity", "high", "low"),
    "goals": ("goals", "high", "low"),
}


# --------------------------------------------------------------------------------------
# Pure logic -- unit-tested in tests/test_steering.py
# --------------------------------------------------------------------------------------
def class_label(lab, name):
    """The class a cached row belongs to for direction `name` (factor scores are binned)."""
    from rubric import factor_bin
    key = DIRECTION_CLASSES[name][0]
    return lab[key] if name == "tier" else factor_bin(float(lab[key]))


def diff_in_means(X, classes, hi, lo):
    """mean(X | hi) - mean(X | lo), plus the two class counts."""
    classes = np.asarray(classes)
    m_hi, m_lo = classes == hi, classes == lo
    if not m_hi.any() or not m_lo.any():
        raise ValueError(f"empty class: n[{hi}]={m_hi.sum()} n[{lo}]={m_lo.sum()}")
    return X[m_hi].mean(0) - X[m_lo].mean(0), int(m_hi.sum()), int(m_lo.sum())


def random_directions(d, n, norm, seed):
    """`n` random directions in R^d, each rescaled to length `norm`."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, d))
    return (v / np.linalg.norm(v, axis=1, keepdims=True) * norm).astype(np.float32)


def stratified_ids(ids, strata, n_per, seed):
    """Up to `n_per` ids from each stratum, seeded, in a stable order."""
    rng = np.random.default_rng(seed)
    out = []
    for s in sorted(set(strata)):
        pool = sorted(i for i, t in zip(ids, strata) if t == s)
        take = rng.choice(len(pool), size=min(n_per, len(pool)), replace=False)
        out += [pool[k] for k in sorted(take)]
    return out


def per_vignette_rho(rows):
    """Spearman(dose, aggressiveness) for each vignette across its doses.

    A vignette whose advice does not move at all has no defined rank correlation; it is
    returned as NaN and counted separately rather than dropped silently.
    """
    from scipy.stats import spearmanr
    by_vid = {}
    for r in rows:
        by_vid.setdefault(r["vignette_id"], []).append((r["dose"], r["aggressiveness"]))
    out = {}
    for vid, pts in by_vid.items():
        pts.sort()
        d, a = zip(*pts)
        out[vid] = float("nan") if len(set(a)) < 2 else float(spearmanr(d, a).correlation)
    return out


def bootstrap_mean_ci(x, n_boot=2000, seed=42):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def summarize_direction(rows, n_boot=2000, seed=42):
    """Dose-response for one direction: mean advice per dose, per-vignette rho, hedging."""
    doses = sorted({r["dose"] for r in rows})
    per_dose = {}
    for d in doses:
        xs = [r["aggressiveness"] for r in rows if r["dose"] == d]
        m, lo, hi = bootstrap_mean_ci(xs, n_boot, seed)
        per_dose[d] = {"mean_aggressiveness": round(m, 4), "ci": [round(lo, 4), round(hi, 4)],
                       "hedge_rate": round(float(np.mean([r["hedged"] for r in rows if r["dose"] == d])), 4),
                       "n": len(xs)}
    rhos = per_vignette_rho(rows)
    vals = list(rhos.values())
    m, lo, hi = bootstrap_mean_ci(vals, n_boot, seed)
    return {
        "per_dose": per_dose,
        "rho_mean": round(m, 4), "rho_ci": [round(lo, 4), round(hi, 4)],
        "n_vignettes": len(vals), "n_flat": int(sum(not np.isfinite(v) for v in vals)),
        "shift_full_range": round(per_dose[doses[-1]]["mean_aggressiveness"]
                                  - per_dose[doses[0]]["mean_aggressiveness"], 4),
    }


def steering_verdict(summaries, cfg_steer):
    """The pre-registered RQ2b call: the tier direction's mean per-vignette rho clears the
    bar AND every random direction stays flat.

    "Flat" is judged on effect SIZE, not rank: a random push that drifts the advice by 0.01
    still has |rho| near 1, because Spearman ignores magnitude. A random direction is flat
    if its full-range shift is at most `control_shift_ratio_max` of the tier direction's.
    (Amended 2026-10-01 after a 3-vignette dry run showed exactly that, before any real run.)
    """
    tier = summaries["tier"]["rho_mean"]
    tier_shift = abs(summaries["tier"]["shift_full_range"])
    randoms = {k: v["shift_full_range"] for k, v in summaries.items() if k.startswith("random_")}
    ratio = cfg_steer["control_shift_ratio_max"]
    flat = bool(randoms) and tier_shift > 0 and all(abs(v) <= ratio * tier_shift for v in randoms.values())
    return {
        "dose_spearman_min": cfg_steer["dose_spearman_min"],
        "control_shift_ratio_max": ratio,
        "tier_rho": tier, "tier_shift": round(tier_shift, 4), "random_shifts": randoms,
        "random_rhos": {k: v["rho_mean"] for k, v in summaries.items() if k.startswith("random_")},
        "tier_pass": bool(tier >= cfg_steer["dose_spearman_min"]),
        "controls_flat": flat,
        "RQ2b_clean_win": bool(tier >= cfg_steer["dose_spearman_min"] and flat),
        "n_tasks": 1,
        "note": "only portfolio_choice is implemented, so the '>= 2/3 tasks' clause reduces "
                "to the one task (deviation logged 2026-10-01, before any steering run)",
    }


# --------------------------------------------------------------------------------------
# Stage 1: directions (CPU, next to the activation cache)
# --------------------------------------------------------------------------------------
def resolve_site(cfg, args, res_dir):
    """Layer/position: CLI override, else the probe's dev-selected best cell."""
    st = cfg["steering"]
    layer = args.layer if args.layer is not None else st["layer"]
    position = args.position or st["position"]
    if layer == "probe_best" or position == "probe_best":
        best = json.loads((res_dir / "probes.json").read_text())["best"]
        layer = best["layer"] if layer == "probe_best" else layer
        position = best["position"] if position == "probe_best" else position
    return int(layer), position


def build_directions(cfg, args):
    from probes import make_splits, split_indices
    from utils.paths import run_dir
    st = cfg["steering"]
    model_id = args.model or cfg["model"]["primary"]
    res_dir = run_dir(cfg, model_id)
    act_dir = Path(args.activations) if args.activations else max(
        (res_dir / "activations").glob("*/acts_all.npy"), key=lambda p: p.stat().st_mtime).parent
    meta = json.loads((act_dir / "meta.json").read_text())
    labels = [json.loads(l) for l in open(act_dir / "labels.jsonl")]
    acts = np.load(act_dir / "acts_all.npy", mmap_mode="r")
    layer, position = resolve_site(cfg, args, res_dir)
    l_idx, p_idx = meta["layers"].index(layer), meta["positions"].index(position)
    print(f"directions: {model_id} cache {act_dir.name}  L{layer} @ {position}")
    plane = np.asarray(acts[:, l_idx, p_idx, :], dtype=np.float32)

    split = make_splits([lab["profile_id"] for lab in labels], seed=cfg["seed"])
    idx = split_indices(labels, split)
    fit_rows = np.concatenate([idx["train"], idx["implicit_train"]])

    names, vecs, info = [], [], {}
    for name in st["directions"]:
        _, hi, lo = DIRECTION_CLASSES[name]
        cls = [class_label(labels[i], name) for i in fit_rows]
        v, n_hi, n_lo = diff_in_means(plane[fit_rows], cls, hi, lo)
        names.append(name)
        vecs.append(v.astype(np.float32))
        info[name] = {"hi": hi, "lo": lo, "n_hi": n_hi, "n_lo": n_lo,
                      "gap_norm": round(float(np.linalg.norm(v)), 4)}
    tier_norm = float(np.linalg.norm(vecs[names.index("tier")]))
    rand = random_directions(plane.shape[1], st["n_random"], tier_norm, cfg["seed"] + 7)
    for k, v in enumerate(rand):
        names.append(f"random_{k}")
        vecs.append(v)
        info[f"random_{k}"] = {"gap_norm": round(tier_norm, 4)}

    # cosine between the named directions: are want / afford / goals separate levers?
    named = [n for n in names if not n.startswith("random_")]
    unit = {n: vecs[names.index(n)] / np.linalg.norm(vecs[names.index(n)]) for n in named}
    cosines = {f"{a}~{b}": round(float(unit[a] @ unit[b]), 4)
               for i, a in enumerate(named) for b in named[i + 1:]}

    # evaluation vignettes: implicit test split only (never seen by the directions)
    it = idx["implicit_test"]
    it_ids = [labels[i]["vignette_id"] for i in it]
    it_tier = [labels[i]["tier"] for i in it]
    core = stratified_ids(it_ids, it_tier, st["eval_per_tier"], cfg["seed"])
    contra = sorted(labels[i]["vignette_id"] for i in it if labels[i]["contradictory"])
    eval_ids = core + [v for v in contra if v not in set(core)] if st["eval_add_contradictory"] else core
    control = stratified_ids(core, [it_tier[it_ids.index(v)] for v in core],
                             st["control_n"] // 3, cfg["seed"] + 1)

    end_norm = float(np.linalg.norm(np.asarray(acts[fit_rows, l_idx,
                                                    meta["positions"].index("profile_end"), :],
                                               dtype=np.float32), axis=1).mean())
    out_dir = res_dir / "steering"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_out = {"model": model_id, "cache_id": meta.get("cache_id"), "layer": layer,
                "position": position, "fit_rows": int(len(fit_rows)), "directions": info,
                "cosines": cosines, "mean_token_norm_profile_end": round(end_norm, 4),
                "tier_gap_over_token_norm": round(tier_norm / end_norm, 4),
                "n_eval": len(eval_ids), "n_eval_core": len(core),
                "n_eval_contradictory": len(contra), "n_control": len(control)}
    np.savez(out_dir / "directions.npz", names=np.array(names), vectors=np.stack(vecs),
             eval_ids=np.array(eval_ids), core_ids=np.array(core),
             contradictory_ids=np.array(contra), control_ids=np.array(control),
             meta=json.dumps(meta_out))
    print(json.dumps(meta_out, indent=2))
    print(f"-> {out_dir / 'directions.npz'}")


def load_directions(path):
    z = np.load(path, allow_pickle=False)
    return {
        "names": [str(n) for n in z["names"]], "vectors": z["vectors"],
        "eval_ids": [str(v) for v in z["eval_ids"]], "core_ids": [str(v) for v in z["core_ids"]],
        "contradictory_ids": [str(v) for v in z["contradictory_ids"]],
        "control_ids": [str(v) for v in z["control_ids"]], "meta": json.loads(str(z["meta"])),
    }


# --------------------------------------------------------------------------------------
# Stage 2: steered advice (GPU)
# --------------------------------------------------------------------------------------
def build_units(dirs, doses, vignettes_by_id, framing, allocation_texts, permutations):
    """One unit per (direction, dose, vignette, perm). Dose 0 is the same unsteered pass for
    every direction, so it is scored once under the name 'none' and shared."""
    from advice import build_user_message, slot_options
    units, seen = [], set()
    for name in dirs["names"]:
        vids = dirs["control_ids"] if name.startswith("random_") else dirs["eval_ids"]
        for dose in doses:
            dname = "none" if dose == 0 else name
            for vid in vids:
                for pi, perm in enumerate(permutations):
                    key = (dname, dose, vid, pi)
                    if key in seen:
                        continue
                    seen.add(key)
                    msg = build_user_message(framing, vignettes_by_id[vid]["text"],
                                             slot_options(allocation_texts, perm))
                    units.append({"direction": dname, "dose": dose, "vid": vid,
                                  "perm_idx": pi, "user_msg": msg})
    return units


def run_steering(cfg, args):
    from advice import (aggregate_allocation_probs, aggressiveness, batched, cyclic_permutations,
                        is_hedged, load_checkpoint, load_model, load_vignettes, make_prompt,
                        parse_shard, resolve_letter_token_ids, score_prompts)
    from utils.hooks import steer
    from utils.paths import run_dir, text_digest
    st, s3 = cfg["steering"], cfg["s3"]
    model_id = args.model or cfg["model"]["primary"]
    out_dir = run_dir(cfg, model_id) / "steering"
    dirs = load_directions(out_dir / "directions.npz")
    layer = dirs["meta"]["layer"]
    if args.dry_run:                    # a few vignettes, every direction and dose
        dirs["eval_ids"] = dirs["eval_ids"][:3]
        dirs["control_ids"] = dirs["control_ids"][:2]
    vec = {n: dirs["vectors"][i] for i, n in enumerate(dirs["names"])}

    allocation_texts = cfg["advice_options"]["portfolio_choice"]
    equity = cfg["advice_options"]["equity_fractions"]
    permutations = cyclic_permutations(len(allocation_texts), s3["n_permutations"])
    vby = {v["vignette_id"]: v for v in load_vignettes(cfg["paths"]["vignettes_dir"])}
    units = build_units(dirs, st["doses"], vby, s3["prompts"]["framing"],
                        allocation_texts, permutations)
    print(f"steering: L{layer}  {len(dirs['names'])} directions x {len(st['doses'])} doses  "
          f"-> {len(units)} prompts")

    # tokenizer first: the checkpoint keys need the templated prompts, and a merge pass
    # (everything already scored by the shards) never needs the model on the GPU at all
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    model = None
    prompts = [make_prompt(tok, u["user_msg"]) for u in units]
    letter_ids = resolve_letter_token_ids(tok, prompts[0], len(allocation_texts))
    vdig = {n: text_digest(vec[n].tolist(), length=8) for n in vec}
    keys = [text_digest(model_id, p, u["direction"], u["dose"], layer,
                        vdig.get(u["direction"], "none"), length=16)
            for u, p in zip(units, prompts)]

    suffix = "_dryrun" if args.dry_run else ""
    shard_i, shard_n = parse_shard(args.shard)
    tag = f".shard{shard_i}of{shard_n}" if shard_n > 1 else ""
    done = {}
    for path in sorted(out_dir.glob(f"checkpoint{suffix}.*jsonl")):
        done.update(load_checkpoint(path))
    results = [done.get(k) for k in keys]
    if any(r is not None for r in results):
        print(f"resuming: {sum(r is not None for r in results)}/{len(units)} already scored")

    groups = {}
    for i, u in enumerate(units):
        if results[i] is None and i % shard_n == shard_i:
            groups.setdefault((u["direction"], u["dose"]), []).append(i)
    batch_size = args.batch_size or s3["batch_size"]
    if groups:
        tok, model = load_model(model_id, cfg["model"]["dtype"], s3["attn_implementation"], args.device)
    with open(out_dir / f"checkpoint{suffix}{tag}.jsonl", "a") as ckpt:
        for (dname, dose), idxs in groups.items():
            idxs.sort(key=lambda i: len(prompts[i]))
            with steer(model, layer, vec[dname] if dname != "none" else None, dose):
                for b in batched(idxs, batch_size):
                    lp, pl, aml = score_prompts(tok, model, [prompts[i] for i in b], letter_ids)
                    for j, i in enumerate(b):
                        results[i] = (lp[j], pl[j], aml[j])
                        ckpt.write(json.dumps({"k": keys[i], "r": results[i]}) + "\n")
                    ckpt.flush()
            print(f"  {dname} dose {dose:+}: {sum(r is not None for r in results)}/{len(units)}",
                  flush=True)
    if shard_n > 1:
        print(f"shard {shard_i}/{shard_n} done. Run without --shard to merge.")
        return

    agg = {}
    for u, res in zip(units, results):
        g = agg.setdefault((u["direction"], u["dose"], u["vid"]), [None] * len(permutations))
        g[u["perm_idx"]] = res
    rows = []
    for (dname, dose, vid), per in agg.items():
        p_alloc = aggregate_allocation_probs([p[0] for p in per], permutations)
        p_letters = sum(p[1] for p in per) / len(per)
        v = vby[vid]
        rows.append({"direction": dname, "dose": dose, "vignette_id": vid,
                     "tier": v["tier"], "risk_score": v["risk_score"],
                     "contradictory": v["contradictory"], "factor_cell": v.get("factor_cell"),
                     "p_alloc": [round(p, 5) for p in p_alloc],
                     "aggressiveness": round(aggressiveness(p_alloc, equity), 5),
                     "p_letters": round(p_letters, 5),
                     "hedged": bool(is_hedged(p_letters, sum(p[2] for p in per) / len(per),
                                              s3["hedge_p_letters_min"]))})
    out = out_dir / f"steering{suffix}.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")
    summarize(cfg, args)


# --------------------------------------------------------------------------------------
# Stage 3: summary + verdict (CPU)
# --------------------------------------------------------------------------------------
def direction_rows(rows, name, vids):
    """This direction's rows plus the shared dose-0 rows, restricted to `vids`."""
    vids = set(vids)
    return [r for r in rows if r["vignette_id"] in vids
            and (r["direction"] == name or (r["direction"] == "none" and r["dose"] == 0))]


def summarize(cfg, args):
    from utils.paths import run_dir
    st = cfg["steering"]
    model_id = args.model or cfg["model"]["primary"]
    out_dir = run_dir(cfg, model_id) / "steering"
    suffix = "_dryrun" if getattr(args, "dry_run", False) else ""
    dirs = load_directions(out_dir / "directions.npz")
    rows = [json.loads(l) for l in open(out_dir / f"steering{suffix}.jsonl")]
    have = {r["vignette_id"] for r in rows}

    summaries, contra = {}, {}
    for name in dirs["names"]:
        vids = dirs["control_ids"] if name.startswith("random_") else dirs["eval_ids"]
        sub = direction_rows(rows, name, [v for v in vids if v in have])
        if sub:
            summaries[name] = summarize_direction(sub, cfg["case_studies"]["n_boot"], cfg["seed"])
        if not name.startswith("random_"):
            sub_c = direction_rows(rows, name, [v for v in dirs["contradictory_ids"] if v in have])
            if sub_c:
                contra[name] = summarize_direction(sub_c, cfg["case_studies"]["n_boot"], cfg["seed"])
    report = {"meta": dirs["meta"], "doses": st["doses"],
              "verdict": steering_verdict(summaries, st) if "tier" in summaries else None,
              "directions": summaries, "say_vs_did_subset": contra}
    (out_dir / f"steering{suffix}.json").write_text(json.dumps(report, indent=2))
    plot_dose_response(summaries, out_dir / f"dose_response{suffix}.png")

    print("=== S7 STEERING (RQ2b) ===")
    for name, s in summaries.items():
        means = "  ".join(f"{d:+g}:{v['mean_aggressiveness']:.3f}" for d, v in s["per_dose"].items())
        print(f"  {name:12s} rho={s['rho_mean']:+.3f} {s['rho_ci']}  {means}")
    if report["verdict"]:
        v = report["verdict"]
        print(f"  [tier rho >= {v['dose_spearman_min']}] {'PASS' if v['tier_pass'] else 'FAIL'}  "
              f"[random shift <= {v['control_shift_ratio_max']}x tier shift] {'PASS' if v['controls_flat'] else 'FAIL'}")
        print(f"  RQ2b clean win: {'YES' if v['RQ2b_clean_win'] else 'NO'}")
    print(f"  -> {out_dir / f'steering{suffix}.json'}")


def plot_dose_response(summaries, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    randoms = [s for n, s in summaries.items() if n.startswith("random_")]
    if randoms:
        doses = sorted(randoms[0]["per_dose"])
        ys = np.array([[s["per_dose"][d]["mean_aggressiveness"] for d in doses] for s in randoms])
        ax.fill_between(doses, ys.min(0), ys.max(0), color="0.85", label="random directions")
    for name, s in summaries.items():
        if name.startswith("random_"):
            continue
        doses = sorted(s["per_dose"])
        ax.plot(doses, [s["per_dose"][d]["mean_aggressiveness"] for d in doses], marker="o", label=name)
    ax.set_xlabel("dose (multiples of the direction's class-mean gap)")
    ax.set_ylabel("mean advice aggressiveness (equity share)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Stage 4: side effects (GPU)
# --------------------------------------------------------------------------------------
def run_side_effects(cfg, args):
    from advice import load_model
    from side_effects import run_suite, side_effect_verdict
    from utils.paths import run_dir
    st = cfg["steering"]
    model_id = args.model or cfg["model"]["primary"]
    out_dir = run_dir(cfg, model_id) / "steering"
    dirs = load_directions(out_dir / "directions.npz")
    vec = dirs["vectors"][dirs["names"].index("tier")]
    tok, model = load_model(model_id, cfg["model"]["dtype"], cfg["s3"]["attn_implementation"], args.device)
    results = run_suite(tok, model, dirs["meta"]["layer"], vec, st["side_effect_doses"], st,
                        args.batch_size or 16, cfg["seed"])
    verdict = side_effect_verdict(results, st["side_effects"])
    (out_dir / "side_effects.json").write_text(json.dumps(
        {"direction": "tier", "layer": dirs["meta"]["layer"], "results": results,
         "bars": st["side_effects"], "verdict": verdict}, indent=2))
    print(f"side effects: safe |dose| <= {verdict['safe_abs_dose']}  -> {out_dir / 'side_effects.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["direction", "run", "summarize", "side-effects"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--activations", default=None, help="cache dir (default: newest for the model)")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--position", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--shard", default=None, help="i/N: score every N-th prompt (one process per GPU)")
    ap.add_argument("--dry-run", action="store_true", help="3 eval + 2 control vignettes, all doses")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    {"direction": build_directions, "run": run_steering, "summarize": summarize,
     "side-effects": run_side_effects}[args.stage](cfg, args)


if __name__ == "__main__":
    main()
