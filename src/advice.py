"""S3: feed each vignette to frozen Gemma-2-9B-it as a financial advisor, read its
4-option portfolio choice as single-token logits, and write a continuous aggressiveness
score per (vignette, condition) to results/advice.jsonl.

Pipeline per vignette:
  - build the advisor prompt (client narrative + 4 lettered allocation options),
  - run it under N cyclic option-order permutations (cancels position + letter-token bias),
  - softmax the A/B/C/D logits, map each letter back to the allocation it held, average
    across permutations -> per-allocation probability -> aggressiveness = sum(p_i * equity_i),
  - flag hedging = probability mass landing OUTSIDE the letter tokens (the constrained
    readout can't "refuse", so refusal shows up as mass elsewhere).

Conditions: every vignette gets a `baseline` row; `contradictory` vignettes also get an
`instruction` row (prompt prepended with the integrate-holistically nudge) for the
two-condition experiment.

The torch/transformers imports are lazy so the pure scoring logic is unit-testable on a
machine without a GPU or PyTorch.

Usage:
    python src/advice.py [--config config.yaml] [--dry-run] [--model ID] [--device cuda]
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

LETTERS = "ABCDE"  # supports up to 5 options; we use 4


# --------------------------------------------------------------------------------------
# Pure logic (no torch) -- unit-tested in tests/test_s3.py
# --------------------------------------------------------------------------------------
def cyclic_permutations(n_options, n_rot=None):
    """`n_rot` cyclic rotations over `n_options` slots. With n_rot == n_options (the
    default) each allocation sits in each slot exactly once -> full bias cancellation."""
    n_rot = n_options if n_rot is None else n_rot
    return [[(slot + r) % n_options for slot in range(n_options)] for r in range(n_rot)]


def slot_options(allocation_texts, permutation):
    """The option strings as they appear in slots A,B,C,D for this permutation."""
    return [allocation_texts[alloc_idx] for alloc_idx in permutation]


def build_user_message(framing, narrative, opts_in_slots, integrate_instruction=None):
    """Assemble the advisor user turn. `opts_in_slots` are option strings in slot order."""
    parts = []
    if integrate_instruction:
        parts.append(integrate_instruction.strip())
    parts.append(framing.strip())
    parts.append(f"\nClient description:\n{narrative}")
    parts.append("\nPortfolio options:")
    parts += [f"{LETTERS[i]}) {opt}" for i, opt in enumerate(opts_in_slots)]
    parts.append("\nRecommend exactly one option for this client. Respond with only the letter.")
    return "\n".join(parts)


def aggregate_allocation_probs(perm_letter_probs, permutations):
    """Map each perm's per-slot letter probs back to allocations and average over perms."""
    n = len(permutations[0])
    acc = [0.0] * n
    for probs, perm in zip(perm_letter_probs, permutations):
        for slot, alloc_idx in enumerate(perm):
            acc[alloc_idx] += probs[slot]
    return [a / len(perm_letter_probs) for a in acc]


def aggressiveness(p_alloc, equity_fractions):
    """Continuous score = probability-weighted average equity share."""
    return float(sum(p * f for p, f in zip(p_alloc, equity_fractions)))


def is_hedged(p_letters_mean, frac_argmax_is_letter, threshold):
    """Hedged if the model puts little mass on the letters, or usually wants to say something else."""
    return (p_letters_mean < threshold) or (frac_argmax_is_letter < 0.5)


# --------------------------------------------------------------------------------------
# Model-dependent (lazy torch/transformers)
# --------------------------------------------------------------------------------------
def load_model(model_id, dtype, attn_impl, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"  # so logits[:, -1] is the real last token for every row
    kw = dict(attn_implementation=attn_impl)
    if device == "cuda":
        kw["device_map"] = "cuda"
    torch_dt = getattr(torch, dtype)
    try:                                            # `dtype` on new transformers
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dt, **kw)
    except TypeError:                               # `torch_dtype` on older transformers
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dt, **kw)
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return tok, model


def make_prompt(tok, user_msg, prefill="("):
    """Chat-templated user turn + assistant prefill so the next token is the choice letter."""
    chat = tok.apply_chat_template([{"role": "user", "content": user_msg}],
                                   tokenize=False, add_generation_prompt=True)
    return chat + prefill


def resolve_letter_token_ids(tok, sample_prompt, n):
    """Token id emitted for each letter right after the prompt prefix (handles spacing)."""
    base = tok(sample_prompt, add_special_tokens=False).input_ids
    ids = []
    for L in LETTERS[:n]:
        full = tok(sample_prompt + L, add_special_tokens=False).input_ids
        i = 0
        while i < len(base) and i < len(full) and base[i] == full[i]:
            i += 1
        if i >= len(full):
            raise ValueError(f"letter {L!r} did not produce a distinct token")
        ids.append(full[i])
    if len(set(ids)) != n:
        raise ValueError(f"letter token ids not distinct: {ids}")
    return ids


def score_prompts(tok, model, prompts, letter_ids):
    """One forward pass per batch. Returns per-prompt (letter_probs[n], p_letters, argmax_is_letter)."""
    import torch
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        logits = model(**enc).logits[:, -1, :].float()
    full = torch.softmax(logits, dim=-1)
    cols = full[:, letter_ids]                                  # [B, n]
    letter_probs = cols / cols.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    p_letters = cols.sum(dim=-1)
    argmax = logits.argmax(dim=-1)
    letter_set = set(letter_ids)
    argmax_is_letter = [int(a.item()) in letter_set for a in argmax]
    return letter_probs.cpu().tolist(), p_letters.cpu().tolist(), argmax_is_letter


def batched(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def load_vignettes(vdir):
    rows = []
    for name in ("explicit", "implicit", "pairs"):
        p = Path(vdir) / f"{name}.jsonl"
        if p.exists():
            rows += [json.loads(l) for l in open(p)]
    return rows


def stratified_sample(rows, n, seed):
    import random
    rng = random.Random(seed)
    by_tier = {}
    for r in rows:
        by_tier.setdefault(r["tier"], []).append(r)
    per = max(1, n // len(by_tier))
    out = []
    for tier, rs in by_tier.items():
        out += rng.sample(rs, min(per, len(rs)))
    # make sure some contradictory vignettes are present
    if not any(r["contradictory"] for r in out):
        contra = [r for r in rows if r["contradictory"]]
        if contra:
            out += rng.sample(contra, min(5, len(contra)))
    return out[:n] if len(out) > n else out


def build_units(vignettes, cfg, allocation_texts, permutations):
    """One unit per (vignette, condition, perm): the prompt text + bookkeeping."""
    framing = cfg["s3"]["prompts"]["framing"]
    instr = cfg["s3"]["prompts"]["integrate_instruction"]
    units = []
    for v in vignettes:
        conditions = [("baseline", None)]
        if v["contradictory"]:
            conditions.append(("instruction", instr))
        for cond, extra in conditions:
            for pi, perm in enumerate(permutations):
                msg = build_user_message(framing, v["text"], slot_options(allocation_texts, perm), extra)
                units.append({"vid": v["vignette_id"], "cond": cond, "perm_idx": pi,
                              "user_msg": msg, "v": v})
    return units


def run(cfg, args):
    s3 = cfg["s3"]
    allocation_texts = cfg["advice_options"]["portfolio_choice"]
    equity = cfg["advice_options"]["equity_fractions"]
    n = len(allocation_texts)
    permutations = cyclic_permutations(n, s3["n_permutations"])

    vignettes = load_vignettes(cfg["paths"]["vignettes_dir"])
    if args.dry_run:
        vignettes = stratified_sample(vignettes, s3["dry_run_n"], cfg["seed"])
    print(f"S3: {len(vignettes)} vignettes "
          f"({sum(v['contradictory'] for v in vignettes)} contradictory) x {n} perms")

    model_id = args.model or cfg["model"]["primary"]
    tok, model = load_model(model_id, cfg["model"]["dtype"], s3["attn_implementation"], args.device)

    units = build_units(vignettes, cfg, allocation_texts, permutations)
    prompts = [make_prompt(tok, u["user_msg"]) for u in units]
    letter_ids = resolve_letter_token_ids(tok, prompts[0], n)
    print(f"model={model_id}  letter token ids={letter_ids}")

    # sort by length for batching efficiency; remember original order
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    results = [None] * len(prompts)
    for batch_idx in batched(order, s3["batch_size"]):
        bp = [prompts[i] for i in batch_idx]
        lp, pl, aml = score_prompts(tok, model, bp, letter_ids)
        for j, i in enumerate(batch_idx):
            results[i] = (lp[j], pl[j], aml[j])
        print(f"  scored {sum(r is not None for r in results)}/{len(prompts)}", end="\r")
    print()

    # group units by (vid, cond) -> aggregate over perms
    groups = {}
    for u, res in zip(units, results):
        key = (u["vid"], u["cond"])
        groups.setdefault(key, {"v": u["v"], "perm": [None] * len(permutations)})
        groups[key]["perm"][u["perm_idx"]] = res

    out_rows = []
    for (vid, cond), g in groups.items():
        v = g["v"]
        letter_probs = [g["perm"][k][0] for k in range(len(permutations))]
        p_letters = [g["perm"][k][1] for k in range(len(permutations))]
        aml = [g["perm"][k][2] for k in range(len(permutations))]
        p_alloc = aggregate_allocation_probs(letter_probs, permutations)
        p_letters_mean = sum(p_letters) / len(p_letters)
        hedged = is_hedged(p_letters_mean, sum(aml) / len(aml), s3["hedge_p_letters_min"])
        out_rows.append({
            "vignette_id": vid, "profile_id": v["profile_id"], "pair_id": v.get("pair_id"),
            "vignette_type": v["vignette_type"], "tier": v["tier"],
            "risk_score": v["risk_score"], "contradictory": v["contradictory"],
            # factor columns: the gate splits on `conflicted`, and patterns/probes
            # need the per-factor targets alongside the blended tier.
            "willingness": v.get("willingness"), "capacity": v.get("capacity"),
            "goals": v.get("goals"), "factor_cell": v.get("factor_cell"),
            "conflicted": bool(v.get("conflicted", False)),
            "pair_field": v.get("pair_field"), "pair_factor": v.get("pair_factor"),
            "condition": cond, "p_alloc": [round(p, 5) for p in p_alloc],
            "aggressiveness": round(aggressiveness(p_alloc, equity), 5),
            "p_letters": round(p_letters_mean, 5), "hedged": bool(hedged),
        })

    # results/<model>/ -- the model is the directory, never the filename, so a 2B
    # dev run can never overwrite a 9B production run.
    from utils.paths import run_dir
    res_dir = run_dir(cfg, model_id)
    suffix = "_dryrun" if args.dry_run else ""
    out_path = res_dir / f"advice{suffix}.jsonl"
    with open(out_path, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    (res_dir / f"advice_meta{suffix}.json").write_text(json.dumps({
        "model": model_id, "n_permutations": len(permutations),
        "attn_implementation": s3["attn_implementation"], "letter_token_ids": letter_ids,
        "equity_fractions": equity, "seed": cfg["seed"],
        "framing": s3["prompts"]["framing"], "integrate_instruction": s3["prompts"]["integrate_instruction"],
    }, indent=2))

    hedge_rate = sum(r["hedged"] for r in out_rows) / len(out_rows)
    print(f"Wrote {len(out_rows)} rows to {out_path}  |  hedge rate: {hedge_rate:.3f}")
    print(f"  per-tier mean aggressiveness (baseline):")
    for tier in cfg["data"]["tiers"]:
        xs = [r["aggressiveness"] for r in out_rows if r["tier"] == tier and r["condition"] == "baseline"]
        if xs:
            print(f"    {tier}: {sum(xs)/len(xs):.3f}  (n={len(xs)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default=None, help="override config model.primary (e.g. google/gemma-2-2b-it)")
    ap.add_argument("--device", default="cuda", help="cuda | cpu")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg, args)


if __name__ == "__main__":
    main()
