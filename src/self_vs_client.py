"""RQ2c: self vs. client. Does the client-risk direction encode the CLIENT's risk tolerance,
or the model's own (instructed) attitude to risk?

Steering (RQ2b) showed the client direction moves the advice. It cannot say whose risk it
carries: "this client is riskier" and "I like risk more" both push advice aggressive. So we
build SELF directions from one-line prefixes ("You are a risk-seeking / cautious person")
and compare them with client directions measured on the SAME tokens:
  lottery context   two-option "sure S vs. p chance of X" choices. The prefix names either
                    the model (S_lot: choose for yourself) or a client (C_lot: recommend for
                    your client); directions are the mean over the option-line tokens, which
                    are identical across prefixes. N_lot (cats / dogs) is an irrelevant null.
  advisor context   the advice prompt, prefixed "You, the advisor, are personally ..." (S_adv,
                    N_adv); mean over the client-narrative tokens, the site of the RQ2b
                    client directions.
Then three tests: geometry (cosine over layers, against a ladder), a 2x2 cross-steer (each
direction pushed on the advice task and on the model's own lottery choices) and a
contamination check (does a "risky advisor" read the same client as riskier?).

Pre-registration: config.yaml `self_vs_client:` (2026-10-20, commit bf05a96) and its
`amended:` block (2026-11-03, after three hostile reviews, before any gate item was scored).

Stages:
  prepare         CPU  fit / eval vignette ids from the cache labels -> ids.json
  pilot           GPU  pilot lotteries (scales disjoint from the real items): off the floor?
  directions      GPU  prefix diff-in-means, both contexts, L10-34 -> directions.npz
  behavior        GPU  dose-0 readouts: gates G0/G1, prefix effects on choices and advice
  cross           GPU  2x2 cross-steer on the shared absolute dose grid (checkpointed,
                       --shard i/N for one process per GPU)
  summarize       CPU  cosines, gates, leaks, verdict -> self_vs_client.json

Usage:
    python src/self_vs_client.py prepare
    python src/self_vs_client.py pilot | directions | behavior | cross [--shard i/N]
    python src/self_vs_client.py summarize
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

LETTERS = "AB"
TRAIT = {"hi": "a risk-seeking person who loves taking chances",
         "lo": "a cautious person who avoids taking chances"}
PREFIXES = {   # name -> (hi prefix, lo prefix); diff-in-means is always hi minus lo
    "S_lot": (f"You are {TRAIT['hi']}.", f"You are {TRAIT['lo']}."),
    "C_lot": (f"Your client is {TRAIT['hi']}.", f"Your client is {TRAIT['lo']}."),
    "N_lot": ("You are a person who loves cats.", "You are a person who loves dogs."),
    "S_adv": (f"You, the advisor, are personally {TRAIT['hi']}.",
              f"You, the advisor, are personally {TRAIT['lo']}."),
    "N_adv": ("You, the advisor, are personally a person who loves cats.",
              "You, the advisor, are personally a person who loves dogs."),
}
LOT_FRAMES = {
    "self": ("This is a controlled research simulation about decision making. There is no "
             "right answer. Choose the option that you yourself would prefer. Do not refuse "
             "and do not explain.", "Which do you choose?"),
    "client": ("This is a controlled research simulation about financial advice. There is no "
               "right answer. Choose the option you recommend for your client. Do not refuse "
               "and do not explain.", "Which do you recommend for your client?"),
}
REFERENT = {"S_lot": "self", "C_lot": "client", "N_lot": "self"}
SOFT = ("Based only on the client's description below,", "Based on the client's description below,")

# Non-numeric life choices, (safe, risky). Main and pilot sets never share an item.
VERBAL = [
    ("Keep a stable salaried job", "Quit to start your own company"),
    ("Rent the same apartment for another year", "Move to a new city without a job lined up"),
    ("Order the dish you always order", "Order a dish you have never heard of"),
    ("Take the guaranteed promotion at your current firm", "Join an early-stage startup for equity"),
    ("Drive the familiar highway route", "Try an unmapped shortcut through back roads"),
    ("Book a package tour with a fixed itinerary", "Travel abroad with no plans or bookings"),
    ("Stay in your current field", "Retrain for a completely different career"),
    ("Keep your savings in a bank account", "Put your savings into a friend's new restaurant"),
    ("Accept the first reasonable job offer", "Turn it down and hope for a better one"),
    ("Buy a reliable used car", "Buy a cheap project car that may not run"),
    ("Finish the degree you started", "Drop out to pursue an unproven idea"),
    ("Take the scheduled flight", "Fly standby for a chance at a free upgrade"),
    ("Sign a fixed-rate contract", "Sign a contract tied to market prices"),
    ("Hike the marked trail", "Go off-trail to find a hidden lake"),
    ("Stay with your current internet provider", "Switch to a new provider with no reviews"),
    ("Keep your hobby as a hobby", "Turn your hobby into your full-time income"),
    ("Live near family", "Move across the world for an adventure"),
    ("Take a salary", "Take commission only, with a higher possible income"),
    ("Cook a recipe you know well for guests", "Cook an ambitious new recipe for guests"),
    ("Keep the house you own", "Sell it and live on a sailboat"),
    ("Choose the well-known university", "Choose the new experimental program"),
    ("Renew your lease", "Buy a fixer-upper at auction"),
    ("Keep your current phone", "Preorder a first-generation device"),
    ("Stay at a chain hotel", "Stay at a stranger's place you found online"),
    ("Take the safe chess move", "Sacrifice your queen for a possible checkmate"),
    ("Accept a settlement offer", "Go to trial for a bigger possible award"),
    ("Keep your manager role", "Take an untested role on a new team"),
    ("Swim at the guarded beach", "Swim at an unguarded cove with better waves"),
    ("Keep a predictable routine", "Say yes to every unplanned invitation for a month"),
    ("Buy travel insurance", "Skip insurance and save the money"),
    ("Freelance for one steady client", "Freelance for many small unknown clients"),
    ("Plant the crops you grew last year", "Plant a new crop that might sell for more"),
    ("Hire the experienced candidate", "Hire the brilliant but unproven candidate"),
    ("Keep your band playing covers", "Switch to only your own new songs"),
    ("Retire at the planned age", "Retire early on a thinner cushion"),
    ("Take the bus you know", "Hitchhike to save time"),
    ("Keep your blog on a free platform", "Spend your savings to launch a media site"),
    ("Stay on the beginner ski slope", "Try the expert slope"),
    ("Take the job close to home", "Take the job abroad on a short visa"),
    ("Keep your name on a waiting list", "Pay a scalper for a ticket that may be fake"),
]
VERBAL_PILOT = [
    ("Stick with your current gym", "Sign up for an extreme training camp"),
    ("Keep your current haircut", "Let a trainee stylist try something radical"),
    ("Rent a car from a big company", "Rent from an unknown local lot at half price"),
    ("Keep your weekend free", "Enter a race you have not trained for"),
    ("Take the standard warranty", "Decline it and keep the cash"),
    ("Watch a movie you liked before", "Watch a random film with no reviews"),
]


# --------------------------------------------------------------------------------------
# Pure logic -- unit-tested in tests/test_self_vs_client.py
# --------------------------------------------------------------------------------------
def numeric_items(probs, ratios, scales, prefix="n", drop_dominated=True):
    """'Sure S vs. a p chance of X' for every (p, ratio, X). ratio = S / (p*X): ratio 1 is
    an exact expected-value tie, ratio > 1 makes the sure option the better deal. Items
    with S >= X are dominance questions, not risk questions, and are dropped."""
    items = []
    for p in probs:
        for ratio in ratios:
            for X in scales:
                S = round(ratio * p * X)
                if drop_dominated and S >= X:
                    continue
                if abs(S / (p * X) - ratio) > 0.01:      # rounding moved it off its label
                    continue
                items.append({"item_id": f"{prefix}_p{p}_r{ratio}_x{X}", "kind": "numeric",
                              "p": p, "ratio": ratio, "X": X, "S": S, "cell": f"p{p}_r{ratio}",
                              "safe": f"Receive ${S:,} for sure.",
                              "risky": f"A {round(p * 100)}% chance to receive ${X:,}, "
                                       f"otherwise nothing."})
    return items


def verbal_items(pairs, prefix="v"):
    return [{"item_id": f"{prefix}_{i}", "kind": "verbal", "cell": f"{prefix}_{i}",
             "safe": s + ".", "risky": r + "."} for i, (s, r) in enumerate(pairs)]


def split_by_cell(items, frac_fit, seed):
    """Fit/eval split by CELL: the payoff scales of one (p, ratio) cell are near-duplicates,
    so they land on the same side. Numeric and verbal are split separately."""
    fit, ev = [], []
    rng = np.random.default_rng(seed)
    for kind in ("numeric", "verbal"):
        sub = [it for it in items if it["kind"] == kind]
        cells = sorted({it["cell"] for it in sub})
        order = rng.permutation(len(cells))
        fit_cells = {cells[i] for i in order[:int(round(frac_fit * len(cells)))]}
        fit += [it for it in sub if it["cell"] in fit_cells]
        ev += [it for it in sub if it["cell"] not in fit_cells]
    return fit, ev


def options_block(item, gamble_first):
    opts = [item["risky"], item["safe"]] if gamble_first else [item["safe"], item["risky"]]
    return "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(opts))


def lottery_message(item, gamble_first, prefix=None, referent="self"):
    """User turn for one item. Returns (message, options text, gamble slot). The options
    text is the token span averaged for the lottery-context directions: it is identical for
    every prefix and both referents, so a diff-in-means there reads how the prefix changes
    the processing of the choice, not the prefix words."""
    framing, question = LOT_FRAMES[referent]
    opts = options_block(item, gamble_first)
    parts = ([prefix] if prefix else []) + [framing, "", question, opts, "",
                                            "Respond with only the letter."]
    return "\n".join(parts), opts, 0 if gamble_first else 1


def advice_message(framing, narrative, opts_in_slots, prefix=None, soft=False):
    from advice import build_user_message
    if soft:
        if SOFT[0] not in framing:
            raise ValueError("framing no longer contains the clause the soft variant edits")
        framing = framing.replace(*SOFT)
    msg = build_user_message(framing, narrative, opts_in_slots)
    return f"{prefix}\n{msg}" if prefix else msg


def p_gamble(letter_probs, gamble_slot):
    return float(letter_probs[gamble_slot])


def spearman(x, y):
    from scipy.stats import spearmanr
    if len(set(y)) < 2 or len(set(x)) < 2:
        return float("nan")
    return float(spearmanr(x, y).correlation)


def unit(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


def cos(a, b):
    return float(unit(a) @ unit(b))


def stratified(ids, strata, n_total, seed):
    """`n_total` ids spread evenly over strata (seeded, stable order)."""
    rng = np.random.default_rng(seed)
    groups = sorted(set(strata))
    per = n_total // len(groups)
    out = []
    for g in groups:
        pool = sorted(i for i, s in zip(ids, strata) if s == g)
        take = rng.choice(len(pool), size=min(per, len(pool)), replace=False)
        out += [pool[k] for k in sorted(take)]
    return out


def deal_tracking(rows):
    """G1 on unprompted numeric rows: mean over (p, X) of the within-cell Spearman(ratio,
    P(gamble)), and mean over (p, X) of P(gamble) at the lowest minus the highest ratio."""
    by = {}
    for r in rows:
        by.setdefault((r["p"], r["X"]), []).append((r["ratio"], r["p_gamble"]))
    rhos, slopes = [], []
    for pts in by.values():
        pts.sort()
        ratios, pg = zip(*pts)
        rho = spearman(ratios, pg)
        rhos.append(0.0 if not np.isfinite(rho) else rho)     # flat cell counts as 0
        slopes.append(pg[0] - pg[-1])
    return {"within_cell_rho": float(np.mean(rhos)), "slope": float(np.mean(slopes)),
            "n_cells": len(by)}


def paired_shifts(rows, kmax):
    """Per item: y at +kmax minus y at -kmax (each row: item, k, y). Items missing an end
    are skipped."""
    by = {}
    for r in rows:
        if abs(r["k"]) == kmax:
            by.setdefault(r["item"], {})[r["k"]] = r["y"]
    return {it: d[kmax] - d[-kmax] for it, d in by.items() if kmax in d and -kmax in d}


def per_item_rho(rows, kmax):
    """Spearman(k, y) per item over |k| <= kmax; a flat item counts as 0, not NaN."""
    by = {}
    for r in rows:
        if abs(r["k"]) <= kmax:
            by.setdefault(r["item"], []).append((r["k"], r["y"]))
    out = []
    for pts in by.values():
        pts.sort()
        k, y = zip(*pts)
        rho = spearman(k, y)
        out.append(0.0 if not np.isfinite(rho) else rho)
    return float(np.mean(out)) if out else float("nan")


def boot_means(values, n_boot, rng):
    v = np.asarray(list(values), float)
    if len(v) == 0:
        return np.full(n_boot, np.nan)
    return rng.choice(v, size=(n_boot, len(v)), replace=True).mean(1)


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [float("nan"), float("nan")]
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def abs_ratio(num, den):
    """|num / den|, with a non-positive on-task denominator (the direction did not push
    its own task the right way) mapped to +inf: undefined, so it can never look small."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, np.abs(num / np.where(den > 0, den, 1.0)), np.inf)


def ci_inf(x):
    """95% percentile interval that keeps +inf draws (order statistics, no interpolation)."""
    x = np.sort(np.asarray(x, float))
    if len(x) == 0 or np.isnan(x).all():
        return [float("nan"), float("nan")]
    lo, hi = x[int(0.025 * (len(x) - 1))], x[int(np.ceil(0.975 * (len(x) - 1)))]
    return [round(float(lo), 4), round(float(hi), 4)]


def leaks(shift, n_boot=2000, seed=42):
    """shift[(direction, task)] = {item: paired shift}. Self = S, client = C. Each task's
    items are resampled independently; the same resample is used for both directions on a
    task, so the ratio within a task column is paired. Leaks are magnitudes: an off-task
    push the WRONG way is still a push."""
    rng = np.random.default_rng(seed)
    out = {}
    for task in ("advice", "lottery"):
        items = sorted(set(shift[("S", task)]) & set(shift[("C", task)]))
        if not items:
            return {"leak_self": float("nan"), "leak_self_ci": [float("nan")] * 2,
                    "leak_client": float("nan"), "leak_client_ci": [float("nan")] * 2,
                    "product": float("nan"), "product_ci": [float("nan")] * 2, "missing": task}
        s = np.array([shift[("S", task)][i] for i in items])
        c = np.array([shift[("C", task)][i] for i in items])
        idx = rng.integers(0, len(items), size=(n_boot, len(items)))
        out[task] = {"S": s.mean(), "C": c.mean(), "S_b": s[idx].mean(1), "C_b": c[idx].mean(1),
                     "n": len(items)}
    a, l = out["advice"], out["lottery"]
    leak_self, leak_self_b = float(abs_ratio(a["S"], a["C"])), abs_ratio(a["S_b"], a["C_b"])
    leak_client, leak_client_b = float(abs_ratio(l["C"], l["S"])), abs_ratio(l["C_b"], l["S_b"])
    prod, prod_b = leak_self * leak_client, leak_self_b * leak_client_b
    res = {"leak_self": round(leak_self, 4), "leak_self_ci": ci_inf(leak_self_b),
           "leak_client": round(leak_client, 4), "leak_client_ci": ci_inf(leak_client_b),
           "product": round(prod, 4), "product_ci": ci_inf(prod_b)}
    for task, d in out.items():
        res[f"{task}_shifts"] = {"S": round(float(d["S"]), 4), "S_ci": ci(d["S_b"]),
                                 "C": round(float(d["C"]), 4), "C_ci": ci(d["C_b"]), "n": d["n"]}
    return res


def leak_verdict(lk):
    """Pre-registered (amended 2026-11-03) cross-steer call on the leak CIs."""
    if "missing" in lk:
        return "MISSING"
    s_lo, s_hi = lk["leak_self_ci"]
    c_lo, c_hi = lk["leak_client_ci"]
    if s_hi <= 1 / 3 and c_hi <= 1 / 3:
        return "DISTINCT"
    if (s_lo >= 2 / 3 and c_hi <= 1 / 3) or (c_lo >= 2 / 3 and s_hi <= 1 / 3):
        return "ASYMMETRIC"            # before CONFLATION: one huge leak can carry the product
    if lk["product_ci"][0] >= 4 / 9:
        return "CONFLATION"
    return "PARTIAL"


def common_kmax(hedge, ks, threshold):
    """Largest k such that every (direction, task) has hedge rate <= threshold at +-k and
    every smaller |k|. hedge[(direction, task, k)] = rate."""
    best = 0
    for k in sorted(x for x in ks if x > 0):
        ok = all(v <= threshold for (d, t, kk), v in hedge.items() if abs(kk) <= k)
        if not ok:
            break
        best = k
    return best


# --------------------------------------------------------------------------------------
# GPU helpers
# --------------------------------------------------------------------------------------
def score_units(tok, model, prompts, n_letters, batch_size):
    """(letter_probs, p_letters, argmax_is_letter) for each prompt, length-sorted batches."""
    from advice import batched, resolve_letter_token_ids, score_prompts
    letter_ids = resolve_letter_token_ids(tok, prompts[0], n_letters)
    out = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for b in batched(order, batch_size):
        lp, pl, aml = score_prompts(tok, model, [prompts[i] for i in b], letter_ids)
        for j, i in enumerate(b):
            out[i] = (lp[j], pl[j], aml[j])
    return out


def capture(tok, model, prompts, span_texts, layers, batch_size):
    """Residual after each layer in `layers`: mean over the token span of `span_texts[i]`
    and the last token. Returns float16 [n, len(layers), 2, d]."""
    import torch
    from activations import narrative_span_tokens
    from advice import batched
    from utils.hooks import residual_cache
    spans = [narrative_span_tokens(tok, p, t)[:2] for p, t in zip(prompts, span_texts)]
    out = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for b in batched(order, batch_size):
        enc = tok([prompts[i] for i in b], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with residual_cache(model, layers) as store, torch.inference_mode():
            try:
                model(**enc, logits_to_keep=1)
            except TypeError:
                model(**enc, num_logits_to_keep=1)
        attn = enc["attention_mask"]
        for r, i in enumerate(b):
            pad = int((attn[r] == 0).sum().item())
            s0, s1 = pad + spans[i][0], pad + spans[i][1]
            out[i] = np.stack([torch.stack([store[l][r, s0:s1 + 1].float().mean(0),
                                            store[l][r, -1].float()]).cpu().numpy()
                               for l in layers]).astype(np.float16)
    return np.stack(out)


def items_all(cfg):
    sv = cfg["self_vs_client"]
    lot, am = sv["lotteries"], sv["amended"]
    items, seen = [], set()
    for it in (numeric_items(lot["probs"], lot["sure_ratio"], am["lotteries"]["scales"])
               + verbal_items(VERBAL)):
        if (it["safe"], it["risky"]) not in seen:          # identical text = one item
            seen.add((it["safe"], it["risky"]))
            items.append(it)
    fit, ev = split_by_cell(items, lot["split"][0], cfg["seed"])
    return items, fit, ev


def load_ids(out_dir):
    return json.loads((out_dir / "ids.json").read_text())


def context(cfg, args):
    from utils.paths import run_dir
    model_id = args.model or cfg["model"]["primary"]
    out_dir = run_dir(cfg, model_id) / "self_vs_client"
    out_dir.mkdir(parents=True, exist_ok=True)
    return model_id, out_dir


def get_model(cfg, args, model_id):
    from advice import load_model
    return load_model(model_id, cfg["model"]["dtype"], cfg["s3"]["attn_implementation"], args.device)


# --------------------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------------------
def run_prepare(cfg, args):
    """Fit / eval vignette ids, from the same split the probes and RQ2b used."""
    from probes import make_splits, split_indices
    model_id, out_dir = context(cfg, args)
    from steering import load_directions
    rq2b = load_directions(out_dir.parent / "steering" / "directions.npz")
    act_dir = Path(args.activations) if args.activations else out_dir.parent / "activations" / rq2b["meta"]["cache_id"]
    labels = [json.loads(l) for l in open(act_dir / "labels.jsonl")]
    idx = split_indices(labels, make_splits([lab["profile_id"] for lab in labels], seed=cfg["seed"]))
    am = cfg["self_vs_client"]["amended"]
    tr = [labels[i] for i in idx["implicit_train"]]
    te = [labels[i] for i in idx["implicit_test"]]
    fit = stratified([l["vignette_id"] for l in tr], [l["tier"] for l in tr], am["fit_vignettes"], cfg["seed"])
    # eval: the RQ2b core vignettes first (so the client vectors are tested where RQ2b
    # tested them), topped up to eval_vignettes from the rest of implicit_test
    core = rq2b["core_ids"]
    assert len(core) <= am["eval_vignettes"]
    rest = [l for l in te if l["vignette_id"] not in set(core)]
    extra = stratified([l["vignette_id"] for l in rest], [l["tier"] for l in rest],
                       am["eval_vignettes"] - len(core), cfg["seed"] + 3)
    ev = list(core) + extra
    rand = stratified(ev, [next(l["tier"] for l in te if l["vignette_id"] == v) for v in ev],
                      200, cfg["seed"] + 5)
    assert not set(fit) & set(ev)
    ids = {"cache_id": act_dir.name, "fit_vignettes": fit, "eval_vignettes": ev, "random_vignettes": rand}
    (out_dir / "ids.json").write_text(json.dumps(ids, indent=1))
    print(f"prepare: fit {len(fit)}  eval {len(ev)} (core {len(core)})  random {len(rand)} -> {out_dir / 'ids.json'}")


def run_pilot(cfg, args):
    sv = cfg["self_vs_client"]
    model_id, out_dir = context(cfg, args)
    lot, am = sv["lotteries"], sv["amended"]
    items = (numeric_items(lot["probs"], lot["sure_ratio"], am["pilot_scales"], prefix="pilot")
             + verbal_items(VERBAL_PILOT, prefix="pilotv"))
    assert not set(am["pilot_scales"]) & set(am["lotteries"]["scales"])
    tok, model = get_model(cfg, args, model_id)
    from advice import make_prompt
    conds = [("none", None, "self"), ("S_hi", PREFIXES["S_lot"][0], "self"),
             ("S_lo", PREFIXES["S_lot"][1], "self"), ("C_hi", PREFIXES["C_lot"][0], "client"),
             ("C_lo", PREFIXES["C_lot"][1], "client")]
    units = [(c, it, gf) for c in conds for it in items for gf in (True, False)]
    prompts, slots = [], []
    for (name, pre, ref), it, gf in units:
        msg, _, slot = lottery_message(it, gf, pre, ref)
        prompts.append(make_prompt(tok, msg))
        slots.append(slot)
    scores = score_units(tok, model, prompts, 2, args.batch_size or 64)
    agg = {}
    for ((name, _, _), it, gf), slot, (lp, pl, aml) in zip(units, slots, scores):
        g = agg.setdefault((name, it["item_id"]), {"it": it, "pg": [], "pl": [], "aml": []})
        g["pg"].append(lp[slot]); g["pl"].append(pl); g["aml"].append(aml)
    rows = [{"cond": n, **{k: g["it"].get(k) for k in ("item_id", "kind", "p", "ratio", "X")},
             "p_gamble": round(float(np.mean(g["pg"])), 5), "p_letters": round(float(np.mean(g["pl"])), 5),
             "order_gap": round(float(g["pg"][0] - g["pg"][1]), 5),
             "hedged": bool(np.mean(g["pl"]) < 0.5 or np.mean(g["aml"]) < 0.5)}
            for (n, _), g in agg.items()]
    with open(out_dir / "pilot.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("=== RQ2c PILOT (pilot items only; no gate uses them) ===")
    for name, _, _ in conds:
        sub = [r for r in rows if r["cond"] == name]
        num = [r for r in sub if r["kind"] == "numeric"]
        dt = deal_tracking(num)
        print(f"  {name:5s} P(gamble) num {np.mean([r['p_gamble'] for r in num]):.3f} "
              f"verbal {np.mean([r['p_gamble'] for r in sub if r['kind'] == 'verbal']):.3f}  "
              f"hedged {np.mean([r['hedged'] for r in sub]):.2f}  p_letters {np.mean([r['p_letters'] for r in sub]):.3f}  "
              f"|order gap| {np.mean([abs(r['order_gap']) for r in sub]):.3f}  "
              f"G1 rho {dt['within_cell_rho']:+.3f} slope {dt['slope']:+.3f}")
    print(f"  -> {out_dir / 'pilot.jsonl'}")


def run_directions(cfg, args):
    """Prefix diff-in-means in both contexts, every layer in meta, span mean + last token.
    Split halves (by cell / by vignette) give the split-half reliability."""
    from advice import load_vignettes, make_prompt, slot_options
    sv = cfg["self_vs_client"]
    model_id, out_dir = context(cfg, args)
    layers = list(range(10, 35))
    _, fit, _ = items_all(cfg)
    if args.dry_run:
        fit = fit[:6]
    tok, model = get_model(cfg, args, model_id)
    bs = args.batch_size or 32
    rng = np.random.default_rng(cfg["seed"] + 11)
    out = {"layers": np.array(layers)}

    # lottery context
    cells = sorted({it["cell"] for it in fit})
    half_a = set(rng.permutation(cells)[: len(cells) // 2])
    rows = [(it, gf) for it in fit for gf in (True, False)]
    in_a = np.array([it["cell"] in half_a for it, _ in rows])
    for name in ("S_lot", "C_lot", "N_lot"):
        acts = []
        for pre in PREFIXES[name]:
            msgs = [lottery_message(it, gf, pre, REFERENT[name]) for it, gf in rows]
            acts.append(capture(tok, model, [make_prompt(tok, m) for m, _, _ in msgs],
                                [o for _, o, _ in msgs], layers, bs).astype(np.float32))
        diff = acts[0] - acts[1]                                  # paired per row
        out[name] = diff.mean(0)
        out[f"{name}_A"], out[f"{name}_B"] = diff[in_a].mean(0), diff[~in_a].mean(0)
        out[f"{name}_rows22"] = diff[:, layers.index(22), 0].astype(np.float16)   # for bootstraps
        out["lot_cells"] = np.array([it["cell"] for it, _ in rows])
        print(f"  {name}: L22 gap {np.linalg.norm(out[name][layers.index(22), 0]):.3f}", flush=True)

    # advisor context
    ids = load_ids(out_dir)
    vby = {v["vignette_id"]: v for v in load_vignettes(cfg["paths"]["vignettes_dir"])}
    fit_v = ids["fit_vignettes"][: 6 if args.dry_run else None]
    opts = slot_options(cfg["advice_options"]["portfolio_choice"], [0, 1, 2, 3])
    framing = cfg["s3"]["prompts"]["framing"]
    half_v = np.arange(len(fit_v)) % 2 == 0
    planes = {}
    for name, pre in [("none", None)] + [(f"{n}_{s}", PREFIXES[n][j]) for n in ("S_adv", "N_adv")
                                          for j, s in enumerate(("hi", "lo"))]:
        msgs = [advice_message(framing, vby[v]["text"], opts, pre) for v in fit_v]
        a = capture(tok, model, [make_prompt(tok, m) for m in msgs],
                    [vby[v]["text"] for v in fit_v], layers, bs).astype(np.float32)
        planes[name] = a
    for name in ("S_adv", "N_adv"):
        diff = planes[f"{name}_hi"] - planes[f"{name}_lo"]
        out[name] = diff.mean(0)
        out[f"{name}_A"], out[f"{name}_B"] = diff[half_v].mean(0), diff[~half_v].mean(0)
        out[f"{name}_rows22"] = diff[:, layers.index(22), 0].astype(np.float16)
        print(f"  {name}: L22 gap {np.linalg.norm(out[name][layers.index(22), 0]):.3f}", flush=True)
    for name, a in planes.items():                                # L22 span mean, for the probe
        out[f"plane22_{name}"] = a[:, layers.index(22), 0].astype(np.float16)
    out["fit_vignettes"] = np.array(fit_v)
    suffix = "_dryrun" if args.dry_run else ""
    np.savez(out_dir / f"directions{suffix}.npz", **out)
    print(f"-> {out_dir / f'directions{suffix}.npz'}")


def run_behavior(cfg, args):
    """Dose 0. Lotteries: every real item unprompted (G0, G1) plus eval items under each
    lottery prefix. Advice: eval vignettes under each advisor prefix, standard and soft
    framing. Rows -> behavior.jsonl."""
    from advice import (aggregate_allocation_probs, aggressiveness, cyclic_permutations,
                        load_vignettes, make_prompt, slot_options)
    sv = cfg["self_vs_client"]
    am = sv["amended"]
    model_id, out_dir = context(cfg, args)
    items, _, ev = items_all(cfg)
    ids = load_ids(out_dir)
    ev_v = ids["eval_vignettes"]
    if args.dry_run:
        items, ev, ev_v = items[:8], ev[:4], ev_v[:3]
    tok, model = get_model(cfg, args, model_id)
    bs = args.batch_size or 64
    rows = []

    lot_conds = [("none", None, "self", items)] + [
        (f"{n}_{s}", PREFIXES[n][j], REFERENT[n], ev) for n in ("S_lot", "C_lot", "N_lot")
        for j, s in enumerate(("hi", "lo"))]
    units, prompts = [], []
    for cname, pre, ref, its in lot_conds:
        for it in its:
            for gf in (True, False):
                msg, _, slot = lottery_message(it, gf, pre, ref)
                units.append((cname, it, slot)); prompts.append(make_prompt(tok, msg))
    scores = score_units(tok, model, prompts, 2, bs)
    agg = {}
    for (cname, it, slot), (lp, pl, aml) in zip(units, scores):
        g = agg.setdefault((cname, it["item_id"]), {"it": it, "pg": [], "pl": [], "aml": []})
        g["pg"].append(lp[slot]); g["pl"].append(pl); g["aml"].append(aml)
    for (cname, _), g in agg.items():
        rows.append({"task": "lottery", "cond": cname, "item": g["it"]["item_id"],
                     **{k: g["it"].get(k) for k in ("kind", "p", "ratio", "X")},
                     "p_gamble": round(float(np.mean(g["pg"])), 5),
                     "order_gap": round(float(g["pg"][0] - g["pg"][1]), 5),   # gamble first - second
                     "p_letters": round(float(np.mean(g["pl"])), 5),
                     "hedged": bool(np.mean(g["pl"]) < cfg["s3"]["hedge_p_letters_min"] or np.mean(g["aml"]) < 0.5)})

    vby = {v["vignette_id"]: v for v in load_vignettes(cfg["paths"]["vignettes_dir"])}
    alloc, equity = cfg["advice_options"]["portfolio_choice"], cfg["advice_options"]["equity_fractions"]
    perms = cyclic_permutations(len(alloc))[:: 4 // am["advice_permutations"]]
    framing = cfg["s3"]["prompts"]["framing"]
    adv_conds = [("none", None, False), ("none_soft", None, True)] + [
        (f"{n}_{s}{'_soft' if soft else ''}", PREFIXES[n][j], soft)
        for n in ("S_adv", "N_adv") for j, s in enumerate(("hi", "lo")) for soft in (False, True)
        if not (n == "N_adv" and soft)]
    units, prompts = [], []
    for cname, pre, soft in adv_conds:
        for v in ev_v:
            for pi, perm in enumerate(perms):
                units.append((cname, v, pi))
                prompts.append(make_prompt(tok, advice_message(framing, vby[v]["text"],
                                                               slot_options(alloc, perm), pre, soft)))
    scores = score_units(tok, model, prompts, len(alloc), bs)
    agg = {}
    for (cname, v, pi), sc in zip(units, scores):
        agg.setdefault((cname, v), [None] * len(perms))[pi] = sc
    for (cname, v), per in agg.items():
        p_alloc = aggregate_allocation_probs([p[0] for p in per], perms)
        pl = float(np.mean([p[1] for p in per]))
        rows.append({"task": "advice", "cond": cname, "item": v, "tier": vby[v]["tier"],
                     "aggressiveness": round(aggressiveness(p_alloc, equity), 5),
                     "p_letters": round(pl, 5),
                     "hedged": bool(pl < cfg["s3"]["hedge_p_letters_min"] or np.mean([p[2] for p in per]) < 0.5)})
    suffix = "_dryrun" if args.dry_run else ""
    with open(out_dir / f"behavior{suffix}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"behavior: {len(rows)} rows -> {out_dir / f'behavior{suffix}.jsonl'}")


def steer_vectors(cfg, out_dir, suffix=""):
    """Every steered direction at L22 as a UNIT vector (the dose grid supplies the norm)."""
    from steering import load_directions, random_directions
    am = cfg["self_vs_client"]["amended"]
    rq2b = load_directions(out_dir.parent / "steering" / "directions.npz")
    assert rq2b["meta"]["layer"] == 22 and rq2b["meta"]["position"] == "profile_mean"
    d = np.load(out_dir / f"directions{suffix}.npz")
    li = list(d["layers"]).index(22)
    vec = {"C_will": rq2b["vectors"][rq2b["names"].index("willingness")],
           "C_tier": rq2b["vectors"][rq2b["names"].index("tier")]}
    for name in ("S_lot", "C_lot", "N_lot", "S_adv"):
        vec[name] = d[name][li, 0]
    vec = {k: unit(v).astype(np.float32) for k, v in vec.items()}
    for i, v in enumerate(random_directions(len(vec["C_will"]), am["n_random"], 1.0, cfg["seed"] + 21)):
        vec[f"random_{i}"] = v
    return vec


def run_cross(cfg, args):
    """Cross-steer: each direction on both tasks over the shared absolute dose grid."""
    from advice import (aggregate_allocation_probs, aggressiveness, batched, cyclic_permutations,
                        load_checkpoint, load_vignettes, make_prompt, parse_shard,
                        resolve_letter_token_ids, score_prompts, slot_options)
    from utils.hooks import steer
    from utils.paths import text_digest
    am = cfg["self_vs_client"]["amended"]
    model_id, out_dir = context(cfg, args)
    suffix = "_dryrun" if args.dry_run else ""
    vec = steer_vectors(cfg, out_dir, suffix)
    _, _, ev = items_all(cfg)
    ids = load_ids(out_dir)
    ev_v, rand_v = ids["eval_vignettes"], ids["random_vignettes"]
    if args.dry_run:
        ev, ev_v, rand_v = ev[:4], ev_v[:3], rand_v[:2]
    vby = {v["vignette_id"]: v for v in load_vignettes(cfg["paths"]["vignettes_dir"])}
    alloc, equity = cfg["advice_options"]["portfolio_choice"], cfg["advice_options"]["equity_fractions"]
    perms = cyclic_permutations(len(alloc))[:: 4 // am["advice_permutations"]]
    framing = cfg["s3"]["prompts"]["framing"]
    unit_norm = am["dose_unit_norm"]
    ks = [k for k in am["doses_k"] if k != 0]
    kmax = max(ks)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    units = []   # (task, direction, k, item, sub, prompt)
    for name in vec:
        is_rand = name.startswith("random_")
        # randoms run at every k: if a dose is later dropped, the random floor must still
        # exist at the common |k|max, or G2's "4x random" would compare against nothing
        for k in ks + ([0] if name == "C_will" else []):
            dname = "none" if k == 0 else name
            for v in (rand_v if is_rand else ev_v):
                for pi, perm in enumerate(perms):
                    units.append(("advice", dname, k, v, pi, make_prompt(
                        tok, advice_message(framing, vby[v]["text"], slot_options(alloc, perm)))))
            for it in ev:
                for gf in (True, False):
                    msg, _, slot = lottery_message(it, gf)
                    units.append(("lottery", dname, k, it["item_id"], slot, make_prompt(tok, msg)))
    vdig = {n: text_digest(v.tolist(), length=8) for n, v in vec.items()}
    keys = [text_digest(model_id, u[5], u[1], u[2], 22, unit_norm, vdig.get(u[1], "none"), length=16)
            for u in units]
    shard_i, shard_n = parse_shard(args.shard)
    tag = f".shard{shard_i}of{shard_n}" if shard_n > 1 else ""
    done = {}
    for path in sorted(out_dir.glob(f"cross_checkpoint{suffix}.*jsonl")):
        done.update(load_checkpoint(path))
    results = [done.get(k) for k in keys]
    print(f"cross: {len(vec)} directions, k {ks}  -> {len(units)} prompts "
          f"({sum(r is not None for r in results)} already scored)", flush=True)
    groups = {}
    for i, u in enumerate(units):
        if results[i] is None and i % shard_n == shard_i:
            groups.setdefault((u[0], u[1], u[2]), []).append(i)
    if groups:
        tok, model = get_model(cfg, args, model_id)
        ids_by_task = {"advice": resolve_letter_token_ids(tok, units[0][5], len(alloc)),
                       "lottery": resolve_letter_token_ids(tok, next(u[5] for u in units if u[0] == "lottery"), 2)}
        bs = {"advice": args.batch_size or 64, "lottery": 2 * (args.batch_size or 64)}
        with open(out_dir / f"cross_checkpoint{suffix}{tag}.jsonl", "a") as ckpt:
            for (task, dname, k), idxs in groups.items():
                idxs.sort(key=lambda i: len(units[i][5]))
                with steer(model, 22, vec[dname] if dname != "none" else None, k * unit_norm):
                    for b in batched(idxs, bs[task]):
                        lp, pl, aml = score_prompts(tok, model, [units[i][5] for i in b], ids_by_task[task])
                        for j, i in enumerate(b):
                            results[i] = (lp[j], pl[j], aml[j])
                            ckpt.write(json.dumps({"k": keys[i], "r": results[i]}) + "\n")
                        ckpt.flush()
                print(f"  {task} {dname} k={k:+}: {sum(r is not None for r in results)}/{len(units)}", flush=True)
    if shard_n > 1:
        print(f"shard {shard_i}/{shard_n} done. Run without --shard to merge.")
        return

    agg = {}
    for u, r in zip(units, results):
        agg.setdefault(u[:4], []).append((u[4], r))
    rows = []
    for (task, dname, k, item), per in agg.items():
        if task == "advice":
            per.sort()
            p_alloc = aggregate_allocation_probs([r[0] for _, r in per], perms)
            y = aggressiveness(p_alloc, equity)
        else:
            y = float(np.mean([r[0][slot] for slot, r in per]))
            by_order = {("gamble_first" if slot == 0 else "gamble_second"): round(r[0][slot], 5)
                        for slot, r in per}
        pl = float(np.mean([r[1] for _, r in per]))
        rows.append({"task": task, "direction": dname, "k": k, "item": item, "y": round(y, 5),
                     **(by_order if task == "lottery" else {}),
                     "p_letters": round(pl, 5),
                     "hedged": bool(pl < cfg["s3"]["hedge_p_letters_min"] or np.mean([r[2] for _, r in per]) < 0.5)})
    with open(out_dir / f"cross{suffix}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"cross: {len(rows)} rows -> {out_dir / f'cross{suffix}.jsonl'}")


# --------------------------------------------------------------------------------------
# Summary (CPU)
# --------------------------------------------------------------------------------------
def direction_rows(rows, task, name):
    """This direction's rows on `task`, plus the shared k=0 rows for the items it ran on."""
    own = [r for r in rows if r["task"] == task and r["direction"] == name]
    items = {r["item"] for r in own}
    return own + [r for r in rows if r["task"] == task and r["k"] == 0 and r["item"] in items]


def spearman_brown(r):
    """Full-sample reliability from a split-half correlation."""
    return 2 * r / (1 + r) if r > -1 else float("nan")


def boot_disattenuated(a_rows, b, groups, rel_a, rel_b, n_boot, seed, b_rows=None, b_groups=None):
    """Bootstrap |cos(mean a, mean b)| / sqrt(rel_a * rel_b), clipped to 1. Rows are
    resampled by group (lottery cell or vignette). `b` is a fixed vector unless `b_rows`
    are given, in which case b is resampled the same way."""
    rng = np.random.default_rng(seed)
    a_rows = np.asarray(a_rows, np.float32)
    g = np.asarray(groups)
    ug = np.unique(g)
    members = {u: np.flatnonzero(g == u) for u in ug}
    den = np.sqrt(rel_a * rel_b) if rel_a > 0 and rel_b > 0 else float("nan")
    vals = []
    for _ in range(n_boot):
        pick = np.concatenate([members[u] for u in rng.choice(ug, len(ug), replace=True)])
        bv = b if b_rows is None else np.asarray(b_rows, np.float32)[pick].mean(0)
        vals.append(min(abs(cos(a_rows[pick].mean(0), bv)) / den, 1.0))
    return np.asarray(vals)


def run_summarize(cfg, args):
    from steering import DIRECTION_CLASSES, class_label, diff_in_means
    from probes import make_splits, split_indices
    sv = cfg["self_vs_client"]
    am, gates = sv["amended"], sv["amended"]["gates"]
    model_id, out_dir = context(cfg, args)
    suffix = "_dryrun" if args.dry_run else ""
    d = np.load(out_dir / f"directions{suffix}.npz")
    layers = [int(x) for x in d["layers"]]
    rep = {"pre_registration": "config.yaml self_vs_client (2026-10-20) + amended (2026-11-03)"}

    # ---- client directions per layer, refit on implicit_train only (cosine side)
    ids = load_ids(out_dir)
    act_dir = out_dir.parent / "activations" / ids["cache_id"]
    meta = json.loads((act_dir / "meta.json").read_text())
    labels = [json.loads(l) for l in open(act_dir / "labels.jsonl")]
    acts = np.load(act_dir / "acts_all.npy", mmap_mode="r")
    idx = split_indices(labels, make_splits([l["profile_id"] for l in labels], seed=cfg["seed"]))
    tr = idx["implicit_train"]
    pi = meta["positions"].index("profile_mean")
    client = {}
    for name in ("willingness", "tier"):
        _, hi, lo = DIRECTION_CLASSES[name]
        cls = [class_label(labels[i], name) for i in tr]
        client[name] = {}
        for l in layers:
            plane = np.asarray(acts[tr, meta["layers"].index(l), pi, :], dtype=np.float32)
            client[name][l] = diff_in_means(plane, cls, hi, lo)[0]
    sv_vec = steer_vectors(cfg, out_dir, suffix)
    rep["client_refit_vs_rq2b"] = {"willingness": round(cos(client["willingness"][22], sv_vec["C_will"]), 4),
                                   "tier": round(cos(client["tier"][22], sv_vec["C_tier"]), 4)}

    # ---- cosines: curve over layers + ladder at L22, raw and disattenuated
    def rel(name, l, pos=0):
        li = layers.index(l)
        return cos(d[f"{name}_A"][li, pos], d[f"{name}_B"][li, pos])
    rng = np.random.default_rng(cfg["seed"] + 31)
    rand = rng.standard_normal((5, d["S_lot"].shape[-1]))
    curve = {}
    for l in layers:
        li = layers.index(l)
        S, C, N, Sa = d["S_lot"][li, 0], d["C_lot"][li, 0], d["N_lot"][li, 0], d["S_adv"][li, 0]
        curve[l] = {"S_lot~C_lot": round(cos(S, C), 4), "S_lot~N_lot": round(cos(S, N), 4),
                    "C_lot~N_lot": round(cos(C, N), 4), "S_adv~C_will": round(cos(Sa, client["willingness"][l]), 4),
                    "S_adv~C_tier": round(cos(Sa, client["tier"][l]), 4),
                    "N_adv~C_will": round(cos(d["N_adv"][li, 0], client["willingness"][l]), 4),
                    "S_adv~S_lot": round(cos(Sa, S), 4), "C_lot~C_will": round(cos(C, client["willingness"][l]), 4),
                    "rel_S_lot": round(rel("S_lot", l), 4), "rel_C_lot": round(rel("C_lot", l), 4),
                    "rel_S_adv": round(rel("S_adv", l), 4),
                    "random~C_will": round(float(np.mean([abs(cos(r, client["willingness"][l])) for r in rand])), 4)}
    rep["cosine_curve"] = curve
    c22 = curve[22]
    li = layers.index(22)
    # split-half reliability of the client refit, by profile halves
    import zlib                                  # stable across runs, unlike hash()
    half = np.array([zlib.crc32(labels[i]["profile_id"].encode()) % 2 == 0 for i in tr])
    def client_rel(name):
        _, hi, lo = DIRECTION_CLASSES[name]
        plane = np.asarray(acts[tr, meta["layers"].index(22), pi, :], dtype=np.float32)
        cls = np.array([class_label(labels[i], name) for i in tr])
        a = diff_in_means(plane[half], cls[half], hi, lo)[0]
        b = diff_in_means(plane[~half], cls[~half], hi, lo)[0]
        return cos(a, b)
    r_cw = client_rel("willingness")
    # disattenuation: Spearman-Brown full-sample reliabilities, bootstrap over lottery
    # cells / advisor vignettes, |cos| clipped to 1; bars are applied to the CI
    sb = {n: spearman_brown(c22[f"rel_{n}"]) for n in ("S_lot", "C_lot", "S_adv")}
    sb["C_will"] = spearman_brown(r_cw)
    n_boot = 1000
    boots = {
        "S_lot~C_lot": boot_disattenuated(d["S_lot_rows22"], None, d["lot_cells"], sb["S_lot"], sb["C_lot"],
                                          n_boot, cfg["seed"], b_rows=d["C_lot_rows22"]),
        "S_adv~C_will": boot_disattenuated(d["S_adv_rows22"], client["willingness"][22],
                                           np.arange(len(d["S_adv_rows22"])), sb["S_adv"], sb["C_will"],
                                           n_boot, cfg["seed"]),
    }
    point = {"S_lot~C_lot": min(abs(c22["S_lot~C_lot"]) / np.sqrt(sb["S_lot"] * sb["C_lot"]), 1.0),
             "S_adv~C_will": min(abs(c22["S_adv~C_will"]) / np.sqrt(sb["S_adv"] * sb["C_will"]), 1.0)}
    geometry = {}
    for k, b in boots.items():
        lo, hi = ci(b)
        geometry[k] = ("distinct" if hi < sv["cosine_distinct_max"] else
                       "conflation" if lo > sv["cosine_conflation_min"] else "partial")
    if c22["rel_C_lot"] < gates["G3_split_half_min"]:
        geometry["S_lot~C_lot"] = "unreliable (rel C_lot below the G3 bar)"
    rep["cosine_L22"] = {"raw": c22, "rel_C_will": round(r_cw, 4),
                         "spearman_brown": {k: round(float(v), 4) for k, v in sb.items()},
                         "disattenuated": {k: round(float(v), 4) for k, v in point.items()},
                         "disattenuated_ci": {k: ci(b) for k, b in boots.items()},
                         "geometry": geometry,
                         "gap_norms": {n: round(float(np.linalg.norm(d[n][li, 0])), 4)
                                       for n in ("S_lot", "C_lot", "N_lot", "S_adv", "N_adv")}}
    Sp = d["S_lot"][li, 0] - (d["S_lot"][li, 0] @ unit(d["C_lot"][li, 0])) * unit(d["C_lot"][li, 0])
    rep["cosine_L22"]["S_perp~C_will_reported_only"] = round(cos(Sp, client["willingness"][22]), 4)

    # ---- contamination: frozen tier probe on S_adv-prefixed client activations
    wpath = out_dir.parent / "weights" / "probe_best_L22_profile_mean.npz"
    if wpath.exists():
        w = np.load(wpath, allow_pickle=True)
        rep["contamination"] = probe_contamination(w, {k[len("plane22_"):]: d[k] for k in d.files
                                                       if k.startswith("plane22_")})

    # ---- behavior: gates G0 / G1, prefix effects
    beh = [json.loads(l) for l in open(out_dir / f"behavior{suffix}.jsonl")]
    lot0 = [r for r in beh if r["task"] == "lottery" and r["cond"] == "none"]
    adv0 = [r for r in beh if r["task"] == "advice" and r["cond"] == "none"]
    dt = deal_tracking([r for r in lot0 if r["kind"] == "numeric"])
    hedge0 = {"lottery": float(np.mean([r["hedged"] for r in lot0])),
              "advice": float(np.mean([r["hedged"] for r in adv0]))}
    # G1 (amendment 2): a READING check under the S_lot prefixes on the eval items
    hi = {r["item"]: r for r in beh if r["task"] == "lottery" and r["cond"] == "S_lot_hi"}
    lo = {r["item"]: r for r in beh if r["task"] == "lottery" and r["cond"] == "S_lot_lo"}
    common = sorted(set(hi) & set(lo))
    gap = float(np.mean([hi[i]["p_gamble"] - lo[i]["p_gamble"] for i in common])) if common else float("nan")
    og = {s_: float(np.mean([abs(m[i]["order_gap"]) for i in common])) if common else float("nan")
          for s_, m in (("hi", hi), ("lo", lo))}
    rep["gates"] = {"G0_hedge_dose0": {k: round(v, 4) for k, v in hedge0.items()},
                    "G0_pass": all(v <= am["hedge_max_dose0"] for v in hedge0.values()),
                    "G1_prefix_gap": round(gap, 4), "G1_order_gap": {k: round(v, 4) for k, v in og.items()},
                    "G1_pass": bool(gap >= am["amended_G1_prefix_gap_min"]
                                    and all(v <= am["amended_G1_order_gap_max"] for v in og.values())),
                    "refit_pass": all(v >= 0.90 for v in rep["client_refit_vs_rq2b"].values())}
    rep["unprompted_deal_tracking_reported_only"] = {
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in dt.items()},
        "mean_abs_order_gap": round(float(np.mean([abs(r["order_gap"]) for r in lot0])), 4),
        "pre_registered_10_20_G1_would_pass": bool(dt["within_cell_rho"] <= gates["G1_within_cell_rho_max"]
                                                   and dt["slope"] >= gates["G1_slope_min"])}
    rep["unprompted_lottery"] = {
        "p_gamble_numeric": round(float(np.mean([r["p_gamble"] for r in lot0 if r["kind"] == "numeric"])), 4),
        "p_gamble_ties": round(float(np.mean([r["p_gamble"] for r in lot0 if r.get("ratio") == 1.0])), 4),
        "p_gamble_verbal": round(float(np.mean([r["p_gamble"] for r in lot0 if r["kind"] == "verbal"])), 4)}
    rep["prefix_effects"] = prefix_effects(beh)

    # ---- cross-steer
    rows = [json.loads(l) for l in open(out_dir / f"cross{suffix}.jsonl")]
    hedge = {}
    for r in rows:
        if r["k"] != 0:
            hedge.setdefault((r["direction"], r["task"], r["k"]), []).append(r["hedged"])
    hedge = {k: float(np.mean(v)) for k, v in hedge.items()}
    ks = sorted({r["k"] for r in rows if r["k"] > 0})
    kmax = common_kmax(hedge, ks, am["hedge_max_any_dose"])
    rep["cross"] = {"common_kmax": kmax, "hedge": {f"{d_}|{t}|{k}": round(v, 4) for (d_, t, k), v in hedge.items()}}
    if kmax == 0:
        rep["verdict"] = "NO VERDICT: every dose breaks some direction (hedge > threshold)"
    else:
        rep["cross"].update(cross_summary(rows, kmax, gates, cfg["seed"]))
        excl = cross_summary([r for r in rows if not r["hedged"]], kmax, gates, cfg["seed"])
        rep["cross"]["hedged_excluded"] = {"verdict": excl["primary"]["verdict"], "G2_pass": excl["G2"]["pass"],
                                           "client_floor_pass": excl["primary"]["client_floor_pass"]}
        rep["gates"]["G2_pass"] = rep["cross"]["G2"]["pass"]
        rep["gates"]["G3_pass"] = bool(c22["rel_S_lot"] >= gates["G3_split_half_min"]
                                       and c22["rel_S_adv"] >= gates["G3_split_half_min"])
        rep["verdict"] = final_verdict(rep, sv)
    (out_dir / f"self_vs_client{suffix}.json").write_text(json.dumps(rep, indent=2, default=float))
    print_report(rep, out_dir / f"self_vs_client{suffix}.json")
    if getattr(args, "wandb", False):
        log_wandb(rep, cfg, out_dir, suffix)


def log_wandb(rep, cfg, out_dir, suffix):
    """Results only (no private notes): config, gates, curves, cross table, verdict, and the
    raw result files as one artifact."""
    import wandb
    from utils.paths import git_commit
    run = wandb.init(project="mechanistic-kyc", job_type="rq2c", name=f"rq2c-self-vs-client{suffix}",
                     tags=["rq2c", "gemma-2-9b-it"],
                     config={"self_vs_client": cfg["self_vs_client"], "git_commit": git_commit()})
    wandb.log({"verdict": wandb.Html(f"<p>{rep['verdict']}</p>")})
    flat = {f"gates/{k}": v for k, v in rep["gates"].items() if isinstance(v, (bool, int, float))}
    wandb.summary.update({**flat, "verdict": rep["verdict"]})
    curve = rep["cosine_curve"]
    cols = ["layer"] + list(next(iter(curve.values())))
    wandb.log({"cosine_curve": wandb.Table(columns=cols, data=[[l] + list(v.values()) for l, v in curve.items()])})
    if "table" in rep.get("cross", {}):
        t = rep["cross"]["table"]
        wandb.log({"cross_table": wandb.Table(columns=["direction|task", "shift", "ci_lo", "ci_hi", "rho", "n"],
                                              data=[[k, v["shift"], *v["ci"], v["rho"], v["n"]] for k, v in t.items()])})
        p = rep["cross"]["pairs"]
        wandb.log({"leaks": wandb.Table(columns=["pair", "leak_self", "leak_client", "product", "verdict"],
                                        data=[[k, v["leak_self"], v["leak_client"], v["product"], v["verdict"]]
                                              for k, v in p.items()])})
        # dose-response per direction and task
        rows = [json.loads(l) for l in open(out_dir / f"cross{suffix}.jsonl")]
        for task in ("advice", "lottery"):
            ks = sorted({r["k"] for r in rows})
            names = sorted({r["direction"] for r in rows if r["direction"] != "none"})
            ys = [[float(np.mean([r["y"] for r in direction_rows(rows, task, n) if r["k"] == k] or [np.nan]))
                   for k in ks] for n in names]
            wandb.log({f"dose_response/{task}": wandb.plot.line_series(xs=ks, ys=ys, keys=names,
                                                                      title=f"{task}: mean readout by dose k",
                                                                      xname="k (x 13.4 norm)")})
    art = wandb.Artifact(f"rq2c-results{suffix}", type="results")
    for name in ("self_vs_client", "cross", "behavior", "pilot"):
        for ext in (".json", ".jsonl"):
            f = out_dir / f"{name}{suffix}{ext}"
            if f.exists():
                art.add_file(str(f))
    run.log_artifact(art)
    print(f"  wandb: {run.url}")
    run.finish()


def probe_contamination(w, planes):
    """Mean predicted P(aggressive) - P(conservative) of the frozen tier probe, per prefix."""
    coef, icpt = w["coef"], w["intercept"]
    mu, sd = w["scaler_mean"], w["scaler_scale"]       # probes.py saves the z-scorer here
    classes = [str(c) for c in w["classes"]]
    out = {}
    for name, X in planes.items():
        Z = (np.asarray(X, np.float32) - mu) / sd
        logits = Z @ coef.T + icpt
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        out[name] = round(float((p[:, classes.index("aggressive")] - p[:, classes.index("conservative")]).mean()), 4)
    return out


def prefix_effects(beh):
    """Dose-0 behavioral effect of each prefix pair (hi minus lo), paired by item."""
    by = {}
    for r in beh:
        by.setdefault((r["task"], r["cond"]), {})[r["item"]] = r.get("p_gamble", r.get("aggressiveness"))
    out = {}
    for (task, cond), vals in by.items():
        if not cond.startswith(("S_", "C_", "N_", "T_")) or "_hi" not in cond:
            continue
        lo = by.get((task, cond.replace("_hi", "_lo")), {})
        common = sorted(set(vals) & set(lo))
        diffs = np.array([vals[i] - lo[i] for i in common])
        rng = np.random.default_rng(7)
        out[f"{task}|{cond.replace('_hi', '')}"] = {"hi_minus_lo": round(float(diffs.mean()), 4),
                                                    "ci": ci(boot_means(diffs, 2000, rng)), "n": len(common)}
    return out


def cross_summary(rows, kmax, gates, seed):
    """Shifts for every direction, G2, and the primary (S_lot, C_will) leaks + verdict."""
    rng = np.random.default_rng(seed)
    names = sorted({r["direction"] for r in rows if r["direction"] != "none"})
    shift = {}
    table = {}
    for name in names:
        for task in ("advice", "lottery"):
            sub = direction_rows(rows, task, name)
            sh = paired_shifts(sub, kmax)
            if not sh:
                continue
            shift[(name, task)] = sh
            table[f"{name}|{task}"] = {"shift": round(float(np.mean(list(sh.values()))), 4),
                                       "abs_shift": round(float(np.mean(np.abs(list(sh.values())))), 4),
                                       "ci": ci(boot_means(sh.values(), 2000, rng)),
                                       "rho": round(per_item_rho(sub, kmax), 4), "n": len(sh)}
    # no random rows at kmax -> NaN, so every floor comparison fails instead of passing
    # mean |per-item shift|: a random push that moves items both ways must not cancel
    rmax = {t: max([table[f"{n}|{t}"]["abs_shift"] for n in names if n.startswith("random_")
                    and f"{n}|{t}" in table] or [float("nan")]) for t in ("advice", "lottery")}
    s_lot = table.get("S_lot|lottery", {"shift": float("nan"), "rho": float("nan")})
    g2 = {"rho": s_lot["rho"], "shift": s_lot["shift"], "random_max": rmax["lottery"],
          "pass": bool(s_lot["rho"] >= gates["G2_rho_min"] and s_lot["shift"] >= gates["G2_shift_min"]
                       and s_lot["shift"] >= gates["G2_random_multiple"] * rmax["lottery"])}
    c_adv = table.get("C_will|advice", {"shift": float("nan")})
    client_floor = bool(c_adv["shift"] >= gates["G2_random_multiple"] * rmax["advice"])
    pairs = {}
    for self_name in ("S_lot", "S_adv"):
        for client_name in ("C_will", "C_tier"):
            if all((n, t) in shift for n in (self_name, client_name) for t in ("advice", "lottery")):
                lk = leaks({("S", t): shift[(self_name, t)] for t in ("advice", "lottery")}
                           | {("C", t): shift[(client_name, t)] for t in ("advice", "lottery")}, seed=seed)
                lk["verdict"] = leak_verdict(lk)
                pairs[f"{self_name}~{client_name}"] = lk
    primary = dict(pairs.get("S_lot~C_will", {"verdict": "MISSING"}))
    primary["client_floor_pass"] = client_floor
    return {"table": table, "random_max": rmax, "G2": g2, "pairs": pairs, "primary": primary}


def final_verdict(rep, sv):
    g = rep["gates"]
    failed = [k for k in ("G0_pass", "G1_pass", "G2_pass", "G3_pass", "refit_pass") if not g.get(k)]
    if failed:
        return f"STOPPED: {', '.join(f.replace('_pass', '') for f in failed)} failed (see gates); no RQ2c verdict"
    prim, excl = rep["cross"]["primary"], rep["cross"]["hedged_excluded"]
    if not prim.get("client_floor_pass"):
        return "STOPPED: client direction's advice shift is not >= 4x random on the shared grid"
    cross = prim["verdict"]
    if (excl["verdict"], excl["G2_pass"], excl["client_floor_pass"]) != (cross, True, True):
        return (f"NO VERDICT: cross-steer {cross} with hedged rows, but hedged-excluded gives "
                f"{excl['verdict']} (G2 {excl['G2_pass']}, client floor {excl['client_floor_pass']})")
    geo = rep["cosine_L22"]["geometry"]
    if cross == "DISTINCT" and all(v == "distinct" for v in geo.values()):
        v = "DISTINCT"
    elif cross == "CONFLATION" or any(v == "conflation" for v in geo.values()):
        v = "CONFLATION"
    elif cross == "ASYMMETRIC":
        v = "ASYMMETRIC"
    else:
        v = "PARTIAL OVERLAP"
    return f"{v} (cross-steer {cross}; geometry {geo})"


def print_report(rep, path):
    print("=== RQ2c SELF vs CLIENT ===")
    print(f"  client refit vs RQ2b: {rep['client_refit_vs_rq2b']}")
    c = rep["cosine_L22"]
    print(f"  L22 cos raw: {c['raw']}")
    print(f"  L22 disattenuated: {c['disattenuated']} CI {c['disattenuated_ci']}  gap norms {c['gap_norms']}")
    print(f"  geometry: {c['geometry']}")
    print(f"  gates: {rep['gates']}")
    print(f"  unprompted deal tracking (reported only): {rep['unprompted_deal_tracking_reported_only']}")
    print(f"  unprompted lottery: {rep['unprompted_lottery']}")
    for k, v in rep["prefix_effects"].items():
        print(f"  prefix {k}: {v}")
    if "contamination" in rep:
        print(f"  contamination (tier probe, aggr-cons): {rep['contamination']}")
    if "table" in rep.get("cross", {}):
        print(f"  common kmax {rep['cross']['common_kmax']}")
        for k, v in rep["cross"]["table"].items():
            print(f"    {k:22s} shift {v['shift']:+.4f} {v['ci']} rho {v['rho']:+.3f} n {v['n']}")
        for k, v in rep["cross"]["pairs"].items():
            print(f"  {k}: leak_self {v['leak_self']} {v['leak_self_ci']}  leak_client {v['leak_client']} "
                  f"{v['leak_client_ci']}  product {v['product']} {v['product_ci']}  -> {v['verdict']}")
    print(f"  VERDICT: {rep['verdict']}")
    print(f"  -> {path}")


# ======================================================================================
# v2 (config amendment 3): confirmatory rerun. Same primary pair as run 1 (S_adv, C_will),
# calibrator prefixes on the same narrative tokens, a lottery readout fixed at its cause
# (forced single letter, piloted), fp32 logits, randoms out of the |k|max rule, a both-orders
# lever rule, and position-restricted steering (after the narrative only) for the mechanism.
# Results go to results/<model>/self_vs_client_v2/.
# ======================================================================================
TRAIT2 = {"hi": "love taking chances", "lo": "avoid taking chances"}
PREFIXES_V2 = {
    "S_adv": PREFIXES["S_adv"],
    "S_adv2": (f"As an advisor, you yourself {TRAIT2['hi']}.", f"As an advisor, you yourself {TRAIT2['lo']}."),
    "C_adv": (f"This client is personally {TRAIT['hi']}.", f"This client is personally {TRAIT['lo']}."),
    "T_adv": (f"Your neighbor is personally {TRAIT['hi']}.", f"Your neighbor is personally {TRAIT['lo']}."),
    "N_adv": PREFIXES["N_adv"],
}
LOT_V2 = {   # (framing, question, reply line, assistant prefill); piloted in config order
    "force_letter": ("This is a controlled research simulation about decision making. There is no "
                     "right answer. Choose the option that you yourself would prefer.",
                     "Which do you choose?", "Reply with exactly one letter: A or B.", "("),
    "force_letter_plain": ("This is a controlled research simulation about decision making. Choose "
                           "one of the two options.", "Which do you choose?",
                           "Reply with exactly one letter: A or B.", "("),
    # amendment 3b: the "(" prefill starts a parenthetical ("As a language model, I can't...")
    # and is the cause of the low letter mass; with no prefill letter mass is ~1.0
    "force_letter_noprefill": ("This is a controlled research simulation about decision making. There "
                               "is no right answer. Choose the option that you yourself would prefer.",
                               "Which do you choose?", "Reply with exactly one letter: A or B.", ""),
    "run1_noprefill": (LOT_FRAMES["self"][0], LOT_FRAMES["self"][1], "Respond with only the letter.", ""),
}
NAMED_V2 = ["C_will", "S_adv", "C_adv", "T_adv", "S_adv2", "S_lot_last"]
POST_V2 = ["S_adv", "C_will"]          # position-restricted, advice only
GATED_V2 = ["C_will", "S_adv", "C_adv", "T_adv", "S_adv2"]   # these set the common |k|max


def lottery_message_v2(item, gamble_first, variant, prefix=None):
    framing, question, reply, _ = LOT_V2[variant]
    opts = options_block(item, gamble_first)
    parts = ([prefix] if prefix else []) + [framing, "", question, opts, "", reply]
    return "\n".join(parts), opts, 0 if gamble_first else 1


def lottery_prompt_v2(tok, item, gamble_first, variant, prefix=None):
    """(templated prompt with this variant's prefill, options text, gamble slot)."""
    from advice import make_prompt
    msg, opts, slot = lottery_message_v2(item, gamble_first, variant, prefix)
    return make_prompt(tok, msg, prefill=LOT_V2[variant][3]), opts, slot


def hedged_v2(p_letters, frac_argmax_letter, rule, base_mass=None):
    """Absolute rule (letter mass < 0.5, or the argmax is usually not a letter); the
    pre-registered fallback is relative to the item's own dose-0 mass, floored at 0.25."""
    if frac_argmax_letter < 0.5:
        return True
    if rule == "absolute" or base_mass is None:
        return p_letters < 0.5
    return p_letters < max(0.25, 0.5 * base_mass)


def order_shift_ci(rows, kmax, n_boot=2000, seed=42):
    """Per option order: mean paired shift y(+kmax) - y(-kmax) with a bootstrap CI."""
    rng = np.random.default_rng(seed)
    out = {}
    for order in ("gamble_first", "gamble_second"):
        by = {}
        for r in rows:
            if abs(r["k"]) == kmax and order in r:
                by.setdefault(r["item"], {})[r["k"]] = r[order]
        sh = [d[kmax] - d[-kmax] for d in by.values() if kmax in d and -kmax in d]
        out[order] = {"shift": round(float(np.mean(sh)), 4) if sh else float("nan"),
                      "ci": ci(boot_means(sh, n_boot, rng)), "n": len(sh)}
    return out


def lever_ok(order_stats):
    """Both option orders move the same way, each with a CI that excludes 0."""
    a, b = order_stats["gamble_first"]["ci"], order_stats["gamble_second"]["ci"]
    return bool((a[0] > 0 and b[0] > 0) or (a[1] < 0 and b[1] < 0))


class Fp32Scorer:
    """Letter probabilities from fp32 final logits: final-norm hidden state x lm_head in fp32,
    Gemma-2 final soft-cap applied (bf16 logits sit on a 0.125 lattice). Optional steering at
    every position, or only from per-row start positions (unpadded indices)."""

    def __init__(self, model):
        self.model = model
        self.W = model.lm_head.weight.detach().float()
        self.cap = getattr(model.config, "final_logit_softcapping", None)

    def __call__(self, tok, prompts, letter_ids, layer=None, vec=None, alpha=0.0, starts=None):
        import contextlib
        import torch
        from utils.hooks import steer, steer_from
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        if vec is None or alpha == 0:
            ctx = contextlib.nullcontext()
        elif starts is None:
            ctx = steer(self.model, layer, vec, alpha)
        else:
            pad = (enc["attention_mask"] == 0).sum(1).tolist()
            ctx = steer_from(self.model, layer, vec, alpha, [p + s for p, s in zip(pad, starts)])
        with ctx, torch.inference_mode():
            h = self.model.model(**enc, use_cache=False).last_hidden_state[:, -1, :].float()
            logits = h @ self.W.T
            if self.cap:
                logits = torch.tanh(logits / self.cap) * self.cap
            full = torch.softmax(logits, dim=-1)
        cols = full[:, letter_ids]
        lp = (cols / cols.sum(-1, keepdim=True).clamp_min(1e-12)).cpu().tolist()
        pl = cols.sum(-1).cpu().tolist()
        am = [int(a) in set(letter_ids) for a in logits.argmax(-1).cpu().tolist()]
        return lp, pl, am


def v2_dir(cfg, args):
    from utils.paths import run_dir
    model_id = args.model or cfg["model"]["primary"]
    d = run_dir(cfg, model_id) / cfg["self_vs_client"]["v2"]["results_subdir"]
    d.mkdir(parents=True, exist_ok=True)
    return model_id, d


def items_v2(cfg):
    sv = cfg["self_vs_client"]
    lot, v2 = sv["lotteries"], sv["v2"]["lottery"]
    items, seen = [], set()
    for it in numeric_items(lot["probs"], lot["sure_ratio"], v2["scales"]) + verbal_items(VERBAL):
        if (it["safe"], it["risky"]) not in seen:
            seen.add((it["safe"], it["risky"]))
            items.append(it)
    fit, ev = split_by_cell(items, lot["split"][0], cfg["seed"] + 100)
    return items, fit, ev


def run_prepare2(cfg, args):
    from probes import make_splits, split_indices
    _, out = v2_dir(cfg, args)
    run1 = load_ids(out.parent / "self_vs_client")
    labels = [json.loads(l) for l in open(out.parent / "activations" / run1["cache_id"] / "labels.jsonl")]
    idx = split_indices(labels, make_splits([l["profile_id"] for l in labels], seed=cfg["seed"]))
    used = set(run1["eval_vignettes"])
    pool = [labels[i] for i in list(idx["implicit_val"]) + list(idx["implicit_test"])
            if labels[i]["vignette_id"] not in used]
    ev = stratified([l["vignette_id"] for l in pool], [l["tier"] for l in pool], 600, cfg["seed"] + 7)
    tier = {l["vignette_id"]: l["tier"] for l in pool}
    rand = stratified(ev, [tier[v] for v in ev], 100, cfg["seed"] + 9)
    assert not set(ev) & used and not set(ev) & set(run1["fit_vignettes"])
    ids = {"cache_id": run1["cache_id"], "fit_vignettes": run1["fit_vignettes"],
           "eval_vignettes": ev, "random_vignettes": rand}
    (out / "ids.json").write_text(json.dumps(ids, indent=1))
    from collections import Counter
    print(f"prepare2: fit {len(ids['fit_vignettes'])} (run-1 fit, implicit_train)  eval {len(ev)} "
          f"{dict(Counter(tier[v] for v in ev))}  random {len(rand)} -> {out / 'ids.json'}")


def _lottery_rows(units, scores, base=None):
    """Collapse (item, cond/direction, k) units over both orders."""
    agg = {}
    for u, (lp, pl, am) in zip(units, scores):
        g = agg.setdefault(u["key"], {"u": u, "by": {}, "pl": [], "am": []})
        g["by"]["gamble_first" if u["slot"] == 0 else "gamble_second"] = lp[u["slot"]]
        g["pl"].append(pl)
        g["am"].append(am)
    rows = []
    for g in agg.values():
        u = g["u"]
        rows.append({**u["meta"], "y": round(float(np.mean(list(g["by"].values()))), 5),
                     **{k: round(v, 5) for k, v in g["by"].items()},
                     "order_gap": round(g["by"].get("gamble_first", np.nan) - g["by"].get("gamble_second", np.nan), 5),
                     "p_letters": round(float(np.mean(g["pl"])), 5),
                     "frac_argmax_letter": round(float(np.mean(g["am"])), 3)})
    return rows


def _score_all(scorer, tok, units, letter_ids, bs, layer=22, vec=None, alpha=0.0):
    out = [None] * len(units)
    order = sorted(range(len(units)), key=lambda i: len(units[i]["prompt"]))
    from advice import batched
    for b in batched(order, bs):
        starts = [units[i]["start"] for i in b] if units[b[0]].get("start") is not None else None
        lp, pl, am = scorer(tok, [units[i]["prompt"] for i in b], letter_ids, layer, vec, alpha, starts)
        for j, i in enumerate(b):
            out[i] = (lp[j], pl[j], am[j])
    return out


def run_pilot2(cfg, args):
    from advice import make_prompt, resolve_letter_token_ids, score_prompts
    sv = cfg["self_vs_client"]
    v2 = sv["v2"]["lottery"]
    model_id, out = v2_dir(cfg, args)
    items = (numeric_items(sv["lotteries"]["probs"], sv["lotteries"]["sure_ratio"], v2["pilot_scales"], prefix="pilot2")
             + verbal_items(VERBAL_PILOT, prefix="pilot2v"))
    used = set(v2["scales"]) | set(sv["amended"]["lotteries"]["scales"]) | set(sv["amended"]["pilot_scales"])
    assert not set(v2["pilot_scales"]) & used
    tok, model = get_model(cfg, args, model_id)
    scorer = Fp32Scorer(model)
    bs = args.batch_size or 64
    report = {"variants": {}}
    for variant in v2["prompt_variants"]:
        stats = {}
        for cname, pre in (("none", None), ("S_hi", PREFIXES["S_lot"][0]), ("S_lo", PREFIXES["S_lot"][1])):
            units = []
            for it in items:
                for gf in (True, False):
                    prompt, _, slot = lottery_prompt_v2(tok, it, gf, variant, pre)
                    units.append({"key": it["item_id"], "slot": slot, "prompt": prompt,
                                  "meta": {"item": it["item_id"], "kind": it["kind"]}})
            ids = resolve_letter_token_ids(tok, units[0]["prompt"], 2)
            if variant == v2["prompt_variants"][0] and cname == "none":   # fp32 vs bf16 sanity
                bf = score_prompts(tok, model, [u["prompt"] for u in units[:32]], ids)
                f32 = scorer(tok, [u["prompt"] for u in units[:32]], ids)
                report["fp32_vs_bf16_max_abs"] = round(float(np.max(np.abs(np.array(bf[0]) - np.array(f32[0])))), 4)
                report["fp32_vs_bf16_letter_mass_max_abs"] = round(float(np.max(np.abs(np.array(bf[1]) - np.array(f32[1])))), 4)
            rows = _lottery_rows(units, _score_all(scorer, tok, units, ids, bs))
            stats[cname] = {"letter_mass": round(float(np.mean([r["p_letters"] for r in rows])), 4),
                            "abs_order_gap": round(float(np.mean([abs(r["order_gap"]) for r in rows])), 4),
                            "p_gamble": round(float(np.mean([r["y"] for r in rows])), 4)}
        ok = (stats["none"]["letter_mass"] >= v2["pilot_letter_mass_min"]
              and stats["none"]["abs_order_gap"] <= v2["pilot_order_gap_max"])
        report["variants"][variant] = {**stats, "passes": ok}
        print(f"  {variant}: {stats}  passes={ok}", flush=True)
    chosen = next((v for v in v2["prompt_variants"] if report["variants"][v]["passes"]), None)
    report["chosen_variant"] = chosen or v2["prompt_variants"][0]
    report["hedge_rule"] = "absolute" if chosen else "relative"
    (out / "pilot.json").write_text(json.dumps(report, indent=2))
    print(f"pilot2: variant {report['chosen_variant']}, hedge rule {report['hedge_rule']}, "
          f"fp32-vs-bf16 max |dp| {report.get('fp32_vs_bf16_max_abs')} -> {out / 'pilot.json'}")


def run_directions2(cfg, args):
    from advice import load_vignettes, make_prompt, slot_options
    model_id, out = v2_dir(cfg, args)
    variant = json.loads((out / "pilot.json").read_text())["chosen_variant"]
    layers = list(range(10, 35))
    _, fit, _ = items_v2(cfg)
    ids = load_ids(out)
    fit_v = ids["fit_vignettes"]
    if args.dry_run:
        fit, fit_v = fit[:6], fit_v[:6]
    tok, model = get_model(cfg, args, model_id)
    bs = args.batch_size or 32
    res = {"layers": np.array(layers)}
    vby = {v["vignette_id"]: v for v in load_vignettes(cfg["paths"]["vignettes_dir"])}
    opts = slot_options(cfg["advice_options"]["portfolio_choice"], [0, 1, 2, 3])
    framing = cfg["s3"]["prompts"]["framing"]
    half_v = np.arange(len(fit_v)) % 2 == 0
    texts = [vby[v]["text"] for v in fit_v]
    for name, (hi, lo) in PREFIXES_V2.items():
        a = [capture(tok, model, [make_prompt(tok, advice_message(framing, t, opts, pre)) for t in texts],
                     texts, layers, bs).astype(np.float32) for pre in (hi, lo)]
        diff = a[0] - a[1]
        res[name] = diff.mean(0)
        res[f"{name}_A"], res[f"{name}_B"] = diff[half_v].mean(0), diff[~half_v].mean(0)
        res[f"{name}_rows22"] = diff[:, layers.index(22), 0].astype(np.float16)
        res[f"plane22_{name}_hi"] = a[0][:, layers.index(22), 0].astype(np.float16)
        res[f"plane22_{name}_lo"] = a[1][:, layers.index(22), 0].astype(np.float16)
        print(f"  {name}: L22 gap {np.linalg.norm(res[name][layers.index(22), 0]):.3f}", flush=True)
    none = capture(tok, model, [make_prompt(tok, advice_message(framing, t, opts)) for t in texts],
                   texts, layers, bs).astype(np.float32)
    res["plane22_none"] = none[:, layers.index(22), 0].astype(np.float16)
    rows = [(it, gf) for it in fit for gf in (True, False)]
    a = []
    for pre in PREFIXES["S_lot"]:
        msgs = [lottery_prompt_v2(tok, it, gf, variant, pre) for it, gf in rows]
        a.append(capture(tok, model, [m for m, _, _ in msgs], [o for _, o, _ in msgs],
                         layers, bs).astype(np.float32))
    res["S_lot"] = (a[0] - a[1]).mean(0)
    print(f"  S_lot: L22 gap mean {np.linalg.norm(res['S_lot'][layers.index(22), 0]):.3f} "
          f"last {np.linalg.norm(res['S_lot'][layers.index(22), 1]):.3f}", flush=True)
    res["fit_vignettes"] = np.array(fit_v)
    suffix = "_dryrun" if args.dry_run else ""
    np.savez(out / f"directions{suffix}.npz", **res)
    print(f"-> {out / f'directions{suffix}.npz'}")


def run_behavior2(cfg, args):
    from advice import (aggregate_allocation_probs, aggressiveness, cyclic_permutations, load_vignettes,
                        make_prompt, resolve_letter_token_ids, slot_options)
    model_id, out = v2_dir(cfg, args)
    variant = json.loads((out / "pilot.json").read_text())["chosen_variant"]
    items, _, ev = items_v2(cfg)
    ev_v = load_ids(out)["eval_vignettes"]
    if args.dry_run:
        items, ev, ev_v = items[:8], ev[:4], ev_v[:3]
    tok, model = get_model(cfg, args, model_id)
    scorer = Fp32Scorer(model)
    bs = args.batch_size or 64
    rows = []
    units = []
    for cname, pre, its in [("none", None, items)] + [(f"S_lot_{s}", PREFIXES["S_lot"][j], ev)
                                                      for j, s in enumerate(("hi", "lo"))]:
        for it in its:
            for gf in (True, False):
                prompt, _, slot = lottery_prompt_v2(tok, it, gf, variant, pre)
                units.append({"key": (cname, it["item_id"]), "slot": slot, "prompt": prompt,
                              "meta": {"task": "lottery", "cond": cname, "item": it["item_id"],
                                       **{k: it.get(k) for k in ("kind", "p", "ratio", "X")}}})
    rows += _lottery_rows(units, _score_all(scorer, tok, units, resolve_letter_token_ids(tok, units[0]["prompt"], 2), bs))

    vby = {v["vignette_id"]: v for v in load_vignettes(cfg["paths"]["vignettes_dir"])}
    alloc, equity = cfg["advice_options"]["portfolio_choice"], cfg["advice_options"]["equity_fractions"]
    perms = cyclic_permutations(len(alloc))[::2]
    framing = cfg["s3"]["prompts"]["framing"]
    conds = [("none", None)] + [(f"{n}_{s}", PREFIXES_V2[n][j]) for n in PREFIXES_V2 for j, s in enumerate(("hi", "lo"))]
    units = [{"cond": c, "v": v, "pi": pi, "prompt": make_prompt(tok, advice_message(framing, vby[v]["text"], slot_options(alloc, perm), pre))}
             for c, pre in conds for v in ev_v for pi, perm in enumerate(perms)]
    scores = _score_all(scorer, tok, units, resolve_letter_token_ids(tok, units[0]["prompt"], len(alloc)), bs)
    agg = {}
    for u, sc in zip(units, scores):
        agg.setdefault((u["cond"], u["v"]), [None] * len(perms))[u["pi"]] = sc
    for (c, v), per in agg.items():
        p_alloc = aggregate_allocation_probs([p[0] for p in per], perms)
        rows.append({"task": "advice", "cond": c, "item": v, "tier": vby[v]["tier"],
                     "aggressiveness": round(aggressiveness(p_alloc, equity), 5),
                     "p_letters": round(float(np.mean([p[1] for p in per])), 5),
                     "frac_argmax_letter": round(float(np.mean([p[2] for p in per])), 3)})
    suffix = "_dryrun" if args.dry_run else ""
    with open(out / f"behavior{suffix}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"behavior2: {len(rows)} rows -> {out / f'behavior{suffix}.jsonl'}")


def steer_vectors_v2(cfg, out, suffix=""):
    from steering import load_directions, random_directions
    rq2b = load_directions(out.parent / "steering" / "directions.npz")
    assert rq2b["meta"]["layer"] == 22 and rq2b["meta"]["position"] == "profile_mean"
    d = np.load(out / f"directions{suffix}.npz")
    li = list(d["layers"]).index(22)
    vec = {"C_will": rq2b["vectors"][rq2b["names"].index("willingness")]}
    for n in ("S_adv", "C_adv", "T_adv", "S_adv2"):
        vec[n] = d[n][li, 0]                      # narrative mean (primary steering site)
    vec["S_lot_last"] = d["S_lot"][li, 1]         # last token (exploratory)
    vec = {k: unit(v).astype(np.float32) for k, v in vec.items()}
    for i, v in enumerate(random_directions(len(vec["C_will"]), cfg["self_vs_client"]["v2"]["n_random"],
                                            1.0, cfg["seed"] + 41)):
        vec[f"random_{i}"] = v
    return vec


def run_cross2(cfg, args):
    from activations import narrative_span_tokens
    from advice import (aggregate_allocation_probs, aggressiveness, batched, cyclic_permutations,
                        load_checkpoint, load_vignettes, make_prompt, parse_shard,
                        resolve_letter_token_ids, slot_options)
    from utils.paths import text_digest
    sv = cfg["self_vs_client"]
    v2 = sv["v2"]
    model_id, out = v2_dir(cfg, args)
    suffix = "_dryrun" if args.dry_run else ""
    variant = json.loads((out / "pilot.json").read_text())["chosen_variant"]
    vec = steer_vectors_v2(cfg, out, suffix)
    _, _, ev = items_v2(cfg)
    ids = load_ids(out)
    ev_v, rand_v = ids["eval_vignettes"], ids["random_vignettes"]
    rng = np.random.default_rng(cfg["seed"] + 13)
    rand_items = [ev[i] for i in sorted(rng.choice(len(ev), size=min(200, len(ev)), replace=False))]
    if args.dry_run:
        ev, ev_v, rand_v, rand_items = ev[:4], ev_v[:3], ev_v[:2], ev[:2]
    vby = {v["vignette_id"]: v for v in load_vignettes(cfg["paths"]["vignettes_dir"])}
    alloc, equity = cfg["advice_options"]["portfolio_choice"], cfg["advice_options"]["equity_fractions"]
    perms = cyclic_permutations(len(alloc))[::2]
    framing = cfg["s3"]["prompts"]["framing"]
    unit_norm = sv["amended"]["dose_unit_norm"]
    ks = [k for k in v2["doses_k"] if k != 0]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    adv_prompt = {}
    for v in ev_v:
        for pi, perm in enumerate(perms):
            p = make_prompt(tok, advice_message(framing, vby[v]["text"], slot_options(alloc, perm)))
            adv_prompt[(v, pi)] = (p, narrative_span_tokens(tok, p, vby[v]["text"])[1] + 1)
    lot_prompt = {}
    for it in ev:
        for gf in (True, False):
            prompt, _, slot = lottery_prompt_v2(tok, it, gf, variant)
            lot_prompt[(it["item_id"], gf)] = (prompt, slot)

    units = []   # task, direction, k, item, sub, prompt, start
    def add_adv(dname, k, vids, post=False):
        for v in vids:
            for pi in range(len(perms)):
                p, start = adv_prompt[(v, pi)]
                units.append(("advice", dname, k, v, pi, p, start if post else None))
    def add_lot(dname, k, its):
        for it in its:
            for gf in (True, False):
                p, slot = lot_prompt[(it["item_id"], gf)]
                units.append(("lottery", dname, k, it["item_id"], slot, p, None))
    add_adv("none", 0, ev_v)
    add_lot("none", 0, ev)
    for name in vec:
        is_rand = name.startswith("random_")
        for k in ks:
            add_adv(name, k, rand_v if is_rand else ev_v)
            add_lot(name, k, rand_items if is_rand else ev)
            if name in POST_V2:
                add_adv(f"{name}@post", k, ev_v, post=True)
    vdig = {n: text_digest(v.tolist(), length=8) for n, v in vec.items()}
    keys = [text_digest(model_id, u[5], u[1], u[2], 22, unit_norm, vdig.get(u[1].split("@")[0], "none"),
                        u[6], "fp32", length=16) for u in units]
    shard_i, shard_n = parse_shard(args.shard)
    tag = f".shard{shard_i}of{shard_n}" if shard_n > 1 else ""
    done = {}
    for path in sorted(out.glob(f"cross_checkpoint{suffix}.*jsonl")):
        done.update(load_checkpoint(path))
    results = [done.get(k) for k in keys]
    print(f"cross2: {len(vec)} directions (+{len(POST_V2)} post-narrative), k {ks} -> {len(units)} prompts "
          f"({sum(r is not None for r in results)} already scored)", flush=True)
    groups = {}
    for i, u in enumerate(units):
        if results[i] is None and i % shard_n == shard_i:
            groups.setdefault((u[0], u[1], u[2]), []).append(i)
    if groups:
        tok, model = get_model(cfg, args, model_id)
        scorer = Fp32Scorer(model)
        lid = {"advice": resolve_letter_token_ids(tok, units[0][5], len(alloc)),
               "lottery": resolve_letter_token_ids(tok, next(u[5] for u in units if u[0] == "lottery"), 2)}
        bs = {"advice": args.batch_size or 48, "lottery": 2 * (args.batch_size or 48)}
        with open(out / f"cross_checkpoint{suffix}{tag}.jsonl", "a") as ckpt:
            for (task, dname, k), idxs in groups.items():
                idxs.sort(key=lambda i: len(units[i][5]))
                base = dname.split("@")[0]
                for b in batched(idxs, bs[task]):
                    starts = [units[i][6] for i in b] if dname.endswith("@post") else None
                    lp, pl, am = scorer(tok, [units[i][5] for i in b], lid[task], 22,
                                        vec[base] if dname != "none" else None, k * unit_norm, starts)
                    for j, i in enumerate(b):
                        results[i] = (lp[j], pl[j], am[j])
                        ckpt.write(json.dumps({"k": keys[i], "r": results[i]}) + "\n")
                    ckpt.flush()
                print(f"  {task} {dname} k={k:+}: {sum(r is not None for r in results)}/{len(units)}", flush=True)
    if shard_n > 1:
        print(f"shard {shard_i}/{shard_n} done. Run without --shard to merge.")
        return
    agg = {}
    for u, r in zip(units, results):
        agg.setdefault(u[:4], []).append((u[4], r))
    rows = []
    for (task, dname, k, item), per in agg.items():
        row = {"task": task, "direction": dname, "k": k, "item": item,
               "p_letters": round(float(np.mean([r[1] for _, r in per])), 5),
               "frac_argmax_letter": round(float(np.mean([r[2] for _, r in per])), 3)}
        if task == "advice":
            per.sort(key=lambda x: x[0])
            row["y"] = round(aggressiveness(aggregate_allocation_probs([r[0] for _, r in per], perms), equity), 5)
        else:
            by = {("gamble_first" if slot == 0 else "gamble_second"): r[0][slot] for slot, r in per}
            row.update({k_: round(v, 5) for k_, v in by.items()})
            row["y"] = round(float(np.mean(list(by.values()))), 5)
        rows.append(row)
    with open(out / f"cross{suffix}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"cross2: {len(rows)} rows -> {out / f'cross{suffix}.jsonl'}")


def run_summarize2(cfg, args):
    sv = cfg["self_vs_client"]
    v2, am = sv["v2"], sv["amended"]
    model_id, out = v2_dir(cfg, args)
    suffix = "_dryrun" if args.dry_run else ""
    pilot = json.loads((out / "pilot.json").read_text())
    rule = pilot["hedge_rule"]
    d = np.load(out / f"directions{suffix}.npz")
    layers = [int(x) for x in d["layers"]]
    li = layers.index(22)
    rep = {"pre_registration": "config.yaml self_vs_client.v2 (amendment 3, 2026-11-03)", "pilot": pilot}

    # geometry (narrative mean), C_will = RQ2b vector at L22; curve over layers for the prefix pairs
    vec = steer_vectors_v2(cfg, out, suffix)
    G = {n: d[n][li, 0] for n in PREFIXES_V2}
    geo = {f"S_adv~{n}": round(abs(cos(G["S_adv"], G[n])), 4) for n in ("C_adv", "T_adv", "S_adv2", "N_adv")}
    geo.update({f"{n}~C_will": round(abs(cos(G[n], vec["C_will"])), 4) for n in PREFIXES_V2})
    geo["S_lot_last~S_adv"] = round(abs(cos(d["S_lot"][li, 1], G["S_adv"])), 4)
    geo["S_lot_last~C_will"] = round(abs(cos(d["S_lot"][li, 1], vec["C_will"])), 4)
    geo["split_half"] = {n: round(cos(d[f"{n}_A"][li, 0], d[f"{n}_B"][li, 0]), 4) for n in PREFIXES_V2}
    geo["gap_norms"] = {n: round(float(np.linalg.norm(G[n])), 4) for n in PREFIXES_V2}
    geo["curve_S_adv~C_adv"] = {l: round(abs(cos(d["S_adv"][i, 0], d["C_adv"][i, 0])), 4) for i, l in enumerate(layers)}
    geo["curve_S_adv~S_adv2"] = {l: round(abs(cos(d["S_adv"][i, 0], d["S_adv2"][i, 0])), 4) for i, l in enumerate(layers)}
    geo["geometry_gate_pass"] = geo["C_adv~C_will"] >= v2["gates"]["geometry_gate_min"]
    rep["geometry"] = geo

    # contamination: frozen tier probe on prefixed client activations, relative to the null prefix
    wpath = out.parent / "weights" / "probe_best_L22_profile_mean.npz"
    if wpath.exists():
        w = np.load(wpath, allow_pickle=True)
        rep["contamination"] = probe_contamination(w, {k[len("plane22_"):]: d[k] for k in d.files if k.startswith("plane22_")})

    # behavior
    beh = [json.loads(l) for l in open(out / f"behavior{suffix}.jsonl")]
    for r in beh:
        if r["task"] == "lottery":
            r["p_gamble"] = r["y"]
    rep["prefix_effects"] = prefix_effects(beh)
    lot0 = [r for r in beh if r["task"] == "lottery" and r["cond"] == "none"]
    adv0 = [r for r in beh if r["task"] == "advice" and r["cond"] == "none"]
    hi = {r["item"]: r for r in beh if r["task"] == "lottery" and r["cond"] == "S_lot_hi"}
    lo = {r["item"]: r for r in beh if r["task"] == "lottery" and r["cond"] == "S_lot_lo"}
    common = sorted(set(hi) & set(lo))
    gap = float(np.mean([hi[i]["y"] - lo[i]["y"] for i in common])) if common else float("nan")
    og = {s: float(np.mean([abs(m[i]["order_gap"]) for i in common])) if common else float("nan")
          for s, m in (("hi", hi), ("lo", lo))}
    # G0: dose-0 rows; under the relative fallback a dose-0 row is its own baseline, so only
    # the absolute 0.25 floor can bite there
    h0 = {"lottery": float(np.mean([hedged_v2(r["p_letters"], r["frac_argmax_letter"], rule, r["p_letters"]) for r in lot0])),
          "advice": float(np.mean([hedged_v2(r["p_letters"], r["frac_argmax_letter"], rule, r["p_letters"]) for r in adv0]))}
    gates = {"G0_hedge_dose0": {k: round(v, 4) for k, v in h0.items()},
             "G0_pass": all(v <= am["hedge_max_dose0"] for v in h0.values()),
             "G1_prefix_gap": round(gap, 4), "G1_order_gap": {k: round(v, 4) for k, v in og.items()},
             "G1_pass": bool(gap >= am["amended_G1_prefix_gap_min"] and all(v <= am["amended_G1_order_gap_max"] for v in og.values()))}
    rep["unprompted_lottery_reported_only"] = {
        **deal_tracking([{**r, "p_gamble": r["y"]} for r in lot0 if r["kind"] == "numeric"]),
        "letter_mass": round(float(np.mean([r["p_letters"] for r in lot0])), 4),
        "abs_order_gap": round(float(np.mean([abs(r["order_gap"]) for r in lot0])), 4),
        "p_gamble": round(float(np.mean([r["y"] for r in lot0])), 4)}

    # cross-steer
    rows = [json.loads(l) for l in open(out / f"cross{suffix}.jsonl")]
    base = {(r["task"], r["item"]): r["p_letters"] for r in rows if r["k"] == 0}
    for r in rows:
        r["hedged"] = hedged_v2(r["p_letters"], r["frac_argmax_letter"], rule, base.get((r["task"], r["item"])))
    names = sorted({r["direction"] for r in rows if r["direction"] != "none"})
    # |k|max is gated by the directions that enter a gate or the verdict; S_lot_last
    # (exploratory) and the @post rows are reported but cannot void the run
    named = [n for n in names if n in GATED_V2]
    hedge = {}
    for r in rows:
        if r["k"] != 0 and r["direction"] in named:
            hedge.setdefault((r["direction"], r["task"], r["k"]), []).append(r["hedged"])
    hedge = {k: float(np.mean(v)) for k, v in hedge.items()}
    ks = sorted({r["k"] for r in rows if r["k"] > 0})
    kmax = common_kmax(hedge, ks, am["hedge_max_any_dose"])
    rep["cross"] = {"hedge_rule": rule, "common_kmax_named": kmax,
                    "hedge": {f"{a}|{b}|{c}": round(v, 4) for (a, b, c), v in hedge.items()}}
    g2_min, g2_mult = am["gates"]["G2_shift_min"], am["gates"]["G2_random_multiple"]
    rep["cross"]["by_k"] = {k: cross_v2_at(rows, k, names, v2, cfg["seed"], g2_min, g2_mult) for k in ks}
    # 3: reliability gate (amendment 1's G3) + disattenuated bootstrap CI for the primary pair
    sb_s = spearman_brown(geo["split_half"]["S_adv"])
    r_cw = spearman_brown(client_split_half(cfg, out, "willingness"))
    boots = boot_disattenuated(d["S_adv_rows22"], vec["C_will"], np.arange(len(d["S_adv_rows22"])),
                               sb_s, r_cw, 1000, cfg["seed"])
    geo["S_adv~C_will_disattenuated_ci"] = ci(boots)
    gates["G3_pass"] = bool(geo["split_half"]["S_adv"] >= am["gates"]["G3_split_half_min"]
                            and geo["split_half"]["C_adv"] >= am["gates"]["G3_split_half_min"])
    if kmax == 0:
        rep["verdict"] = "NO VERDICT: every dose breaks some gated direction"
    else:
        at = rep["cross"]["by_k"][kmax]
        # 1: the verdict must not change when hedged rows are excluded (amendment 1)
        excl = cross_v2_at([r for r in rows if not r["hedged"]], kmax, names, v2, cfg["seed"], g2_min, g2_mult)
        rep["cross"]["hedged_excluded"] = {"verdict": excl["pairs"].get("S_adv~C_will", {}).get("verdict"),
                                           "G2_pass": excl["G2"]["pass"], "client_floor_pass": excl["client_floor_pass"]}
        gates.update({"G2_pass": at["G2"]["pass"], "client_floor_pass": at["client_floor_pass"]})
        rep["gates"] = gates
        rep["verdict"] = final_verdict_v2(rep, at, kmax)
    rep["gates"] = gates
    (out / f"self_vs_client{suffix}.json").write_text(json.dumps(rep, indent=2, default=float))
    print_report_v2(rep, out / f"self_vs_client{suffix}.json")


def client_split_half(cfg, out, name):
    """Split-half cosine of the RQ2b-style client direction at L22 profile_mean, implicit_train,
    halves by a stable hash of profile_id (as run 1)."""
    import zlib
    from probes import make_splits, split_indices
    from steering import DIRECTION_CLASSES, class_label, diff_in_means
    act_dir = out.parent / "activations" / load_ids(out)["cache_id"]
    meta = json.loads((act_dir / "meta.json").read_text())
    labels = [json.loads(l) for l in open(act_dir / "labels.jsonl")]
    tr = split_indices(labels, make_splits([l["profile_id"] for l in labels], seed=cfg["seed"]))["implicit_train"]
    acts = np.load(act_dir / "acts_all.npy", mmap_mode="r")
    plane = np.asarray(acts[tr, meta["layers"].index(22), meta["positions"].index("profile_mean"), :], dtype=np.float32)
    _, hi, lo = DIRECTION_CLASSES[name]
    cls = np.array([class_label(labels[i], name) for i in tr])
    half = np.array([zlib.crc32(labels[i]["profile_id"].encode()) % 2 == 0 for i in tr])
    return cos(diff_in_means(plane[half], cls[half], hi, lo)[0], diff_in_means(plane[~half], cls[~half], hi, lo)[0])


def cross_v2_at(rows, k, names, v2, seed, g2_min=0.15, g2_mult=4):
    """Shifts, random floor (non-hedged randoms), lever rule, G2, leaks for one dose."""
    rng = np.random.default_rng(seed)
    table, shift = {}, {}
    for name in names:
        for task in ("advice", "lottery"):
            sub = direction_rows(rows, task, name)
            if not sub:
                continue
            if name.startswith("random_"):
                sub = [r for r in sub if not r["hedged"]]
            sh = paired_shifts(sub, k)
            if not sh:
                continue
            shift[(name, task)] = sh
            entry = {"shift": round(float(np.mean(list(sh.values()))), 4),
                     "abs_shift": round(float(np.mean(np.abs(list(sh.values())))), 4),
                     "ci": ci(boot_means(sh.values(), 2000, rng)), "n": len(sh)}
            if task == "lottery":
                entry["by_order"] = order_shift_ci(sub, k, seed=seed)
                entry["lever"] = lever_ok(entry["by_order"])
            table[f"{name}|{task}"] = entry
    floor = {t: max([table[f"{n}|{t}"]["abs_shift"] for n in names if n.startswith("random_")
                     and f"{n}|{t}" in table] or [float("nan")]) for t in ("advice", "lottery")}
    s = table.get("S_adv|lottery", {"shift": float("nan"), "lever": False})
    g2 = {"shift": s["shift"], "floor": floor["lottery"], "lever": s.get("lever", False),
          "pass": bool(s["shift"] >= g2_min and s["shift"] >= g2_mult * floor["lottery"] and s.get("lever", False))}
    c = table.get("C_will|advice", {"shift": float("nan")})
    pairs = {}
    for self_name, client_name in (("S_adv", "C_will"), ("S_adv", "C_adv"), ("S_adv2", "C_will"), ("T_adv", "C_will")):
        if all((n, t) in shift for n in (self_name, client_name) for t in ("advice", "lottery")):
            lk = leaks({("S", t): shift[(self_name, t)] for t in ("advice", "lottery")}
                       | {("C", t): shift[(client_name, t)] for t in ("advice", "lottery")}, seed=seed)
            lk["verdict"] = leak_verdict(lk)
            pairs[f"{self_name}~{client_name}"] = lk
    post = {n: {"full": table.get(f"{n}|advice", {}).get("shift"), "post_narrative": table.get(f"{n}@post|advice", {}).get("shift"),
                "post_ci": table.get(f"{n}@post|advice", {}).get("ci")} for n in POST_V2}
    return {"table": table, "random_floor": floor, "G2": g2,
            "client_floor_pass": bool(c["shift"] >= g2_mult * floor["advice"]), "pairs": pairs, "position_restricted": post}


def final_verdict_v2(rep, at, kmax):
    g = rep["gates"]
    failed = [n for n in ("G0_pass", "G1_pass", "G2_pass", "G3_pass", "client_floor_pass") if not g.get(n)]
    if failed:
        return f"STOPPED at k={kmax}: {', '.join(f.replace('_pass', '') for f in failed)} failed; no RQ2c verdict"
    cross = at["pairs"].get("S_adv~C_will", {}).get("verdict", "MISSING")
    excl = rep["cross"]["hedged_excluded"]
    if (excl["verdict"], excl["G2_pass"], excl["client_floor_pass"]) != (cross, True, True):
        return (f"NO VERDICT at k={kmax}: cross-steer {cross} with hedged rows, hedged-excluded gives "
                f"{excl['verdict']} (G2 {excl['G2_pass']}, client floor {excl['client_floor_pass']})")
    geo = rep["geometry"]
    if not geo["geometry_gate_pass"]:
        geo_call = "not interpretable for referent (cos(C_adv, C_will) below gate)"
    else:
        lo, hi = geo["S_adv~C_will_disattenuated_ci"]
        geo_call = "distinct" if hi < 0.5 else "conflation" if lo > 0.8 else "partial"
    if cross == "DISTINCT" and geo_call == "distinct":
        v = "DISTINCT"
    elif cross == "CONFLATION" or geo_call == "conflation":
        v = "CONFLATION"
    elif cross == "ASYMMETRIC":
        v = "ASYMMETRIC"
    else:
        v = "PARTIAL OVERLAP"
    return f"{v} at k={kmax} (cross-steer {cross}; geometry {geo_call})"


def print_report_v2(rep, path):
    print("=== RQ2c v2 SELF vs CLIENT (confirmatory) ===")
    print(f"  pilot: variant {rep['pilot']['chosen_variant']}  hedge rule {rep['pilot']['hedge_rule']}")
    g = rep["geometry"]
    print(f"  geometry: { {k: v for k, v in g.items() if not k.startswith('curve')} }")
    print(f"  gates: {rep['gates']}")
    print(f"  unprompted lottery: {rep['unprompted_lottery_reported_only']}")
    for k, v in rep["prefix_effects"].items():
        print(f"  prefix {k}: {v}")
    if "contamination" in rep:
        print(f"  contamination: {rep['contamination']}")
    print(f"  common kmax (named): {rep['cross']['common_kmax_named']}")
    for k, at in rep["cross"]["by_k"].items():
        print(f"  -- k={k}: floor {at['random_floor']}  G2 {at['G2']}  client floor {at['client_floor_pass']}")
        for name, e in at["table"].items():
            if not name.startswith("random_"):
                extra = f" lever {e['lever']} {e['by_order']['gamble_first']['shift']:+.3f}/{e['by_order']['gamble_second']['shift']:+.3f}" if "lever" in e else ""
                print(f"     {name:20s} {e['shift']:+.4f} {e['ci']}{extra}")
        for name, lk in at["pairs"].items():
            print(f"     {name}: leak_self {lk['leak_self']} {lk['leak_self_ci']} leak_client {lk['leak_client']} {lk['leak_client_ci']} -> {lk['verdict']}")
        print(f"     position-restricted: {at['position_restricted']}")
    print(f"  VERDICT: {rep['verdict']}")
    print(f"  -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "pilot", "directions", "behavior", "cross", "summarize",
                                      "prepare2", "pilot2", "directions2", "behavior2", "cross2", "summarize2"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--activations", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--wandb", action="store_true", help="summarize: also log results to W&B")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    {"prepare": run_prepare, "pilot": run_pilot, "directions": run_directions, "behavior": run_behavior,
     "cross": run_cross, "summarize": run_summarize,
     "prepare2": run_prepare2, "pilot2": run_pilot2, "directions2": run_directions2,
     "behavior2": run_behavior2, "cross2": run_cross2, "summarize2": run_summarize2}[args.stage](cfg, args)


if __name__ == "__main__":
    main()
