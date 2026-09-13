"""S4: cache frozen Gemma-2-9B-it residual-stream activations for every vignette, at two
positions, for the config layers, written as fp16 shards for S5 (probes).

Positions:
  - profile_end:  last token of the CLIENT NARRATIVE, before the options appear. The
    integration site -- the model has read the whole profile but not yet seen the task.
    Permutation-independent (options come after it).
  - profile_mean: mean of the residual over the narrative token span. Decoder last-token is
    a known information bottleneck, so this is a cheap aggregation-robustness variant
    (computed on-device from the same hidden states -- zero extra forward cost).
  - decision:     last token of the full advisor prompt (the assistant prefill "("), i.e.
    the exact position whose logits S3 read for the advice. Indexed as -1.

We reuse S3's prompt construction verbatim so P2 lines up with what S3 scored. Each vignette
is cached ONCE under the identity option order (perm 0); P1 doesn't depend on option order,
and the probe just needs one canonical representation per vignette.

Position indexing is the documented failure mode, so P1 is located via the tokenizer's
char->token offset mapping (find the token covering the narrative's last character) and the
pure helper is unit-tested in tests/test_s4.py. Left-padding is handled by offsetting the
per-row pad count, so batching never moves P1.

Shards (results/activations/<model_tag>/shard_XXXX.npz) hold:
  acts  float16 [n_rows, n_layers, n_positions, d]   (positions order = ["profile_end","decision"])
  plus per-row label columns (vignette_id, profile_id, pair_id, tier, risk_score,
  vignette_type, contradictory). A meta.json records model/layers/positions/d and row order.
Resume: vignette_ids already present in shards are skipped.

torch/transformers are imported lazily (via advice) so the pure logic here is testable
without a GPU.

Usage:
    python src/activations.py [--config config.yaml] [--dry-run] [--model ID]
        [--layers 20,25,31] [--device cuda] [--shard-size 512] [--overwrite]
"""

import argparse
import json
import re
from pathlib import Path

import yaml

from utils.paths import cache_fingerprint, git_commit, model_tag, run_dir
from advice import (
    build_user_message, cyclic_permutations, load_model, load_vignettes,
    make_prompt, slot_options, stratified_sample,
)

POSITIONS = ["profile_end", "profile_mean", "decision"]   # order = the activation axis order


# --------------------------------------------------------------------------------------
# Pure logic (no torch) -- unit-tested in tests/test_s4.py
# --------------------------------------------------------------------------------------
def token_covering_char(offsets, char_idx):
    """Index of the token whose char span contains `char_idx`.

    `offsets` is a HuggingFace fast-tokenizer offset_mapping: a list of (start, end) char
    spans, one per token. Special/zero-width tokens have start == end and are skipped.
    Returns None if no token covers the position.
    """
    for i, (s, e) in enumerate(offsets):
        if s == e:                      # zero-width: special token, skip
            continue
        if s <= char_idx < e:
            return i
    return None


def narrative_start_char(prompt_str, narrative):
    """Char index of the FIRST character of `narrative` in the full prompt string (last
    occurrence, mirroring narrative_end_char)."""
    return prompt_str.rindex(narrative)      # raises ValueError if absent -> caught by caller


def narrative_end_char(prompt_str, narrative):
    """Char index of the LAST character of `narrative` as it sits in the full prompt string.

    The chat template inserts the narrative verbatim, so rindex finds it; we take the last
    occurrence to be robust to incidental substrings earlier in the framing.
    """
    return prompt_str.rindex(narrative) + len(narrative) - 1


def narrative_span_tokens(tok, prompt_str, narrative):
    """Unpadded token span (start, end) of the narrative for one prompt, plus the unpadded
    token count.

    `end` is the profile_end (last-token) position; the inclusive range [start, end] is the
    profile_mean span. Tokenizes with offsets and no special tokens (the chat template wrote
    them as text). Raises if the narrative can't be located or a boundary char has no token,
    so a silent off-by-one can't slip through.
    """
    try:
        c0, c1 = narrative_start_char(prompt_str, narrative), narrative_end_char(prompt_str, narrative)
    except ValueError as exc:
        raise ValueError("narrative not found verbatim in templated prompt") from exc
    enc = tok(prompt_str, add_special_tokens=False, return_offsets_mapping=True)
    start = token_covering_char(enc["offset_mapping"], c0)
    end = token_covering_char(enc["offset_mapping"], c1)
    if start is None or end is None or start > end:
        raise ValueError(f"bad narrative token span (start={start}, end={end})")
    return start, end, len(enc["input_ids"])


def label_row(v):
    """The per-row metadata S5 needs as probe labels / split keys."""
    return {
        "vignette_id": v["vignette_id"], "profile_id": v["profile_id"],
        "pair_id": v.get("pair_id"), "tier": v["tier"],
        "risk_score": v["risk_score"], "vignette_type": v["vignette_type"],
        "contradictory": bool(v["contradictory"]),
        # factor targets (the probe labels) ...
        "willingness": v.get("willingness", -1.0), "capacity": v.get("capacity", -1.0),
        "goals": v.get("goals", -1.0), "factor_cell": v.get("factor_cell", ""),
        "conflicted": bool(v.get("conflicted", False)),
        # ... which field a counterfactual pair varies ...
        "pair_field": v.get("pair_field") or "", "pair_factor": v.get("pair_factor") or "",
        # ... and the confound controls.
        "template_id": v.get("template_id", ""), "name": v.get("name") or "",
        "n_chars": int(v.get("n_chars", len(v.get("text", "")))),
    }


# model_tag / run_dir / cache_fingerprint live in utils.paths (re-exported here so
# existing call sites keep working).


def parse_layers(spec, n_layers):
    """Resolve a layer spec to a concrete list. `spec` is "all" (-> every layer), a list of
    ints (from config), or a comma string (from --layers)."""
    if isinstance(spec, str):
        spec = spec.strip()
        if spec == "all":
            return list(range(n_layers))
        return [int(x) for x in spec.split(",") if x != ""]
    return [int(x) for x in spec]


def done_vignette_ids(shard_dir):
    """vignette_ids already cached, so a resumed run skips them."""
    import numpy as np
    done = set()
    for shard in sorted(Path(shard_dir).glob("shard_*.npz")):
        with np.load(shard, allow_pickle=True) as z:
            done.update(str(x) for x in z["vignette_id"])
    return done


def next_shard_index(shard_dir):
    existing = sorted(Path(shard_dir).glob("shard_*.npz"))
    if not existing:
        return 0
    return int(existing[-1].stem.split("_")[1]) + 1


# --------------------------------------------------------------------------------------
# Model-dependent driver (lazy torch via advice.load_model)
# --------------------------------------------------------------------------------------
def write_shard(shard_dir, idx, acts, labels):
    """Write one .npz: acts[n,L,P,d] fp16 + parallel label columns."""
    import numpy as np
    keys = ["vignette_id", "profile_id", "pair_id", "tier", "risk_score",
            "vignette_type", "contradictory"]
    cols = {k: np.array([row[k] for row in labels], dtype=object) for k in keys}
    out = Path(shard_dir) / f"shard_{idx:04d}.npz"
    np.savez(out, acts=np.asarray(acts, dtype=np.float16), **cols)
    return out


def run(cfg, args):
    import numpy as np
    import torch

    allocation_texts = cfg["advice_options"]["portfolio_choice"]
    n_opts = len(allocation_texts)
    identity_perm = cyclic_permutations(n_opts, 1)[0]      # [0,1,...] -- canonical order
    framing = cfg["s3"]["prompts"]["framing"]
    batch_size = args.batch_size      # forward-pass batch; caching grabs full [B,T,D] hidden states

    vignettes = load_vignettes(cfg["paths"]["vignettes_dir"])
    if args.dry_run:
        vignettes = stratified_sample(vignettes, cfg["s3"]["dry_run_n"], cfg["seed"])

    model_id = args.model or cfg["model"]["primary"]
    # The cache id covers the prompt, the data files, the layer set and the dtype --
    # anything that changes an activation. Change any of them and this resolves to a
    # different directory, so a resume can never silently mix incompatible runs.
    vignette_files = sorted(Path(cfg["paths"]["vignettes_dir"]).glob("*.jsonl"))
    fingerprint, cache_id = cache_fingerprint(
        model_id, cfg["s3"]["prompts"]["framing"], vignette_files,
        args.layers or cfg["extraction"]["layers"], POSITIONS, cfg["model"]["dtype"])
    shard_dir = run_dir(cfg, model_id) / "activations" / cache_id
    if args.overwrite and shard_dir.exists():
        for f in shard_dir.glob("shard_*.npz"):
            f.unlink()
        (shard_dir / "meta.json").unlink(missing_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    already = set() if args.overwrite else done_vignette_ids(shard_dir)
    todo = [v for v in vignettes if v["vignette_id"] not in already]
    print(f"S4: {len(vignettes)} vignettes, {len(already)} already cached, {len(todo)} to do  model={model_id}")
    if not todo:
        print("nothing to do.")
        return

    tok, model = load_model(model_id, cfg["model"]["dtype"], cfg["s3"]["attn_implementation"], args.device)
    n_layers_model = model.config.num_hidden_layers
    d = model.config.hidden_size
    layers = parse_layers(args.layers or cfg["extraction"]["layers"], n_layers_model)
    bad = [l for l in layers if not 0 <= l < n_layers_model]
    if bad:
        raise ValueError(f"layers {bad} out of range for {model_id} "
                         f"(has {n_layers_model} layers; pass --layers to override)")
    print(f"    layers={layers[0]}..{layers[-1]} ({len(layers)})  positions={POSITIONS}  d={d}")

    # Precompute prompt + narrative token span per vignette (CPU; asserts offsets early).
    prepared = []
    for v in todo:
        user_msg = build_user_message(framing, v["text"], slot_options(allocation_texts, identity_perm))
        prompt = make_prompt(tok, user_msg)
        span_start, span_end, unpadded_len = narrative_span_tokens(tok, prompt, v["text"])
        prepared.append({"v": v, "prompt": prompt, "span_start": span_start,
                         "span_end": span_end, "unpadded_len": unpadded_len})

    shard_idx = next_shard_index(shard_dir)
    buf_acts, buf_labels = [], []
    n_done = 0

    def flush():
        nonlocal shard_idx, buf_acts, buf_labels
        if not buf_acts:
            return
        out = write_shard(shard_dir, shard_idx, buf_acts, buf_labels)
        print(f"    wrote {out.name} ({len(buf_acts)} rows)")
        shard_idx += 1
        buf_acts, buf_labels = [], []

    # Batch by similar length for padding efficiency.
    prepared.sort(key=lambda x: x["unpadded_len"])
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start:start + batch_size]
        prompts = [b["prompt"] for b in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        attn = enc["attention_mask"]
        seq_len = attn.shape[1]
        from utils.hooks import residual_cache
        with residual_cache(model, layers) as store, torch.inference_mode():
            model(**enc)
        # store[l]: [B, seq_len, d] on device. Index positions per row, accounting for left pad.
        for r, b in enumerate(batch):
            n_pad = int((attn[r] == 0).sum().item())     # left padding count for this row
            s0, s1 = n_pad + b["span_start"], n_pad + b["span_end"]   # narrative span, padded

            def pos_vec(l, p):
                if p == "profile_mean":
                    return store[l][r, s0:s1 + 1].float().mean(0).cpu().numpy()
                idx = s1 if p == "profile_end" else seq_len - 1        # decision = last token
                return store[l][r, idx].float().cpu().numpy()

            # [L, P, d]
            row = np.stack([
                np.stack([pos_vec(l, p) for p in POSITIONS]) for l in layers
            ]).astype(np.float16)
            buf_acts.append(row)
            buf_labels.append(label_row(b["v"]))
            n_done += 1
        if len(buf_acts) >= args.shard_size:
            flush()
        print(f"    cached {n_done}/{len(prepared)}", end="\r")
    print()
    flush()

    meta = {
        "model": model_id, "layers": layers, "positions": POSITIONS,
        "hidden_size": d, "num_hidden_layers": n_layers_model,
        "dtype_stored": "float16", "option_order": "identity_perm0",
        "n_vignettes_cached": len(already) + n_done, "seed": cfg["seed"],
        "attn_implementation": cfg["s3"]["attn_implementation"],
        # provenance: what this cache is, and what it was built from
        "cache_id": cache_id, "fingerprint": fingerprint,
        "git_commit": git_commit(),
    }
    (shard_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"S4 done: {n_done} new rows -> {shard_dir}  (total {meta['n_vignettes_cached']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default=None, help="override config model.primary (e.g. google/gemma-2-2b-it)")
    ap.add_argument("--layers", default=None, help="comma list to override config extraction.layers")
    ap.add_argument("--device", default="cuda", help="cuda | cpu")
    ap.add_argument("--shard-size", type=int, default=512, help="rows per .npz shard")
    ap.add_argument("--batch-size", type=int, default=8, help="forward-pass batch size (cache holds full hidden states; keep small)")
    ap.add_argument("--overwrite", action="store_true", help="delete existing shards first")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg, args)


if __name__ == "__main__":
    main()
