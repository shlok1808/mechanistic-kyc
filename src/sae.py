"""S9: name the steering directions with Gemma Scope SAE features (CPU, no GPU).

Two rankings per direction, as in the design spec:
  decoder cosine   which SAE features' decoder vectors point along the steering direction
  activation diff  which features fire more for the high class than the low class, encoding
                   real per-token activations (profile_end) -- an SAE is trained on single
                   tokens, so mean-pooled profile_mean vectors are not valid SAE input
Feature labels come from Neuronpedia's auto-interp explanations.

Caveat recorded in the output: these are the BASE-model (pt) SAEs applied to the
instruction-tuned model. Gemma Scope's IT SAEs do not cover layer 22, and DeepMind report
base-to-IT transfer is usually adequate, but the labels are approximate.

Usage:
    python src/sae.py [--sae layer_22/width_16k/average_l0_65] [--top 10]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def jumprelu_encode(X, p):
    pre = X @ p["W_enc"] + p["b_enc"]
    return pre * (pre > p["threshold"])


def top_k(scores, k):
    idx = np.argsort(-scores)[:k]
    return [(int(i), round(float(scores[i]), 4)) for i in idx]


def neuronpedia_label(model, sae_id, idx):
    import ssl
    import urllib.request
    url = f"https://www.neuronpedia.org/api/feature/{model}/{sae_id}/{idx}"
    try:
        import certifi                           # python.org builds ship without a CA bundle
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = None
    try:
        with urllib.request.urlopen(url, timeout=15, context=ctx) as r:
            d = json.load(r)
        ex = d.get("explanations") or []
        return ex[0]["description"].strip() if ex else "(no explanation)"
    except Exception as e:                       # offline or rate-limited: keep the index
        return f"(lookup failed: {type(e).__name__})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sae", default="layer_22/width_16k/average_l0_65")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    from huggingface_hub import hf_hub_download
    from probes import make_splits, split_indices
    from steering import DIRECTION_CLASSES, class_label, load_directions
    from utils.paths import run_dir

    res = run_dir(cfg, cfg["model"]["primary"])
    dirs = load_directions(res / "steering" / "directions.npz")
    layer = dirs["meta"]["layer"]
    if f"layer_{layer}/" not in args.sae + "/":
        raise ValueError(f"SAE {args.sae} is not for steering layer {layer}")
    p = dict(np.load(hf_hub_download("google/gemma-scope-9b-pt-res", f"{args.sae}/params.npz")))
    p = {k: v.astype(np.float32) for k, v in p.items()}

    act_dir = res / "activations" / dirs["meta"]["cache_id"]
    meta = json.loads((act_dir / "meta.json").read_text())
    labels = [json.loads(l) for l in open(act_dir / "labels.jsonl")]
    acts = np.load(act_dir / "acts_all.npy", mmap_mode="r")
    idx = split_indices(labels, make_splits([l["profile_id"] for l in labels], seed=cfg["seed"]))
    rows = np.concatenate([idx["train"], idx["implicit_train"]])
    X = np.asarray(acts[rows, meta["layers"].index(layer), meta["positions"].index("profile_end"), :],
                   dtype=np.float32)
    F = jumprelu_encode(X, p)                                  # [n, 16k]
    print(f"SAE {args.sae}: mean L0 on these tokens = {(F > 0).sum(1).mean():.1f}")

    W_dec = p["W_dec"] / np.linalg.norm(p["W_dec"], axis=1, keepdims=True)
    sae_tag = f"{layer}-gemmascope-res-16k"
    out = {"sae": args.sae, "layer": layer, "caveat": "base-model SAE on the IT model", "directions": {}}
    for name in ["tier", "willingness", "capacity", "goals"]:
        v = dirs["vectors"][dirs["names"].index(name)]
        cos = W_dec @ (v / np.linalg.norm(v))
        _, hi, lo = DIRECTION_CLASSES[name]
        cls = np.array([class_label(labels[i], name) for i in rows])
        diff = F[cls == hi].mean(0) - F[cls == lo].mean(0)
        entry = {"decoder_cosine_up": top_k(cos, args.top), "decoder_cosine_down": top_k(-cos, args.top),
                 "activation_diff_up": top_k(diff, args.top), "activation_diff_down": top_k(-diff, args.top)}
        feats = {i for lst in entry.values() for i, _ in lst}
        entry["labels"] = {i: neuronpedia_label("gemma-2-9b", sae_tag, i) for i in sorted(feats)}
        out["directions"][name] = entry
        print(f"\n== {name} ({hi} vs {lo})")
        for key in ("decoder_cosine_up", "activation_diff_up", "activation_diff_down"):
            print(f"  {key}:")
            for i, s in entry[key][:6]:
                print(f"    #{i:5d} {s:+.3f}  {entry['labels'][i]}")
    path = res / "sae_features.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
