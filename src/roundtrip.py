"""Per-field roundtrip: did the rendered text actually keep the facts the label claims?

The banned-lexicon scan and the template-vs-paraphrase numeric diff only cover part of
the surface. Most rubric fields reach the page as prose with no digits in them, and
nothing else in QC checks whether that prose is still recoverable.

This reads each vignette back with an LLM, asks it to classify the client on every
rubric field, and scores the answers against ground truth. Two design points matter:

  PER LEVEL, not just per field. A field can average 0.9 while one of its levels is
  unreadable, because the other levels carry the average. That is exactly how the
  precarious/variable collapse hid: income_stability looked like 0.75 overall while
  73% of precarious clients were being read as variable.

  EXPLICIT IS THE CONTROL. A miss is either the text losing the fact or the reader
  failing to see it. Explicit vignettes name these facts in plain risk vocabulary, so
  the explicit score is roughly the reader's ceiling and the explicit-implicit gap is
  what implicit rendering costs.

A level that scores near zero is not a model failure. It means two phrasings in
templates.py describe the same thing at different volumes, so the fact was never in the
text for a probe to find. Fix the phrasing, not the model.

Usage:
    python src/roundtrip.py [--n 300] [--vignette-type implicit|explicit|both]
                            [--reader MODEL] [--config config.yaml]
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

# Rubric fields that reach the page as prose. `age` is excluded: it is rendered as a
# literal number in the opening line and is already covered by the template-vs-output
# numeric diff. Ordered by rubric weight, heaviest first.
ENUM_FIELDS = ["past_drawdown_reaction", "investing_experience", "income_stability",
               "stated_goal"]
BAND_FIELDS = ["horizon_years", "emergency_fund_months", "dependents"]
FIELDS = ENUM_FIELDS + BAND_FIELDS


def band_label(lo, hi, field):
    unit = {"horizon_years": "years", "emergency_fund_months": "months",
            "dependents": "dependents"}[field]
    return f"{lo}-{hi} {unit}" if lo != hi else f"{lo} {unit}"


def field_options(cfg):
    """{field: [ordered label, ...]} plus a value->label map, built from the templates.

    Options come from the same tables that rendered the text, so this can never drift
    from what the vignettes actually say.
    """
    import templates as T
    opts, to_label = {}, {}
    for f in ENUM_FIELDS:
        levels = list(cfg["rubric"]["fields"][f]["levels"])
        opts[f] = levels
        to_label[f] = dict(zip(levels, levels))
    for f in BAND_FIELDS:
        labels, mapping = [], {}
        for lo, hi, _ in T.BAND_PHRASINGS[f]:
            lab = band_label(lo, hi, f)
            labels.append(lab)
            for v in range(lo, hi + 1):
                mapping[v] = lab
        opts[f], to_label[f] = labels, mapping
    return opts, to_label


def build_extract_prompt(text, options):
    lines = "\n".join(f"  {f}: one of {options[f]}" for f in FIELDS)
    return (
        "Below is how a person described themselves to a financial advisor. Classify "
        "them on each attribute using ONLY the listed options. Judge only what the text "
        "says; do not guess from vibe, and do not infer one attribute from another.\n\n"
        f"{lines}\n\n"
        "Return ONLY a JSON object with exactly these keys and no other text.\n\n"
        f"Description:\n{text}"
    )


def parse_reply(raw):
    """Pull the JSON object out of a reply, tolerating code fences or stray prose."""
    s = (raw or "").strip()
    if "{" in s and "}" in s:
        s = s[s.index("{"): s.rindex("}") + 1]
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def score(rows, preds, options):
    """Per-field and PER-LEVEL accuracy, ordinal distance, and >=2-step flip rate."""
    out = {}
    for f in FIELDS:
        labels = options[f]
        idx = {v: i for i, v in enumerate(labels)}
        hits = dists = flips = n = unparsed = 0
        confusion = Counter()
        per_level = defaultdict(lambda: {"n": 0, "hits": 0, "confusions": Counter()})
        for r, p in zip(rows, preds):
            want = r["_truth"].get(f)
            if want is None:
                continue
            if p is None or f not in p or p[f] not in idx:
                unparsed += 1
                continue
            got = p[f]
            n += 1
            per_level[want]["n"] += 1
            if got == want:
                hits += 1
                per_level[want]["hits"] += 1
            else:
                confusion[f"{want} -> {got}"] += 1
                per_level[want]["confusions"][got] += 1
            d = abs(idx[got] - idx[want])
            dists += d
            flips += (d >= 2)
        out[f] = {
            "n": n, "unreadable": unparsed,
            "accuracy": round(hits / n, 4) if n else None,
            "chance": round(1 / len(labels), 4),
            "mean_ordinal_distance": round(dists / n, 4) if n else None,
            "flip_rate": round(flips / n, 4) if n else None,
            "top_confusions": confusion.most_common(3),
            "per_level": {
                lvl: {"n": d["n"],
                      "accuracy": round(d["hits"] / d["n"], 4) if d["n"] else None,
                      "mistaken_for": d["confusions"].most_common(2)}
                for lvl, d in sorted(per_level.items(), key=lambda kv: idx.get(kv[0], 99))
            },
        }
    return out


def weakest_levels(report, threshold=0.85):
    """Every (split, field, level) whose text is not reliably recoverable."""
    bad = []
    for split, res in report["splits"].items():
        for f in FIELDS:
            for lvl, d in res[f]["per_level"].items():
                if d["accuracy"] is not None and d["n"] >= 10 and d["accuracy"] < threshold:
                    bad.append((split, f, lvl, d["accuracy"], d["n"], d["mistaken_for"]))
    return sorted(bad, key=lambda x: x[3])


def run(cfg, args):
    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI()

    options, to_label = field_options(cfg)
    pcfg = cfg["vignette"]["paraphrase"]
    reader = args.reader or pcfg["model"]

    profiles_path = cfg["paths"].get("profiles", "data/profiles.jsonl")
    profiles = {json.loads(l)["profile_id"]: json.loads(l) for l in open(profiles_path)}

    kinds = ["explicit", "implicit"] if args.vignette_type == "both" else [args.vignette_type]
    report = {"reader": reader, "paraphrase_model": pcfg["model"],
              "n_requested": args.n, "splits": {}}

    cache_dir = Path(cfg["paths"]["cache_dir"]) / "roundtrip"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for kind in kinds:
        rows = [json.loads(l) for l in
                open(Path(cfg["paths"]["vignettes_dir"]) / f"{kind}.jsonl")]
        rows = [r for r in rows if r.get("pair_id") is None and r["profile_id"] in profiles]
        random.Random(cfg["seed"]).shuffle(rows)
        rows = rows[: args.n]
        for r in rows:
            p = profiles[r["profile_id"]]
            r["_truth"] = {f: to_label[f].get(p[f]) for f in FIELDS}

        def one(r):
            key = cache_dir / f"{reader}_{len(FIELDS)}f_{r['vignette_id']}.json"
            if key.exists():
                return parse_reply(key.read_text())
            kw = {}
            if pcfg.get("reasoning_effort"):
                kw["reasoning_effort"] = pcfg["reasoning_effort"]
            try:
                resp = client.chat.completions.create(
                    model=reader,
                    messages=[{"role": "user",
                               "content": build_extract_prompt(r["text"], options)}],
                    **kw)
                raw = resp.choices[0].message.content
            except Exception as e:
                print(f"    call failed ({type(e).__name__}); counted as unreadable")
                return None
            key.write_text(raw)
            return parse_reply(raw)

        print(f"  {kind}: reading {len(rows)} vignettes with {reader} "
              f"({len(FIELDS)} fields each) ...")
        with ThreadPoolExecutor(max_workers=pcfg.get("max_workers", 8)) as ex:
            preds = list(ex.map(one, rows))
        report["splits"][kind] = score(rows, preds, options)

    out = Path(cfg["paths"]["results_dir"]) / "roundtrip.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("\n=== per-field roundtrip ===")
    for kind, res in report["splits"].items():
        print(f"\n{kind}:")
        print(f"  {'field':24s} {'n':>5s} {'acc':>7s} {'chance':>7s} {'dist':>6s} {'flips':>7s}")
        for f in FIELDS:
            d = res[f]
            if d["accuracy"] is None:
                print(f"  {f:24s}  no readable predictions")
                continue
            print(f"  {f:24s} {d['n']:5d} {d['accuracy']:7.3f} {d['chance']:7.3f} "
                  f"{d['mean_ordinal_distance']:6.2f} {d['flip_rate']:7.3f}")

    bad = weakest_levels(report, args.threshold)
    print(f"\n=== levels below {args.threshold:.0%} (the text does not carry these) ===")
    if not bad:
        print("  none -- every level is recoverable from its own rendering")
    for split, f, lvl, acc, n, conf in bad:
        seen_as = ", ".join(f"{k} x{v}" for k, v in conf) or "-"
        print(f"  [{split:8s}] {f:24s} {lvl:22s} {acc:6.3f}  (n={n:3d})  read as: {seen_as}")

    if "explicit" in report["splits"] and "implicit" in report["splits"]:
        print("\n  explicit is the reader ceiling; the gap is the cost of implicit rendering:")
        for f in FIELDS:
            e = report["splits"]["explicit"][f]["accuracy"]
            i = report["splits"]["implicit"][f]["accuracy"]
            if e is not None and i is not None:
                flag = "  <-- look" if (e - i) > 0.10 else ""
                print(f"    {f:24s} {e:.3f} -> {i:.3f}   (gap {e - i:+.3f}){flag}")
    print(f"\nreport -> {out}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n", type=int, default=300, help="vignettes per split")
    ap.add_argument("--vignette-type", default="both",
                    choices=["implicit", "explicit", "both"])
    ap.add_argument("--reader", default=None, help="override the reader model")
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="per-level accuracy below this is reported as unreadable")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg, args)


if __name__ == "__main__":
    main()
