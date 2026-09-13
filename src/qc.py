"""S2B: QC gate for rendered vignettes. Blocks the pipeline (exit 1) on hard failure.

Checks (design-doc §6.7):
  - banned-lexicon scan over implicit + implicit-tier pairs == 0 hits        [HARD]
  - factor-cell (willingness x capacity) balance within +/- tier_balance_tol
  - mean token length matched across tiers & types within +/- token_len_tol
  - MinHash near-duplicate fraction < dup_fraction_max
  - age/experience coherence: young clients don't assert a long investing tenure (soft)
  - round-trip tier-recovery on a sample of implicit vignettes (signal survived?)

Writes results/qc_report.json and prints a summary.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

from templates import build_banned_regex, find_banned

# A client sampled young (age <= TENURE_YOUNG_MAX) whose text asserts a long PAST investing
# tenure ("been investing for two decades") is incoherent -- it implies they started as a
# child. age and investing_experience are sampled independently, so this is a real risk.
# The pattern is deliberately tight: a past-investing verb, then for/since, then a LONG
# duration (decades, or >=10 years). It must NOT fire on future horizon ("won't need this
# money for several decades"), short plausible tenures ("investing for a few years"), or
# vague phrasing ("investing for a while"). Flagged soft so it never blocks the pipeline.
TENURE_YOUNG_MAX = 30
_PAST_INVEST = r"(?:been\s+(?:actively\s+)?investing|been\s+an\s+investor|investing|invested)"
_CONNECT = r"\s+(?:for|since)\s+(?:the\s+|past\s+|last\s+|about\s+|nearly\s+|over\s+|almost\s+|roughly\s+)*"
_LONG_DUR = (r"(?:(?:a\s+couple\s+of\s+|two|three|four|five|six|seven|eight|nine|several|many|"
             r"ten|fifteen|twenty|thirty|\d{2})\s+decades?"
             r"|(?:ten|fifteen|twenty|thirty|\d{2})\s+years?)")
TENURE_PAT = re.compile(_PAST_INVEST + _CONNECT + _LONG_DUR, re.I)


def read_jsonl(path):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


# -- tokenization (subject tokenizer if available, else whitespace proxy) ---------------
def get_token_counter():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
        return (lambda s: len(tok.encode(s))), "gemma-2-9b-it"
    except Exception:
        return (lambda s: len(s.split())), "whitespace-proxy"


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def tenure_incoherences(rows, profiles_by_id):
    """Vignettes where a young client (<= TENURE_YOUNG_MAX) asserts a long investing tenure."""
    hits = []
    for r in rows:
        p = profiles_by_id.get(r["profile_id"])
        if p and p.get("age", 99) <= TENURE_YOUNG_MAX and TENURE_PAT.search(r["text"]):
            m = TENURE_PAT.search(r["text"])
            hits.append({"vignette_id": r["vignette_id"], "age": p["age"],
                         "snippet": r["text"][max(0, m.start() - 15):m.end() + 15].strip()})
    return hits


# -- MinHash near-dup (datasketch if present, else sampled shingle-Jaccard) -------------
def shingles(text, n=5):
    toks = text.lower().split()
    return {" ".join(toks[i:i + n]) for i in range(max(1, len(toks) - n + 1))}


def dup_fraction(texts, threshold=0.7):
    try:
        from datasketch import MinHash, MinHashLSH
        lsh = MinHashLSH(threshold=threshold, num_perm=64)
        mhs = []
        for i, t in enumerate(texts):
            m = MinHash(num_perm=64)
            for sh in shingles(t):
                m.update(sh.encode())
            lsh.insert(str(i), m)
            mhs.append(m)
        dup = sum(1 for i, m in enumerate(mhs) if len(lsh.query(m)) > 1)
        return dup / len(texts) if texts else 0.0, "datasketch"
    except ImportError:
        # Fallback: sampled O(k^2) Jaccard near-dup estimate.
        import random
        sample = texts if len(texts) <= 400 else random.Random(0).sample(texts, 400)
        sh = [shingles(t) for t in sample]
        dup = 0
        for i in range(len(sh)):
            for j in range(i + 1, len(sh)):
                inter = len(sh[i] & sh[j])
                uni = len(sh[i] | sh[j]) or 1
                if inter / uni >= threshold:
                    dup += 1
                    break
        return dup / len(sample) if sample else 0.0, "sampled-jaccard"


# -- round-trip tier recovery (needs an API key; skipped gracefully otherwise) ----------
def roundtrip_recovery(implicit_rows, k, cfg):
    import random
    sample = implicit_rows if len(implicit_rows) <= k else random.Random(0).sample(implicit_rows, k)
    backend = cfg["vignette"]["paraphrase"]["backend"]
    prompt_tail = ("\n\nBased only on this description, classify the person's investment "
                   "risk profile as exactly one word: conservative, moderate, or aggressive. "
                   "Answer with the single word only.")
    try:
        if backend == "gemini":
            import google.generativeai as genai
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                return None
            genai.configure(api_key=key)
            model = genai.GenerativeModel(cfg["vignette"]["paraphrase"]["model"])
            def ask(t):
                return model.generate_content(t + prompt_tail).text.strip().lower()
        elif backend == "openai":
            from openai import OpenAI
            if not os.environ.get("OPENAI_API_KEY"):
                return None
            client = OpenAI()
            def ask(t):
                r = client.chat.completions.create(
                    model=cfg["vignette"]["paraphrase"]["model"],
                    messages=[{"role": "user", "content": t + prompt_tail}])
                return r.choices[0].message.content.strip().lower()
        else:
            return None
    except Exception:
        return None

    # Raw accuracy is the wrong headline on a skewed label. Profiles are balanced over
    # the willingness x capacity grid, so the DERIVED tier comes out near 28/59/13 and
    # a model that answers "moderate" every time scores ~0.59. Reporting accuracy
    # against a hardcoded 1/3 chance would make a below-baseline result look like a
    # pass. Macro recall (mean per-tier recall) is the metric that survives the skew.
    from collections import Counter
    correct, per_tier_hits, per_tier_n, n_asked = 0, Counter(), Counter(), 0
    for r in sample:
        try:
            guess = ask(r["text"])
        except Exception:
            continue
        n_asked += 1
        per_tier_n[r["tier"]] += 1
        if r["tier"] in guess:
            correct += 1
            per_tier_hits[r["tier"]] += 1
    if not n_asked:
        return None
    recalls = {t: per_tier_hits[t] / per_tier_n[t] for t in per_tier_n}
    majority = max(per_tier_n.values()) / n_asked
    return {"n": n_asked,
            "accuracy": correct / n_asked,
            "majority_baseline": round(majority, 4),
            "macro_recall": round(sum(recalls.values()) / len(recalls), 4),
            "per_tier_recall": {t: round(v, 4) for t, v in recalls.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--vignettes-dir", default=None,
                    help="override config paths.vignettes_dir (e.g. data/vignettes_edge)")
    ap.add_argument("--profiles", default=None,
                    help="profiles.jsonl to join for the age/experience check "
                         "(default: <data_dir>/profiles.jsonl)")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    vdir = Path(args.vignettes_dir) if args.vignettes_dir else Path(cfg["paths"]["vignettes_dir"])
    qc = cfg["vignette"]["qc"]
    regex = build_banned_regex(cfg["vignette"]["banned_lexicon"])

    profiles_path = (Path(args.profiles) if args.profiles
                     else Path(cfg["paths"]["data_dir"]) / "profiles.jsonl")
    profiles_by_id = {p["profile_id"]: p for p in read_jsonl(profiles_path)}

    explicit = read_jsonl(vdir / "explicit.jsonl")
    implicit = read_jsonl(vdir / "implicit.jsonl")
    pairs = read_jsonl(vdir / "pairs.jsonl")
    report = {"counts": {"explicit": len(explicit), "implicit": len(implicit), "pairs": len(pairs)}}
    hard_fail = []

    # 1. Banned-lexicon scan (hard) over everything rendered in the implicit tier.
    implicit_like = implicit + [p for p in pairs if p["vignette_type"] == "implicit"]
    leaks = [{"vignette_id": r["vignette_id"], "hits": find_banned(r["text"], regex)}
             for r in implicit_like if find_banned(r["text"], regex)]
    report["banned_leaks"] = {"count": len(leaks), "examples": leaks[:10]}
    if leaks:
        hard_fail.append(f"{len(leaks)} banned-lexicon leaks in implicit set")

    # 2. Factor-cell balance.
    #
    # The blended tier is NOT balanced any more, on purpose: profiles are balanced
    # over the willingness x capacity grid, and balancing a weighted sum cannot
    # balance its parts, so the derived tier mix comes out near 28/58/13. Checking
    # tier balance here would fail on correct data every run and train everyone to
    # ignore QC. The grid is what must be flat, so that is what is checked; the tier
    # mix is reported for information only.
    rows_all = explicit + implicit
    cell_counts = Counter(r.get("factor_cell") for r in rows_all if r.get("factor_cell"))
    total = sum(cell_counts.values()) or 1
    shares = {c: n / total for c, n in cell_counts.items()}
    expected = 1.0 / len(cell_counts) if cell_counts else 0.0
    bal_ok = bool(cell_counts) and all(
        abs(s - expected) <= qc["tier_balance_tol"] for s in shares.values())
    tier_counts = Counter(r["tier"] for r in rows_all)
    tier_total = sum(tier_counts.values()) or 1
    report["factor_cell_balance"] = {
        "shares": shares, "expected": expected, "tol": qc["tier_balance_tol"],
        "pass": bal_ok,
        "derived_tier_mix": {t: c / tier_total for t, c in tier_counts.items()},
        "conflicted_share": sum(bool(r.get("conflicted")) for r in rows_all) / tier_total,
    }

    # 3. Token-length match across tiers & types.
    count_tokens, tok_name = get_token_counter()
    groups = {"explicit": explicit, "implicit": implicit}
    means = {g: mean([count_tokens(r["text"]) for r in rs]) for g, rs in groups.items() if rs}
    if means:
        mlo, mhi = min(means.values()), max(means.values())
        len_ok = (mhi - mlo) / (mhi or 1) <= qc["token_len_tol"]
    else:
        len_ok = True
    report["token_length"] = {"tokenizer": tok_name, "means": means,
                              "tol": qc["token_len_tol"], "pass": len_ok}

    # 4. Near-duplicate fraction.
    frac, dup_method = dup_fraction([r["text"] for r in (explicit + implicit)])
    dup_ok = frac < qc["dup_fraction_max"]
    report["duplicates"] = {"method": dup_method, "fraction": frac,
                            "max": qc["dup_fraction_max"], "pass": dup_ok}

    # 5. Age/experience coherence (soft): young clients asserting a long investing tenure.
    if profiles_by_id:
        tenure_hits = tenure_incoherences(explicit + implicit + pairs, profiles_by_id)
        coh_ok = not tenure_hits
        report["age_experience_coherence"] = {
            "count": len(tenure_hits), "examples": tenure_hits[:10],
            "young_age_max": TENURE_YOUNG_MAX, "pass": coh_ok}
    else:
        coh_ok = True
        report["age_experience_coherence"] = {"skipped": f"profiles not found at {profiles_path}"}

    # 6. Round-trip signal check (soft; skipped without an API key).
    rt = roundtrip_recovery(implicit, qc["roundtrip_sample"], cfg)
    report["roundtrip"] = rt if rt else {"skipped": "no API key / backend unavailable"}

    # -- write + summarize -------------------------------------------------------
    res_dir = Path(cfg["paths"]["results_dir"])
    res_dir.mkdir(parents=True, exist_ok=True)
    # Report filename tracks the vignettes dir so an edge run doesn't clobber the main one:
    # data/vignettes -> qc_report.json, data/vignettes_edge -> s2_qc_report_edge.json.
    tag = re.sub(r"^vignettes_?", "", vdir.name)
    report_path = res_dir / (f"qc_report_{tag}.json" if tag else "qc_report.json")
    report_path.write_text(json.dumps(report, indent=2))

    def mark(b):
        return "PASS" if b else "FAIL"
    print("=== S2B QC ===")
    print(f"counts: {report['counts']}")
    print(f"[{mark(not leaks)}] banned-lexicon leaks: {len(leaks)}")
    fb = report["factor_cell_balance"]
    print(f"[{mark(bal_ok)}] factor-cell balance: {len(shares)} cells, "
          f"expected {fb['expected']:.3f} each, "
          f"spread {min(shares.values(), default=0):.3f}-{max(shares.values(), default=0):.3f}")
    print(f"       derived tier mix (informational, imbalance is intended): "
          f"{{ {', '.join(f'{t}:{s:.3f}' for t, s in fb['derived_tier_mix'].items())} }}")
    print(f"       conflicted share: {fb['conflicted_share']:.3f}")
    print(f"[{mark(len_ok)}] token length ({tok_name}): {means}")
    print(f"[{mark(dup_ok)}] near-dup fraction ({dup_method}): {frac:.4f} < {qc['dup_fraction_max']}")
    coh = report["age_experience_coherence"]
    if "skipped" in coh:
        print(f"[----] age/experience coherence: skipped ({coh['skipped']})")
    else:
        print(f"[{mark(coh_ok)}] age/experience coherence: {coh['count']} young clients "
              f"asserting long investing tenure")
    if rt:
        beats = rt["accuracy"] >= rt["majority_baseline"]
        print(f"[----] round-trip tier recovery: acc {rt['accuracy']:.2f} vs "
              f"majority-guess {rt['majority_baseline']:.2f} "
              f"({'above' if beats else 'BELOW'}), macro-recall {rt['macro_recall']:.2f} "
              f"on n={rt['n']}")
        print(f"       per-tier recall: {rt['per_tier_recall']}")
        print( "       note: this asks for the BLENDED tier, which is a weighted sum of 8 "
               "fields.")
        print( "       Low recovery here is expected and is not a text-quality signal -- "
               "see roundtrip.py")
        print( "       for whether the individual FACTS survived, which is the thing QC "
               "can actually act on.")
    else:
        print("[----] round-trip tier recovery: skipped (no API key)")
    print(f"report -> {report_path}")

    soft = [n for n, ok in (("factor-cell balance", bal_ok), ("token length", len_ok),
                            ("duplicates", dup_ok), ("age/experience coherence", coh_ok)) if not ok]
    if soft:
        print(f"SOFT WARNINGS: {', '.join(soft)}")
    if hard_fail:
        print(f"HARD FAIL: {'; '.join(hard_fail)}")
        sys.exit(1)
    print("QC gate passed.")


if __name__ == "__main__":
    main()
