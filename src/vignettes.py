"""S2: render synthetic client profiles into explicit/implicit vignettes + pairs.

Reads data/profiles.jsonl (from S1), renders each profile as an explicit + implicit twin,
paraphrases both (same backend, §5.3), and writes:
    data/vignettes/explicit.jsonl
    data/vignettes/implicit.jsonl
    data/vignettes/pairs.jsonl     (matched conservative/aggressive counterfactuals)

Vignettes store the CLIENT NARRATIVE ONLY; advice-task options are composed at inference
(S3). Run S2B QC afterwards as the go/no-go gate.

Usage:
    python src/vignettes.py [--config config.yaml] [--limit N]
                                      [--no-paraphrase] [--backend gemini|openai]
"""

import argparse
import json
import random
from pathlib import Path

import yaml

from rubric import (
    factor_bin, factor_cell, factor_scores, is_conflicted, is_contradictory,
    risk_score, risk_tier, sample_decoy_fields, sample_rubric_fields,
)
from templates import build_banned_regex, find_banned, pick_template_id, render
from paraphrase import ParaphraseClient


def load_profiles(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def draw_tier_rubric(rubric, rng, target_tier, max_tries=2000):
    """Reject-sample a rubric-field draw whose tier == target_tier."""
    for _ in range(max_tries):
        fields = sample_rubric_fields(rubric, rng)
        score = risk_score(fields, rubric)
        if risk_tier(score, rubric) == target_tier:
            return fields, score
    raise RuntimeError(f"could not draw a {target_tier} profile")


def field_extremes(spec):
    """(low_value, high_value) for one rubric field: the ends of its scale.

    `invert: true` fields are returned already flipped, so low_value always means
    "supports lower risk" and high_value always means "supports higher risk".
    """
    if "levels" in spec:
        low, high = spec["levels"][0], spec["levels"][-1]
    else:
        lo, hi = spec["range"]
        low, high = lo, hi
    return (high, low) if spec.get("invert") else (low, high)


def build_twin_jobs(profiles, seed, rubric):
    """One render job per (profile, tier). Returns list of job dicts with template_text."""
    jobs = []
    for p in profiles:
        tid = pick_template_id(p["profile_id"], seed)
        contradictory = is_contradictory(p, rubric)
        for vtype in ("explicit", "implicit"):
            rng = random.Random(f"{seed}|{p['profile_id']}|{vtype}")
            jobs.append({
                "vignette_id": f"v_{p['profile_id']}_{vtype}",
                "profile_id": p["profile_id"],
                "pair_id": None, "pair_field": None,
                "pair_factor": None, "pair_side": None,
                "tier": p["tier"],
                "risk_score": p["risk_score"],
                "vignette_type": vtype,
                "template_id": tid,
                "contradictory": contradictory,
                **profile_columns(p, rubric),
                "template_text": render(p, tid, vtype, rng),
            })
    return jobs


def profile_columns(profile, rubric):
    """Factor scores + confound-control columns copied onto each vignette row.

    Probes need the factor targets, and the confound controls (template, name,
    length) have to be on the row so a probe report can show the target is not just
    being read off surface form.
    """
    cols = {name: round(v, 4) for name, v in factor_scores(profile, rubric).items()}
    cols.update({f"{n}_bin": factor_bin(v) for n, v in factor_scores(profile, rubric).items()})
    cols["factor_cell"] = "/".join(factor_cell(profile, rubric))
    cols["conflicted"] = is_conflicted(profile, rubric)
    cols["name"] = profile.get("name")
    return cols


def build_pair_jobs(rubric, n_pairs, render_tier, seed):
    """One-factor counterfactual pairs: two profiles differing in EXACTLY one field.

    The June pilot drew each side of a pair independently, so a "matched pair"
    differed on a median of 7 of 8 rubric fields (never fewer than 4). Patching
    between those cannot attribute an effect to any single factor -- it only shows
    that two unrelated clients produce different behaviour.

    Here both sides share one base profile: same name, occupation, city, template,
    and same value for every rubric field except the one being tested, which is set
    to each end of its own scale. Target fields are cycled so every field gets equal
    coverage, and `pair_field` / `pair_factor` travel with each row so downstream
    patching knows exactly what moved.
    """
    rng = random.Random(f"{seed}|pairs")
    target_fields = list(rubric["fields"])
    jobs = []
    for i in range(n_pairs):
        pair_id = f"pair_{i:04d}"
        decoy = sample_decoy_fields(rng)                       # shared name/occupation/city/...
        tid = pick_template_id(pair_id, seed)
        base = sample_rubric_fields(rubric, rng)               # shared on every other field
        field = target_fields[i % len(target_fields)]          # equal coverage per field
        factor = next(f for f, names in rubric["factors"].items() if field in names)
        low_value, high_value = field_extremes(rubric["fields"][field])
        for side, value in (("lo", low_value), ("hi", high_value)):
            fields = dict(base, **{field: value})
            score = risk_score(fields, rubric)
            profile = {"profile_id": f"{pair_id}_{side}", **fields, **decoy,
                       "risk_score": round(score, 2), "tier": risk_tier(score, rubric)}
            # SAME rng for both sides: the template picks identical phrasing for every
            # sentence, so the only textual difference is the sentence carrying the one
            # field under test. Seeding per side instead would reword every sentence and
            # reintroduce exactly the surface confound one-factor pairs exist to remove.
            prng = random.Random(f"{seed}|{pair_id}")
            jobs.append({
                "vignette_id": f"v_{pair_id}_{side}",
                "profile_id": profile["profile_id"],
                "pair_id": pair_id,
                "pair_field": field,          # the ONE field that differs
                "pair_factor": factor,        # which factor that field belongs to
                "pair_side": side,
                "rubric_fields": dict(fields),   # intermediate; finalize() drops it
                "tier": profile["tier"],
                "risk_score": profile["risk_score"],
                "vignette_type": render_tier,
                "template_id": tid,
                "contradictory": is_contradictory(profile, rubric),
                **profile_columns(profile, rubric),
                # stable_seed: per-field rng, so flipping the field under test cannot
                # reword any other sentence (see templates.render).
                "template_text": render(profile, tid, render_tier, prng,
                                        stable_seed=f"{seed}|{pair_id}"),
            })
    return jobs


def finalize(job, text, model, banned_regex):
    """Attach paraphrase output + banned scan, drop the intermediate template_text."""
    banned = find_banned(text, banned_regex) if job["vignette_type"] == "implicit" else []
    return {
        "vignette_id": job["vignette_id"],
        "profile_id": job["profile_id"],
        "pair_id": job["pair_id"],
        "pair_field": job.get("pair_field"),      # the one field a pair varies
        "pair_factor": job.get("pair_factor"),    # willingness / capacity / goals
        "pair_side": job.get("pair_side"),
        "tier": job["tier"],
        "risk_score": job["risk_score"],
        "vignette_type": job["vignette_type"],
        "template_id": job["template_id"],
        "contradictory": job["contradictory"],
        "paraphrase_model": model,
        "banned_terms_found": banned,
        **{k: job[k] for k in ("willingness", "capacity", "goals", "willingness_bin",
                               "capacity_bin", "goals_bin", "factor_cell", "conflicted",
                               "name") if k in job},
        "n_chars": len(text),          # length confound control
        "text": text,
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def submit_batch_and_exit(client, all_jobs, results_dir):
    """Write uncached jobs to a Batch API input file, submit, persist batch_id + sidecar."""
    results_dir.mkdir(parents=True, exist_ok=True)
    lines, sidecar = client.build_batch([(j["template_text"], j["vignette_type"]) for j in all_jobs])
    cached = len(all_jobs) - len(lines)
    if not lines:
        print(f"All {len(all_jobs)} jobs already cached -- nothing to batch. "
              f"Run normally to render + QC.")
        return

    input_path = results_dir / "batch_input.jsonl"
    with open(input_path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    (results_dir / "batch_sidecar.json").write_text(json.dumps(sidecar))

    batch_id = client.submit_batch(input_path)
    (results_dir / "batch_id.txt").write_text(batch_id)
    print(f"Submitted {len(lines)} uncached jobs ({cached} already cached) to Batch API.")
    print(f"  batch_id: {batch_id}  ->  {results_dir / 'batch_id.txt'}")
    print(f"  collect with: python3 src/collect_batch.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--profiles", default=None,
                    help="input profiles.jsonl (default: <data_dir>/profiles.jsonl)")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: config paths.vignettes_dir)")
    ap.add_argument("--n-pairs", type=int, default=None,
                    help="override pair count (0 to skip pairs, e.g. for edge sets)")
    ap.add_argument("--limit", type=int, default=None, help="cap profiles (dev)")
    ap.add_argument("--no-paraphrase", action="store_true", help="template-only, no API")
    ap.add_argument("--backend", default=None, help="override config paraphrase backend")
    ap.add_argument("--batch", action="store_true",
                    help="submit uncached jobs to the OpenAI Batch API and exit "
                         "(collect later with src/collect_batch.py)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    rubric = cfg["rubric"]
    vcfg = cfg["vignette"]
    pcfg = vcfg["paraphrase"]
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["paths"]["vignettes_dir"])

    profiles_path = (Path(args.profiles) if args.profiles
                     else Path(cfg["paths"]["data_dir"]) / "profiles.jsonl")
    profiles = load_profiles(profiles_path)
    # S1 writes profiles in generation order (aggressive tier clusters at the end), so
    # shuffle deterministically before any --limit slice keeps dev subsets tier-representative.
    random.Random(seed).shuffle(profiles)
    if args.limit:
        # Stratify by factor cell, not a head slice. A raw slice makes a smoke run look
        # cell-imbalanced and trips the QC gate on data that is actually fine.
        by_cell = {}
        for pr in profiles:
            by_cell.setdefault(pr.get("factor_cell", "?"), []).append(pr)
        per_cell = max(1, args.limit // len(by_cell))
        profiles = [pr for cell in sorted(by_cell)
                    for pr in by_cell[cell][:per_cell]][: args.limit]

    banned_words = vcfg["banned_lexicon"]
    banned_regex = build_banned_regex(banned_words)

    if args.n_pairs is not None:
        n_pairs = args.n_pairs
    elif args.limit:
        n_pairs = max(2, args.limit // 4)
    else:
        n_pairs = vcfg["n_pairs"]
    twin_jobs = build_twin_jobs(profiles, seed, rubric)
    pair_jobs = build_pair_jobs(rubric, n_pairs, vcfg["pair_render_tier"], seed)
    all_jobs = twin_jobs + pair_jobs
    print(f"Rendered {len(twin_jobs)} twin + {len(pair_jobs)} pair templates; paraphrasing...")
    print("    (pairs are NOT paraphrased -- see below)")

    client = ParaphraseClient(
        backend=args.backend or pcfg["backend"], model=pcfg["model"],
        banned_words=banned_words, banned_regex=banned_regex,
        cache_dir=Path(cfg["paths"]["cache_dir"]) / "paraphrase",
        temperature=pcfg["temperature"], max_retries=pcfg["max_retries"],
        max_workers=pcfg.get("max_workers", 8), enabled=not args.no_paraphrase,
        reasoning_effort=pcfg.get("reasoning_effort"),
    )

    # Counterfactual pairs are deliberately NOT paraphrased.
    #
    # Paraphrase exists to stop probes memorising template phrasing, and pairs never
    # enter the probe splits (probes.py excludes any row with a pair_id). What pairs
    # ARE for is activation patching, which needs the two sides to differ in exactly
    # one sentence. Paraphrasing each side independently rewords unrelated sentences
    # and destroys that, which is the same surface confound one-factor pairs exist to
    # remove. So pairs keep their template rendering, where the diff is provably
    # minimal. It is also 2 x n_pairs fewer API calls.
    if args.batch:
        submit_batch_and_exit(client, twin_jobs, Path(cfg["paths"]["results_dir"]))
        return

    outputs = client.paraphrase_batch([(j["template_text"], j["vignette_type"]) for j in twin_jobs])
    rows = [finalize(j, t, m, banned_regex) for j, (t, m) in zip(twin_jobs, outputs)]
    rows += [finalize(j, j["template_text"], "template-only-counterfactual", banned_regex)
             for j in pair_jobs]

    by_file = {"explicit": [], "implicit": [], "pairs": []}
    for r in rows:
        if r["pair_id"] is not None:
            by_file["pairs"].append(r)
        else:
            by_file[r["vignette_type"]].append(r)

    for name, rs in by_file.items():
        write_jsonl(out_dir / f"{name}.jsonl", rs)

    leaks = sum(1 for r in rows if r["banned_terms_found"])
    fallbacks = sum(1 for r in rows if r["paraphrase_model"] == "template-only" and not args.no_paraphrase)
    contra = sum(1 for r in rows if r["contradictory"])
    print(f"Wrote {len(by_file['explicit'])} explicit, {len(by_file['implicit'])} implicit, "
          f"{len(by_file['pairs'])} pair rows to {out_dir}")
    print(f"  contradictory: {contra} rows  |  banned-term leaks: {leaks}  |  "
          f"template-only fallbacks: {fallbacks}")
    if leaks:
        print("  WARNING: leaks present -- S2B QC will hard-fail until resolved.")


if __name__ == "__main__":
    main()
