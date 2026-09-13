"""S2 batch collector: poll an OpenAI batch, cache its results, then render + QC.

Pairs with `python3 src/vignettes.py --batch`, which submitted the uncached
paraphrase jobs and wrote results/batch_id.txt + results/batch_sidecar.json.

This script:
  1. polls the batch until it finishes (or is already done),
  2. downloads the output and writes each response into the paraphrase cache
     (re-running the banned-word check; leaks/errors fall back to the clean template),
  3. runs the normal render (now an all-cache-hit, no-API pass) and the S2B QC gate.

Usage:
    python3 src/collect_batch.py [--batch-id ID] [--poll-interval 60]
                                    [--profiles ...] [--out-dir ...] [--n-pairs N]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from templates import build_banned_regex
from paraphrase import ParaphraseClient

TERMINAL = {"completed", "failed", "expired", "cancelled"}


def make_client(cfg):
    vcfg = cfg["vignette"]
    pcfg = vcfg["paraphrase"]
    banned = vcfg["banned_lexicon"]
    return ParaphraseClient(
        backend=pcfg["backend"], model=pcfg["model"],
        banned_words=banned, banned_regex=build_banned_regex(banned),
        cache_dir=Path(cfg["paths"]["cache_dir"]) / "paraphrase",
        temperature=pcfg["temperature"], max_retries=pcfg["max_retries"],
        reasoning_effort=pcfg.get("reasoning_effort"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--batch-id", default=None, help="default: read results/batch_id.txt")
    ap.add_argument("--poll-interval", type=int, default=60, help="seconds between polls")
    # forwarded to the render pass so it matches what was submitted
    ap.add_argument("--profiles", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-pairs", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    res_dir = Path(cfg["paths"]["results_dir"])

    batch_id = args.batch_id or (res_dir / "batch_id.txt").read_text().strip()
    sidecar = json.loads((res_dir / "batch_sidecar.json").read_text())
    client = make_client(cfg)

    # 1. poll ---------------------------------------------------------------------
    while True:
        b = client.batch_status(batch_id)
        rc = getattr(b, "request_counts", None)
        done = getattr(rc, "completed", "?") if rc else "?"
        total = getattr(rc, "total", "?") if rc else "?"
        print(f"batch {batch_id}: {b.status} ({done}/{total} done)")
        if b.status in TERMINAL:
            break
        time.sleep(args.poll_interval)

    if b.status != "completed" and not getattr(b, "output_file_id", None):
        print(f"Batch ended as '{b.status}' with no output. "
              f"Check error_file_id={getattr(b, 'error_file_id', None)}; nothing cached.")
        sys.exit(1)

    # 2. cache results ------------------------------------------------------------
    summary = client.collect_batch(batch_id, sidecar)
    print(f"Cached {summary['cached']} responses "
          f"({summary['fallbacks']} template-only fallbacks, {summary['errors']} errors).")
    if b.status == "expired":
        print("WARNING: batch expired -- only the completed subset was cached. "
              "Re-run --batch to submit the remainder, then collect again.")

    # 3. render (all-cache-hit) + QC ----------------------------------------------
    render = [sys.executable, "src/vignettes.py"]
    qc = [sys.executable, "src/qc.py"]
    for flag, val in (("--profiles", args.profiles), ("--out-dir", args.out_dir),
                      ("--n-pairs", args.n_pairs)):
        if val is not None:
            render += [flag, str(val)]
    if args.profiles:
        qc += ["--profiles", args.profiles]
    if args.out_dir:
        qc += ["--vignettes-dir", args.out_dir]

    print(f"\n$ {' '.join(render)}")
    subprocess.run(render, check=True)
    print(f"\n$ {' '.join(qc)}")
    subprocess.run(qc, check=True)


if __name__ == "__main__":
    main()
