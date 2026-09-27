#!/usr/bin/env bash
# Runs ON the GPU box (started by `scripts/lambda.sh advice`). One advice.py process per
# GPU via --shard i/N, then an unsharded pass that merges the checkpoints and writes output.
# Safe to re-run after a crash: every pass resumes from its checkpoint.
set -uo pipefail
source ~/kyc-venv/bin/activate
BS="${1:-128}"
n=$(nvidia-smi -L | wc -l)
if [ "$n" -gt 1 ]; then
  for i in $(seq 0 $((n - 1))); do
    CUDA_VISIBLE_DEVICES=$i python src/advice.py --batch-size "$BS" --shard "$i/$n" > "advice.shard$i.log" 2>&1 &
  done
  wait
fi
python src/advice.py --batch-size "$BS" 2>&1 | tee advice.log
