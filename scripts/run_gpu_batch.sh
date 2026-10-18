#!/usr/bin/env bash
# Runs ON a multi-GPU box (started by `scripts/lambda.sh batch`). One process per GPU:
#   GPUs 0..N-3  steering, sharded (--shard i/K)
#   GPU  N-2     verbal elicitation, then the edge-profile advice run
#   GPU  N-1     steering side effects (MMLU / perplexity / refusals)
# Then an unsharded steering pass merges the shard checkpoints and writes the summary.
# Every stage resumes from its checkpoint, so a re-run after a crash only redoes the gap.
set -uo pipefail
source ~/kyc-venv/bin/activate
export PYTHONPATH=src
BS="${1:-64}"
n=$(nvidia-smi -L | wc -l)
k=$((n - 2))
mkdir -p logs
for i in $(seq 0 $((k - 1))); do
  CUDA_VISIBLE_DEVICES=$i python src/steering.py run --batch-size "$BS" --shard "$i/$k" > "logs/steer.shard$i.log" 2>&1 &
done
CUDA_VISIBLE_DEVICES=$((n - 2)) bash -c "python src/elicit.py > logs/elicit.log 2>&1; \
  python src/advice.py --vignettes-dir data/vignettes_edge --suffix _edge --batch-size $BS > logs/edge.log 2>&1" &
CUDA_VISIBLE_DEVICES=$((n - 1)) python src/steering.py side-effects --batch-size 32 > logs/side_effects.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=0 python src/steering.py run --batch-size "$BS" 2>&1 | tee logs/steer.merge.log
echo "ALL DONE"
