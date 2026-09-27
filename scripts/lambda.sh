#!/usr/bin/env bash
# Lambda Cloud helper for GPU stages. Run from the repo root on the Mac.
#
#   export LAMBDA_API_KEY=...   (cloud.lambda.ai -> API keys)
#   export HF_TOKEN=...         (needs Gemma-2 license accepted on HF)
#
#   scripts/lambda.sh launch      # pick a 1-GPU H100 (A100 fallback), launch, wait for IP
#   scripts/lambda.sh sync        # push src/, config.yaml, data/ to the box
#   scripts/lambda.sh setup       # venv + deps + HF login on the box
#   scripts/lambda.sh dryrun      # 100-vignette smoke pass (few minutes)
#   scripts/lambda.sh advice      # full advice run inside tmux session "advice"
#   scripts/lambda.sh log         # tail the run log
#   scripts/lambda.sh pull        # copy results/ back to the Mac
#   scripts/lambda.sh terminate   # kill the instance (stops billing)
set -euo pipefail
# keys can also live outside the repo in ~/.kyc-secrets (export lines, chmod 600)
[ -f "$HOME/.kyc-secrets" ] && source "$HOME/.kyc-secrets"

API="https://cloud.lambda.ai/api/v1"
KEY="$HOME/.ssh/lambda_final"
STATE=".lambda_instance"            # gitignored-by-dotfile convention; holds "id ip"
PREFER=(gpu_1x_h100_sxm5 gpu_1x_h100_pcie gpu_1x_gh200 gpu_1x_a100_sxm4 gpu_1x_a100 gpu_1x_a10)
REMOTE_DIR="mechanistic-kyc"
# Lambda's driver supports CUDA 12.8; default PyPI torch targets a newer CUDA and fails
# with "NVIDIA driver too old". Override with TORCH_CUDA=cu130 etc. if the image changes.

api() { curl -fsS -u "${LAMBDA_API_KEY:?set LAMBDA_API_KEY}:" "$@"; }
ip()  { awk '{print $2}' "$STATE"; }
rsh() { ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@$(ip)" "$@"; }

case "${1:-}" in
launch)
  types=$(api "$API/instance-types")
  pick=""; region=""
  for t in "${PREFER[@]}"; do
    region=$(echo "$types" | python3 -c "import json,sys;d=json.load(sys.stdin)['data'];r=d.get('$t',{}).get('regions_with_capacity_available',[]);print(r[0]['name'] if r else '')")
    if [ -n "$region" ]; then pick=$t; break; fi
  done
  [ -n "$pick" ] || { echo "No capacity for any of: ${PREFER[*]}"; exit 1; }
  price=$(echo "$types" | python3 -c "import json,sys;print(json.load(sys.stdin)['data']['$pick']['instance_type']['price_cents_per_hour']/100)")
  echo "Picked $pick in $region at \$$price/hr"
  [ "$pick" = gpu_1x_a10 ] && echo "WARNING: A10 is 24GB -- fine for advice, tight for the activation cache step."

  # reuse the registered key that matches ~/.ssh/lambda_final.pub, else upload it
  pub=$(awk '{print $2}' "$KEY.pub")
  keyname=$(api "$API/ssh-keys" | python3 -c "import json,sys;print(next((k['name'] for k in json.load(sys.stdin)['data'] if '$pub' in k['public_key']),''))")
  if [ -z "$keyname" ]; then
    keyname="mac-lambda-final"
    api -X POST "$API/ssh-keys" -H 'Content-Type: application/json' \
      -d "{\"name\":\"$keyname\",\"public_key\":\"$(cat "$KEY.pub")\"}" >/dev/null
    echo "Uploaded SSH key as $keyname"
  fi

  id=$(api -X POST "$API/instance-operations/launch" -H 'Content-Type: application/json' \
    -d "{\"region_name\":\"$region\",\"instance_type_name\":\"$pick\",\"ssh_key_names\":[\"$keyname\"],\"quantity\":1,\"name\":\"kyc-advice\"}" \
    | python3 -c "import json,sys;print(json.load(sys.stdin)['data']['instance_ids'][0])")
  echo "Launched $id -- waiting for it to boot (usually 2-5 min)..."
  while :; do
    read -r status addr < <(api "$API/instances/$id" | python3 -c "import json,sys;d=json.load(sys.stdin)['data'];print(d['status'],d.get('ip') or '-')")
    [ "$status" = active ] && [ "$addr" != - ] && break
    sleep 15
  done
  echo "$id $addr" > "$STATE"
  echo "Active: ubuntu@$addr  (ssh -i $KEY ubuntu@$addr)"
  ;;
sync)
  rsh "mkdir -p $REMOTE_DIR"
  rsync -az -e "ssh -i $KEY" --exclude __pycache__ \
    src scripts config.yaml requirements.txt conftest.py tests data "ubuntu@$(ip):$REMOTE_DIR/"
  echo "Synced to $(ip):$REMOTE_DIR"
  ;;
setup)
  rsh "cd $REMOTE_DIR && python3 -m venv ~/kyc-venv && source ~/kyc-venv/bin/activate \
    && pip install -q --upgrade pip \
    && pip install -q torch --index-url https://download.pytorch.org/whl/${TORCH_CUDA:-cu128} \
    && pip install -q -r requirements.txt \
    && mkdir -p ~/.cache/huggingface && (umask 077; echo '${HF_TOKEN:?set HF_TOKEN}' > ~/.cache/huggingface/token) \
    && python -c 'import torch;print(\"cuda\",torch.cuda.is_available(),torch.cuda.get_device_name(0))' \
    && python -m pytest -q"
  ;;
dryrun)
  rsh "cd $REMOTE_DIR && source ~/kyc-venv/bin/activate && python src/advice.py --dry-run --batch-size ${BATCH:-128}"
  ;;
advice)
  rsh "cd $REMOTE_DIR && tmux new -d -s advice 'bash scripts/run_advice.sh ${BATCH:-128}'"
  echo "Running in tmux session 'advice'. Watch: scripts/lambda.sh log"
  echo "Attach: ssh -i $KEY ubuntu@$(ip) -t tmux attach -t advice   (detach: Ctrl+B D)"
  ;;
log)
  rsh "tail -c 2000 $REMOTE_DIR/advice.log | tr '\r' '\n' | tail -5"
  ;;
pull)
  rsync -az -e "ssh -i $KEY" "ubuntu@$(ip):$REMOTE_DIR/results/" results/
  echo "Pulled results/ from $(ip)"
  ;;
terminate)
  id=$(awk '{print $1}' "$STATE")
  api -X POST "$API/instance-operations/terminate" -H 'Content-Type: application/json' \
    -d "{\"instance_ids\":[\"$id\"]}" >/dev/null && rm "$STATE" && echo "Terminated $id"
  ;;
*) sed -n 2,17p "$0"; exit 1 ;;
esac
