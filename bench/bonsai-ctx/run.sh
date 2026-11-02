#!/usr/bin/env bash
# How much context Bonsai 2 27B (PTQ1_0) fits with a q8_0 KV cache on an 8 GB card, on our fork.
#   bash bench/bonsai-ctx/run.sh                    # 16K, 32K, 48K, 64K
#   bash bench/bonsai-ctx/run.sh 32768              # just one
# For each size: start the server with its documented launch (QMODEL=bonsai2), read VRAM, then
# fill all but 4,000 tokens of the context with the repo's docs and ask one question about them.
# Results: bench/bonsai-ctx/results/<date>/summary.txt.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="${OUT:-$HERE/results/$(date +%F)}"
mkdir -p "$OUT"
cd "$REPO" || exit 1
[ $# -gt 0 ] || set -- 16384 32768 49152 65536

stop() { systemctl --user stop llama-ctxtest.scope 2>/dev/null || true; sleep 4; }
trap stop EXIT

for ctx in "$@"; do
  stop
  systemd-run --user --scope --unit=llama-ctxtest -p MemoryMax=26G \
    env QMODEL=bonsai2 CTX="$ctx" bash scripts/run-qwen.sh > "$OUT/.server-$ctx.log" 2>&1 &
  pid=$!
  up=0
  for _ in $(seq 120); do
    curl -sf localhost:8080/health >/dev/null && { up=1; break; }
    kill -0 "$pid" 2>/dev/null || break
    sleep 2
  done
  {
    echo "== ctx $ctx, KV ${KVQUANT:-q8_0 (the bonsai2 default)}"
    if [ "$up" = 0 ]; then
      echo "did not start: $(grep -m1 -iE 'failed to|out of memory|error' "$OUT/.server-$ctx.log" | cut -c1-160)"
    else
      echo "VRAM at start: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
      timeout 1800 python3 "$HERE/longprompt.py" $((ctx - 4000)) 2>&1 | tail -2
      echo "VRAM after: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    fi
  } | tee -a "$OUT/summary.txt"
done
rm -f "$OUT"/.server-*.log
