#!/usr/bin/env bash
# Quick checks of the non-chat models, each launched the documented way (scripts/run-qwen.sh).
#   bash bench/smoke/run.sh [clefflash d1 eg2 locate]      # default all, writes bench/smoke/rerun/
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="${SMOKE_OUT:-$HERE/rerun}"
mkdir -p "$OUT"
[ $# -gt 0 ] || set -- clefflash d1 eg2 locate
stop() { systemctl --user stop llama-smoke.scope 2>/dev/null || true; }
trap stop EXIT
for m in "$@"; do
  stop
  case "$m" in clefflash|d1) test=(smoke.py decide); args=() ;;
               eg2) test=(smoke.py embed); args=() ;;
               locate) test=(locate_shapes.py); args=(--image) ;;
               *) echo "unknown: $m"; exit 2 ;; esac
  QMODEL=$m systemd-run --user --scope --unit=llama-smoke bash "$REPO/scripts/run-qwen.sh" "${args[@]}" > "$OUT/server-$m.log" 2>&1 &
  for _ in $(seq 120); do curl -sf localhost:8080/health >/dev/null && break; sleep 2; done
  if ! curl -sf localhost:8080/health >/dev/null; then echo "$m: server did not come up, see $OUT/server-$m.log"; exit 1; fi
  { echo "VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; timeout 600 python3 "$HERE/${test[0]}" "${test[@]:1}"; } | tee "$OUT/$m.txt"
done
