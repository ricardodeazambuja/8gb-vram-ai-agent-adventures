#!/usr/bin/env bash
# Launch each chat model the documented way, then run the speed, tool-call and code checks.
#   bash bench/recheck/run.sh              # every model, about 1.5 hours
#   bash bench/recheck/run.sh flint groot  # some of them
# Results go to bench/recheck/results/<date>/, one file per model, plus summary.txt.
# One model at a time, each server in its own scope so a memory kill stays inside it.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="${RECHECK_OUT:-$HERE/results/$(date +%F)}"
mkdir -p "$OUT"
cd "$REPO" || exit 1
ALL=(flint gemma26b tcgemma bolt qwen36 distill groot lfm25 ornith bonsai2 bonsai2q2 spark lfm3 lfm3vl e4b qwen3vl dory coder agentic vibe3b vibe15)
[ $# -gt 0 ] || set -- "${ALL[@]}"

launch() {
  case "$1" in
    gemma26b) echo "env BACKEND=cuda bash scripts/run-server.sh" ;;
    tcgemma)  echo "env BACKEND=cuda MODEL=models/thinkingcap-gemma4-26b-a4b/${TC_FILE:-ThinkingCap-gemma-4-26B-A4B-it-IQ4_XS.gguf} bash scripts/run-server.sh" ;;
    coder)    echo "env BACKEND=cuda bash scripts/run-12b-coder.sh" ;;
    agentic)  echo "env BACKEND=cuda bash scripts/run-12b-agentic.sh" ;;
    *)        echo "env QMODEL=$1 bash scripts/run-qwen.sh" ;;
  esac
}
stop() { systemctl --user stop llama-recheck.scope 2>/dev/null || true; sleep 4; }
trap stop EXIT

for m in "$@"; do
  stop
  f="$OUT/$m.txt"; : > "$f"
  echo "launch: $(launch "$m")" >> "$f"
  # shellcheck disable=SC2046
  systemd-run --user --scope --unit=llama-recheck -p MemoryMax=26G $(launch "$m") > "$OUT/.server-$m.log" 2>&1 &
  pid=$!
  up=0
  for _ in $(seq 240); do
    curl -sf localhost:8080/health >/dev/null && { up=1; break; }
    kill -0 "$pid" 2>/dev/null || break
    sleep 2
  done
  if [ "$up" = 0 ]; then
    { echo "FAILED to start"; tail -4 "$OUT/.server-$m.log" | cut -c1-200; } >> "$f"
    printf '%-10s FAILED\n' "$m" | tee -a "$OUT/summary.txt"; continue
  fi
  { echo "VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    echo "pressure: $(head -1 /proc/pressure/memory)"
    timeout 600 python3 scripts/qwen-bench.py
    echo "--- code check"
    timeout 1200 python3 scripts/qwen-codecheck.py --max-tokens 16000 --out "$OUT/.merge-$m.py"
  } >> "$f" 2>&1
  printf '%-10s %s | %s | %s\n' "$m" "$(grep -m1 '^VRAM' "$f")" "$(grep -m1 '== speed' "$f" | cut -c4-40)" "$(grep -E 'vs reference|NO CODE' "$f" | tail -1 | cut -c1-60)" | tee -a "$OUT/summary.txt"
done
rm -f "$OUT"/.server-*.log "$OUT"/.merge-*.py
sed -i "s#$REPO/##g" "$OUT"/*.txt   # keep local paths out of the saved results
