#!/usr/bin/env bash
# Launch each chat model the documented way, then run the speed, tool-call and code checks.
#   bash bench/recheck/run.sh              # every model, about 1.5 hours
#   bash bench/recheck/run.sh flint groot  # some of them
#   LAUNCH=llm bash bench/recheck/run.sh   # through scripts/llm instead: its defaults (expert cache, Gemma's
#                                          # q8_0 KV and template fix), plus `scripts/llm status` and, for chat
#                                          # models, a pi one-shot (`scripts/llm NAME -p ...`) before the checks
# Results go to bench/recheck/results/<date>/, one file per model, plus summary.txt.
# One model at a time, each server in its own scope so a memory kill stays inside it.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO" || exit 1
ALL=(flint gemma26b tcgemma bolt qwen36 distill groot lfm25 ornith bonsai2 bonsai2q2 spark lfm3 lfm3vl e4b qwen3vl dory vibe3b vibe15)
LAUNCH="${LAUNCH:-backend}"
OUT="${RECHECK_OUT:-$HERE/results/$(date +%F)$([ "$LAUNCH" = llm ] && echo -llm)}"
mkdir -p "$OUT"
if [ "$LAUNCH" = llm ]; then
  ALL=(flint gemma thinkingcap bolt qwen36 distill groot lfm25 ornith bonsai2 spark lfm3 lfm3vl e4b qwen3vl dory twil vibe3b vibe15)
fi
[ $# -gt 0 ] || set -- "${ALL[@]}"
PI_PROMPT="Reply with the single word READY and nothing else. Do not use any tools."

launch() {
  case "$1" in
    gemma26b) echo "env BACKEND=cuda bash scripts/run-server.sh" ;;
    tcgemma)  echo "env BACKEND=cuda MODEL=models/thinkingcap-gemma4-26b-a4b/${TC_FILE:-ThinkingCap-gemma-4-26B-A4B-it-IQ4_XS.gguf} bash scripts/run-server.sh" ;;
    *)        echo "env QMODEL=$1 bash scripts/run-qwen.sh" ;;
  esac
}
stop() {
  if [ "$LAUNCH" = llm ]; then scripts/llm stop >/dev/null 2>&1 || true
  else systemctl --user stop llama-recheck.scope 2>/dev/null || true; fi
  sleep 4
}
trap stop EXIT

# scripts/llm: start the server (and run the pi one-shot for chat models), then the same checks.
llm_up() { # name file
  local m=$1 f=$2 job
  job=$(scripts/llm list --md | grep -F "\`scripts/llm $m\`" | awk -F'|' '{gsub(/ /, "", $3); print $3}')
  echo "launch: scripts/llm $m" >> "$f"
  # TwIL: no pi before its pibench run (docs/security.md). VibeThinker: pi goes through its own shim launcher,
  # outside scripts/llm's scope, so only the server is checked here.
  if [ "$job" = agent ] && [ "$m" != twil ] && [[ $m != vibe* ]]; then
    echo "--- pi one-shot: scripts/llm $m -p \"$PI_PROMPT\"" >> "$f"
    LLM_LOG="$OUT/.server-$m.log" timeout 1200 scripts/llm "$m" -p "$PI_PROMPT" > "$OUT/.pi-$m.log" 2>&1
    echo "exit $?; pi said: $(grep -v '^>>\|^   ' "$OUT/.pi-$m.log" | grep -v '^$' | tail -3 | tr '\n' ' ' | cut -c1-200)" >> "$f"
    grep -E '^>> (MoE cache|starting|.*ready)' "$OUT/.pi-$m.log" >> "$f"
  else
    LLM_LOG="$OUT/.server-$m.log" timeout 1200 scripts/llm "$m" serve > "$OUT/.pi-$m.log" 2>&1
    grep -E '^>> (MoE cache|starting|.*ready)' "$OUT/.pi-$m.log" >> "$f"
  fi
  { echo "--- scripts/llm status"; scripts/llm status; } >> "$f" 2>&1
  curl -sf localhost:8080/health >/dev/null
}

for m in "$@"; do
  stop
  f="$OUT/$m.txt"; : > "$f"
  if [ "$LAUNCH" = llm ]; then
    if ! llm_up "$m" "$f"; then
      { echo "FAILED to start"; tail -6 "$OUT/.pi-$m.log" "$OUT/.server-$m.log" 2>/dev/null | cut -c1-200; } >> "$f"
      printf '%-10s FAILED\n' "$m" | tee -a "$OUT/summary.txt"; continue
    fi
  else
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
  fi
  { echo "VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    echo "pressure: $(head -1 /proc/pressure/memory)"
    timeout 600 python3 scripts/qwen-bench.py
    echo "--- code check"
    timeout 1200 python3 scripts/qwen-codecheck.py --max-tokens 16000 --out "$OUT/.merge-$m.py"
  } >> "$f" 2>&1
  printf '%-10s %s | %s | %s\n' "$m" "$(grep -m1 '^VRAM' "$f")" "$(grep -m1 '== speed' "$f" | cut -c4-40)" "$(grep -E 'vs reference|NO CODE' "$f" | tail -1 | cut -c1-60)" | tee -a "$OUT/summary.txt"
done
rm -f "$OUT"/.server-*.log "$OUT"/.pi-*.log "$OUT"/.merge-*.py
sed -i "s#$REPO/##g" "$OUT"/*.txt   # keep local paths out of the saved results
