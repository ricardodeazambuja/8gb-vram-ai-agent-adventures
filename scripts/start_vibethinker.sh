#!/usr/bin/env bash
#
# start_vibethinker.sh — VibeThinker as a pi agent: server, tool-call recovery shim, then pi.
# It never learned OpenAI tool calls, so the shim turns its own formats into real ones, and pi
# runs without extensions, because their reminders derail it. See docs/models/vibethinker.md.
#
#   bash scripts/start_vibethinker.sh                 # 3B, interactive
#   SIZE=1.5B bash scripts/start_vibethinker.sh       # the math model
#   bash scripts/start_vibethinker.sh -p "write ..."  # one shot
#
# Env: SIZE 3B | 1.5B (default 3B), SHIM_PORT (default 8088), KEEP_EXTENSIONS=1 keeps pi's
# extensions. Other run-qwen.sh knobs pass through (CTX, TEMP, ...).
set -euo pipefail
for _arg in "$@"; do case "$_arg" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \?//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
esac; done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIZE="${SIZE:-3B}"
case "$SIZE" in 3B) QM=vibe3b ;; 1.5B) QM=vibe15 ;; *) echo "ERROR: SIZE=3B or 1.5B"; exit 1 ;; esac
MODEL_ID="vibethinker-${SIZE,,}"
PORT="${PORT:-8080}"
SHIM_PORT="${SHIM_PORT:-8088}"
export PORT QMODEL="$QM" EXTRA_ARGS="${EXTRA_ARGS:---reasoning-budget 600}"

SERVER_PID="" SHIM_PID=""
cleanup() {
  [ -n "$SHIM_PID" ] && kill "$SHIM_PID" 2>/dev/null
  [ -n "$SERVER_PID" ] && bash "$REPO_ROOT/scripts/stop-server.sh" >/dev/null 2>&1
  return 0
}
trap cleanup EXIT

if ! curl -sf "localhost:$PORT/health" >/dev/null; then
  bash "$REPO_ROOT/scripts/run-qwen.sh" > "${SERVER_LOG:-/tmp/vibethinker-server.log}" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 120); do curl -sf "localhost:$PORT/health" >/dev/null && break; sleep 1; done
  curl -sf "localhost:$PORT/health" >/dev/null || { echo "ERROR: the server did not start"; exit 1; }
fi

python3 - "$MODEL_ID" "$SHIM_PORT" <<'PY'
import json, os, sys
model, port = sys.argv[1], sys.argv[2]
path = os.path.expanduser("~/.pi/agent/models.json")
d = json.load(open(path)) if os.path.exists(path) else {"providers": {}}
d.setdefault("providers", {})["vtshim"] = {
    "api": "openai-completions", "apiKey": "dummy", "baseUrl": f"http://127.0.0.1:{port}/v1",
    "compat": {"supportsDeveloperRole": False},
    "models": [{"id": model, "contextWindow": 40960, "name": f"{model} (via tool shim)",
                "input": ["text"], "reasoning": True}]}
os.makedirs(os.path.dirname(path), exist_ok=True)
json.dump(d, open(path, "w"), indent=2)
PY

UPSTREAM="http://127.0.0.1:$PORT" PORT="$SHIM_PORT" python3 "$REPO_ROOT/scripts/vibethinker-toolshim.py" 2>"${SHIM_LOG:-/tmp/vibethinker-shim.log}" &
SHIM_PID=$!
sleep 1

LEAN=(--no-extensions)
[ "${KEEP_EXTENSIONS:-0}" = 1 ] && LEAN=()
pi --provider vtshim --model "$MODEL_ID" "${LEAN[@]}" "$@"
