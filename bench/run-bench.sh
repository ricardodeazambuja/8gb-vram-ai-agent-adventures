#!/usr/bin/env bash
#
# run-bench.sh — run the toy agentic benchmark: every model × every task × N repeats, each run
# in a fresh sandboxed container. Results land in bench/results/raw/ (one folder per run, with
# the full pi transcript) and are summarised by bench/summarize.py.
#
#   bash bench/run-bench.sh                              # all models, all tasks, 3 repeats
#   MODELS="flint qwen36" REPEATS=1 bash bench/run-bench.sh
#   TASKS="scrape-books" MODELS=spark bash bench/run-bench.sh
#
# Resumable: a run whose score.json exists is skipped, so re-running continues where it stopped.
# Needs: the pibench image (docker build -t pibench:1 bench/), the pibench network and the
# firewall rules (bench/README.md). It refuses to start if the sandbox can reach the LAN.
#
# Config (env vars):
#   MODELS   space-separated model keys   (default: flint qwen36 distill spark gemma26b bonsai2)
#   TASKS    space-separated task names   (default: every folder in bench/tasks)
#   REPEATS  runs per model × task        (default: 3)
#   TIME_LIMIT  seconds per run           (default: 1200)
#   PREFLIGHT_ONLY=1  only check the sandbox, then exit
#
set -euo pipefail

for _arg in "$@"; do case "$_arg" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \?//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
esac; done

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$BENCH")"
GW=172.30.0.1
NET=pibench
IMAGE=pibench:1
MODELS="${MODELS:-flint qwen36 distill spark gemma26b bonsai2}"
TASKS="${TASKS:-$(find "$BENCH/tasks" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | tr '\n' ' ')}"
REPEATS="${REPEATS:-3}"
TIME_LIMIT="${TIME_LIMIT:-1200}"
RAW="$BENCH/results/raw"
SERVER_LOG="$BENCH/results/server.log"
mkdir -p "$RAW"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# --- the sandbox must be sealed before any model gets tools ----------------------------------
preflight() {
  # A throwaway listener on another Docker bridge address of this host. Without the firewall a
  # pibench container reaches it; with the firewall (INPUT drop) it must time out.
  local probe_ip; probe_ip="$(ip -4 -o addr show docker0 | awk '{print $4}' | cut -d/ -f1)"
  python3 -m http.server 18099 --bind "$probe_ip" >/dev/null 2>&1 &
  local probe_pid=$!
  sleep 1
  local out
  out="$(docker run --rm -i --network "$NET" "$IMAGE" python3 - "$probe_ip" <<'PY' 2>&1 || true
import socket, sys, urllib.request
def reach(host, port):
    try:
        socket.create_connection((host, port), timeout=4).close(); return True
    except OSError:
        return False
lan = [("host-other-bridge", sys.argv[1], 18099), ("router-a", "192.168.0.1", 80), ("router-b", "192.168.1.1", 80)]
bad = [name for name, h, p in lan if reach(h, p)]
try:
    urllib.request.urlopen("https://books.toscrape.com/", timeout=15).read(100); net = True
except Exception:
    net = False
print("LEAK " + ",".join(bad) if bad else "SEALED", "INTERNET" if net else "NO-INTERNET")
PY
)"
  kill "$probe_pid" 2>/dev/null || true
  log "preflight: $out"
  case "$out" in
    *"SEALED INTERNET"*) ;;
    *) echo "ERROR: sandbox check failed ($out). Install the rules: sudo sh bench/firewall.sh add"; exit 1 ;;
  esac
}

# --- model servers ------------------------------------------------------------------------------
start_server() {  # start_server KEY — launch on the bridge address, wait until healthy
  local key="$1"
  if curl -fsS -m 2 "http://127.0.0.1:8080/health" >/dev/null 2>&1 || curl -fsS -m 2 "http://$GW:8080/health" >/dev/null 2>&1; then
    echo "ERROR: something is already serving on port 8080; stop it first (scripts/stop-server.sh)"; exit 1
  fi
  log "starting $key"
  case "$key" in
    gemma26b)
      # The best measured 64K setting (30.5 tok/s; docs/how-it-fits.md).
      (cd "$REPO_ROOT" && BACKEND=cuda CTX=65536 NCMOE=20 KVQUANT=q8_0 HOST=$GW \
        setsid bash scripts/run-server.sh >>"$SERVER_LOG" 2>&1 &) ;;
    tcgemma)
      # bottlecapai's ThinkingCap fine-tune of the same model. Its IQ4_XS file keeps the non-expert
      # tensors at higher precision (2.6 GB vs 1.4 GB), so 3 more layers' experts go to RAM.
      (cd "$REPO_ROOT" && BACKEND=cuda CTX=65536 NCMOE=23 KVQUANT=q8_0 HOST=$GW \
        MODEL="$REPO_ROOT/models/thinkingcap-gemma4-26b-a4b/ThinkingCap-gemma-4-26B-A4B-it-IQ4_XS.gguf" \
        ALIAS=thinkingcap-gemma-4-26b-a4b setsid bash scripts/run-server.sh >>"$SERVER_LOG" 2>&1 &) ;;
    *)
      (cd "$REPO_ROOT" && QMODEL="$key" HOST=$GW setsid bash scripts/run-qwen.sh >>"$SERVER_LOG" 2>&1 &) ;;
  esac
  STARTED=1
  for _ in $(seq 300); do
    curl -fsS -m 2 "http://$GW:8080/health" >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "ERROR: $key did not come up; see $SERVER_LOG"; stop_server; exit 1
}

STARTED=0
stop_server() {  # stops only a server this script started
  [ "$STARTED" = 1 ] || return 0
  STARTED=0
  local pids
  pids="$(ss -ltnpH "sport = :8080" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u || true)"
  [ -n "$pids" ] || return 0
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  for _ in $(seq 30); do
    ss -ltnH "sport = :8080" | grep -q . || return 0
    sleep 1
  done
  # shellcheck disable=SC2086
  kill -9 $pids 2>/dev/null || true
  sleep 2
}

write_models_json() {  # write_models_json PATH — pi provider config for whatever is being served
  python3 - "$1" "$GW" <<'PY'
import json, sys, urllib.request
path, gw = sys.argv[1], sys.argv[2]
mid = json.load(urllib.request.urlopen(f"http://{gw}:8080/v1/models"))["data"][0]["id"]
ctx = json.load(urllib.request.urlopen(f"http://{gw}:8080/props"))["default_generation_settings"]["n_ctx"]
cfg = {"providers": {"bench": {
    "api": "openai-completions", "apiKey": "dummy", "baseUrl": f"http://{gw}:8080/v1",
    "compat": {"supportsDeveloperRole": False},
    "models": [{"id": mid, "name": mid, "contextWindow": ctx, "input": ["text"], "reasoning": True}]}}}
json.dump(cfg, open(path, "w"), indent=1)
print(mid, ctx)
PY
}

# --- one run ------------------------------------------------------------------------------------
run_one() {  # run_one MODEL_KEY MODEL_ID TASK REP
  local key="$1" mid="$2" task="$3" rep="$4"
  local dir="$RAW/$key/$task/r$rep" tdir="$BENCH/tasks/$task"
  [ -f "$dir/score.json" ] && { log "skip $key $task r$rep (done)"; return 0; }
  rm -rf "$dir"; mkdir -p "$dir/work"
  if [ -d "$tdir/files" ]; then cp -r "$tdir/files/." "$dir/work/"; fi
  cp "$BENCH/results/models.json" "$dir/models.json"

  local prompt t0 t1 rc=0
  prompt="$(cat "$BENCH/prompt-header.md" "$tdir/prompt.md")"
  log "run  $key $task r$rep"
  t0=$(date +%s.%N)
  docker run --rm --name "pibench-$key-$task-r$rep" --network "$NET" \
    --cap-drop ALL --security-opt no-new-privileges --pids-limit 512 \
    --memory 4g --cpus 4 \
    -v "$dir/work:/work" -v "$dir/models.json:/bench/models.json:ro" \
    "$IMAGE" timeout --kill-after=10 "$TIME_LIMIT" \
    pi -p --no-session -ne -np --no-mcp --provider bench --model "$mid" --mode json \
    "${THINK_ARGS[@]}" "$prompt" \
    >"$dir/pi.jsonl" 2>"$dir/pi.stderr" || rc=$?
  t1=$(date +%s.%N)

  # Score in a fresh container with no network: copy the work, add the hidden tests, run them.
  docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges \
    -v "$dir/work:/work:ro" -v "$tdir/tests:/tests:ro" "$IMAGE" sh -c '
      cp -r /work /tmp/s && cp -r /tests/. /tmp/s/ && cd /tmp/s &&
      timeout 300 python3 -I -m pytest -q -p no:cacheprovider --noconftest -c /dev/null --rootdir /tmp/s --junitxml=/tmp/junit.xml test_hidden.py >/dev/null 2>&1
      cat /tmp/junit.xml 2>/dev/null' >"$dir/junit.xml" || true

  python3 "$BENCH/score.py" "$dir" "$key" "$mid" "$task" "$rep" "$rc" "$t0" "$t1" "$TIME_LIMIT"

  # A run that failed at the API before producing a single token measures the setup, not the
  # model: keep it aside for diagnosis and stop, instead of recording a score.
  if python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s["output_tokens"]==0 and s["api_errors"]>0 else 1)' "$dir/score.json"; then
    rm -rf "$dir.invalid"; mv "$dir" "$dir.invalid"
    echo "ERROR: $key $task r$rep failed at the API with no output; see $dir.invalid/pi.jsonl"; exit 1
  fi
}

# --- main ---------------------------------------------------------------------------------------
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "ERROR: build the image: docker build -t $IMAGE bench/"; exit 1; }
docker network inspect "$NET" >/dev/null 2>&1 || {
  echo "ERROR: create the network: docker network create --subnet 172.30.0.0/24 --gateway $GW -o com.docker.network.bridge.name=br-pibench $NET"; exit 1; }
preflight
[ "${PREFLIGHT_ONLY:-0}" = 1 ] && exit 0
trap stop_server EXIT

for key in $MODELS; do
  todo=0
  for task in $TASKS; do for rep in $(seq "$REPEATS"); do
    [ -f "$RAW/$key/$task/r$rep/score.json" ] || todo=1
  done; done
  [ "$todo" = 1 ] || { log "skip $key (all done)"; continue; }
  start_server "$key"
  read -r mid ctx < <(write_models_json "$BENCH/results/models.json")
  log "serving $mid, n_ctx=$ctx"
  # pi asks for reasoning effort "medium" by default. Spark's chat template accepts only high or
  # low (and rejects the request otherwise); its card evaluates with high.
  THINK_ARGS=()
  [ "$key" = spark ] && THINK_ARGS=(--thinking high)
  for rep in $(seq "$REPEATS"); do
    for task in $TASKS; do run_one "$key" "$mid" "$task" "$rep"; done
  done
  stop_server
done
log "all done"
python3 "$BENCH/summarize.py"
