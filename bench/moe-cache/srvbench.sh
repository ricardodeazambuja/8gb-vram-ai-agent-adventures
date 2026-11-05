#!/usr/bin/env bash
# usage: srvbench.sh LABEL [server args...]   -> starts server, measures prefill (long prompt) + decode, stops it
# env: MODEL (default Qwen3.6), CTX, BATCH, RUNS, AFTER (a command to run instead of nothing after srvbench.py),
#      ADV_LLAMA (our fork, default ../llama.cpp next to this repo), OUT (log folder, default logs/ here)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
S="${OUT:-$HERE/logs}"; mkdir -p "$S"
L=$1; shift
cd "${ADV_LLAMA:-$REPO/../llama.cpp}" || exit 1
M=${MODEL:-$REPO/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}
systemd-run --user --scope -q --unit=moebench-$$ mamba run --no-capture-output -n llamacpp-cuda build/bin/llama-server -m "$M" -ngl 99 -c ${CTX:-131072} -ctk q8_0 -ctv q8_0 -fa on -t 6 -b ${BATCH:-512} -ub ${BATCH:-512} --parallel 1 --jinja --port 8090 --host 127.0.0.1 "$@" > "$S/srv-$L.log" 2>&1 &
for _ in $(seq 300); do curl -sf 127.0.0.1:8090/health >/dev/null && break; sleep 1; done
VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)
python3 -I "$HERE/srvbench.py" "$L" "$VRAM" ${RUNS:-1}
${AFTER:-true}
systemctl --user stop moebench-$$.scope 2>/dev/null; pkill -f 'llama-server.*--port 8090'; sleep 3
grep -E 'hit rate' "$S/srv-$L.log" | tail -2
