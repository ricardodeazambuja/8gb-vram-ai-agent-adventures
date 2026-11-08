#!/usr/bin/env bash
# Step 19b (2026-11-08): speculative decoding with Qwen3.8's MTP head (our only local model with one; qwen35moe
# drafts one step at a time on every build) on the then-default build 3f2b2d60a, the upstream sync, and the sync with
# LLAMA_KV_ROWS_DISABLE=1 (upstream's new kv_rows attention, on by default with a unified KV cache and flash
# attention, off). Checks the sync's speculative rollback on a hybrid model and kv_rows; srvbench x2 (greedy
# 400-token decode), and the server log's draft acceptance and any rollback warnings.
# Expert cache at the launcher's settings (2770 MiB, batch 2048, limit 256, threshold 257).
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ADV="$(cd "$Q/../../.." && pwd)"; LD="$Q/../longctx"
pgrep -f "[m]emguard.py" >/dev/null || { echo "memguard not running, refusing"; exit 1; }
: "${NEW_LLAMA:?set NEW_LLAMA (llama.cpp dir of the sync build)}" "${NEW_BIN:?set NEW_BIN (server path relative to NEW_LLAMA)}"
: "${OLD_LLAMA:?set OLD_LLAMA: a llama.cpp checkout at tag known-good-2026-11-08-merged, built (srv.sh labels runs with its commit)}"
OLD_BIN=${OLD_BIN:-build/bin/llama-server}
export LOGDIR=${LOGDIR:-$Q/logs} RUNS=2 BATCH=2048 GGML_OP_OFFLOAD_MIN_BATCH=257
export MODEL=$ADV/models/qwen38-35b-a3b-distill/Qwen3.8-35B-A3B-IQ4_XS.gguf
A=(-cmoe --moe-cache-mib 2770 --moe-cache-cpu-share 0.95 --moe-cache-max-batch 256 --spec-type draft-mtp --spec-draft-p-min 0)
tail_log() { grep -aiE "draft acceptance|accepted|MTP chained|roll back|rollback|full reprocess" "$LOGDIR/srv-$1.log" | sed -E 's/^[0-9.]+ [IWE] //' | sort | uniq -c | sort -rn | head -6 | sed 's/^/    /'; }
LLAMA=$OLD_LLAMA BIN=$OLD_BIN "$LD"/srv.sh s19b-q38-mtp-merged "${A[@]}"; tail_log s19b-q38-mtp-merged
LLAMA=$NEW_LLAMA BIN=$NEW_BIN "$LD"/srv.sh s19b-q38-mtp-sync "${A[@]}"; tail_log s19b-q38-mtp-sync
LLAMA_KV_ROWS_DISABLE=1 LLAMA=$NEW_LLAMA BIN=$NEW_BIN "$LD"/srv.sh s19b-q38-mtp-sync-nokvrows "${A[@]}"; tail_log s19b-q38-mtp-sync-nokvrows
LLAMA=$OLD_LLAMA BIN=$OLD_BIN "$LD"/srv.sh s19b-q38-nomtp-merged -cmoe --moe-cache-mib 2770 --moe-cache-cpu-share 0.95 --moe-cache-max-batch 256
