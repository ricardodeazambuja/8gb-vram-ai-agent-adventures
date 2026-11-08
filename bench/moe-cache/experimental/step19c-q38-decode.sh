#!/usr/bin/env bash
# Step 19c (2026-11-08): step 19 had Qwen3.8 decoding ~9% slower on the sync (47.1 vs 51.2-52.3 tok/s, one sync run)
# while Qwen3.6 matched. One instrumented set to tell the candidates apart, decode only (srvbench x2):
#   noise            sync and merged each run twice, interleaved
#   #31888           sync without upstream's moe-cache prefetch line (build-sync-no31888, NO31888_BIN)
#   CUDA graphs      GGML_CUDA_DISABLE_GRAPHS=1 on both builds (#30662 changed graph eviction)
# kv_rows is not a candidate: kv_unified = false in these runs, so it is off. Launcher settings, threshold 257.
# Then MTP (step 19b OOMed at 2770 MiB): Qwen3.8 with --spec-type draft-mtp at a 2000 MiB cache, merged vs sync,
# and both without MTP at the same size; the log's draft acceptance and any rollback lines.
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ADV="$(cd "$Q/../../.." && pwd)"; LD="$Q/../longctx"
pgrep -f "[m]emguard.py" >/dev/null || { echo "memguard not running, refusing"; exit 1; }
: "${NEW_LLAMA:?sync llama.cpp dir}" "${NEW_BIN:?sync server, relative}" "${NO31888_BIN:?no-#31888 server, relative to NEW_LLAMA}"
OLD_LLAMA=${OLD_LLAMA:-$ADV/../llama.cpp} OLD_BIN=${OLD_BIN:-build/bin/llama-server}
export LOGDIR=${LOGDIR:-$Q/logs} RUNS=2 BATCH=2048 GGML_OP_OFFLOAD_MIN_BATCH=257
export MODEL=$ADV/models/qwen38-35b-a3b-distill/Qwen3.8-35B-A3B-IQ4_XS.gguf
C=(-cmoe --moe-cache-mib 2770 --moe-cache-cpu-share 0.95 --moe-cache-max-batch 256)
old() { LLAMA=$OLD_LLAMA BIN=$OLD_BIN "$LD"/srv.sh "$@"; }
new() { LLAMA=$NEW_LLAMA BIN=$NEW_BIN "$LD"/srv.sh "$@"; }
old s19c-merged "${C[@]}"
new s19c-sync "${C[@]}"
LLAMA=$NEW_LLAMA BIN=$NO31888_BIN "$LD"/srv.sh s19c-sync-no31888 "${C[@]}"
GGML_CUDA_DISABLE_GRAPHS=1 new s19c-sync-nographs "${C[@]}"
GGML_CUDA_DISABLE_GRAPHS=1 old s19c-merged-nographs "${C[@]}"
new s19c-sync-b "${C[@]}"
old s19c-merged-b "${C[@]}"
M=(-cmoe --moe-cache-mib 2000 --moe-cache-cpu-share 0.95 --moe-cache-max-batch 256)
acc() { grep -aiE "draft acceptance|acceptance rate|accepted|roll ?back|full reprocess|MTP" "$LOGDIR/srv-$1.log" | sed -E 's/^[0-9.]+ [IWE] //' | sort | uniq -c | sort -rn | head -5 | sed 's/^/    /'; }
old s19c-mtp-merged "${M[@]}" --spec-type draft-mtp --spec-draft-p-min 0; acc s19c-mtp-merged
new s19c-mtp-sync "${M[@]}" --spec-type draft-mtp --spec-draft-p-min 0; acc s19c-mtp-sync
old s19c-c2000-merged "${M[@]}"
new s19c-c2000-sync "${M[@]}"
