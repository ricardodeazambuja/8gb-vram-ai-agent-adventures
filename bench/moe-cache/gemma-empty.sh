#!/usr/bin/env bash
# Base Gemma 4 26B with config/gemma4-26b-nonl.jinja (no '\n' after the thinking, empty thought blocks kept):
# divergence2.py (thinking off and on), then the agent loop with thinking off, then on. logs/gemma-step4.log.
# Run 2026-11-05 on the fork's moe-cache-max-batch build (69ca796dd), 32K, q8_0 KV, cache 3000 MiB.
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ADV="$(cd "$D/../.." && pwd)"
export LOGDIR=${LOGDIR:-$D/logs}
export MODEL=$ADV/models/gemma4-26b-a4b-qat/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf CTX=32768 BATCH=512
export PI_CAPTURE=$ADV/bench/gemma12-probe/capture/agentic12/0001-req.json MAXTOK=6000
export TASK='Write /work/inventory.py from scratch: a complete Python module with an Inventory class (add, remove, restock, total value, JSON save/load) using the write tool. Then write /work/test_inventory.py with thorough pytest tests using the write tool. Then run the tests with bash.'
PY="${PY:-python3} -I"
AFTER="$PY $D/divergence2.py; echo '-- agent loop, thinking off'; $PY $D/agentloop.py http://127.0.0.1:8090 5; echo '-- agent loop, thinking on'; THINK=1 $PY $D/agentloop.py http://127.0.0.1:8090 5" \
  "$D"/longctx/srv.sh gemma-empty -cmoe --moe-cache-mib 3000 --moe-cache-cpu-share 0.95 --chat-template-file "$ADV/config/gemma4-26b-nonl.jinja"
