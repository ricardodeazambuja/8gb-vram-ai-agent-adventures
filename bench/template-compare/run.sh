#!/usr/bin/env bash
# run.sh: start Bonsai 2 twice (its own template, then Unsloth's via --chat-template-file), have llama-server
# render the 18 captured pi requests with each (/apply-template, nothing is generated), then compare.
# Rendering needs no GPU, so the model loads on the CPU (NGL=0, 4K context: ~6 GB of RAM, no VRAM). Port 8091,
# so a server on 8080 is left alone. About 1 minute.
set -euo pipefail
B="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ADV="$(cd "$B/../.." && pwd)"
export PORT=8091; TMP=$(mktemp -d); mapfile -t REQS < <(ls "$ADV"/bench/gemma12-probe/capture/*/*-req.json)
for arm in bonsai2 unsloth; do
  extra=""; [ $arm = unsloth ] && extra="--chat-template-file $B/templates/unsloth-2026-11-11.jinja"
  NGL=0 CTX=4096 EXTRA_ARGS="$extra" "$ADV/scripts/llm" bonsai2 serve > "$TMP/serve-$arm.log" 2>&1 \
    || { tail -5 "$TMP/serve-$arm.log"; "$ADV/scripts/llm" stop > /dev/null 2>&1; exit 1; }
  python3 -I "$B/render.py" "$PORT" "$TMP/rendered-$arm.json" "${REQS[@]}"
  "$ADV/scripts/llm" stop > /dev/null 2>&1; sleep 3
done
python3 -I "$B/compare.py" "$TMP/rendered-bonsai2.json" "$TMP/rendered-unsloth.json" "$B/results"
rm -rf "$TMP"
