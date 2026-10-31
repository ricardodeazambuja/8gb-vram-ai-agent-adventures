#!/bin/bash
# Bake an adapter into the weights and make a GGUF: bake_gguf.sh HF_DIR ADAPTER|- OUT.gguf [QUANT]
# QUANT empty: stop at BF16/F16 (LFM2.5's GGUF is F16). HF scratch dir is removed at the end.
set -euo pipefail
HF=$(realpath "$1"); AD=$2; OUT=$(realpath -m "$3"); Q=${4:-}
L=${ADV_LLAMA:-$(dirname "$0")/../../../llama.cpp}; L=$(realpath "$L")
PY=~/miniforge3/envs/vlft/bin/python
TMP=$OUT.hf; mkdir -p "$TMP"
$PY "$(dirname "$0")/bake.py" --model "$HF" ${AD:+$([ "$AD" != - ] && echo --adapter "$(realpath "$AD")")} --out "$TMP"
RAW=$OUT; [ -n "$Q" ] && RAW=$OUT.raw
(cd "$L" && PYTHONPATH=gguf-py CUDA_VISIBLE_DEVICES= "$PY" convert_hf_to_gguf.py "$TMP" --outfile "$RAW" --outtype "${OUTTYPE:-bf16}" 2>&1 | tr "\r" "\n" | tail -1)
if [ -n "$Q" ]; then
  "$L/build/bin/llama-quantize" "$RAW" "$OUT" "$Q" 2>&1 | tail -2
  rm -f -- "$RAW"
fi
rm -rf -- "$TMP"
ls -la "$OUT"
