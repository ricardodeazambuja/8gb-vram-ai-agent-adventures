#!/bin/bash
# sandbox.sh: run train_glance.py where the model repo's own Python may run (trust_remote_code).
#
#   bash bench/finetune/sandbox.sh fit --model /model
#   OUT=bench/finetune/results bash bench/finetune/sandbox.sh train --model /model --out /out/adapter
#
# GPU yes; no network, read-only root, not root. The vlft env, the code and the weights
# (MODEL_DIR, pinned by sha256) are mounted read-only. Only OUT (a scratch folder by default) is
# writable, at /out.
set -eu
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
ENV_DIR="${VLFT_ENV:-$HOME/miniforge3/envs/vlft}"
MODEL_DIR="$(realpath "${MODEL_DIR:-$REPO/models/lfm3-vl-2b-hf}")"
OUT="$(realpath -m "${OUT:-$REPO/bench/finetune/sandbox-out}")"
mkdir -p "$OUT"
exec docker run --rm --gpus all --network none --read-only --tmpfs /tmp:exec \
  --user "$(id -u):$(id -g)" --cap-drop ALL --security-opt no-new-privileges \
  -v "$ENV_DIR:$ENV_DIR:ro" -v "$MODEL_DIR:/model:ro" -v "$REPO/bench:/bench:ro" \
  -v /usr/share/fonts:/usr/share/fonts:ro -v "$OUT:/out" \
  -e ADV_SANDBOX=1 -e ADV_RES=/out -e HF_HOME=/tmp/hf -e HF_HUB_OFFLINE=1 -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  ubuntu:22.04 "$ENV_DIR/bin/python" /bench/finetune/train_glance.py "$@"
