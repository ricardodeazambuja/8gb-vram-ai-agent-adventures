#!/bin/bash
# sandbox.sh: run train_glance.py where the model repo's own Python may run (trust_remote_code).
#
#   bash bench/finetune/sandbox.sh fit --model /model
#   OUT=models/lfm3-vl-2b-adapters bash bench/finetune/sandbox.sh train --model /model --out /out/adapter
#
# GPU yes; no network, read-only root, not root. The vlft env, the code and the weights
# (MODEL_DIR, pinned by sha256) are mounted read-only. Only OUT (a scratch folder by default) is
# writable, at /out: code in the container can plant symlinks there, so OUT may not be inside bench/,
# and any symlink or special file found afterwards fails the run. Check before copying results out.
set -eu
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
ENV_DIR="${VLFT_ENV:-$HOME/miniforge3/envs/vlft}"
MODEL_DIR="$(realpath "${MODEL_DIR:-$REPO/models/lfm3-vl-2b-hf}")"
OUT="$(realpath -m "${OUT:-$REPO/models/sandbox-out}")"
case "$OUT/" in "$REPO/bench/"*) echo "ERROR: OUT must be outside bench/ ($OUT)" >&2; exit 2 ;; esac
mkdir -p "$OUT"
rc=0
docker run --rm --gpus all --network none --read-only --tmpfs /tmp:exec \
  --user "$(id -u):$(id -g)" --cap-drop ALL --security-opt no-new-privileges \
  -v "$ENV_DIR:$ENV_DIR:ro" -v "$MODEL_DIR:/model:ro" -v "$REPO/bench:/bench:ro" \
  -v /usr/share/fonts:/usr/share/fonts:ro -v "$OUT:/out" \
  -e ADV_SANDBOX=1 -e ADV_RES=/out -e HF_HOME=/tmp/hf -e HF_HUB_OFFLINE=1 -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  ubuntu:22.04 "$ENV_DIR/bin/python" /bench/finetune/train_glance.py "$@" || rc=$?
# Scan even if the run failed, and treat a folder we can't fully read as a failure.
if ! bad="$(find "$OUT" ! -type f ! -type d -print -quit)"; then
  echo "ERROR: could not scan all of $OUT; don't copy anything out" >&2
  exit 3
fi
if [ -n "$bad" ]; then
  echo "ERROR: symlink or special file in $OUT; don't copy anything out:" >&2
  find "$OUT" ! -type f ! -type d -ls >&2
  exit 3
fi
exit "$rc"
