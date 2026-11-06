#!/bin/bash
# sandbox.sh: run a script against Cactus's engine (libcactus.so) and .cact weights with NO network.
#
#   bash bench/cactus/sandbox.sh /bench/cactus/mirage_smoke.py
#
# Docker --network none, read-only root, not root, CPU only. Mounted read-only: the `local` mamba env,
# Cactus's Python wrapper (models/cactus/py, pinned commit in py/SOURCE), the engine and weights
# (models/cactus, fetched from Hugging Face only, sha256-checked) and bench/. Telemetry is off and
# HF is offline as well, so nothing tries to leave; with no network nothing could. Only OUT is writable.
set -eu
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
ENV_DIR="${PY_ENV:-$HOME/miniforge3/envs/local}"
CX="$REPO/models/cactus"
OUT="$(realpath -m "${OUT:-$REPO/models/cactus/out}")"
case "$OUT/" in "$REPO/bench/"*) echo "ERROR: OUT must be outside bench/ ($OUT)" >&2; exit 2 ;; esac
mkdir -p "$OUT"
rc=0
docker run --rm --network none --read-only --tmpfs /tmp:exec \
  --user "$(id -u):$(id -g)" --cap-drop ALL --security-opt no-new-privileges \
  -v "$ENV_DIR:$ENV_DIR:ro" -v "$CX/py:/cx:ro" -v "$CX/engine:/engine:ro" -v "$CX:/weights:ro" \
  -v "$REPO/bench:/bench:ro" -v "$REPO/models/image-gen/out:/images:ro" -v /usr/share/fonts:/usr/share/fonts:ro -v "$OUT:/out" \
  -e PYTHONPATH=/cx -e CACTUS_LIB_PATH=/engine/libcactus.so \
  -e CACTUS_MIRAGE_WEIGHTS=/weights/mirage.cact -e CACTUS_WHISTLE_WEIGHTS=/weights/whistle.cact \
  -e CACTUS_NEEDLE_WEIGHTS=/weights/needle3.cact \
  -e CACTUS_TELEMETRY=0 -e DO_NOT_TRACK=1 -e HF_HUB_OFFLINE=1 -e HF_HOME=/tmp/hf -e HOME=/tmp \
  -e PYTHONDONTWRITEBYTECODE=1 -e ADV_RES=/out \
  ubuntu:22.04@sha256:076e7bf82319619f28dd0aa9c02ab34b055ad37100395b4f49a2ffba04fb5e11 "$ENV_DIR/bin/python" "$@" || rc=$?
if ! bad="$(find "$OUT" ! -type f ! -type d -print -quit)"; then echo "ERROR: could not scan $OUT" >&2; exit 3; fi
if [ -n "$bad" ]; then echo "ERROR: symlink or special file in $OUT; don't copy anything out" >&2; exit 3; fi
exit "$rc"
