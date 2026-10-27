#!/usr/bin/env bash
#
# audit-gguf.sh — thin wrapper that runs utils/audit-gguf.py inside the
# project's conda env (so numpy + gguf are available), falling back to a plain
# python3 if mamba isn't around.
#
# Usage:
#   bash utils/audit-gguf.sh <file.gguf> [--ref BASE] [--dump-template OUT] [--sha256 HEX]
#
# Examples:
#   bash utils/audit-gguf.sh models/gemma4-26b-a4b-qat/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
#   bash utils/audit-gguf.sh models/gemma4-26b-a4b-qat/mmproj-BF16.gguf
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/audit-gguf.py"
ENV_NAME="${ENV_NAME:-llamacpp}"

if command -v mamba >/dev/null 2>&1; then
  exec mamba run --no-capture-output -n "$ENV_NAME" python "$PY" "$@"
else
  exec python3 "$PY" "$@"
fi
