#!/usr/bin/env bash
#
# download-hf-weights.sh — the safetensors repos fine-tuning trains on, at pinned revisions.
#
#   bash scripts/download-hf-weights.sh lfm25vl     # LFM2.5-VL-450M, 0.9 GB
#   bash scripts/download-hf-weights.sh lfm3vl      # LFM3-VL-2B, 4 GB
#   bash scripts/download-hf-weights.sh --check     # only verify what's on disk
#
# Unlike download-models.sh this also fetches LFM3-VL's Python. That code may run only through
# bench/finetune/sandbox.sh (docs/security.md). Every file is checked against
# envs/hf-weights.sha256, and a mismatch stops the script.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUMS="$REPO_ROOT/envs/hf-weights.sha256"
ENV_NAME="${ENV_NAME:-llamacpp}"
declare -A HF=([lfm25vl]="LiquidAI/LFM2.5-VL-450M fc6221ca597f3315e4f82fc2df606783267b34ba lfm2.5-vl-450m-hf"
               [lfm3vl]="LiquidAI/LFM3-VL-2B ba44bd51db20a2753f28c273a1571ccaa210d954 lfm3-vl-2b-hf")

check() { # models/ subfolder
  (cd "$REPO_ROOT/models" && grep -F "  $1/" "$SUMS" | sha256sum --check --quiet) && echo ">> $1: every file matches"
}

case "${1:-}" in
  --check) for k in "${!HF[@]}"; do read -r _ _ dir <<<"${HF[$k]}"; check "$dir"; done; exit 0 ;;
  lfm25vl|lfm3vl) ;;
  *) sed -n '3,11p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 2 ;;
esac
read -r repo rev dir <<<"${HF[$1]}"
PREFIX="$(mamba run -n "$ENV_NAME" printenv CONDA_PREFIX)"
"$PREFIX/bin/hf" download "$repo" --revision "$rev" --local-dir "$REPO_ROOT/models/$dir"
rm -rf "$REPO_ROOT/models/$dir/.cache"
check "$dir"
