#!/usr/bin/env bash
# fetch.sh: re-download the two outside templates at pinned revisions, re-extract Bonsai 2's from our GGUF,
# and check all three against the sha256s below. Only the first 32 MB of Unsloth's smallest GGUF is fetched
# (the header holds the template); nothing is run.
set -euo pipefail
B="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ADV="$(cd "$B/../.." && pwd)"; T="$B/templates"; TMP=$(mktemp -d)
curl -fsSL -o "$T/qwen-official.jinja" \
  "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/37561816e66b5b9153b1b64c10968c18c1e6f4a0/chat_template.jinja"
curl -fsSL -r 0-33554431 -o "$TMP/head.gguf" \
  "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/ce331efc48d5722b4b8041f72793e530eae18d9b/Qwen3.8-27B-UD-IQ1_S.gguf"
python3 -I "$B/gguf_kv.py" "$TMP/head.gguf" "$T/unsloth-2026-11-11.jinja" > /dev/null
python3 -I "$B/gguf_kv.py" "$ADV/models/bonsai2-27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf" "$T/bonsai2.jinja" > /dev/null
rm -rf "$TMP"
cd "$T" && sha256sum -c <<'SUMS'
af16f654968d47d09a6c604e00370b12dbc21f76456f59c9e98b82e84d9f6375  qwen-official.jinja
12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce  unsloth-2026-11-11.jinja
c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041  bonsai2.jinja
SUMS
