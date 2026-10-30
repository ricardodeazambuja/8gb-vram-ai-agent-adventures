#!/usr/bin/env bash
#
# download-models.sh — fetch + sha256-verify the GGUFs for every model run by scripts/run-qwen.sh
# (the Gemma 4 models have their own downloads). Each entry is pinned; vision models get their projector:
#
#   qwen36   Qwen3.6-35B-A3B          unsloth UD-IQ4_XS        ~17.7 GB  MoE, 3B active
#   qwen38   Qwen3.8-27B (dense)      ISTA-DASLab GSQ-RCO      ~10.4 GB  IQ3_XXS + MTP head
#   distill  Qwen3.8-35B-A3B-Distill  empero-ai IQ4_XS         ~19.6 GB  3.6 base, 3.8 traces
#
# Optional (not in the default set):
#   dflash   DFlash2 block-diffusion drafter for the 27B (z-lab, Q4_K_M, ~1.1 GB)
#   bonsai2  Ternary-Bonsai-2-27B PTQ1_0 (PrismML, ~5.95 GB + 0.63 GB vision) — Qwen3.8-27B
#            in ternary weights; needs the PrismML llama.cpp fork (docs/models/bonsai2-27b.md)
#   qwen3vl  Qwen3-VL-8B-Instruct Q4_K_M (Qwen, ~5.0 GB + 0.75 GB vision Q8_0) — dense 8B
#            vision-language model, fits fully on the 8 GB card (docs/models/qwen3-vl-8b.md)
#   dory     NVIDIA-Nemotron-Labs-Dory1.0-4B Q5_K_M (NVIDIA, ~3.67 GB) — looped 4B text model;
#            needs the Dory llama.cpp PR build (docs/models/dory1.0-4b.md)
#   spark    Spark-X2.6-4B Q8_0 (XHToken, ~4.4 GB) — dense 4B agentic text model; upstream
#            llama.cpp since 2026-09 (the Dory build has it; docs/models/spark-x2.6-4b.md)
#   flint    Recursive Flint UD-Q3_K_XL (Unsloth, ~18.5 GB) — 40B MoE, 2.5B active; upstream
#            llama.cpp since PR #30928 (2026-10-21; docs/models/flint.md)
#   bolt     NVIDIA Nemotron 3.5 Bolt 74B-A9B "24GB" (NVIDIA's own GGUF, ~22.0 GB + 1.7 GB vision):
#            experts in IQ2 trained for 2 bits (quantization-aware distillation); hybrid
#            Mamba/MoE, 9B active; upstream llama.cpp since PR #30520 (2026-10-14)
#   e4b      NVIDIA Nemotron 3.5 Bolt E4B-4GB Q4_K_M (NVIDIA, ~6.2 GB) — Gemma-4-E4B-based 8B
#            (4B effective), text only, 64K; all on the GPU
#   groot    Groot-19B-A3B-261023 Q4_K_M (GrootTeam, ~11.7 GB + 0.9 GB vision) — Qwen3.6-35B-A3B
#            pruned to 19B and post-trained for agents (docs/models/groot-19b-a3b.md)
#   ornith   Ornith-1.5-35B-A3B Q4_K_M (ornith-ai, ~21.7 GB + 0.9 GB vision) — Qwen3.6-class 35B-A3B,
#            self-improvement post-training (docs/models/ornith-1.5-35b-a3b.md)
#   lfm25    LFM2.5-24B-A2B QAD Q4_0 (LiquidAI, ~13.6 GB) — conv/attention hybrid MoE, 2B active;
#            Q4_0 trained with quantization-aware distillation (docs/models/lfm2.5-24b-a2b.md)
#   tcgemma  ThinkingCap gemma-4-26B-A4B-it IQ4_XS (bottlecap.ai, ~13.6 GB + 1.2 GB vision) —
#            Gemma 4 26B post-trained to think less (docs/up-next.md)
#   lfm3     LFM3-1.7B Q8_0 (LiquidAI, ~1.9 GB) — tri-hybrid 1.7B for agents; needs llama.cpp >= 2026-10-27
#   lfm3vl   LFM3-VL-2B Q8_0 (LiquidAI, ~1.9 GB + 0.4 GB vision)
#   lfm3audio LFM3-Audio-2B Q8_0 (LiquidAI, ~1.9 GB + 0.6 GB audio): speech in and out
#   d1       d1-2B Q8_0 (LiquidAI, ~1.9 GB) — decision model
#   eg2      EmbeddingGemma 2 Q8_0 (ggml-org, ~0.3 GB) — embeddings
#   clefflash Clef-Flash Q4_K_M (ggml-org, ~6.7 GB) — Cloudflare decision model (/v1/systemone)
#   locate   LocateAnything-3B Q8_0 (yuuko-eth port of nvidia's, ~3.6 GB + 0.9 GB vision) — grounding
#            specialist; non-commercial research license (docs/up-next.md)
#
# Reproducible by construction: every file is downloaded at a PINNED Hugging Face repo
# revision and checked against a PINNED sha256 (the exact bytes tested in
# docs/models/qwen3.6-35b-a3b.md). A GGUF can have the right size and a valid header and still be
# corrupt (see docs/how-it-fits.md), so a mismatch deletes the file and exits non-zero.
# Re-running skips files that already verify. FOLLOW_MAIN=1 downloads the repos' current
# main instead and checks against the API's current LFS sha256 (newer, untested bytes).
#
# Needs the `hf` CLI (package huggingface_hub) INSIDE the conda env ENV_NAME — the
# script refuses an `hf` that merely happens to be on PATH, so results don't depend
# on what else is installed on the machine. Fix with:
#   mamba install -n llamacpp -c conda-forge huggingface_hub
#
# Usage:
#   bash scripts/download-models.sh              # all three
#   bash scripts/download-models.sh qwen38       # just one (qwen36 | qwen38 | distill | dflash | bonsai2 | qwen3vl | dory | spark | flint | bolt | e4b | groot | ornith | lfm25 | tcgemma | lfm3 | lfm3vl | lfm3audio | d1 | eg2 | clefflash | locate)
#
# Overrides: ENV_NAME (env with huggingface_hub, default llamacpp), FOLLOW_MAIN=1.
#
set -euo pipefail

# -h / --help: print this script's header comment block and exit.
for _arg in "$@"; do case "$_arg" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \?//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
esac; done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-llamacpp}"
FOLLOW_MAIN="${FOLLOW_MAIN:-0}"

# name | HF repo | pinned revision | local dir (under models/) | file:sha256 ...
MODELS=(
  "qwen36|unsloth/Qwen3.6-35B-A3B-GGUF|a483e9e6cbd595906af30beda3187c2663a1118c|qwen36-35b-a3b|Qwen3.6-35B-A3B-UD-IQ4_XS.gguf:649d7508507b84638732c4f52c24c8b15843c6dca2f3ff793ae07c14a67ebbb3 mmproj-F16.gguf:8971ee4f331ff0a4c609374f32984b3d4e6dc086c0aa35f1d637fad1829e887f"
  "qwen38|ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF|d562806dbafae37109975e970aae91b43e73b440|qwen38-27b-gsq|Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf:63f29a2189a6b4cc31f81e093d3856ad293a5583114439f41ae1ed7af4093262 mmproj-Qwen3.8-27B-BF16.gguf:13cb7bebccbd04afc8f4090cb949ecf8937cdf7377c5799b1a0c594e7c0d3e16"
  "distill|empero-ai/Qwen3.8-35B-A3B-Distill-GGUF|b1f9d1dcc3de8aa867669b0ab919384aeeb9b8d5|qwen38-35b-a3b-distill|Qwen3.8-35B-A3B-IQ4_XS.gguf:b645af45431ef9b41f43cae51c9323b2d0ca84f23031d483d2105c643ae58d65 mmproj-Qwen3.8-35B-A3B-F16.gguf:4381cb5110074396c2c7b39221fffae0c31886aa95b674de8e103d97edf58b94"
  "bonsai2|prism-ml/Ternary-Bonsai-2-27B-gguf|b072e1d3b35a0a630cece372c2127528e0994386|bonsai2-27b|Ternary-Bonsai-2-27B-PTQ1_0.gguf:53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3 Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf:6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903"
  "qwen3vl|Qwen/Qwen3-VL-8B-Instruct-GGUF|f982a07559d4a2f6c8744d840bf6fccab30eea96|qwen3vl-8b|Qwen3VL-8B-Instruct-Q4_K_M.gguf:67d1659bfe71b89d50b45a4ad1a9e5b997e5bb16ce5da66a6a6167abd569e9e2 mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf:c6ba85508d82f42590e6eb77d5340369ab6fecf107a7561d809523d8aa5f3bfd"
  "dory|nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF|0963c8000791e77b2b1b49aa387d92a6da9988d6|dory1-4b|NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf:96080be49ffd67b916018b7bdef8c484762576fb52780e10ab98ac4978a660e0"
  "spark|XHToken/Spark-X2.6-4B-GGUF|fe31f8d64890302cc3a1d365d700357b4b75ba77|spark-x2.6-4b|Spark-X2.6-4B-Q8_0.gguf:145c2aa606820663f26ad530b8fc141ed885e2e9ca29f9be7c3bfe8931d58b81"
  "flint|unsloth/flint-GGUF|20468edfc38ab43d6914cc88b08cbecb05d0fc62|flint-40b-a2.5b|flint-UD-Q3_K_XL.gguf:91b8d85ab6ff7fb095d9a972f9afa6f3fa2b9acb642867a079f3e013ea3c350c"
  "bolt|nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF|e0fecb69cb3e5e4c5e8a056fa2aec98c78e91246|nemotron-bolt-74b-a9b|NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf:91d63363e782c8a3d5f4be42816eba8e0d42dd2530bd07a7d0d2dc440599af75 mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf:0ee891eadc7bf0cae6289b249c74eb1ef420534351f6ef8b64d49f0ca6a926b0"
  "e4b|nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB|1db54d5b12533ff1bcaa8e42cf4c667988a1f1be|nemotron-bolt-e4b|Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf:c8af2b8aea10a690a1d9491db744a9446660173cb38fc489612b556615a69612"
  "groot|GrootTeam/Groot-19B-A3B-261023-GGUF|2fa1ec958e3bbf7312cf0af97802ff26bfdc5260|groot-19b-a3b|Groot-19B-A3B-261023-Q4_K_M.gguf:0716cae1f19a39a786a463f7b8c068d50b3d7a74532fce73e5ea23ed98130796 mmproj-Groot-19B-A3B-261023-BF16.gguf:05a5c8a279b2589306824864d76d663063ba69ff1f399e6a283fae988fdff360"
  "ornith|ornith-ai/Ornith-1.5-35B-A3B-GGUF|12393612fd4f730ff5aadc23e9b8f9648aa49ceb|ornith-1.5-35b-a3b|Ornith-1.5-35B-Q4_K_M.gguf:42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f mmproj-Ornith-1.5-35B-BF16.gguf:1921a36a85aee56cd2abd27f46701802c9d85a33474792e600df6c3b282a135d"
  "lfm25|LiquidAI/LFM2.5-24B-A2B-GGUF|0912d33141f77e10705345a6732bfd83f0252fdc|lfm2.5-24b-a2b|LFM2.5-24B-A2B-QAD-Q4_0.gguf:d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5"
  "tcgemma|bottlecapai/ThinkingCap-gemma-4-26B-A4B-it-GGUF|bb52b322579d218a9931261d6c9cd7403d80eb96|thinkingcap-gemma4-26b-a4b|ThinkingCap-gemma-4-26B-A4B-it-IQ4_XS.gguf:c84bb3c8b997cc7c8fbd5922f3abb362f92ebf5d290654a90bcef5a10c021085 mmproj-ThinkingCap-gemma-4-26B-A4B-it-f16.gguf:4030b92de0d559d7a0cde19659761adc3d8454f06503adb8dac425acfd67c239"
  "eg2|ggml-org/embeddinggemma-2-GGUF|58faf90940886df2d62be555c8765d00dac362c5|embeddinggemma-2|embeddinggemma-2-Q8_0.gguf:2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135"
  "clefflash|ggml-org/Clef-Flash-GGUF|6e2a164d70aae2af7fce86005da3f91805565b1d|clef-flash|Clef-Flash-Q4_K_M.gguf:eef2968c238a80d49d340430f22410ea3db9934ad1c6fdb233a503cbf9a9b54e"
  "lfm3|LiquidAI/LFM3-1.7B-GGUF|b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf|lfm3-1.7b|LFM3-1.7B-Q8_0.gguf:8f6b24182258205e9e01e21f1f24a1f208c1e8115578825d202ce39976af20b6"
  "lfm3vl|LiquidAI/LFM3-VL-2B-GGUF|7c3b4a43c9023e80bf234fec5e826a3365c8dda8|lfm3-vl-2b|LFM3-VL-2B-Q8_0.gguf:e138e54189a751d90ea8c0b236360490613ecd465d28d68fa6094d3db7fa6727 mmproj-LFM3-VL-2B-F16.gguf:1d8e6e58eefc093e09d6332f081516e1ddbd5b35cfca7f35463fb270db230b1e"
  "lfm3audio|LiquidAI/LFM3-Audio-2B-GGUF|fcfae9027c00a6f3f872c48a841cfeb77e2b5a1c|lfm3-audio-2b|LFM3-Audio-2B-Q8_0.gguf:d4e36902ba8f0ee7b333596030a59d650f88ac76660b4d9ba5ff77c3c2258fc0 mmproj-LFM3-Audio-2B-F16.gguf:bd0fce0670c96e405a753f996701dee6568bc6a1318a80091d74fb58f7510d79"
  "locate|yuuko-eth/LocateAnything-3B-GGUF|c3260717380c24cd2fc11f66c43e2feeec3e32fe|locateanything-3b|LocateAnything-3B-Q8_0.gguf:3d039dea2f121006d5dffe2a0371b4979c67dc17ac9d50a557834a064c9e107f mmproj-LocateAnything-3B-BF16.gguf:0216a6647317bc7d5bd152b66bd8b9248472ad090893f75d2f4a689d1789ee80"
  "d1|LiquidAI/d1-2B-GGUF|63be75e6586882defeb706ed4bda812d90256a6d|d1-2b|d1-2B-Q8_0.gguf:073313d92b53b0ac8193db6e906cb3c78a98183d244b15b7b8f5426115c6738c"
  "dflash|z-lab/Qwen3.8-27B-DFlash2-GGUF|2d9571f8ce46e151f61c6499c99dee6079e1d610|qwen38-27b-gsq|Qwen3.8-27B-DFlash2-Q4_K_M.gguf:1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd"
)

WANT=("$@")
[ ${#WANT[@]} -eq 0 ] && WANT=(qwen36 qwen38 distill)

command -v mamba >/dev/null 2>&1 || { echo "ERROR: 'mamba' not found on PATH (install Miniforge)."; exit 1; }

# The hf CLI must come from ENV_NAME itself, not from some other install on PATH.
ENV_PREFIX="$(mamba run -n "$ENV_NAME" printenv CONDA_PREFIX 2>/dev/null || true)"
[ -n "$ENV_PREFIX" ] || { echo "ERROR: conda env '$ENV_NAME' not found. Create it: mamba create -y -n $ENV_NAME -c conda-forge huggingface_hub"; exit 1; }
HF_BIN="$ENV_PREFIX/bin/hf"
[ -x "$HF_BIN" ] || { echo "ERROR: no 'hf' CLI in env '$ENV_NAME' ($HF_BIN). Install: mamba install -y -n $ENV_NAME -c conda-forge huggingface_hub"; exit 1; }

# Current-main sha256 of one file (FOLLOW_MAIN=1 only), from the HF API (LFS oid).
main_sha() {
  local repo="$1" file="$2"
  curl -fsSL "https://huggingface.co/api/models/$repo/paths-info/main" \
       --data-urlencode "paths=$file" \
    | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d[0]["lfs"]["oid"] if d and d[0].get("lfs") else "")'
}

FAIL=0
for want in "${WANT[@]}"; do
  found=0
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r name repo rev dir files <<<"$entry"
    [ "$name" = "$want" ] || continue
    found=1
    dest="$REPO_ROOT/models/$dir"
    mkdir -p "$dest"
    [ "$FOLLOW_MAIN" = 1 ] && rev=main
    for spec in $files; do
      f="${spec%%:*}"
      if [ "$FOLLOW_MAIN" = 1 ]; then
        want_sha="$(main_sha "$repo" "$f")"
        [ -n "$want_sha" ] || { echo "ERROR: no sha256 from the HF API for $repo/$f"; FAIL=1; continue; }
      else
        want_sha="${spec#*:}"
      fi
      if [ -f "$dest/$f" ]; then
        echo ">> $name: $f exists, verifying ..."
        if [ "$(sha256sum "$dest/$f" | cut -d' ' -f1)" = "$want_sha" ]; then
          echo "   OK (already verified)"; continue
        fi
        echo "   sha256 mismatch on existing file, re-downloading"
        rm -f "$dest/$f"
      fi
      echo ">> $name: downloading $repo@${rev:0:7}/$f -> models/$dir/"
      "$HF_BIN" download "$repo" "$f" --revision "$rev" --local-dir "$dest"
      got_sha="$(sha256sum "$dest/$f" | cut -d' ' -f1)"
      if [ "$got_sha" = "$want_sha" ]; then
        echo "   sha256 OK"
      else
        echo "ERROR: sha256 mismatch for $f"
        echo "   expected $want_sha"
        echo "   got      $got_sha"
        rm -f "$dest/$f"
        FAIL=1
      fi
    done
  done
  [ "$found" = 1 ] || { echo "ERROR: unknown model '$want' (use qwen36 | qwen38 | distill | dflash | bonsai2 | qwen3vl | dory | spark | flint | bolt | e4b | groot | ornith | lfm25 | tcgemma | lfm3 | lfm3vl | lfm3audio | d1 | eg2 | clefflash | locate)"; FAIL=1; }
done

exit "$FAIL"
