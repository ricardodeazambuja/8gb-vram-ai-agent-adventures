# 🪨 Bonsai 2 27B — a 27B model on a 1.7-bit diet

> **TL;DR**
>
> | | |
> |---|---|
> | Size on disk | 5.95 GB (`PTQ1_0`) + 0.63 GB vision projector |
> | Where it lives | everything on the GPU (7.45 GB used) |
> | Context | **16K** (and that is the catch) |
> | Generation | ~27 tok/s |
> | Prompt processing | 269 tok/s (15.6K-token prompt) |
> | [pibench](../pibench.md) | 14/18 solved, 13 of 18 runs ran out of context |
> | [Vision probe](../vision-probe.md) | **98%**, the best at boxes in pixels (as asked) |
> | Launch | `bash scripts/start_bonsai2.sh` |

## What is it?

Ternary-Bonsai-2-27B is PrismML's re-encoding of **Qwen3.8-27B**, Alibaba Qwen's dense 27B model
(Apache-2.0, published 2026-09-16). The architecture is untouched: 64 layers in Qwen's hybrid
layout, where three of every four layers are Gated DeltaNet (linear attention with a fixed-size
state) and only 16 use full attention with a KV cache. A vision tower comes as a separate,
optional `mmproj` file.

What's new is the diet. Every weight is −1, 0 or +1, with one FP16 scale per 128 weights,
stored in a Hadamard-rotated basis: about **1.72 bits per weight**. The same model is ~54 GB at
FP16 and 17.6 GB as a 4-bit `UD-Q4_K_XL`. Bonsai squeezes it into 5.95 GB, and by PrismML's own
numbers keeps 98.2% of the FP16 score doing it.

## How does it fit in 8 GB?

It's dense, so there's no MoE trick of parking experts in RAM: weights, KV cache and compute
buffers all have to fit on the card. The ternary weights make that possible. The price is
context: the KV cache has to be **f16** (see Gotchas), which is ~1.14 GB at 16K and ~2.3 GB at
32K, and 32K doesn't fit. So you get 16K, with 7.45 GB of VRAM in use.

Before Bonsai, the only way to run Qwen3.8-27B here was a 10.4 GB `IQ3_XXS` with half its
layers in system RAM: **2.5 tok/s**. Bonsai does **27 tok/s**, about 11× faster. More on the
general tricks in [How it fits](../how-it-fits.md).

## Install & run

It needs **PrismML's llama.cpp fork**: stock llama.cpp doesn't know the `PTQ1_0` type. Build it
once (about 15 minutes) with the repo's CUDA toolchain, into its own folder so it never
touches your other builds:

```bash
# 1. Build the PrismML fork into vendor/prism-llama.cpp
CUDA_VER=12.2 LLAMA_REPO=https://github.com/PrismML-Eng/llama.cpp \
  LLAMA_REF=prism-b10777-e3de715 SRC_DIR=$PWD/vendor/prism-llama.cpp \
  bash scripts/build-llama-cuda.sh

# 2. Download the model + vision projector (~6.6 GB), pinned revision, sha256-checked
bash scripts/download-models.sh bonsai2

# 3. Serve it and open pi (thinking level "medium" by default)
bash scripts/start_bonsai2.sh

# One question, no chat:
bash scripts/start_bonsai2.sh -p "explain @README.md" --no-session </dev/null

# Done? Free the GPU:
bash scripts/stop-server.sh
```

The download script needs the `hf` CLI inside the conda env it uses (`ENV_NAME`, default
`llamacpp`); see [Getting started](../getting-started.md). Free the GPU before building:
the build's smoke test fails with a misleading "failed to load model" if another server is
holding the VRAM.

## Official numbers

PrismML's own evaluation (EvalScope + vLLM on an H100, thinking mode), **not re-measured here**:

| Benchmark | Qwen3.8-27B FP16 | UD-Q4_K_XL | IQ2_XXS | Bonsai 2 (5.95 GB) |
|---|---:|---:|---:|---:|
| LiveCodeBench | 90.05 | 87.96 | 56.40 | 90.07 |
| AIME26 | 94.58 | 93.00 | 57.50 | 95.83 |
| MMLU-Redux | 91.46 | 93.35 | 88.93 | 89.09 |
| BFCL v3 (tool calling) | 76.74 | — | — | 74.92 |
| Average of 14 benchmarks | 86.32 | 85.18 | 72.59 | 84.78 (98.2% of FP16) |

For agentic coding you have to look at the **base model**, Qwen3.8-27B: SWE-bench Verified 75.6
(measured by Recursive), Terminal-Bench 2.1 73.0 (Qwen's card). That makes it the strongest
model in this repo on paper. Source: the
[PrismML card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) and the
[Qwen3.8-27B card](https://huggingface.co/Qwen/Qwen3.8-27B); comparisons across vendors are in
the [scoreboard](../scoreboard.md).

## Our numbers

| Test | Result |
|---|---|
| `llama-bench` | pp512 322.3 tok/s, tg64 28.47 tok/s |
| Server, f16 KV, 16K | **27.0 tok/s**, 15.6K-token prompt at 269 tok/s, 7.45 GB VRAM |
| Server, q8_0 KV | aborts on the first request (Turing, see Gotchas) |
| Tool call | ✓ |
| Code check (`merge_intervals`, 500 random tests) | correct, 0/500 wrong, 1,597 tokens, 60 s |

**[pibench](../pibench.md): 14/18 solved, 90% of hidden tests passed, median 4.9 min per run.**
All code tasks 3/3, login 3/3, the JavaScript page 2/3, and the big-page book scraper 0/3.

**[Vision probe](../vision-probe.md): 98% overall.** OCR 100%, boxes 100% (and 67% in pixels
exactly as asked, the best of any model), understanding 100%, CAD 93%. On one drawing it read
the hole pitch and edge offset as 25 instead of 30 and 20. 38 s per image (the projector runs
on the CPU).

## How it behaves

- **Smart, and constantly out of room.** In pibench, 12 of its 18 runs ended because the
  conversation filled the 16K context, and one more was rejected outright when a page dump
  pushed a request to 22,482 tokens. Only 5 runs finished on their own.
- **It often finishes the job just before the lights go out.** Nine of those cut-off runs
  still passed, because it had already written the files. One run thought for 22,000
  characters in a single block, wrote the class and its tests, ran pytest, saw a failure, and
  was cut off before it could react. On a longer job, that's an unfinished job.
- **Big web pages are its kryptonite.** All three book-scraping runs failed: one context
  overflow, one garbled "Noah’s" plus a cut-off before it could notice, and one crashed
  script with an empty CSV.
- **It rarely runs tests** (0.3 pytest calls per run, the fewest), probably because there's no
  room left to read the output.
- **When it fits, the reasoning is good.** Its exploration of the sites was sound, and it was
  the model that best followed the "pixels please" instruction on the bounding boxes.

## Gotchas

- **Use f16 KV, not q8_0.** The fork's flash-attention kernel for a q8_0 cache has no code for
  Turing (sm_75), and the server aborts on the first request: `CUDA kernel flash_attn_ext_f16
  has no device code compatible with CUDA arch 750`. f16 is the default for this model.
- **pi's thinking levels need translating.** The template only accepts `reasoning_effort`
  `xhigh`, `medium` or `low` (and the card says `low` behaves like `xhigh`). pi's `high` gets
  an HTTP 500. The pi model entry in `config/pi-provider.json` maps minimal/low/medium →
  `medium` and high/xhigh → `xhigh`.
- **Stick to `medium` thinking** (the launcher's default). At `xhigh` it thinks for minutes per
  turn at 27 tok/s.
- **16K fills fast.** One fetched JSON page once came back as ~146K tokens. A session that
  overflowed stays broken (every request resends the oversized turn): start a new one with
  `/new`.
- **Hangs that aren't hangs.** If a client gives up on a request, llama-server keeps generating
  it, and the next requests queue behind it on the single slot. Check
  `curl -s http://127.0.0.1:8080/slots` (`is_processing`) before blaming the client.

## Knobs

All optional; set them in front of the launcher (`CTX=… bash scripts/start_bonsai2.sh`).

| Variable | Default | What it does |
|---|---|---|
| `CTX` | 16384 | Context window. 32K doesn't fit with f16 KV. |
| `KVQUANT` | `f16` | KV-cache type. Don't use `q8_0` on a Turing card. |
| `NGL` | 99 | Layers on the GPU (all of them). |
| `TEMP` / `TOP_P` / `TOP_K` / `MIN_P` | 1.0 / 0.95 / 20 / 0.05 | Sampling, from the card. |
| `--thinking` (pi flag) | `medium` | pi's thinking level; `high` maps to the slow `xhigh`. |
