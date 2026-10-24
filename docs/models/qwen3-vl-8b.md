# 👀 Qwen3-VL-8B-Instruct — the one that reads your screenshots in six seconds

> **TL;DR**
>
> | | |
> |---|---|
> | Size on disk | 5.03 GB (`Q4_K_M`) + 0.75 GB vision projector (`Q8_0`) |
> | Where it lives | everything on the GPU, projector included (~7.4 GB used) |
> | Context | 16K |
> | Generation | ~38 tok/s (26 tok/s with the context full) |
> | Prompt processing | ~830–900 tok/s |
> | [pibench](../pibench.md) | not run (an image model without a thinking mode) |
> | [Vision probe](../vision-probe.md) | **89%**, perfect OCR, ~6 s per image |
> | Launch | `bash scripts/start_qwen3vl.sh` |

## What is it?

Qwen3-VL-8B-Instruct is the 8B member of Alibaba Qwen's Qwen3-VL vision-language family. It's
the **Instruct** edition, so it answers straight away instead of thinking first. Under the hood:
a 36-layer transformer with standard attention in every layer (8 KV heads × 128), plus a vision
encoder that ships as a separate `mmproj` file. The card adds Interleaved-MRoPE positions and
DeepStack (features from several levels of the vision transformer), a native 256K context, OCR
in 32 languages, GUI-agent use and code generation from images.

Qwen publishes official GGUFs that run on **stock llama.cpp**: no fork, no patch, no rebuild.

## How does it fit in 8 GB?

The naive setup doesn't. `Q8_0` weights alone are 8.7 GB. With `Q4_K_M` (5.03 GB) and the
projector (0.75 GB), the KV cache becomes the limit: 2.42 GB at 16K in f16, and even at q8_0,
32K (2.57 GB) is too much next to everything else.

Then the image encoder. On the CPU, a 4K photo took **270 s**. On the GPU, an uncapped 4K photo
becomes 4,096 image tokens, and the encoder runs out of memory and **takes the server down with
it**.

What works is plain configuration: Q4_K_M with all 36 layers on the GPU, a 16K q8_0 KV cache,
the projector on the GPU (`MMPROJ_GPU=1`), and images capped at 1,024 tokens
(`IMAGE_MAX_TOKENS=1024`). That's ~7.4 GB, and the same 4K photo now takes **5.4 s**. Both
settings are the defaults for this model. See [How it fits](../how-it-fits.md).

## Install & run

It uses the stock llama.cpp build in `vendor/llama.cpp` from
[Getting started](../getting-started.md). Then:

```bash
# 1. Download the model + projector (5.8 GB), pinned revision, sha256-checked
bash scripts/download-models.sh qwen3vl

# 2. Serve it (vision on) and open pi
bash scripts/start_qwen3vl.sh

# One question about an image, no chat:
bash scripts/start_qwen3vl.sh -p "describe @photo.jpg" --no-session </dev/null

# Server only, OpenAI-compatible API on :8080:
QMODEL=qwen3vl bash scripts/run-qwen.sh --image

# Done? Free the GPU:
bash scripts/stop-server.sh
```

In pi, attach an image with `@file.png`. That works because the repo's pi model entry for
`qwen3-vl-8b` says `"input": ["text", "image"]`; without it, pi treats the model as text-only,
doesn't send the image, and the model politely tells you it "can't view the image".

## Official numbers

From Qwen's [Qwen3-VL card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) (text benchmarks of
the Instruct model, non-thinking), **not re-measured here**:

| Benchmark | Qwen3-VL-8B-Instruct |
|---|---:|
| MMLU | 80.7 |
| MMLU-Pro | 71.6 |
| AIME25 | 45.9 |
| LiveCodeBench v6 (2025-02 to 05) | 39.3 |
| IFEval | 83.7 |
| BFCL-v3 (tool calling) | 66.3 |

The LiveCodeBench window is older and non-thinking, so don't line it up against the thinking
models' scores on the [scoreboard](../scoreboard.md).

## Our numbers

| Test | Prompt | Prompt speed | Generation | Result |
|---|---:|---:|---:|---|
| Text: Fibonacci function | 22 tok | 386 tok/s | 36.5 tok/s | correct |
| 640×400 synthetic image, projector on CPU | 290 tok | 57 tok/s | 38.7 tok/s | all shapes, colours and order; OCR exact |
| 4K photo, projector on CPU, uncapped | 4,096 tok | 16 tok/s (**270 s**) | 31.8 tok/s | correct |
| 4K photo, projector on GPU, uncapped | — | — | — | **out of memory, server aborts** |
| 4K photo, GPU, `IMAGE_MAX_TOKENS=1024` | 1,027 tok | 480 tok/s (**5.4 s** total) | 38.7 tok/s | correct |
| 4K photo, GPU, `IMAGE_MAX_TOKENS=2048` | 1,999 tok | 377 tok/s (8.8 s) | 36.7 tok/s | correct, a bit more detail |
| Tool call | — | — | — | ✓ |
| Needle in a 15,922-token prompt | 15,922 tok | 828 tok/s | 26.3 tok/s | found exactly |

VRAM: 7.39 GB with the projector on the GPU (idle and with the context full), 7.54 GB peak
during a 2,048-token image.

**[Vision probe](../vision-probe.md): 89% overall, at ~6 s per image** (6–12× faster than the
thinking models, which encode on the CPU and reason first):

| OCR | Boxes, any format | Boxes in pixels, as asked | Understanding | CAD |
|---:|---:|---:|---:|---:|
| 100% | 100% | 7% | 61% | 96% |

**[pibench](../pibench.md): not run.** The coding benchmark is for the agentic models; Qwen3-VL
is here for its eyes.

## How it behaves

- **OCR is flawless**, including the blurred, rotated "scanned" invoice.
- **It knows where things are, in its own coordinates.** Asked for pixel boxes, it answered on
  Qwen's 0–1000 grid every time. Converted, every box was right (mean overlap 0.79–0.89). If you
  use it for detection, ask for its native format or convert from it.
- **Counting is where no-thinking shows.** It counted 12 blue dots instead of 11, got the
  "how many cities above 50" chart question wrong, and named the wrong object "directly below
  the green triangle". The thinking models got those right (mostly). It's fast because it
  doesn't stop to check.
- **Engineering drawings read well:** perfect on two plates, and on the third it read the hole
  pitch as 20 instead of 30, where three dimension labels sit close together.

## Gotchas

- **Cap the image tokens** (`IMAGE_MAX_TOKENS`, default 1024). Uncapped, a big photo crashes the
  server. 2048 fits, but leaves less headroom for whatever your desktop is doing.
- **Keep the projector on the GPU** (`MMPROJ_GPU=1`, the default here), unless you enjoy
  waiting 4.5 minutes per photo.
- **No thinking mode.** pi's entry says `reasoning: false`; thinking flags do nothing.
- **`pi -p` can hang** if stdin is left open (for example under a script). Add `</dev/null`.
- **`Q8_0` weights (8.7 GB) don't fit** fully on the card; partial offload wasn't tried.

## Knobs

| Variable | Default | What it does |
|---|---|---|
| `CTX` | 16384 | Context window. 32K doesn't fit. |
| `IMAGE_MAX_TOKENS` | 1024 | Tokens per image; more detail, more VRAM. |
| `MMPROJ_GPU` | 1 | Vision projector on the GPU (1) or CPU (0). |
| `KVQUANT` | `q8_0` | KV-cache type. |
| `TEMP` / `TOP_P` / `TOP_K` / `PRESENCE_PENALTY` | 0.7 / 0.8 / 20 / 1.5 | Sampling, the card's VL preset. |
