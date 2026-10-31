# 💎 Gemma 4 26B-A4B (QAT) — the one that started it all

> **TL;DR**
>
> | | |
> |---|---|
> | File | `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`, ~14 GB (13.26 GiB) |
> | Where it lives | attention + KV cache on the GPU, most MoE experts in system RAM |
> | Context | 32K by default; 64K for free; up to 128K with a quantized KV cache |
> | Generation | **~23.5 tok/s** (CUDA) · ~4.9 tok/s (Vulkan) · ~1.9 tok/s (CPU only) |
> | Prompt processing | ~63 tok/s (llama-bench, CUDA) |
> | [pibench](../pibench.md) | **18/18 solved** (100% of hidden tests), median 4.8 min per run |
> | [Vision probe](../vision-probe.md) | **100%** overall (OCR, boxes, understanding, CAD), ~71 s per image |
> | Launch | `BACKEND=cuda bash scripts/start.sh` |

## What is it?

Google's Gemma 4 26B-A4B is a **mixture-of-experts** model: 25.2B parameters in total, but each
token only wakes up **8 of 128 experts** per layer, so only about **4B parameters do work per
token**. That "A4B" (active 4B) is the whole reason it runs on a laptop.

We use Unsloth's **QAT** GGUF. QAT means the model was fine-tuned with 4-bit quantization
simulated during training, so the 4-bit weights keep much more quality than a plain after-the-fact
quant. The `UD-Q4_K_XL` flavour is Unsloth's "Dynamic" quant: important tensors get more bits.
Per Unsloth, a naive `Q4_0` conversion of this model scores 70.2% top-1 accuracy against 85.6% for
the dynamic one, and the dynamic one is smaller. The repo is public: no Hugging Face token needed.

Under the hood: 30 layers, 128 experts with top-8 routing, a 256K trained context, and
**sliding-window attention** where every 6th layer is a full-attention layer. It can see images too
(with a separate vision projector, see below).

This is the model the whole project started with. Everything else in this repo is what happened
when we asked "ok, what *else* fits?"

## How does it fit in 8 GB?

A 14 GB model on an 8 GB card sounds like a magic trick, and it sort of is. llama.cpp's
`--cpu-moe` flag puts the **expert weights in system RAM** and keeps the small, always-used parts
(attention, embeddings, router, norms) **plus the KV cache** on the GPU. Experts are most of the
parameters but only 8 of 128 fire per token, so the CPU only streams tens of MB per layer per
token. Then `--n-cpu-moe N` (the `NCMOE` knob) moves some experts back to the GPU for speed: each
layer's experts cost ~0.45 GB, and at 32K context about 8 layers fit (`NCMOE=22`). The full story,
with numbers, is in [How it fits](../how-it-fits.md).

## Install & run

Prerequisites (mamba, an NVIDIA driver, `pi`) are in [Getting started](../getting-started.md).

```bash
# 1. Env + model download (~14 GB) + build llama.cpp against YOUR driver's CUDA (~20 min, once)
BACKEND=cuda bash scripts/setup.sh

# 2. Register the local server with pi (once)
bash scripts/configure-pi.sh

# 3. Server + pi, in one go
BACKEND=cuda bash scripts/start.sh
```

Don't want to build anything? `bash scripts/setup.sh` (Vulkan, the default) works on any driver,
it's just ~5× slower. More on that below.

Other ways in:

```bash
bash scripts/start.sh --menu                        # guided setup: backend, context, KV quant, images
bash scripts/start.sh -p "summarize @README.md"     # one-shot; extra args go to pi
CTX=131072 KVQUANT=q8_0 bash scripts/start.sh       # long context with a quantized KV cache
bash scripts/start.sh --image                       # accept images (needs the projector, below)
bash scripts/stop-server.sh                         # stop the server when you're done
```

When `pi` exits, `start.sh` offers to stop the server it started (`STOP_ON_EXIT=1` always stops,
`0` never does). On the first launch it can also **auto-tune** the expert split for your GPU and
remember the result.

**Images.** The main GGUF is text only; vision comes from a separate projector (~1.2 GB, BF16,
images only, no audio). Download it once:

```bash
curl -L -o models/gemma4-26b-a4b-qat/mmproj-BF16.gguf \
  https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/mmproj-BF16.gguf
```

`--image` keeps the projector on the CPU (there's no VRAM left for it), so encoding an image is a
one-off CPU cost, and token generation stays at full speed.

## Official numbers

From [Google's model card](https://huggingface.co/google/gemma-4-26B-A4B-it) (Google's own runs,
full-precision model):

| Benchmark | Gemma 4 26B-A4B |
|---|---:|
| MMLU Pro | 82.6% |
| AIME 2026 (no tools) | 88.3% |
| LiveCodeBench v6 | 77.1% |
| Codeforces ELO | 1718 |
| GPQA Diamond | 82.3% |
| τ²-bench (average over 3) | 68.2% |
| HLE (no tools) | 8.7% |
| MMMU Pro (vision) | 73.8% |

Other vendors put it in their tables too, and they disagree loudly on agentic coding:
**SWE-bench Verified is 17.4 in Qwen's table and 57.4 in NVIDIA's.** Terminal-Bench is 34.2
(2.0, Qwen's table) and 35 (2.1, NVIDIA's). Same model, different harness, wildly different
number. Keep that in mind before you crown or bury anything based on one row.
The full comparison is on the [scoreboard](../scoreboard.md).

## Our numbers

**Speed by backend** (RTX 2070, `CTX=16384`, `NCMOE=22`, same model, same split):

| Backend | Prompt (llama-bench) | Generation |
|---|---:|---:|
| CPU only | 3.1 tok/s | **1.9 tok/s** |
| Vulkan (stock conda build) | 47 tok/s | **4.35 tok/s** (4.9 via the server) |
| **CUDA (built from source)** | 63 tok/s | **25.25 tok/s** (23.5 via the server) |

Yes, CUDA is ~5× Vulkan on this card at the *same* CPU/GPU split. It isn't RAM bandwidth either:
the Vulkan MoE path on Turing is simply slow. Moral: measure your backends.

**Context vs speed** (CUDA):

| Config | Context | Speed |
|---|---:|---:|
| `NCMOE=22` (default) | 32K | ~23 tok/s |
| `NCMOE=20`, `KVQUANT=q8_0` | 64K | 30.5 tok/s |
| `NCMOE=22`, `KVQUANT=q8_0` | 128K | 27.0 tok/s |
| `--cpu-moe` (all experts in RAM), f16 KV | 128K | slower |

A q8_0 KV cache frees ~1 GB at 64K and ~2 GB at 128K, which buys a faster expert split. Below
~64K it isn't worth it. Thanks to sliding-window attention, 64K decodes about as fast as 32K.

**MTP (speculative decoding).** Gemma 4 ships a tiny 0.25 GB "draft" head that guesses the next
few tokens. It's lossless, and at **greedy** it's a real win: 79–88% of drafts accepted and
**+19–31%** faster (≈21 → 27 tok/s). At the default temperature 1.0, acceptance drops to 66–74%
and the gain disappears into this laptop's ±13% noise. A draft-confidence floor (`p_min > 0`)
turned the output into `fmt-fmt…` loops, so don't. Want faster coding? Lower the temperature and
try `MTP=1`. (The EAGLE3 draft was slower and crashier; it's in the [graveyard](../graveyard.md).)
On our fork, `--spec-draft-sampling probabilistic` gave 27 tok/s at temperature 1.0 against 21-22
without a draft, in one run, with no change in scores
([details](../up-next.md)).

**[pibench](../pibench.md)** (64K, `NCMOE=20`, q8_0 KV): **18/18 solved**, 100% of hidden tests,
median **4.8 min** and ~7.1K output tokens per run. Tied for first with Flint, which is a fine
plot twist given those 17.4 SWE-bench vibes. It was the slowest per run of the first six (Nemotron Bolt 74B later took that title at 6.4 min).

**[Vision probe](../vision-probe.md)**: **100% overall.** Perfect OCR (including the blurred,
rotated "scan"), perfect chart/counting/spatial answers, perfect CAD readings. Boxes were spot on
too, but in its own format (more below). About 71 s per image; the bounding-box images made it
think for 4–7K tokens each.

## How it behaves

- **Thinks first, acts later.** Long reasoning before acting (~12K thinking characters per
  pibench run), then the **fewest tool calls** of all models (~12 per run).
- **Chases weird data.** In the book-scraping task it noticed a garbled "Noah’s" in its own CSV,
  grepped the live page for it, checked the response headers with `curl -I`, found the missing
  charset, fixed the decoding and re-checked. That exact trap caught two other models.
- **Scouts before scraping.** On the JavaScript page it wrote a throw-away `check_structure.py` to
  inspect the rendered page, used it, then deleted it.
- **Clumsy hands, good recovery.** Of the first six models, it's the only one that sent malformed tool calls: 8 in 18
  runs, all `edit`/`write` calls missing the `path` argument. Every time, it shrugged and rewrote
  the whole file with `write`. It also likes making new files (`test_durations_new.py`) instead of
  editing the one it was given.
- **Boxes in its own dialect.** Asked for pixel coordinates, it answered with its native `box_2d`
  key: `[y_min, x_min, y_max, x_max]` on a 0–1000 grid. Converted, every box was right. If you
  use it for detection, ask for (or convert from) that format.

## Gotchas

- **It only sees images it actually reads.** The projector being loaded isn't enough: the model
  has to call pi's `read` tool on the image file. If you ask it to judge a picture it just
  generated, tell it to open the PNG, or it may confidently describe an image it never looked at.
- **Small `max_tokens` = empty answer.** The thinking goes to `reasoning_content` first; a tiny
  budget runs out before the visible answer. Not a bug, just a chatty brain.
- **Why pi feels slower than the tok/s.** A real pi turn has a big prompt (one example: 5,639
  tokens, ~14 s just to read it), hidden reasoning, and several tool-call round trips. The decode
  rate is real; it's just not the whole story. Lower `--thinking` if you want snappier turns.
- **Old drivers vs prebuilt CUDA.** The conda CUDA builds crash with `device kernel image is
  invalid` on drivers older than their CUDA version. Build from source (`BACKEND=cuda`) or use
  Vulkan. See [troubleshooting](../troubleshooting.md).
- **256K doesn't fit.** The model was trained for 256K, but on 8 GB the practical ceiling is
  ~128K (≈160K hard limit, all experts in RAM).
- Harmless load warnings about control-token types (`<|tool_response>`, `</s>`) and
  `failed to fit params ... n_gpu_layers already set by user` can be ignored.

## Knobs

All are env vars for `scripts/start.sh` (forwarded to `scripts/run-server.sh`; `-h` lists them all).

| Knob | What it does | Default |
|---|---|---|
| `BACKEND` | `cuda` \| `vulkan` \| `cpu` | auto (CUDA if built) |
| `CTX` | context window in tokens | 32768 |
| `NCMOE` | expert layers kept in RAM; lower = faster, more VRAM | all (or the auto-tuned value) |
| `KVQUANT` | KV cache type: `q8_0` for long context | f16 (off) |
| `TEMP` / `TOP_P` / `TOP_K` | sampling (Unsloth's recommendation) | 1.0 / 0.95 / 64 |
| `MTP` / `NMAX` | speculative decoding with the MTP head; draft length | off / 2 |
| `AUTOTUNE` | measure the fastest `NCMOE` for your GPU (`1` force, `0` never) | ask once |
| `STOP_ON_EXIT` | stop the server when pi exits (`1`/`0`) | ask |
| `--image` | load the vision projector | off |
