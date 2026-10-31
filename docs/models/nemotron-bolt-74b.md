# ⚡ Nemotron 3.5 Bolt 74B-A9B — the biggest thing we've stuffed into this laptop

> **TL;DR**
>
> | | |
> |---|---|
> | File | NVIDIA's own "24GB" GGUF, 22.0 GB (+ 1.7 GB vision projector) |
> | Where it lives | Always-on layers on the GPU (6.6 GB VRAM), every expert in RAM |
> | Context | 128K (the model does 256K) |
> | Speed | **14 tok/s** generating, **310 tok/s** reading a prompt |
> | pibench | **15/18** (93 % of hidden tests) |
> | Vision probe | **91 %** |
> | Launcher | `bash scripts/start_bolt.sh` |

## What is it?

NVIDIA's Nemotron 3.5 Bolt is a 74-billion-parameter model that only uses about 9 billion per
token. It's a hybrid: 40 Mamba-2 layers (a recurrent kind of layer with a small, fixed-size
memory), 40 mixture-of-experts layers (240–432 experts each, 9 picked per token), and only 8
ordinary attention layers. It reads text, images and video, calls tools, and reasons before
answering. NVIDIA compressed it from Nemotron 3.5 Super VL. License:
[OpenMDW 1.1](https://openmdw.ai/license/1-1/).

We run **NVIDIA's own 2-bit file**, not a community quant. NVIDIA trained the model *for* 2 bits
(quantization-aware distillation): every expert, routed and shared, is IQ2_XXS at 2.06 bits per
weight, and the rest stays at higher precision. That's how 74B parameters fit in a 22 GB file.

## How does it fit in 8 GB?

Two lucky breaks. First, context is nearly free: only 8 of its 88 layers keep a KV cache, with 2
KV heads each, so 128K of q8_0 cache is about half a gigabyte, and the Mamba layers' memory is a
fixed ~160 MB. Second, the split between GPU and CPU work is unusually kind. Per token it touches
~6.8B "always-on" parameters (Mamba, attention, shared experts) and only ~1.7B routed-expert
parameters. The always-on part sits on the GPU; the 66B-parameter expert pool sits in RAM, and the
CPU only does a little work with it per token.

That's also why moving experts onto the GPU barely helps (13–15 tok/s whatever we tried): the GPU
side sets the pace. The one knob that matters is the batch size, which doubles prompt speed.
[How it fits](../how-it-fits.md) explains the general trick.

## Install & run

It runs on the same llama.cpp build as Flint (Bolt support landed upstream on 2026-10-14):

```bash
# once: our llama.cpp (fork, adventures branch; see getting-started for the toolchain)
CUDA_VER=12.2 LLAMA_REPO=https://github.com/ricardodeazambuja/llama.cpp \
  LLAMA_REF=3cb2992b39cbcc7a6fd90002a39f694f4bc03a73 SRC_DIR=$PWD/../llama.cpp \
  bash scripts/build-llama-cuda.sh

# once: 22 GB + 1.7 GB, pinned and sha256-checked
bash scripts/download-models.sh bolt

# go
bash scripts/start_bolt.sh                     # server + pi
bash scripts/start_bolt.sh --image             # with vision (projector on the CPU)
bash scripts/start_bolt.sh -p "explain @README.md" --no-session </dev/null
bash scripts/stop-server.sh
```

You need about 22 GB of free RAM for the experts. On our 32 GB laptop it fit next to a desktop
session with room to spare.

## Official numbers

From [NVIDIA's model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF),
for **this exact 2-bit file**, against full-precision rivals (NVIDIA's harness, NVIDIA's numbers):

| Benchmark | Qwen3.6-35B-A3B (BF16) | Gemma 4 26B-A4B (BF16) | **Bolt 74B-A9B 24GB (IQ2)** |
|---|---:|---:|---:|
| Terminal-Bench 2.1 | 44.38 | 37.22 | **50.98** |
| GPQA Diamond | 83.40 | 79.61 | **86.43** |
| HLE (no tools) | 19.56 | 17.42 | **25.30** |
| SciCode | 35.33 | 40.28 | **41.02** |
| MCP-Atlas | 51.20 | — | **52.20** |
| PinchBench | **88.07** | 74.70 | 87.82 |
| AA-Omniscience accuracy | 19.47 | **22.17** | 16.90 |

Bold claims for a 2-bit model. The usual grain of salt applies: same vendor, its own harness.

## Our numbers

| Setup | VRAM | Generation | Prompt (15K tokens) |
|---|---:|---:|---:|
| All experts in RAM, batch 512 | 5.96 GB | 14.0 tok/s | 136 tok/s |
| 1 expert layer on the GPU, batch 2048 | 7.10 GB | 14.9 tok/s | 293 tok/s |
| 2 expert layers on the GPU, batch 512 | 7.07 GB | 13.1 tok/s | 139 tok/s |
| **All experts in RAM, batch 2048 (default)** | **6.62 GB** | **14.0 tok/s** | **310 tok/s** |

Tool calls work; the `merge_intervals` code check passed (0/500 wrong, 2,580 tokens of reasoning).

**[pibench](../pibench.md): 15/18.** Bug fix 2/3, CLI 3/3, feature 3/3, login 3/3, books 1/3,
JS page 3/3. Median 6.4 minutes per attempt (the slowest model), one timeout, never ran out of
context.

**[Vision probe](../vision-probe.md): 91 %.** OCR, understanding (chart, counting, grid) and CAD
drawings all 100 %, including the dot count that tripped Qwen3-VL and the distill. Bounding boxes
are its weak spot: 21 % in pixels as asked, 64 % even in its own format. It takes its time: median
115 s per image, up to 11 minutes on one.

## How it compares

Against the other big models that run 128K here (all our measurements; full tables on the
[scoreboard](../scoreboard.md)):

| | **Bolt 74B** | Flint | Gemma 4 26B | Qwen3.6 | Qwen3.8 distill |
|---|---:|---:|---:|---:|---:|
| Parameters (active) | **74B (9B)** | 40B (2.5B) | 26B (4B) | 35B (3B) | 35B (3B) |
| Generation | 14.0 t/s | 24.3 t/s | 23.5 t/s | 22.8 t/s | **26.2 t/s** |
| Prompt speed | 310 t/s | **542 t/s** | — | 221 t/s | 167 t/s |
| pibench | 15/18 | **18/18** | **18/18** | 16/18 | 16/18 |
| Median time per attempt | 6.4 min | 4.3 min | 4.8 min | 3.4 min | **2.5 min** |
| Vision probe | 91 % | text only | **100 %** | **100 %** | 88 % |
| Terminal-Bench 2.1 (vendor) | 51.0 | **54.9** | 35 | 43–45 | — |

On paper it's second only to Flint; on this laptop it's slower and less reliable than the three
35B-class models. Pick it when you want the most knowledge in one model and can wait for it.

## How it behaves

- **Looks before it leaps**, with little `python3 -c "requests…"` probes rather than curl, then
  writes the real script.
- **Tests edge cases nobody asked for:** empty files, `--top 0`, "O'Brien", names with line breaks
  in a CSV. They all passed.
- **The charset trap got it twice.** The books site doesn't declare its character set; Bolt printed
  a CSV with a garbled "Noahâs" and called it done once, and the second time chased the symptom
  (a broken price parser, "must be a non-breaking space") until the 20-minute budget ran out. On
  the third attempt it set the encoding and solved it.
- **Thinks long, then moves fast** on things it knows: the JavaScript-quotes chore took under three
  minutes, every time.
- **Boxes run tall.** In pixel mode its boxes get the x positions and top edges right and the bottoms
  far too low, about twice the real height.

## Gotchas

- **RAM, not VRAM, is the limit.** The experts need ~22 GB of RAM. Close the browser tabs.
- **It's slow per job.** 14 tok/s plus long reasoning means 5–10 minutes per agent task. Great for
  "go do this while I get coffee", less great for chatting.
- **Vision is slow too:** the 1.7 GB projector runs on the CPU (there's no VRAM left for it).
- **Not the Unsloth file.** Unsloth also publishes a 22 GB 2-bit quant; we use NVIDIA's, which was
  distilled for 2 bits. Same size, presumably smarter.
- NVIDIA's recipe adds a DFlash speculative-decoding drafter; we haven't tried it on this card.

## Knobs

| Variable | Default | What it does |
|---|---|---|
| `CTX` | 131072 | Context window. Cheap here, so there's little reason to lower it. |
| `NCMOE` | (empty: all experts in RAM) | First N layers' experts stay in RAM. Experts on the GPU barely help; leave it. |
| `BATCH` / `UBATCH` | 2048 | Prompt speed. 512 halves it. |
| `KVQUANT` | q8_0 | KV cache type. |
| `--image` | off | Loads the vision projector (on the CPU). |
