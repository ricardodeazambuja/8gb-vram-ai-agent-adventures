# 🌱 Groot-19B-A3B — Qwen3.6 with half the experts, and faster for it

> **TL;DR**
>
> | | |
> |---|---|
> | On disk | 11.7 GB (`Q4_K_M`) + 0.9 GB vision projector |
> | Where it lives | experts of 25 of 40 layers in RAM, everything else on the GPU (7.7 GB VRAM) |
> | Context | **128K** |
> | Generation | **39.5 tok/s** (42 tok/s at batch 512) |
> | Prompt processing | **887 tok/s** (15K-token prompt) |
> | Code check | correct twice, 0/500 wrong, 2.3K–3.0K tokens |
> | [pibench](../pibench.md) | **18/18 solved** (100% of hidden tests), 3.1 min median per run |
> | [Vision probe](../vision-probe.md) | 74% overall: OCR and CAD fine, boxes poor |
> | [Security audit](../security.md) | clean |
> | Launch | `QMODEL=groot bash scripts/start-qwen.sh` |

## What is it?

A release by **GrootTeam** (2026-10-23, Apache 2.0). They took
[Qwen3.6-35B-A3B](qwen3.6-35b-a3b.md), cut each MoE layer from 256 experts to 128 (8 still
active per token, about 3B active), and post-trained it on "fewer than 0.1B tokens of open-source
data" for agents and tool use. The rest of the architecture is untouched: 40 layers, 30 Gated
DeltaNet + 10 full attention, the same vocabulary, the same vision tower. The GGUF also carries
the MTP (multi-token prediction) block; llama.cpp ignores it for now.

The card's pitch is "smaller and better": it claims to beat Qwen3.6-35B-A3B and Gemma-4-26B-A4B
on 12 of 16 benchmarks with 23% fewer output tokens than Qwen3.6.

## How does it fit in 8 GB?

Like Qwen3.6, the experts are the bulk of the file. With half as many, each layer's experts weigh
about half as much, so twice as many layers fit on the GPU (15 against Qwen3.6's 8), and less work
happens on the CPU per token. That's where the extra speed comes from. Like
[Flint](flint.md), it uses batch 2048 so the expert maths for long prompts goes to the GPU; the
bigger compute buffer costs a few layers of experts. KV cache: the same as Qwen3.6, 128K at q8_0
is about 1.4 GB. More in [how-it-fits.md](../how-it-fits.md).

## Install & run

After [getting-started.md](../getting-started.md):

```bash
# 1. The newest upstream llama.cpp build (skip if you built it for Flint)
CUDA_VER=12.2 LLAMA_REF=a37cc46104eb50900d5ab4a9ba02a0abdac039db \
  SRC_DIR=$PWD/vendor/flint-llama.cpp bash scripts/build-llama-cuda.sh

# 2. Download + sha256-check (~12.6 GB with the projector), then audit it
bash scripts/download-models.sh groot
bash utils/audit-gguf.sh models/groot-19b-a3b/Groot-19B-A3B-261023-Q4_K_M.gguf

# 3. Server + pi
QMODEL=groot bash scripts/start-qwen.sh

# ...with images
QMODEL=groot bash scripts/start-qwen.sh --image

# Stop it
bash scripts/stop-server.sh
```

Server only, for any OpenAI-compatible client: `QMODEL=groot bash scripts/run-qwen.sh`.

## Official numbers

From the [card](https://huggingface.co/GrootTeam/Groot-19B-A3B-261023) (thinking mode, BF16,
256K context, temp 1.0):

| Benchmark | **Groot-19B-A3B** | Qwen3.6-35B-A3B | Gemma-4-26B-A4B |
|---|---:|---:|---:|
| Terminal-Bench 2.1 | **52.10** | 43.82 | 26.52 |
| NL2Repo-Bench | **35.65** | 28.77 | 16.55 |
| Benchmarks won (of 16) | **12** | | |
| Output tokens vs Qwen3.6 | **−23%** | | |

The rest are only in the card's charts.

## Our numbers

RTX 2070 Max-Q (8 GB), 32 GB RAM, q8_0 KV, flash attention on, sampling from the original repo's
`generation_config.json` (temp 1.0, top-p 0.95, top-k 20). `scripts/qwen-bench.py`, single runs.

| Config | VRAM | Generation | Prompt (15K tokens) | Tool call |
|---|---:|---:|---:|---|
| `NCMOE=26`, batch 512 | 6.8 GB | 38.6 tok/s | 364 tok/s | ✓ |
| `NCMOE=23`, batch 512 | 7.5 GB | 42.1 tok/s | 393 tok/s | ✓ |
| `NCMOE=22`, batch 512 | 7.75 GB | 42.4 tok/s | 403 tok/s | ✓ |
| `NCMOE=19`, batch 512 | — | — | — | **out of memory at load** |
| **`NCMOE=25`, batch 2048 (default)** | **7.7 GB** | **39.5 tok/s** | **887 tok/s** | ✓ |

The fastest generation of any big model here (Qwen3.6 23, the distill 26, Flint 24 tok/s), and
the fastest prompt reading.

**Code check** (`merge_intervals` vs 500 random inputs, run with `--sandbox`): correct twice,
own doctests pass, 0/500 wrong, in **2,315 and 2,988 tokens**. That's a bit under Qwen3.6
(3,558) and well above the distill (557–784).

**[Vision probe](../vision-probe.md):** 74% overall. OCR 97% (read "HSS" as "RSS" once), CAD
100%, understanding 78% (counted 13 blue dots where there are 11), but **boxes 21%** in any
format. It answers in pixels as asked, with x roughly right and y shifted and stretched
unevenly, so few boxes reach IoU 0.5. Qwen3.6, its parent, scores 100% on the same boxes, so the
pruning or post-training cost it some spatial precision. Median 49 s per image.

**[pibench](../pibench.md): 18/18 solved**, 100% of hidden tests, median **3.1 min** per run
(the fastest of the perfect scores: Flint 4.3, Gemma 4.8), 5.2K output tokens per run. Not one
hidden test failed in 18 attempts. The slowest chore was the books scrape (3.6–7.7 min: it
visits every book page). Run at `NCMOE=27` (see Gotchas).

## Security audit

`utils/audit-gguf.py`, before it was ever loaded ([security.md](../security.md)):

- sha256 matches the pinned revision; header clean (753 tensors in bounds, aligned, no overlaps,
  no trailing data); 8 metadata strings, nothing suspicious; vocabulary identical to Qwen3.6.
- The chat template is identical to the one in the original safetensors repo. Compared with
  Qwen's official Qwen3.6 template, the only change is that **every earlier turn's reasoning
  stays in the prompt** (Qwen's drops reasoning from turns before the last user message). No
  added text, no hidden characters, no Jinja internals.
- Pad token 248044 (`<|endoftext|>`, as in the original repo) instead of the 248055 in
  Unsloth's Qwen3.6 GGUF. Harmless.
- The projector is clean as well.

## Gotchas

- Its template keeps all past reasoning in the prompt, so long agent sessions fill the context
  faster than with Qwen3.6.
- Don't trust its bounding boxes.
- `NCMOE=25` leaves ~0.47 GB of VRAM; if your desktop uses more, go to 26 or 27. During pibench,
  with Firefox holding ~185 MB of VRAM, 25 ran out of memory at load and 26 crashed on the first
  request (cuBLAS workspace); 27 ran all 18 attempts.

## Knobs

Env vars for `scripts/start-qwen.sh` / `scripts/run-qwen.sh` with `QMODEL=groot`.

| Knob | Default | What it does |
|---|---|---|
| `NCMOE` | 25 | layers whose experts stay in RAM (19 runs out of VRAM; 22–23 at batch 512) |
| `BATCH` / `UBATCH` | 2048 / 2048 | 512 gives ~42 tok/s but half the prompt speed |
| `CTX` | 131072 | context window |
| `KVQUANT` | q8_0 | KV cache type |
| `TEMP` / `TOP_P` / `TOP_K` | 1.0 / 0.95 / 20 | sampling |
| `--image` / `MMPROJ_GPU` | off / 0 | vision projector, on the CPU by default |
