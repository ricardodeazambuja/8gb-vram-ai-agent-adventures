# 🐦 Ornith-1.5-35B-A3B — a Qwen3.6-class MoE that taught itself on its own tasks

> **TL;DR**
>
> | | |
> |---|---|
> | On disk | 21.7 GB (`Q4_K_M`) + 0.9 GB vision projector |
> | Where it lives | experts of 34 of 40 layers in RAM, everything else on the GPU (7.1 GB VRAM) |
> | Context | **128K** (the model does 256K) |
> | Generation | **~31 tok/s** (32–33 tok/s with one more layer on the GPU) |
> | Prompt processing | ~490–510 tok/s (15K-token prompt) |
> | Code check | correct twice, 0/500 wrong, **1.4K–1.8K tokens** |
> | [pibench](../pibench.md) | running |
> | [Vision probe](../vision-probe.md) | running |
> | [Security audit](../security.md) | clean |
> | Launch | `QMODEL=ornith bash scripts/start-qwen.sh` |

## What is it?

**Ornith-1.5** (ornith-ai, MIT license) is the second round of a "self-improvement" project:
Ornith-1.0 was continued pre-training, mid-training and post-training on top of Qwen3.5 and
Gemma 4; 1.5 extends the loop so the model **generates its own training tasks**, builds the
scaffolds to solve them, and learns from the rollouts with reinforcement learning. The 35B-A3B
member has the Qwen3.6-35B-A3B architecture unchanged: 40 layers (30 Gated DeltaNet + 10 full
attention), 256 experts with 8 active per token (about 3B active), the same vocabulary and the
same vision tower. It always thinks first (`<think> … </think>`).

## How does it fit in 8 GB?

The same way as [Qwen3.6](qwen3.6-35b-a3b.md): the experts stay in RAM, everything else goes to
the GPU. This file is a plain `Q4_K_M` (21.7 GB, against the 17.7 GB `UD-IQ4_XS` we use for
Qwen3.6), so each layer's experts weigh about 0.5 GB and only 6 layers' worth fit on the GPU next
to a 128K q8_0 KV cache and batch 2048's compute buffer. It needs about **18 GB of free RAM** for
the rest: close the browser first. The first load of the 21.7 GB file, with a download running at
the same time, once pushed the laptop's memory pressure high enough for systemd-oomd to kill the
whole terminal; load it on a quiet machine.

## Install & run

After [getting-started.md](../getting-started.md):

```bash
# 1. The newest upstream llama.cpp build (skip if you built it for Flint or Groot)
CUDA_VER=12.2 LLAMA_REF=a37cc46104eb50900d5ab4a9ba02a0abdac039db \
  SRC_DIR=$PWD/vendor/flint-llama.cpp bash scripts/build-llama-cuda.sh

# 2. Download + sha256-check (~22.6 GB with the projector), then audit it
bash scripts/download-models.sh ornith
bash utils/audit-gguf.sh models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf

# 3. Server + pi (close the browser first: ~18 GB of RAM)
QMODEL=ornith bash scripts/start-qwen.sh

# ...with images
QMODEL=ornith bash scripts/start-qwen.sh --image

# Stop it
bash scripts/stop-server.sh
```

Server only, for any OpenAI-compatible client: `QMODEL=ornith bash scripts/run-qwen.sh`.

## Official numbers

From the [card](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) (BF16, temp 1.0):

| Benchmark | **Ornith-1.5-35B-A3B** | Ornith-1.0-35B-A3B | Qwen3.6-35B-A3B | Gemma-4-31B |
|---|---:|---:|---:|---:|
| Terminal-Bench 2.1 (Terminus-2) | **67.8** | 64.2 | 52.5 | 42.1 |
| SWE-bench Verified | **79.0** | 75.6 | 73.4 | 52.0 |
| SWE-bench Pro | **59.6** | 50.4 | 49.5 | 35.7 |
| NL2Repo | **46.2** | 34.6 | 29.4 | 15.5 |
| GPQA Diamond | **89.2** | 86.2 | 86.0 | 84.3 |
| HLE (no tools) | **25.6** | 20.8 | 21.4 | 19.5 |
| MCP-Atlas | **70.2** | 64.4 | 62.8 | 55.0 |
| Toolathlon-Verified | **48.7** | 42.4 | 41.7 | 40.8 |
| ClawEval | **72.5** | 69.8 | 68.7 | 48.5 |

Big claims for a 3B-active model: ten points over Qwen3.6 on SWE-bench Pro and Terminal-Bench.
The card has more rows (DeepSWE, BrowseComp, WideSearch, ...).

## Our numbers

RTX 2070 Max-Q (8 GB), 32 GB RAM, q8_0 KV, flash attention on, 128K context, the card's
benchmark sampling (temp 1.0, top-p 0.95, top-k 20). `scripts/qwen-bench.py`, single runs.

| Config | VRAM | Generation | Prompt (15K tokens) | Tool call |
|---|---:|---:|---:|---|
| `NCMOE=37`, batch 2048 | 5.6 GB | 26.2 tok/s | 471 tok/s | ✓ |
| `NCMOE=36`, batch 2048 | 6.1 GB | 27.1 tok/s | 479 tok/s | ✓ |
| `NCMOE=35`, batch 2048 | 6.6 GB | 29.5 tok/s | 489 tok/s | ✓ |
| **`NCMOE=34`, batch 2048 (default)** | **7.1 GB** | **~31 tok/s** ¹ | — | |
| `NCMOE=34`, batch 512 | 6.6 GB | 31.2 tok/s | 176 tok/s | ✓ |
| `NCMOE=33`, batch 2048 | 7.5 GB | 32.2 tok/s | 507 tok/s | ✓ |
| `NCMOE=33`, batch 512 | 7.0 GB | 31.3 tok/s | 179 tok/s | ✓ |
| `NCMOE=32`, batch 512 | 7.5 GB | 33.0 tok/s | 183 tok/s | ✓ |

¹ From the two code-check runs (31.4 and 30.9 tok/s).

The default leaves ~0.9 GB of VRAM free on purpose: [Groot](groot-19b-a3b.md) at 7.7 GB ran out
of memory during pibench with a browser open. `NCMOE=33` is 1 tok/s faster if your desktop is lean.
Prompt reading is about half of Groot's (887 tok/s): more of each layer's experts live in RAM.

**Code check** (`merge_intervals` vs 500 random inputs, run with `--sandbox`): correct twice,
own doctests pass, 0/500 wrong, in **1,401 and 1,813 tokens**: half of Qwen3.6's 3,558 for the
same architecture, and fewer than Groot (2.3K–3.0K).

## Security audit

`utils/audit-gguf.py`, before it was ever loaded ([security.md](../security.md)):

- sha256 matches the pinned revision; header clean (753 tensors in bounds, aligned, no overlaps,
  no trailing data); 12 metadata strings, nothing suspicious; special tokens all short markers.
- The chat template is Unsloth's patched Qwen3.6 template (the one in our Qwen3.6 GGUF) minus
  the branch that drops reasoning from earlier turns, the same change as Groot's. No added text,
  no hidden characters, no Jinja internals.
- The projector is clean as well.

## Gotchas

- 21.7 GB file: about 18 GB of RAM for the experts. Close the browser, and don't load it while a
  download or another big job is running (see "How does it fit").
- Its template keeps all past reasoning in the prompt, so long agent sessions fill the context
  faster than with Qwen3.6.
- The card suggests temp 0.6 for general chat; the launcher uses its benchmark setting, 1.0.

## Knobs

Env vars for `scripts/start-qwen.sh` / `scripts/run-qwen.sh` with `QMODEL=ornith`.

| Knob | Default | What it does |
|---|---|---|
| `NCMOE` | 34 | layers whose experts stay in RAM (33 at batch 2048 and 32 at batch 512 also fit, with less headroom) |
| `BATCH` / `UBATCH` | 2048 / 2048 | 512 costs nothing in generation but reads prompts ~2.8× slower |
| `CTX` | 131072 | context window |
| `KVQUANT` | q8_0 | KV cache type |
| `TEMP` / `TOP_P` / `TOP_K` | 1.0 / 0.95 / 20 | sampling (temp 0.6 for chat, per the card) |
