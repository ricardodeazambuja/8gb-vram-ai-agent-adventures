# 💧 LFM2.5-24B-A2B — not a Qwen: Liquid AI's conv/attention MoE, trained inside agent harnesses

> **TL;DR**
>
> | | |
> |---|---|
> | On disk | 13.6 GB (`Q4_0`, quantization-aware distilled) |
> | Where it lives | experts of 25 of 38 MoE layers in RAM, everything else on the GPU (7.6 GB VRAM) |
> | Context | **128K** |
> | Generation | **31.5 tok/s** (32.4 tok/s at batch 512) |
> | Prompt processing | **~940 tok/s** (15K-token prompt) |
> | Code check | correct twice, 0/500 wrong, 3.1K–3.3K tokens |
> | [pibench](../pibench.md) | 10/18 solved (86% of hidden tests), 4.2 min median per run |
> | Vision | none (text only) |
> | [Security audit](../security.md) | clean |
> | Launch | `QMODEL=lfm25 bash scripts/start-qwen.sh` |

## What is it?

The largest and last model of **Liquid AI**'s LFM2.5 family (LFM Open License v1.0: free, but
commercial use only below $10M annual revenue). A different lineage from everything else here:

- **40 layers: 30 double-gated short-convolution blocks + 10 GQA attention layers** (attention at
  layers 2, 6, 10, … 38). Only the attention layers have a KV cache.
- **The first 2 layers are dense**; the other 38 each have **64 experts, 4 active** (2.45B active
  of 24B).
- 128K vocabulary, 128K context, text only.
- **Always thinks**: the template opens `<think>` on every answer and has no switch to turn it off
  (`enable_thinking: false` is ignored).
- Post-trained with agentic RL **inside agent harnesses, Pi among them**, so it has seen pi's tools
  and system prompt.
- Writes tool calls as Python (`[read(path="...")]` between special tokens); llama.cpp's LFM2
  parser turns them into ordinary OpenAI tool calls.

The `QAD` file is Liquid's own Q4_0, distilled from the full model to win back what Q4_0 loses.

## How does it fit in 8 GB?

The same trick as the Qwen MoEs: everything but the experts goes on the GPU, plus the experts of
the last 13 layers. `NCMOE=27` counts the 2 dense layers too, so 25 layers' experts live in RAM.
Each layer's experts weigh ~0.34 GB, so every step of `NCMOE` is a big jump; 26 only fits at
batch 512. KV cache is small: 10 attention layers × 8 KV heads, ~1.4 GB at 128K q8_0.

Why it's slower than [Groot](groot-19b-a3b.md) despite fewer active parameters: its experts are
three times as wide (1536 vs 512), so each token reads ~530 MB of expert weights from RAM
against Groot's ~380 MB. Generation here is bound by RAM bandwidth, and the ratio (1.39) matches
the speed gap (1.25) roughly.

## Install & run

After [getting-started.md](../getting-started.md):

```bash
# 1. The newest upstream llama.cpp build (skip if you built it for Flint or Groot)
CUDA_VER=12.2 LLAMA_REF=a37cc46104eb50900d5ab4a9ba02a0abdac039db \
  SRC_DIR=$PWD/vendor/flint-llama.cpp bash scripts/build-llama-cuda.sh

# 2. Download + sha256-check (~13.6 GB), then audit it
bash scripts/download-models.sh lfm25
bash utils/audit-gguf.sh models/lfm2.5-24b-a2b/LFM2.5-24B-A2B-QAD-Q4_0.gguf

# 3. Server + pi
QMODEL=lfm25 bash scripts/start-qwen.sh

# Stop it
bash scripts/stop-server.sh
```

Server only, for any OpenAI-compatible client: `QMODEL=lfm25 bash scripts/run-qwen.sh`.

## Official numbers

From the [card](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) (vLLM, full precision; most evals
at temp 0.6, tool benchmarks at 0–0.001):

| Benchmark | **LFM2.5-24B-A2B** | Qwen3.6-35B-A3B | Gemma-4-26B-A4B |
|---|---:|---:|---:|
| AutomationBench | **27.69** | 8.72 | 5.46 |
| BFCLv4 | **65.01** | 65.00 | 57.09 |
| ToolSandbox | **80.71** | 75.02 | 72.09 |
| LiveClaw | **48.09** | 38.19 | 41.25 |
| PinchBench | 83.18 | **89.93** | 77.12 |
| WebArena Verified | 26.75 | **37.62** | 25.62 |
| IFBench | **74.69** | 64.21 | 47.44 |
| GPQA Diamond | 74.42 | **83.69** | 74.90 |
| MATH-500 | 96.04 | **96.73** | 95.66 |
| SWE-Bench Verified (OpenCode) | 51.00 | **66.20** | 25.80 |
| SciCode | 32.71 | **37.35** | 35.42 |

Strong on tool calling and instruction following, behind Qwen3.6 on coding and knowledge. Liquid
itself says to pair it with retrieval for knowledge-heavy work.

## Our numbers

RTX 2070 Max-Q (8 GB), 32 GB RAM, q8_0 KV, flash attention on, the card's sampling (temp 0.1,
top-k 50, repeat-penalty 1.05, no top-p). `scripts/qwen-bench.py`, single runs.

| Config | VRAM | Generation | Prompt (15K tokens) | Tool call |
|---|---:|---:|---:|---|
| `NCMOE=28`, batch 2048 | 7.3 GB | 30.2 tok/s | 922 tok/s | ✓ |
| **`NCMOE=27`, batch 2048 (default)** | **7.6 GB** | **31.2–31.9 tok/s** | **939–945 tok/s** | ✓ |
| `NCMOE=26`, batch 2048 | — | — | — | **out of memory at load** |
| `NCMOE=26`, batch 512 | 7.5 GB | 32.4 tok/s | 325 tok/s | |
| `NCMOE=25`, batch 512 | — | — | — | **out of memory at load** |

Slower to generate than Groot (39.5), faster than Qwen3.6 (23); prompt reading about the same as
Groot.

**Code check** (`merge_intervals` vs 500 random inputs, run with `--sandbox`): correct twice,
own doctests pass, 0/500 wrong, in **3,290 and 3,071 tokens**: as wordy as Qwen3.6 (3,558).

**[pibench](../pibench.md): 10/18 solved**, 86% of hidden tests, median 4.2 min per run, 6.7K
output tokens. Perfect on the feature, the JavaScript page and (2 of 3) the CLI and the login,
and two clean zeros, each the same mistake three times:

- 🐛 **bug fix, 0/3**: it fixed the two reported bugs and stopped. The parser still accepts
  `5x`, `abc` and `1h 2` (22–25 of 27 tests), so the bugs nobody reported stayed.
- 📚 **books, 0/3**: every run got prices, stock and UPCs but **mangled the titles**: it read the
  UTF-8 pages as Latin-1 (`Noahâ\x80\x99s Ark`) and left HTML entities in (`It&#39;s`). It
  printed those titles each time and walked past them. One run also stored "In stock" as the
  stock count. The slowest run took 20 minutes and 70 tool calls.

Same lesson as the rest of the [pibench](../pibench.md) field: it runs its code, but doesn't read
what comes out. (Run with a concurrent llama.cpp build and downloads on the same laptop, so the
times may be a little high.)

## Security audit

`utils/audit-gguf.py --ref chat_template.jinja` (the original repo's), before it was ever loaded
([security.md](../security.md)):

- sha256 matches the pinned revision; header clean (428 tensors in bounds, aligned, no overlaps,
  no trailing data); 34 metadata strings, nothing suspicious; 124 control tokens, all short.
- The chat template is **byte-identical** to the original repo's. No hidden characters, no Jinja
  internals.
- The GGUF carries its sampling defaults (`general.sampling.*`), the same as the card's.

## Gotchas

- It always thinks. A request with a small `max_tokens` can end inside the reasoning with an
  empty answer.
- The temp 0.1 default is the card's; Liquid ran most of its own benchmarks at 0.6.
- `NCMOE=27` leaves ~0.6 GB of VRAM; if your desktop uses more, go to 28.
- The license is Liquid's own (LFM Open License v1.0): Apache-like, but commercial use is only
  licensed for organizations under $10M annual revenue.

## Knobs

Env vars for `scripts/start-qwen.sh` / `scripts/run-qwen.sh` with `QMODEL=lfm25`.

| Knob | Default | What it does |
|---|---|---|
| `NCMOE` | 27 | layers (dense ones included) whose experts stay in RAM; 26 needs batch 512 |
| `BATCH` / `UBATCH` | 2048 / 2048 | 512 gives +1 tok/s but a third of the prompt speed |
| `CTX` | 131072 | context window |
| `KVQUANT` | q8_0 | KV cache type |
| `TEMP` / `TOP_P` / `TOP_K` / `REPEAT_PENALTY` | 0.1 / 1.0 / 50 / 1.05 | sampling |
