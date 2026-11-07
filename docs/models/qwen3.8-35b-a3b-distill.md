# ⚡ Qwen3.8-35B-A3B Distill — Qwen3.6's brain, with fewer words

> **TL;DR**
>
> | | |
> |---|---|
> | On disk | 19.6 GB (`IQ4_XS`) + 0.9 GB vision projector |
> | Where it lives | every expert in RAM with a 2770 MiB VRAM cache of the recently used ones, everything else on the GPU (`scripts/llm`); `--static`: experts of 34 of 40 layers in RAM (7.2–7.3 GB VRAM) |
> | Context | **128K** |
> | Generation | **49.0 tok/s** with `scripts/llm` (expert cache, light desktop; 43.1 next to a busy daytime one); 26.2 with `--static` |
> | Prompt processing | 167 tok/s (15.6K-token prompt) |
> | [pibench](../pibench.md) | 16/18 solved (98% of hidden tests), **2.5 min per run** |
> | [Vision probe](../vision-probe.md) | 88% overall |
> | [Security audit](../security.md) | clean |
> | Launch | `scripts/llm distill` |

## What is it?

A community fine-tune by **empero-ai**. It keeps the exact architecture of
[Qwen3.6-35B-A3B](qwen3.6-35b-a3b.md) (35B total, about 3B active per token, 40 layers, mostly
Gated DeltaNet with full attention on 10 of them) and was trained on curated chain-of-thought
traces from Qwen3.8 teacher models, weighted towards math and code, with tool use included.

Its training examples were 8,192 tokens long, so it learned to get to the point. On a laptop
generating ~26 tokens per second, "gets to the point" is a feature, not a personality flaw.
The vision part is untouched from the base model.

## How does it fit in 8 GB?

Same trick as Qwen3.6: the experts mostly live in system RAM and the GPU holds the rest. Its
experts are a bit bigger (19.6 GB file vs 17.7 GB), so it needs `NCMOE=34`, keeping the experts
of 34 of the 40 layers in RAM. `NCMOE=32` runs out of VRAM at load (a 545 MiB `cudaMalloc`
fails). The KV cache is identical to the base model: 128K at q8_0 is about 1.42 GB. More in
[how-it-fits.md](../how-it-fits.md).

**Faster, same VRAM (2026-11-05, the default in `scripts/llm distill`):** a GPU cache of the recently used experts
instead of fixed layers. `MOE_CACHE=auto` uses 2770 MiB with batch 2048 (long prompts ~2.3× faster than batch 512)
and `--moe-cache-max-batch 256`: **49.0 tok/s** decode with a light desktop, 43.1 next to a ~1.7 GB one. At a light
desktop's limit (`NCMOE=31` against a 3661 MiB cache, batch 512, both ~7.2 GB) it was 52.7 vs 39.3 tok/s.
`--static` gives the `NCMOE=34` split. More in
[how-it-fits §12](../how-it-fits.md#12-let-a-cache-pick-the-experts-the-default-for-the-big-moes) and
[`bench/moe-cache/`](../../bench/moe-cache/README.md#five-more-moe-models-models-2026-11-05).

## Install & run

Build our llama.cpp once: [getting-started.md](../getting-started.md). Then:

```bash
# 1. Hugging Face CLI env for the download script (it wants `hf` inside this env)
mamba create -y -n llamacpp -c conda-forge huggingface_hub

# 2. Download + sha256-check (~20.5 GB with the projector)
bash scripts/download-models.sh distill
bash utils/audit-gguf.sh models/qwen38-35b-a3b-distill/Qwen3.8-35B-A3B-IQ4_XS.gguf   # then audit it (and the mmproj)

# 3. Server + pi (expert cache on; add --static for the per-layer split)
scripts/llm distill

# ...with images
scripts/llm distill --image

# One-shot, leaves the server running
scripts/llm distill -p "explain @README.md"

# Stop it
scripts/llm stop
```

Server only, for any OpenAI-compatible client: `scripts/llm distill serve`. (The building block:
`QMODEL=distill bash scripts/run-qwen.sh`; every knob in [Knobs](#knobs) below still applies, e.g.
`CTX=65536 scripts/llm distill`.)

## Official numbers

The [card](https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill) only reports general
knowledge benchmarks against its base:

| Task | Metric | Qwen3.6-35B-A3B (base) | **Distill** | Δ |
|---|---|---:|---:|---:|
| MMLU (57 subjects) | acc | 0.838 | 0.834 | −0.004 |
| ARC-Challenge | acc | 0.548 | 0.582 | +0.034 |
| ARC-Challenge | acc_norm | 0.548 | 0.591 | +0.044 |
| ARC-Easy | acc | 0.819 | 0.830 | +0.011 |

No agentic or coding numbers. In other words: roughly its base model on paper, and the rest
you have to find out yourself. We did (see below).

## Our numbers

RTX 2070 Max-Q (8 GB), 32 GB RAM, q8_0 KV, flash attention on, sampling from the card (temp
0.6, top-p 0.95, top-k 20). Single runs, ±13% noise.

| Config | VRAM | Generation | Prompt (15.6K tokens) | Tool call |
|---|---:|---:|---:|---|
| `NCMOE=32`, 128K | — | — | — | **out of memory at load** |
| `NCMOE=34`, 128K (`--static`) | 7.2–7.3 GB | 26.2 tok/s | 167 tok/s | ✓ |
| same, clean-room rebuild from scratch | — | 29.0 tok/s | 197 tok/s | ✓ |
| **`MOE_CACHE=auto` (2770 MiB, batch 2048), 128K (default)** | — | **49.0 tok/s** ¹ | — | |

¹ [Step 16](../../bench/moe-cache/README.md) of the cache tests, 400-token runs, light (~0.5 GB) desktop; 43.1 tok/s
with the cache shrunk to 1770 MiB for a ~1.7 GB desktop. At batch 2048 the cache read 6K / 34K-token prompts at
517 / 495 tok/s (3174 MiB cache, [batch test](../../bench/moe-cache/README.md#qwen38-batch-512-or-2048-modelsstep6-qwen38-ubatchsh-step7-turnssh)).
`NCMOE=31` fits only next to a light desktop (~0.33–0.41 GB of VRAM in use).

**Code check** (`merge_intervals` vs 500 random inputs): correct twice, 0/500 wrong, in
**557 and 784 tokens**, about 6× fewer than Qwen3.6 (3,558) for the same correct function.

**[pibench](../pibench.md):** 16/18 solved, 98% of hidden tests, and **among the quickest big models
per run: median 2.5 minutes** (Ornith took 2.2), median 3,250 output tokens. Ties Qwen3.6 on solved runs.

**[Vision probe](../vision-probe.md):** 88% overall. Perfect OCR and CAD, boxes 87% once you
read them in its 0–1000 format (13% in pixels as asked), and understanding 67%: it counted
8 red and 12 blue dots when there were 7 and 11. Median 35 s per image.

The author also drove it by hand in pi for a while. Verdict: "not bad".

## How it behaves

- **Writes from memory, and usually gets away with it.** It almost never inspects a site
  before scraping it (0.1 curl calls per run, against Flint's 2.8). On the JavaScript quotes
  page it checked Playwright was installed, wrote the scraper with the right selectors blind,
  ran it, printed a sample, and was done in 48 seconds with 4 tool calls. 5/5.
- **Test-driven on code.** It runs pytest more than any other model (1.5 times per run), adds
  its own test cases, and used failing tests to find every hidden bug in the bug-fix task
  (3/3).
- **Both misses were values it printed and didn't read.** In the login task it collected the
  "(about)" links instead of the Goodreads links, `cat`-ed the file showing the wrong URLs, and
  only checked how many keys there were. In the bookshop task, `requests` decoded a page as
  Latin-1 (the site doesn't declare a charset), so "Noah’s" became mojibake in one title. One
  wrong title, three failed tests.
- It looked at the *content* of its own output in only half of its runs.

## Gotchas

- With `--static`, `NCMOE=32` does not fit; stick with 34 (or go higher if your desktop eats VRAM).
- Its short answers are great until a task genuinely needs long generation; the card only
  covers MMLU/ARC, so long sessions are less proven than the base model.
- Double-check anything it scrapes: fast and confident is not the same as right.
- Boxes come back on a 0–1000 grid as `bbox_2d`. See [vision-probe.md](../vision-probe.md).

## Knobs

Env vars for `scripts/llm distill` / `scripts/run-qwen.sh` with `QMODEL=distill`.

| Knob | Default | What it does |
|---|---|---|
| `NCMOE` | 34 | with `--static`: layers whose experts stay in RAM (32 runs out of VRAM) |
| `MOE_CACHE` | auto in `scripts/llm` (off with `--static` or in `run-qwen.sh` alone) | VRAM cache of the experts instead of `NCMOE`'s fixed layers |
| `CTX` | 131072 | context window |
| `KVQUANT` | q8_0 | KV cache type |
| `TEMP` / `TOP_P` / `TOP_K` | 0.6 / 0.95 / 20 | sampling |
| `--image` / `MMPROJ_GPU` | off / 0 | vision projector, on the CPU by default |
