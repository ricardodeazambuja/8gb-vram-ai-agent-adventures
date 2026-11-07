# 🌱 Groot-19B-A3B — Qwen3.6 with half the experts, and faster for it

> **TL;DR**
>
> | | |
> |---|---|
> | On disk | 11.7 GB (`Q4_K_M`) + 0.9 GB vision projector |
> | Where it lives | every expert in RAM, a 2.8 GB cache of the busy ones on the GPU (6.7 GB of VRAM of its own at its peak; `--static`: experts of 27 of 40 layers in RAM, 7.0 GB) |
> | Context | **128K** |
> | Generation | **47–48 tok/s** with the expert cache (39–40 with `--static`) |
> | Prompt processing | **870–890 tok/s** (15K-token prompt) |
> | Code check | correct twice, 0/500 wrong, 2.3K–3.0K tokens |
> | [pibench](../pibench.md) | **18/18 solved** (100% of hidden tests), 3.1 min median per run |
> | [Vision probe](../vision-probe.md) | 74% overall: OCR and CAD fine, boxes poor |
> | [Security audit](../security.md) | clean |
> | Launch | `scripts/llm groot` |

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
about half as much, so with the static split more layers fit on the GPU (13 of 40 against Qwen3.6's 8), and
less work happens on the CPU per token. That's where the extra speed comes from. Like
[Flint](flint.md), it uses batch 2048 so the expert maths for long prompts goes to the GPU; the
bigger compute buffer costs a few layers of experts. KV cache: the same as Qwen3.6, 128K at q8_0
is about 1.4 GB. More in [how-it-fits.md](../how-it-fits.md).

## Install & run

Build our llama.cpp once: [getting-started.md](../getting-started.md). Then:

```bash
# 1. Download + sha256-check (~12.6 GB with the projector), then audit it
bash scripts/download-models.sh groot
bash utils/audit-gguf.sh models/groot-19b-a3b/Groot-19B-A3B-261023-Q4_K_M.gguf

# 2. Server + pi
scripts/llm groot

# ...with images
scripts/llm groot --image

# Stop it
scripts/llm stop
```

Server only, for any OpenAI-compatible client: `scripts/llm groot serve`. (The building block:
`QMODEL=groot bash scripts/run-qwen.sh`; every knob in [Knobs](#knobs) below still applies, e.g.
`CTX=65536 scripts/llm groot`.)

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
| `NCMOE=25`, batch 2048 | 7.7 GB | 39.5 tok/s | 887 tok/s | ✓ |
| `NCMOE=26`, batch 2048 | 7.4 GB | 40.2 tok/s | 888 tok/s | ✓ |
| `NCMOE=27`, batch 2048 (`--static`; the default until 2026-11-07) | 7.2 GB | 39.2 tok/s | 871 tok/s | ✓ |
| **`MOE_CACHE=auto` (2821 MiB), batch 2048 (default)** | **6.7 GB** | **47.4 tok/s** | same within noise | |

The fastest generation of any big model here with the static split (Qwen3.6 23, the distill 26, Flint 24 tok/s),
and very fast prompt reading (LFM2.5 reached about 940). With the expert cache on every model, as `scripts/llm` runs
them, it's second only to the distill (49.0): Groot 47.4, LFM2.5 43.5, Ornith 40.4, Qwen3.6 39.4 tok/s.

**Expert cache: the launcher default since 2026-11-07.** `NCMOE=27` peaks at ~7.03 GB of its own, which leaves
the desktop only ~360 MiB by the [VRAM rule](../../bench/moe-cache/README.md#five-more-moe-models-models-2026-11-05)
(that's why `NCMOE=25` ran out of memory in pibench with a browser open). A 2821 MiB cache peaks at 6.66 GB, leaves
~730 MiB, and decodes faster: **47.4 vs 39.5 tok/s (+20%)**; turns and prompt reading the same within noise
([step 17](../../bench/moe-cache/README.md#groot-and-lfm25-sizes-for-the-launcher-modelsstep17-groot-lfm25-sizesh-2026-11-07)). `MOE_CACHE=auto` uses 2821 MiB and shrinks it for a busier desktop; `scripts/llm groot --static`
gives the old split. A bigger cache (3321 MiB) did 51.4 tok/s but leaves the desktop only ~230 MiB.

**Code check** (`merge_intervals` vs 500 random inputs, run with `--sandbox`): correct twice,
own doctests pass, 0/500 wrong, in **2,315 and 2,988 tokens**. That's a bit under Qwen3.6
(3,558) and well above the distill (557–784).

**[Vision probe](../vision-probe.md):** 74% overall. OCR 97% (read "HSS" as "RSS" once), CAD
100%, understanding 78% (counted 13 blue dots where there are 11), but **boxes 21%** in any
format. It answers in pixels as asked, with x roughly right and y shifted and stretched
unevenly, so few boxes reach IoU 0.5. Qwen3.6, its parent, scores 100% on the same boxes, so the
pruning or post-training cost it some spatial precision. Median 44 s per image.

**[pibench](../pibench.md): 18/18 solved**, 100% of hidden tests, median **3.1 min** per run
(the fastest of the perfect scores: Flint 4.3, Gemma 4.8), 5.2K output tokens per run. Not one
hidden test failed in 18 attempts. Run at `NCMOE=27`, the static split that was the default then; the expert cache
wasn't in pibench yet.

## How it behaves

- **Thorough on scrapes.** The slowest chore was the books scrape (3.6–7.7 min): it visits every book page.
- **Boxes in pixels, but off.** It answers in pixels as asked, with x roughly right and y shifted and
  stretched (see the vision probe above).

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
- With `--static`, keep `NCMOE=27` or higher: it leaves the desktop only ~360 MiB. During pibench, with Firefox
  holding ~185 MB of VRAM, 25 ran out of memory at load and 26 crashed on the first request (cuBLAS workspace);
  27 ran all 18 attempts. The default expert cache leaves ~730 MiB and shrinks itself for a busier desktop.

## Knobs

Env vars for `scripts/llm groot` / `scripts/run-qwen.sh` with `QMODEL=groot`.

| Knob | Default | What it does |
|---|---|---|
| `NCMOE` | 27 | with `--static`: layers whose experts stay in RAM (25–26 fit with less headroom; 19 runs out of VRAM; 22–23 at batch 512) |
| `MOE_CACHE` | auto in `scripts/llm` (off with `--static` or in `run-qwen.sh` alone) | VRAM cache of the experts instead of `NCMOE`'s fixed layers |
| `BATCH` / `UBATCH` | 2048 / 2048 | 512 gives ~42 tok/s but half the prompt speed |
| `CTX` | 131072 | context window |
| `KVQUANT` | q8_0 | KV cache type |
| `TEMP` / `TOP_P` / `TOP_K` | 1.0 / 0.95 / 20 | sampling |
| `--image` / `MMPROJ_GPU` | off / 0 | vision projector, on the CPU by default |
