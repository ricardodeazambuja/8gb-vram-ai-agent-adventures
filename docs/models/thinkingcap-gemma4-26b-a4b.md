# 🧢 ThinkingCap Gemma 4 26B-A4B — the same Gemma, told to think less

> **TL;DR**
>
> | | |
> |---|---|
> | On disk | 13.6 GB (`IQ4_XS` experts, `Q8_0` everything else) + 1.2 GB vision projector |
> | Where it lives | experts of 23 of 30 layers in RAM, everything else on the GPU (7.0 GB VRAM) |
> | Context | 64K (as we run the base Gemma in pibench) |
> | Generation | **17–18 tok/s** (the base Gemma file: ~30 tok/s with the same settings) |
> | Prompt processing | ~350 tok/s (16K-token prompt) |
> | Code check | correct twice, 0/500 wrong, 2.2K–2.4K tokens |
> | [pibench](../pibench.md) | **17/18 solved** (98% of hidden tests), **47% less thinking** than the base, 5.5 min median per run |
> | [Security audit](../security.md) | clean |
> | Launch | `MODEL=models/thinkingcap-gemma4-26b-a4b/ThinkingCap-gemma-4-26B-A4B-it-IQ4_XS.gguf ALIAS=thinkingcap-gemma-4-26b-a4b NCMOE=23 BACKEND=cuda bash scripts/run-server.sh` |

## What is it?

[bottlecap.ai](https://huggingface.co/bottlecapai)'s "ThinkingCap" post-training applied to
[Gemma 4 26B-A4B](gemma4-26b-a4b.md) (Apache 2.0, under Google's Gemma terms). Same architecture,
same vocabulary, same vision tower; the training only aims to **shorten the reasoning**. The
[card](https://huggingface.co/bottlecapai/ThinkingCap-gemma-4-26B-A4B-it) reports 38.9% fewer
thinking tokens on average over twelve benchmarks (14% to 69%), at 75.0% average accuracy against
the base's 75.2%, and far fewer runaway traces (truncated 0.7% vs 4.4%, looping 0.8% vs 3.6%).
As in the base, thinking is off unless the request turns it on; pi turns it on.

## How does it fit in 8 GB?

Like the base Gemma: experts in RAM, the rest on the GPU. But this GGUF keeps every non-expert
tensor (attention, the dense feed-forward block, the embedding) at `Q8_0`, 2.6 GB against 1.4 GB
in the Unsloth QAT file we use for the base. That extra 1.2 GB pushes three more layers' experts
to RAM (`NCMOE=23` instead of 20; 20 and 18 run out of VRAM at load). The bytes of experts read
from RAM per token come out about the same as the base's (smaller `IQ4_XS` experts, more layers
of them), so the drop to 17–18 tok/s from ~30 most likely comes from the CPU's `IQ4_XS` kernels
being slower than the QAT file's; we haven't profiled it.

The card recommends its `MXFP4_MOE` file (the one it evaluated); we took `IQ4_XS` as the
smallest 4-bit option. A file with the non-expert part at Q4–Q6 would get most of the speed
back.

## Install & run

After [getting-started.md](../getting-started.md):

```bash
# Download + sha256-check (~14.8 GB with the projector), then audit it
bash scripts/download-models.sh tcgemma
bash utils/audit-gguf.sh models/thinkingcap-gemma4-26b-a4b/ThinkingCap-gemma-4-26B-A4B-it-IQ4_XS.gguf

# Server (the base Gemma's launcher, pointed at this file)
BACKEND=cuda CTX=65536 NCMOE=23 KVQUANT=q8_0 ALIAS=thinkingcap-gemma-4-26b-a4b \
  MODEL=models/thinkingcap-gemma4-26b-a4b/ThinkingCap-gemma-4-26B-A4B-it-IQ4_XS.gguf \
  bash scripts/run-server.sh
```

## Official numbers

From the [card](https://huggingface.co/bottlecapai/ThinkingCap-gemma-4-26B-A4B-it) (BF16, thinking
on, temp 1.0, top-p 0.95, top-k 64):

| Benchmark | Base accuracy | **ThinkingCap** | Base thinking tokens | **ThinkingCap** | Reduction |
|---|---:|---:|---:|---:|---:|
| GPQA-Diamond | 79.9 | 79.0 | 7,699 | 4,284 | −44% |
| AIME 2026 | 86.2 | 86.7 | 14,766 | 8,963 | −39% |
| LiveCodeBench v6 | 77.0 | 77.0 | 11,560 | 8,375 | −28% |
| IFBench | 76.8 | 78.7 | 6,470 | 5,585 | −14% |
| τ²-bench | 64.3 | 63.7 | 4,374 | 3,466 | −21% |
| Macro average (12) | 75.2 | 75.0 | 8,773 | 5,398 | −39% |

## Our numbers

RTX 2070 Max-Q (8 GB), 32 GB RAM, 64K context, q8_0 KV, flash attention on, the base's sampling
(temp 1.0, top-p 0.95, top-k 64). `scripts/qwen-bench.py`, single runs.

| Config | VRAM | Generation | Prompt (16K tokens) |
|---|---:|---:|---:|
| `NCMOE=18` or `20` | — | **out of memory at load** | — |
| **`NCMOE=23` (used)** | **7.0 GB** | **17.4 tok/s** | **352 tok/s** |
| `NCMOE=24` | 6.6 GB | 17.0 tok/s | 344 tok/s |
| `NCMOE=26` | 5.9 GB | 16.0 tok/s | 327 tok/s |

**Code check** (`merge_intervals` vs 500 random inputs, run with `--sandbox`): correct twice,
own doctests pass, 0/500 wrong, in 2,163 and 2,370 tokens.

**[pibench](../pibench.md): 17/18 solved**, 98% of hidden tests. The base Gemma got 18/18.

| Per pibench attempt (18 each) | Base Gemma 4 26B | **ThinkingCap** |
|---|---:|---:|
| Thinking, total characters | 218,695 | **115,203 (−47%)** |
| Thinking, median per attempt | 11,919 | **7,524 (−37%)** |
| Output tokens, median | 7,114 | **5,303 (−25%)** |
| Median time | **4.8 min** | 5.5 min |

The card's claim holds up on agentic work: about half the thinking, the same results. The
biggest cut was on the bug fix (23K → 9K characters), and twice it wrote the CLI tool with
almost no thinking at all. The one miss was the books scrape, where it read the UTF-8 pages as
Latin-1 (`Noahâ\x80\x99s Ark`), the same trap that caught [LFM2.5](lfm2.5-24b-a2b.md) and
[Ornith](ornith-1.5-35b-a3b.md). It's slower per job only because this file decodes at 17 tok/s
on this laptop instead of ~30. A file quantized like our base Gemma file could turn the shorter
thinking into shorter jobs; we haven't tried one.

## Security audit

`utils/audit-gguf.py --ref` the base Gemma QAT GGUF, before it was ever loaded
([security.md](../security.md)):

- sha256 matches the pinned revision; header clean; vocabulary identical (262,144 tokens).
- The chat template is a newer Gemma 4 template (null arguments, continuation tracking, images in
  tool responses, a `preserve_thinking` option). Nothing injected: the only new emitted text is
  a `raise_exception` message about tool-call arguments.
- `eot_token_id` (106) is unset; we saw no runaway turns.
- The projector is clean as well.

## Gotchas

- `NCMOE=20`, the base's setting, runs out of VRAM: use 23.
- Thinking is off unless the client asks for it, as with the base; there's no saving to be had
  without it.
