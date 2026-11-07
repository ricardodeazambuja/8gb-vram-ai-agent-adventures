# 🧮 TwIL-8B — a small math and code reasoner

> **TL;DR**
>
> | | |
> |---|---|
> | On disk | 5.35 GB (`Q4_K_M`, the verbose variant) |
> | Where it lives | entirely on the GPU (5.0 GiB of weights + the KV cache) |
> | Context | 24K with a q4_0 KV cache (the card wants 40K for hard problems) |
> | Speed | not measured yet |
> | [Security audit](../security.md) | clean |
> | Licence | webAI Non-Commercial 1.0: non-commercial research and education only |
> | Launch | `scripts/llm twil serve` (server only until it has run in pibench's sandbox) |

## What is it?

[webAI's TwIL-8B](https://huggingface.co/webAI-Official/TwIL-8B): IBM's Granite 4.2 8B (dense, 40 layers,
8 KV heads, 131K context), post-trained with full-parameter SFT and GRPO/DAPO reinforcement learning for **math,
code and instruction following**. It thinks (`<think>…</think>`) before answering. Two variants share the
architecture: *balanced* (shorter reasoning) and *verbose* (longer); we run **verbose** Q4_K_M.

The card's claims, on its own harness and the bf16 weights, not the GGUF: MATH-500 98.2, AIME 2025 88.3,
LiveCodeBench 81.6 and IFBench 74.4 (both measured on verbose), against Qwen3.5-9B's 65.6 and 64.5. It is weak on
knowledge (GPQA Diamond 70.2) and has **no tool-use training**; tool use and agentic tasks were not evaluated.
It writes long: ~19.5K tokens per AIME 2026 problem on balanced, more on verbose.

Where it could fit here: the code writer in the "bigger model plans, small model writes" idea that
[VibeThinker 3B](vibethinker.md) was brought in for ([up-next](../up-next.md)).

## Install & run

Build our llama.cpp once: [getting-started.md](../getting-started.md). Then:

```bash
bash scripts/download-models.sh twil      # pinned revision 58d49f0, sha256 checked
bash utils/audit-gguf.sh models/twil-8b/TwIL-8B-Q4_K_M-verbose.gguf   # then audit it
scripts/llm twil serve                    # port 8080, 24K, q4_0 KV, temp 1.0, top-p 0.95
```

(The building block: `QMODEL=twil bash scripts/run-qwen.sh`; its knobs still apply. `scripts/llm twil` serves it
without pi for now: pi has shell tools, and per [security.md](../security.md) a new model's first shell access is
in pibench's sandbox.)

**Memory.** The KV cache costs 160 KiB a token in f16, 85 in q8_0 and 45 in q4_0 (40 layers x 8 KV heads x
128). Next to 5.0 GiB of weights, 24K at q4_0 (~1.1 GB) leaves room for a ~0.7 GB desktop by estimate;
32K would need the desktop under ~350 MiB, and q8_0 at 32K (2.7 GB) doesn't fit. These are estimates until the
first run measures the compute buffer. With a daytime desktop (~1.8 GB with Chrome open) it does not fit on the
GPU at all.

## Our numbers

Not run yet. Next: the tool-less speed test (`scripts/qwen-bench.py`), then code checks in the sandbox
(`scripts/qwen-codecheck.py`).
