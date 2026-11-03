# 🚀 Gemma 4 12B "Agentic" (fable5 v2) — retired

> **Retired on 2026-11-02.** 0/18 on [pibench](../pibench.md), and the only evidence that it should do better is
> a 20-task benchmark its author ran. Its launchers, its download entry and the TurboQuant build that existed only
> for it are gone. This page records what we measured.
>
> | | |
> |---|---|
> | File | `gemma4-v2-Q4_K_M.gguf`, 6.87 GB (dense 12B) |
> | Where it lived | 40 of 48 layers on the GPU on our fork (32K, ~13 tok/s); all 48 with the TurboQuant fork's `turbo3` V cache (16K, ~28 tok/s) |
> | [pibench](../pibench.md) | **0/18** at 32K on our fork (2026-11-02) |

## What was it?

The v2 successor to the [12B Coder](gemma4-12b-coder.md), by the same hobbyist:
[`yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF`](https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF),
pinned at `190a3136`. On top of v1's coding data it added "real multi-step tool-use trajectories… in Gemma 4's
native tool protocol".

Its one benchmark is the author's own: **τ²-bench telecom, ~55% against ~15%** for the base `gemma-4-12B-it`
(20 tasks, greedy, a self-simulated user, Q8_0; "not directly comparable to published tau2-bench leaderboard
figures"). That ratio is the "3.5x" in its name. The author also reports the base model ahead on τ²-bench retail and
a little ahead on MMLU-Pro, and that v2 "still flails a little sometimes (over-trying, retrying)".

## Why it was retired

**pibench, 2026-11-02**, with the old `run-12b-agentic.sh` on our fork (32K, 40 layers on the GPU, about 13 tok/s,
repetition penalty 1.1 as the card advises): **0/18**. It worked as an agent, busily: 31 tool calls a job, pytest
runs. But:

- **Time.** 11 of 18 jobs hit the 20-minute limit.
- **Broken tool calls.** 57 calls had no file path, and about 15 tool errors a job. Some argument names arrived
  mangled, with Gemma's own markup inside them (`content=<|"|>import csv…`, `<channel|><|tool_call>call`).
- **Close, not done.** 25/27 and 26/27 on the bug fix; one attempt ran past 32K.
- **A loop.** In at least one job it wrote the same three lines over and over inside one `write` call (over 30,000
  characters) until time ran out.

**The harness?** ([probe](../../bench/gemma12-probe/README.md)). Not shown to be. Its first tool call came out in
clean native syntax in 16 of 16 raw samples, at repetition penalty 1.1 and 1.0, and all 15 calls in a captured
pibench job were well formed. The mangled calls are intermittent. We stopped there: no evidence pointed at llama.cpp
or pi, and the model's own claims didn't justify more GPU time.

## What we measured

| Mode (Q4_K_M) | Build | Layers on GPU | KV cache (K / V) | Context | VRAM | Generation |
|---|---|---:|---|---:|---:|---:|
| TurboQuant | TheTom/llama-cpp-turboquant | **48 (all)** | q8_0 / turbo3 | 16K | 7.76 / 8 GB | **~28.3 tok/s** |
| stock | mainline llama.cpp | 40–42 | q8_0 / q8_0 | 32K | ~7.1 GB | ~16.5 tok/s |
| our fork, 2026-11-01 | `adventures` | 40 | q8_0 / q8_0 | 32K | — | 13.4 tok/s |

`turbo3` is a ~3.5-bit V cache (Walsh-Hadamard rotated, from Google's TurboQuant research) that isn't in mainline
llama.cpp; the 25K context the author uses fit on this card only sometimes. More in
[How it fits](../how-it-fits.md#9-turboquant-an-even-smaller-v-cache).

One lesson worth keeping: our first download, done with a 16-connection `aria2c` and interrupted, had the right size
and a valid header but wrong tensor data. It loaded and printed `<unused32><unused32>…` forever, which looked like a
TurboQuant bug. `sha256sum` against Hugging Face's checksum caught it. **Garbage from a fresh GGUF? Check the
checksum before you blame the engine.**
