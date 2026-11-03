# 🧑‍💻 Gemma 4 12B "Coder" (fable5 v1) — retired

> **Retired on 2026-11-02.** It isn't an agent (0/18 on [pibench](../pibench.md), never wrote a file), its own author
> replaced it with the [v2 "agentic" model](gemma4-12b-agentic.md), and as a chat coder it's slower than models we
> keep. Its launcher and download entry are gone. This page records what we measured.
>
> | | |
> |---|---|
> | File | `gemma4-coding-Q4_K_M.gguf`, 7.38 GB (dense 12B) |
> | Where it lived | 40 of 48 layers on the GPU, the last 8 in system RAM, 32K, q8_0 KV cache |
> | Generation | 13–17 tok/s |
> | [pibench](../pibench.md) | **0/18** (2026-11-02) |

## What was it?

A coding fine-tune of Google's `gemma-4-12B-it` by yuxinlu1, a hobby project ("shared as-is, no warranty"):
[`yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF`](https://huggingface.co/yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF),
pinned at `1380be17`. Its card describes training on "algorithmic / function-level problems that come with
deterministic tests", with reasoning traces from two teacher models kept only when the code passed. Nothing about
tool use, no benchmark of its own, and *"not safety-aligned, reduced refusals"*. The author later wrote that they
had "owned v1's training problems".

## Why it was retired

**pibench, 2026-11-02 (our fork, 32K): 0/18.** In 18 jobs it made 6 tool calls (4 `read`, 2 `bash`) and never a
`write` or `edit`. It put its code in the reply, usually within a minute (median 0.9 min), and stopped. Its 28% of
hidden tests passed is what the untouched starter files score.

**Not the harness** ([probe](../../bench/gemma12-probe/README.md)). We captured pi's requests and replayed them with
no grammar or parser in the way. The template and tool declarations it gets are standard Gemma 4, and in 7 of 8
samples it never emitted Gemma's `<|tool_call>` token: it narrated steps as text (`<channel|>read
inventory.py<channel|>edit inventory.py`) and then wrote the code out. llama.cpp made that look worse (its parser
cuts a reply at a stray `<channel|>`, so pi often saw an empty turn), but there was no tool call to lose. The one
real call it wrote used JSON strings (`read{path: "inventory.py"}`), which Gemma 4's format doesn't use.

## What we measured

All on CUDA, 32K context, q8_0 KV cache, flash attention on, greedy, 200 generated tokens:

| Quant | Size | Max layers on GPU | VRAM used | Prompt | Generation | Usable? |
|---|---:|---:|---:|---:|---:|---|
| Q2_K | 4.83 GB | all (99) | 6.3 GB | 103 tok/s | 19.5 tok/s | **no, garbage** |
| **Q4_K_M** | 7.38 GB | **42** | 7.67 GB | 66.9 tok/s | **17.4 tok/s** | yes |
| Q6_K | 9.79 GB | 32 | 7.78 GB | 30.0 tok/s | 7.2 tok/s | yes |
| Q8_0 | 12.67 GB | 24 | 7.75 GB | 12.9 tok/s | 4.5 tok/s | yes |

A dense model has no experts to park in RAM, so the lever is `-ngl`, how many layers live on the GPU; each step up
in quant pushes more layers off and the speed falls off a cliff ([How it fits](../how-it-fits.md)). On our fork on
2026-11-01 it did 13.0–13.8 tok/s and passed the [code check](../../bench/recheck/README.md) (0/500 wrong).

In chat it wrote careful small-scale code (an `is_prime` with edge cases, sensible O(1) LRU caches), though an LRU
`put` sometimes lost its last line. Q2_K loads and runs at 19.5 tok/s and produces `l. l. l.`, CJK noise, or an
empty answer: see the [graveyard](../graveyard.md#gemma-4-12b-coder-at-q2_k-fits-runs-talks-nonsense).
