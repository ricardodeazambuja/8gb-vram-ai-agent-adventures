# 🐇 Nemotron 3.5 Bolt E4B-4GB — fast, and it knows it

> **TL;DR**
>
> | | |
> |---|---|
> | File | NVIDIA's own Q4_K_M GGUF, 6.2 GB |
> | Where it lives | Entirely on the GPU (4.4 GB VRAM at 64K) |
> | Context | 64K |
> | Speed | **52 tok/s** generating, **1,531 tok/s** reading a prompt |
> | pibench | **7/18** (78 % of hidden tests) |
> | Vision probe | text only |
> | Launcher | `bash scripts/start_e4b.sh` |

## What is it?

The small sibling of [Nemotron 3.5 Bolt 74B](nemotron-bolt-74b.md), and a completely different
animal: NVIDIA's fine-tune of Google's Gemma 4 E4B, 8B parameters (4B effective), text in and text
out, 64K context, with switchable reasoning. NVIDIA built it for **local voice and device
assistants, interactive personas and embedded interfaces**, not for coding. License:
[OpenMDW 1.1](https://github.com/OpenMDW/OpenMDW/blob/main/1.1/LICENSE.OpenMDW-1.1).

## How does it fit in 8 GB?

It just does. The 6.2 GB file loads entirely onto the GPU and the whole thing, 64K context
included, uses 4.4 GB of VRAM. No experts in RAM, no tricks, no drama. That's also why it's fast:
nothing waits on the CPU.

## Install & run

```bash
# once: our llama.cpp (fork, adventures branch; the same one Flint and Bolt use; see getting-started)
CUDA_VER=12.2 LLAMA_REPO=https://github.com/ricardodeazambuja/llama.cpp \
  LLAMA_REF=3cb2992b39cbcc7a6fd90002a39f694f4bc03a73 SRC_DIR=$PWD/../llama.cpp \
  bash scripts/build-llama-cuda.sh

# once: 6.2 GB, pinned and sha256-checked
bash scripts/download-models.sh e4b

# go
bash scripts/start_e4b.sh                      # server + pi
bash scripts/start_e4b.sh -p "explain @README.md" --no-session </dev/null
bash scripts/stop-server.sh
```

## Official numbers

From [NVIDIA's model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB), the Q4_K_M
column (the file we run) against the Q4_K_M versions of its rivals. NVIDIA's harness; reasoning
on except where noted. A selection:

| Benchmark | **Bolt E4B (Q4_K_M)** | Gemma 4 E4B (Q4_K_M) | Qwen3.5 4B (Q4_K_M) |
|---|---:|---:|---:|
| BFCL-v3 (tool calling) | 67.31 | 65.03 | **70.56** |
| τ²-bench | 52.32 | 38.72 | **61.74** |
| ToolSandbox | 74.13 | 68.31 | **74.51** |
| IFBench | 64.00 | 42.67 | **66.67** |
| MultiChallenge | **55.60** | 39.36 | 52.23 |
| RoleBench | **71.21** | 60.51 | 52.35 |
| LongBench v2 (48K) | 48.67 | 45.70 | **58.28** |

Notice what's missing: no coding benchmarks at all. That's a hint about what it's for.

## Our numbers

| | |
|---|---|
| VRAM at 64K | 4.3 GB loaded, 4.4 GB after a 15K-token prompt |
| Generation | 52.0 tok/s (the fastest model in this repo) |
| Prompt speed | 1,531 tok/s (15.8K tokens in 10 s) |
| Tool calls | work |
| Code check | answered without a code block, so there was nothing to run |

**[pibench](../pibench.md): 7/18, last place.** Bug fix 1/3, CLI 0/3, feature 3/3, login 1/3,
books 0/3, JS page 2/3. Median 2.0 minutes per attempt (the quickest), one timeout.

## How it compares

Against the other small models that sit entirely on the GPU (our measurements; full tables on
the [scoreboard](../scoreboard.md)):

| | **Bolt E4B** | Spark-X2.6-4B | Dory1.0-4B | Qwen3-VL-8B |
|---|---:|---:|---:|---:|
| Size | 8B (4B effective) | 4B | 4B, looped | 8B |
| VRAM | **4.4 GB** | 7.6 GB | 5.7 GB | 7.4 GB |
| Context | 64K | **108K** | 64K | 16K |
| Generation | **52 t/s** | 48–50 t/s | 29 t/s | 38 t/s |
| Prompt speed | 1,531 t/s | **1,450–1,620 t/s** | 852 t/s | 830–900 t/s |
| pibench | 7/18 | **14/18** | not run | not run |
| Vision | text only | text only | text only | **89 %** |

Spark is nearly as fast and solves twice as many chores; for agent coding on a small model, use
Spark. The E4B's edge is the VRAM it leaves free.

## How it behaves

- **Declares victory on empty output.** It wrote `{}` to `authors.json`, opened it, and announced
  "The task is complete". Twice it delivered a CSV with only the header row. This is the purest
  form of the pattern every failing model showed: the evidence was on screen, unread.
- **Skims the spec.** The CLI spec says `'quoted'` counts as the word `quoted`; it kept the
  apostrophes all three times and never tested that case. It kept curly quotes it was told to strip.
- **Fumbles the tools:** 5.5 tool errors per attempt, the most of any model, mostly `edit` calls
  whose text didn't match the file and malformed tool calls.
- **Inconsistent.** It solved the bug fix perfectly, hidden bugs included, in under four minutes,
  then missed hidden bugs the next two times. It failed the login chore twice, then solved it in
  54 seconds. Once it finished a task early and kept going until the 20-minute limit.
- **Fast enough that retries are cheap.** At 52 tok/s an attempt takes a couple of minutes; a human
  checking its work closely would still be ahead.

## Gotchas

- **It's not a coding model.** Use it for what NVIDIA built it for: snappy assistants, tool calls
  in a short conversation, personas. For agent coding, the bigger models in this repo are far more
  dependable.
- **Text only**, despite the Gemma 4 base.
- **Reasoning control** (`enable_thinking`) is a chat-template option; pi can't switch it, so it
  thinks by default.
- NVIDIA also publishes a DFlash drafter for it; we haven't tried it.

## Knobs

| Variable | Default | What it does |
|---|---|---|
| `CTX` | 65536 | Context window (its maximum). |
| `KVQUANT` | q8_0 | KV cache type; there's VRAM to spare for f16 if you like. |
| `TEMP` / `TOP_P` | 1.0 / 0.95 | NVIDIA's recommended sampling. |
