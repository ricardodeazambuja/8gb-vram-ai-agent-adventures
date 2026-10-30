# 🐟 Nemotron-Labs Dory1.0-4B — the model with a "think harder" dial

> **TL;DR**
>
> | | |
> |---|---|
> | Size on disk | 3.67 GB (`Q5_K_M`) |
> | Where it lives | everything on the GPU (5.7 GB used at 64K) |
> | Context | **64K** (128K with `LOOP_KV=last_loop`) |
> | Generation | 29.3 tok/s at 2 loops, 17.7 tok/s at 4 |
> | Prompt processing | 852 tok/s at 2 loops (15.8K-token prompt) |
> | [pibench](../pibench.md) | not run |
> | [Vision probe](../vision-probe.md) | not run (text only) |
> | Launch | `bash scripts/start_dory.sh` |

## What is it?

[NVIDIA-Nemotron-Labs-Dory1.0-4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16)
is a 4B text model NVIDIA released on 2026-10-16 (OpenMDW-1.1 license, English only). It's a
*looped* model: the stack is an input block (8 layers), a recurrent block (20 layers) and an
output block (8 layers), and **the recurrent block runs several times per token**, with the
input fed back in on every loop. Attention alternates between sliding-window (1024 tokens) and
full-attention layers, so most layers keep only a short KV cache. It supports up to 256K
context.

The fun part: **the loop count (1–4) is a runtime setting, not a different model.** More loops
spend more compute per token and, per NVIDIA's card, buy accuracy and need fewer tokens to
solve a task. Same checkpoint, adjustable effort. There's also a *last-loop* KV-cache mode that
keeps one cache per recurrent layer instead of one per loop, so extra loops don't cost extra
KV memory.

## How does it fit in 8 GB?

Size isn't the problem: 3.67 GB. Two settings are.

1. **`-ngl 999`, not 99.** This llama.cpp build expands the loops into *virtual* layers that
   share the same weights (112 at 2 loops, 192 at 4), so the usual `-ngl 99` silently leaves
   weights on the CPU. At 4 loops that kept 2.9 GB of weights and most of the KV cache in
   system RAM: **3.8 tok/s** instead of 19.
2. **Flash attention on, q8_0 KV**, unlike the card's Mac example. That shrinks the compute
   buffer from 1,140 MiB to 93–364 MiB and halves the KV cache, which is what leaves room for
   64K context (or 128K with the last-loop cache).

Both are the defaults here. More on the general tricks in [How it fits](../how-it-fits.md).

## Install & run

No llama.cpp release supports Dory yet. NVIDIA's pull request
[ggml-org/llama.cpp#30306](https://github.com/ggml-org/llama.cpp/pull/30306) does. Our llama.cpp
fork carries it: branch [`adventures`](https://github.com/ricardodeazambuja/llama.cpp/tree/adventures)
is upstream of 2026-10-28 with that PR (at `43e865ec`, the version measured here) merged in, and
it serves the other new models too ([getting-started](../getting-started.md#3-build-llamacpp-once-per-flavour)).
The numbers on this page were first measured on the PR's own base (upstream of 2026-10-15, kept
as `patches/llama.cpp-dory-pr30306.patch`); the fork build gives the same, 29.7 tok/s and
867 tok/s prompt (2026-10-30).

```bash
# 1. Build our llama.cpp (fork, adventures branch, pinned) next to this repo (~20 min)
CUDA_VER=12.2 LLAMA_REPO=https://github.com/ricardodeazambuja/llama.cpp \
  LLAMA_REF=f0456c64d7b891ada1a0a924716ce3244122e92b SRC_DIR=$PWD/../llama.cpp \
  bash scripts/build-llama-cuda.sh

# 2. Download the Q5_K_M GGUF (3.67 GB), pinned revision, sha256-checked
bash scripts/download-models.sh dory

# 3. Serve it and open pi (2 loops, 64K)
bash scripts/start_dory.sh

# Turn the dial: 4 loops is more accurate and ~40% slower
LOOPS=4 bash scripts/start_dory.sh

# 128K context at 2 loops
CTX=131072 LOOP_KV=last_loop bash scripts/start_dory.sh

# One question, no chat:
bash scripts/start_dory.sh -p "explain @README.md" --no-session </dev/null

# Done? Free the GPU:
bash scripts/stop-server.sh
```

The same build also runs [Spark-X2.6-4B](spark-x2.6-4b.md), so you get two models for one
compile.

## Official numbers

From [NVIDIA's card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16),
measured by NVIDIA under its own harness (Terminal-Bench 2.1 with Terminus-2), **not
re-measured here**:

| Benchmark | 1 loop | 2 loops | 4 loops | Qwen3.6-35B-A3B (same table) |
|---|---:|---:|---:|---:|
| SWE-bench Verified | 50.3 | 59.5 | 61.8 | 73.4 (reported) |
| Terminal-Bench 2.1 | 12.5 | 18.1 | 22.2 | 43 |
| GPQA Diamond (no tools) | 69.19 | 72.41 | 72.16 | 85.35 |
| τ³-bench (Banking) | 11.6 | 12.89 | 14.69 | 9.54 |
| IFBench | 74.29 | 77.94 | 79 | 57.44 |

Most of the gain comes from the second loop; NVIDIA recommends 2 as the default and 4 when
accuracy matters more, "particularly for agentic tasks". A high SWE-bench Verified with a low
Terminal-Bench is worth keeping in mind: see the [scoreboard](../scoreboard.md).

## Our numbers

| Loops | Generation | Prompt (15.8K tokens) | Peak VRAM | Tool call |
|---|---:|---:|---:|---|
| **2** (per-loop KV, 64K) | **29.3 tok/s** | 852 tok/s | 5.7 GB | ✓ |
| 4 (last-loop KV, 64K) | 17.7 tok/s | 492 tok/s | 5.1 GB | ✓ |

Short-story speed by loop count: 43 / 31 / 22 / 19 tok/s for 1–4 loops. 128K at 2 loops with
the last-loop cache: 6.1 GB, 30.1 tok/s. From pi, a one-shot "read this script and explain it"
made one `read` call and gave a correct one-sentence answer.

**Did the speed settings hurt the model?** Perplexity on 6K tokens of text, with KL divergence
against the card's own settings:

| Config | Perplexity | KL divergence vs card | Same top token |
|---|---:|---:|---:|
| Card settings (2 loops) | 6.728 | — | — |
| Flash attention on, q8_0 KV | 6.699 | 0.0069 | 96.4% |
| …plus last-loop cache | 6.674 | 0.0157 | 94.4% |
| 1 loop, for scale | 7.138 | 0.0942 | 85.8% |
| 4 loops, last-loop | 6.599 | 0.0371 | 91.3% |

Flash attention and q8_0 change its predictions about 13× less than dropping one loop. One text,
6K tokens: enough to rule out a broken setting, not a benchmark.

**[pibench](../pibench.md) and the [vision probe](../vision-probe.md): not run.** Dory joined
the lineup before the benchmark existed and wasn't in the final run, so there's no measured
quality score for it yet beyond the checks above.

## How it behaves

We only have short smoke tests, so no character study yet. What we saw: thinking is on by
default (llama-server returns it as `reasoning_content`), tool calls work, and it answered the
pi one-shot correctly with one tool call. The loop dial behaves as advertised: more loops,
lower perplexity, lower speed.

## Gotchas

- **`-ngl 999`.** If you launch it by hand, `-ngl 99` quietly puts layers on the CPU. The
  giveaway: VRAM use going *down* as you add loops.
- **Per-loop caches for 3–4 loops don't fit**, so `LOOPS=3/4` switch to the last-loop cache
  automatically.
- **The llama.cpp support is an unmerged PR.** Its own stated limits: no context shifting, no
  loop counts above 4, Metal untested, long-context qualification incomplete.
- With flash attention off you'll see a warning about padding the V cache; it only appears
  then.

## Knobs

| Variable | Default | What it does |
|---|---|---|
| `LOOPS` | 2 | Passes through the recurrent block, 1–4. |
| `LOOP_KV` | `per_loop` (1–2 loops), `last_loop` (3–4) | One KV cache per loop, or one shared cache. |
| `CTX` | 65536 | Context window. 131072 needs `LOOP_KV=last_loop`. |
| `KVQUANT` | `q8_0` | KV-cache type. |
| `NGL` | 999 | Must cover every virtual layer. |
| `FA` | `on` | Flash attention. Off costs a 1.1 GB compute buffer. |
| `TEMP` / `TOP_P` / `TOP_K` | 1.0 / 0.95 / 0 (off) | Sampling, from the card. |
