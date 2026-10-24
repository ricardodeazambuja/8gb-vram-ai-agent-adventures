# 🚀 Gemma 4 12B "Agentic" (fable5 v2) — fully on the GPU, thanks to a squeezed cache

> **TL;DR**
>
> | | |
> |---|---|
> | File | `gemma4-v2-Q4_K_M.gguf`, 6.87 GB |
> | Where it lives | **all 48 layers on the GPU** (with the TurboQuant build) |
> | Context | 16K (25K works only sometimes) |
> | Generation | **~28.3 tok/s** with TurboQuant · ~16.5 tok/s on the stock build |
> | [pibench](../pibench.md) | not benchmarked yet |
> | [Vision probe](../vision-probe.md) | not run (no vision projector set up for it) |
> | Launch | `bash scripts/start-turbo.sh` |

## What is it?

The v2 successor to the [12B Coder](gemma4-12b-coder.md), by the same author:
[`yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF`](https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF).
Same base (`google/gemma-4-12B-it`) and same body: a **dense** 48-layer Gemma 4 with a native
context of 262144 tokens and a custom thinking + tool-call template (needs `--jinja`). The focus
moved from "write code" to **coding plus agentic, terminal-style tool use**.

Quants: Q3_K_M (5.7 GB), **Q4_K_M (6.87 GB, recommended)**, Q6_K (9.11 GB), Q8_0 (11.8 GB). No
Q2_K this time: per the author, it failed their stress tests (and on v1 it was garbage anyway).

Same warning as v1, straight from the card: *"not safety-aligned, reduced refusals"*. Review what
it does.

## How does it fit in 8 GB?

Dense model, so the lever is `-ngl` (layers on the GPU), just like v1. The new trick is the
**TurboQuant `turbo3` V cache**: a ~3.5-bit cache format (Walsh-Hadamard rotated polar
quantization, from Google's TurboQuant research) that shrinks the KV cache enough to keep **all**
48 layers on the GPU. The key cache stays at `q8_0` (keys are sensitive; values tolerate the
squeeze). That's ~1.7× the speed of the partial-offload setup.

`turbo3` is a **runtime** cache option, not a model format, so it works with any Gemma 4 GGUF.
But it isn't in mainline llama.cpp: it needs the
[`TheTom/llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant) fork. The recipe
is the model author's own (~30 tok/s on their RTX 4060). More on the general tricks in
[How it fits](../how-it-fits.md).

## Install & run

```bash
# 1. Build the TurboQuant fork into its own folder (CUDA; leaves vendor/llama.cpp alone)
git clone --depth 1 --branch feature/turboquant-kv-cache \
  https://github.com/TheTom/llama-cpp-turboquant vendor/llama-cpp-turboquant
SRC_DIR=$PWD/vendor/llama-cpp-turboquant LLAMA_REF=feature/turboquant-kv-cache \
  ENV_NAME=llamacpp-cuda bash scripts/build-llama-cuda.sh

# 2. Download the recommended quant
mamba run -n llamacpp hf download yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF \
    gemma4-v2-Q4_K_M.gguf --local-dir models/gemma4-12b-agentic

# 3. Server + pi, with the TurboQuant recipe baked in
bash scripts/start-turbo.sh
bash scripts/start-turbo.sh -p "explain @README.md"   # one-shot (leaves the server up)

# 4. Stop the server when you're done
bash scripts/stop-server.sh
```

`start-turbo.sh` is the zero-config preset (TurboQuant, Q4_K_M, 16K, then pi; it offers to stop
the server when pi exits). Prefer to pick your own settings? `start-agentic.sh` is the same
launcher with every knob open, and `run-12b-agentic.sh` starts just the server:

```bash
TURBO=1 bash scripts/start-agentic.sh                # same as start-turbo, all knobs available
bash scripts/run-12b-agentic.sh                       # stock build, partial offload (no fork needed)
CTX=25000 bash scripts/start-turbo.sh                 # the author's 25K context (marginal on 8 GB)
```

## Official numbers

The author ran one benchmark, honestly labelled as their own local setup: **τ²-bench "telecom"**,
20 tasks, same harness for both models, all Q8_0.

| Model | τ²-bench telecom (author's harness) |
|---|---:|
| base `gemma-4-12B-it` | ~15% |
| **this model (v2)** | **~55%** |

The author also reports the trade-off: on τ²-bench *retail* the base model scores a bit higher,
and on MMLU-Pro this model lands a little below the base. These numbers are not comparable with published
leaderboards (different harness), only with each other. For the base model's official numbers
(Google: MMLU Pro 77.2%, LiveCodeBench v6 72.0%, GPQA Diamond 78.8%), see the
[scoreboard](../scoreboard.md).

## Our numbers

| Mode (Q4_K_M) | Build | Layers on GPU | KV cache (K / V) | Context | VRAM | Generation |
|---|---|---:|---|---:|---:|---:|
| **TurboQuant** | turboquant fork | **48 (all)** | q8_0 / **turbo3** | 16K | 7.76 / 8 GB | **~28.3 tok/s** |
| stock | mainline llama.cpp | 40–42 | q8_0 / q8_0 | 32K | ~7.1 GB | ~16.5 tok/s |

Fully offloaded, it's coherent on thinking and coding: `merge_intervals`, a hash map and string
reversal all came out correct. `q8_0` keys + `turbo3` values were first checked on the v1 coder
(same architecture, head size 256) to make sure the fork's CUDA kernel works on this Turing GPU. It
does.

It hasn't been through [pibench](../pibench.md) or the [vision probe](../vision-probe.md) yet.

## How it behaves

- **Fast for a dense 12B on 8 GB.** Everything on the GPU means no CPU round trips; ~28 tok/s is
  close to the author's ~30 on a newer card.
- **A thinker with a native tool protocol.** Reasoning lands in `reasoning_content`, tool calls
  come out as proper calls (with `--jinja`). Give it a generous `max_tokens`.
- **Can get stuck repeating `0000…`.** The card's fix is a repetition penalty of 1.1, and the
  launcher uses it by default (`REP_PEN`).

## Gotchas

- **The "it's broken!" that wasn't.** Our first download, done with a 16-connection `aria2c` and
  then interrupted, had the right size and a valid header but **wrong tensor data**. It loaded fine
  and printed `<unused32><unused32>…` forever, which looked exactly like a TurboQuant bug. It was
  the file: `sha256sum` against Hugging Face's checksum caught it. **Garbage from a fresh GGUF?
  Check the checksum before you blame the engine.** (`hf download` verifies it for you.)
- **Full offload needs small batches.** `-ngl 99` runs out of memory with llama.cpp's default 4
  parallel slots and 2048 batch (a ~0.5 GB compute buffer). The TurboQuant mode defaults to
  `--parallel 1 -b 512 -ub 512`; leave those alone unless you like OOMs.
- **25K context is on a knife's edge.** It fit once and failed once, depending on how much VRAM the
  desktop happened to use. 16K is the reliable default. For 25K, close other GPU apps or try
  `BATCH=256 UBATCH=256`.
- **Q6_K and Q8_0 don't fit fully**, even with turbo3: their weights alone exceed 8 GB, so they
  fall back to partial offload.
- **MTP drafts in the repo aren't wired in.** Per the card, only one specific llama.cpp build
  loads them cleanly; newer ones crash. The scripts don't use them.
- **pi shows the wrong name.** There's no pi entry for this model, so pi uses the default model id
  (the 26B's). It works, because llama-server serves whatever is loaded; the label is just wrong.

## Knobs

Env vars for `scripts/start-agentic.sh` / `scripts/run-12b-agentic.sh` (`start-turbo.sh` sets the
first three for you; `-h` lists everything):

| Knob | What it does | Default |
|---|---|---|
| `TURBO` | `1` = TurboQuant build + turbo3 V cache + all layers on GPU | 0 (`start-turbo.sh`: 1) |
| `QUANT` | `Q3_K_M` \| `Q4_K_M` \| `Q6_K` \| `Q8_0` | Q4_K_M |
| `CTX` | context window | 16384 with TURBO, else 32768 |
| `NGL` | layers on the GPU | 99 with TURBO, else 40 (per quant) |
| `KVQUANT` / `CTK` / `CTV` | KV cache types (TURBO sets K=q8_0, V=turbo3) | q8_0 |
| `REP_PEN` | repetition penalty (fixes `0000…`) | 1.1 |
| `TEMP` / `TOP_P` / `TOP_K` | sampling (the card's recommendation) | 1.0 / 0.95 / 64 |
| `NP` / `BATCH` / `UBATCH` | memory knobs for full offload | 1 / 512 / 512 |
