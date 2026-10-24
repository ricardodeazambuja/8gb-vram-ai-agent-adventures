# 🧑‍💻 Gemma 4 12B "Coder" (fable5 v1) — the dense one that needs a different trick

> **TL;DR**
>
> | | |
> |---|---|
> | File | `gemma4-coding-Q4_K_M.gguf`, 7.38 GB |
> | Where it lives | 42 of 48 layers on the GPU, the last 6 in system RAM |
> | Context | 32K (q8_0 KV cache) |
> | Generation | **~17 tok/s** |
> | Prompt processing | ~67 tok/s (llama-bench) |
> | [pibench](../pibench.md) | not benchmarked yet |
> | [Vision probe](../vision-probe.md) | not run (no vision projector set up for it) |
> | Launch | `bash scripts/run-12b-coder.sh`, then `bash scripts/run-pi.sh` |

## What is it?

A coding fine-tune of Google's `gemma-4-12B-it` by yuxinlu1
([`yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF`](https://huggingface.co/yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF)),
license `gemma`. About 11.9B parameters, **dense** (every parameter works on every token), 48
layers, a native context of 131072 tokens, and a custom chat template with a thinking channel and
tool calls (it needs `--jinja`). The repo holds only GGUF files and a README: nothing executable.

Fair warning from its own card: *"not safety-aligned, reduced refusals"*. It's a task-focused
coder. Read the code it writes before you run it (you were going to anyway, right?).

It has since been succeeded by the [v2 "agentic" model](gemma4-12b-agentic.md), which is faster
on this laptop thanks to a fancier KV cache.

## How does it fit in 8 GB?

The MoE trick from the [26B](gemma4-26b-a4b.md) does nothing here: a dense model has no experts to
park in RAM. Instead the lever is `-ngl`, **how many of the 48 layers live on the GPU**; the rest
run on the CPU. At Q4_K_M (7.38 GB of weights) the measured maximum is **42 layers** at 32K
context with a q8_0 KV cache; 44 runs out of memory. Gemma's sliding-window attention keeps the KV
cache small (~0.5 GB at 32K), which is what makes 32K possible at all. The model card says
8 GB / Q4_K_M is "tight, ~2–4K context", but that assumes everything on the GPU; spilling 6 layers
to RAM gets you 32K. More in [How it fits](../how-it-fits.md).

## Install & run

You need the standard CUDA build of llama.cpp in `vendor/llama.cpp` (from
`BACKEND=cuda bash scripts/setup.sh` or `bash scripts/build-llama-cuda.sh`; see
[Getting started](../getting-started.md)).

```bash
# 1. Download the recommended quant (hf checks the SHA-256 for you)
mamba run -n llamacpp hf download yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF \
    gemma4-coding-Q4_K_M.gguf --local-dir models/gemma4-12b-coder

# 2. Start the server (Q4_K_M, 40 layers on the GPU, 32K, q8_0 KV, port 8080)
bash scripts/run-12b-coder.sh

# 3. In another terminal: pi
bash scripts/run-pi.sh                                  # interactive
bash scripts/run-pi.sh -p "explain this code @file.py"  # one-shot

# 4. Done? Stop the server
bash scripts/stop-server.sh
```

Variations:

```bash
NGL=42 bash scripts/run-12b-coder.sh                            # the measured max: ~17 tok/s, tighter VRAM
TEMP=0 bash scripts/run-12b-coder.sh                            # greedy, deterministic coding
CTX=65536 KVQUANT=q4_0 NGL=38 bash scripts/run-12b-coder.sh     # longer context
QUANT=Q6_K bash scripts/run-12b-coder.sh                        # more fidelity, much slower
```

The script defaults to 40 layers rather than 42 on purpose: it leaves some headroom so a VRAM
spike from your desktop doesn't knock over an unattended server.

## Official numbers

The fine-tune's card has no benchmark table of its own. For the **base model**, Google's card lists
Gemma 4 12B at MMLU Pro 77.2%, LiveCodeBench v6 72.0%, GPQA Diamond 78.8% and τ²-bench 69.0%;
another vendor's table gives the base SWE-bench Verified 44.2. Those are the base model's numbers,
not this fine-tune's. Details on the [scoreboard](../scoreboard.md).

## Our numbers

All on CUDA, 32K context, q8_0 KV cache, flash attention on, greedy, 200 generated tokens:

| Quant | Size | Max layers on GPU | VRAM used | Prompt | Generation | Usable? |
|---|---:|---:|---:|---:|---:|---|
| Q2_K | 4.83 GB | all (99) | 6.3 GB | 103 tok/s | 19.5 tok/s | **no, garbage** |
| **Q4_K_M** | 7.38 GB | **42** | 7.67 GB | 66.9 tok/s | **17.4 tok/s** | **yes, excellent** |
| Q6_K | 9.79 GB | 32 | 7.78 GB | 30.0 tok/s | 7.2 tok/s | yes |
| Q8_0 | 12.67 GB | 24 | 7.75 GB | 12.9 tok/s | 4.5 tok/s | yes |

"VRAM used" includes ~0.5 GB of desktop. Q4_K_M is the clear winner: only 6 layers spill to RAM.
Each step up in quality pushes more layers off the GPU, and the speed falls off a cliff.

It hasn't been through [pibench](../pibench.md) or the [vision probe](../vision-probe.md) yet.

## How it behaves

- **Good, careful code.** Spot checks at Q4_K_M were correct: an `is_prime` with edge cases, a
  square-root bound and even-number skipping, correct one-liners, and sensible O(1) LRU cache
  designs (hash map plus a doubly linked list with dummy head and tail, or an `OrderedDict`).
- **…but not infallible.** Across runs, the LRU cache's `put` method sometimes lost its final
  insertion line. A small model, so review and run what it writes.
- **Tool calls work.** Through `/v1/chat/completions` with tools (the path pi uses) it returned a
  clean `tool_calls` response with a well-formed call. Good enough to drive an agent.
- **A thinker.** Its reasoning goes to `reasoning_content`; give it a generous `max_tokens`.

## Gotchas

- **Skip Q2_K.** It loads, it's fast, and it produces `l. l. l.`, CJK noise, or rambling with an
  empty answer. The launcher refuses it. Q4_K_M and up are fine.
- **Always talk to it through the chat template.** A raw prompt without the chat structure drives
  even Q4_K_M into garbage. That's expected for an instruct model, not a bug. pi already uses the
  right path.
- **Don't reuse the 26B launcher.** `--cpu-moe` does nothing on a dense model and `-ngl 99` runs
  out of memory. That's why it has its own script.
- **pi shows the wrong name.** There's no pi entry for this model, so `run-pi.sh` uses the default
  model id (the 26B's). It still works, because llama-server serves whatever model is loaded; the
  label in pi is just wrong. `MODEL_ID=gemma-4-12b-coder` won't help unless you add a matching
  entry to `config/pi-provider.json`.
- **Context sync.** pi's context window comes from `config/pi-provider.json` (via
  `scripts/configure-pi.sh`), not from the server. Keep them consistent if you change `CTX`.

## Knobs

Env vars for `scripts/run-12b-coder.sh` (`-h` lists them all):

| Knob | What it does | Default |
|---|---|---|
| `QUANT` | `Q4_K_M` \| `Q6_K` \| `Q8_0` | Q4_K_M |
| `NGL` | layers on the GPU (the main lever) | 40 / 30 / 22 per quant |
| `CTX` | context window | 32768 |
| `KVQUANT` | KV cache type (`q4_0` for ~2× context) | q8_0 |
| `TEMP` / `TOP_P` / `TOP_K` | sampling (the card's recommendation) | 1.0 / 0.95 / 64 |
| `BACKEND` | `cuda` \| `vulkan` | CUDA if built |
| `MODEL_DIR` | where the GGUFs live | `models/gemma4-12b-coder` |
