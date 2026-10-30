# 🔥 Recursive Flint — 40 billion parameters, 2.5 billion at a time

> **TL;DR**
>
> | | |
> |---|---|
> | On disk | 18.5 GB (`UD-Q3_K_XL`) |
> | Where it lives | experts of 42 of 49 layers in RAM, everything else on the GPU (7.25 GB VRAM) |
> | Context | **128K** |
> | Generation | 24.3 tok/s |
> | Prompt processing | **542 tok/s** (14.4K-token prompt) |
> | [pibench](../pibench.md) | **18/18 solved** (100% of hidden tests) |
> | Vision | none, text only |
> | Launch | `bash scripts/start_flint.sh` |

## What is it?

Flint is an **agentic-coding Mixture-of-Experts** model from Recursive
([RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint), Apache-2.0). In plain words:

- **39.8B parameters in total, about 2.5B active per token.**
- 49 layers: 2 dense ones, then 47 MoE layers with 256 experts each (6 routed plus 1 shared
  per token).
- Only **14 layers use full attention** (32 query heads, 4 KV heads, head size 128, and no
  positional encoding at all); the other 35 only look at the last 513 tokens.
- 262,144-token context, text in, text out. It's a reasoning model and thinking is **always
  on**: its chat template has no off switch.

Support for it landed in upstream llama.cpp on 2026-10-21 (PR #30928). Unsloth's guide builds
their own fork, but you don't need it for the K-quants used here.

## How does it fit in 8 GB?

About 95% of Flint's weights are experts, and only 6 of 256 experts per layer fire for each
token, so the experts can live in system RAM. With `NCMOE=42`, the experts of the first 42
layers stay in RAM and the last 7 layers' experts go on the GPU. The KV cache is tiny: 14
full-attention layers × 4 KV heads × 128 is about 15 KiB per token at q8_0, so **128K of
context costs ~1.9 GB**.

The fun discovery: **batch size is the real knob, not expert placement.** With experts in
RAM, llama.cpp ships the expert math for big batches to the GPU, so going from batch 512 to
2048 took prompt processing from **192 to 542 tok/s**. The bigger compute buffer costs one
layer of experts, hence `NCMOE=42` instead of 41. See [how-it-fits.md](../how-it-fits.md).

**Why the 3-bit file:** the laptop has 32 GB of RAM with about 20 GB actually available. The
4-bit `UD-Q4_K_XL` is 25 GB and would leave the page cache gasping; `UD-Q3_K_XL` at 18.5 GB is
also what Unsloth recommends for 24 GB machines.

## Install & run

After [getting-started.md](../getting-started.md):

```bash
# 1. The build toolchain env (skip if you already have it)
mamba create -y -n llamacpp-cuda --file envs/llamacpp-cuda.linux-64.lock

# 2. Flint needs a newer llama.cpp than the stock b9606 build: our llama.cpp (fork, adventures
#    branch, pinned; upstream of 2026-10-28) next to this repo, so nothing else gets overwritten
CUDA_VER=12.2 LLAMA_REPO=https://github.com/ricardodeazambuja/llama.cpp \
  LLAMA_REF=f0456c64d7b891ada1a0a924716ce3244122e92b SRC_DIR=$PWD/../llama.cpp \
  bash scripts/build-llama-cuda.sh

# 3. Hugging Face CLI env for the download script (it wants `hf` inside this env)
mamba create -y -n llamacpp -c conda-forge huggingface_hub

# 4. Download + sha256-check (18.5 GB)
bash scripts/download-models.sh flint

# 5. Server + pi, no env vars needed
bash scripts/start_flint.sh

# One-shot
bash scripts/start_flint.sh -p "explain @README.md" --no-session </dev/null

# Stop it (it keeps running after pi exits, so the next start is instant)
bash scripts/stop-server.sh
```

`start_flint.sh` refuses to start if another model is already serving on port 8080, so pi
never talks to the wrong model. Stop the other one first.

## Official numbers

From [Recursive's card](https://huggingface.co/RecursiveSI/flint) (BF16 checkpoint):

| Benchmark | **Flint** | Qwen3.6-35B-A3B (same chart) |
|---|---:|---:|
| SWE-bench Verified | **74.2** | 73.4 |
| Terminal-Bench 2.1 | **54.9** | 44.9 |
| SWE-bench Pro v2 † | **89.7** | 73.8 |
| LiveCodeBench v6 | 80.3 | 80.4 |
| IFBench | **89.2** | 67.9 |
| IFEval | **95.1** | 89.6 |

† measured by Recursive. Their settings: temp 1.0, top-p 0.95, top-k 20; Terminal-Bench with
Terminus 2, SWE-bench with mini-swe-agent, pass@1 averaged over at least five attempts. They
also note their Terminal-Bench score counts answers found by web retrieval as failures, which
other developers may not do. Vendor numbers, full precision, their harness.

## Our numbers

RTX 2070 Max-Q (8 GB), 32 GB RAM, 128K context, q8_0 KV, single runs (±13% noise):

| `NCMOE` | batch | VRAM | Generation | Prompt (14.4K tokens) |
|---|---|---:|---:|---:|
| all experts in RAM | 512 | 4.4 GB | 22.2 tok/s | 168 tok/s |
| 42 | 512 | 6.6 GB | 24.9 tok/s | 186 tok/s |
| 40 | 512 | 7.2 GB | 26.1 tok/s | 192 tok/s |
| 40 | 1024 | 7.4 GB | 24.9 tok/s | 340 tok/s |
| 41 | 2048 | 7.6 GB (~0.2 GB spare) | 25.4 tok/s | 549 tok/s |
| **42** | **2048** | **7.25 GB** | **24.3 tok/s** | **542 tok/s** |

Generation barely moves between rows (that's within noise); prompt speed nearly triples.

At the default settings:

| Test | Result |
|---|---|
| Tool call | well-formed, `finish=tool_calls` |
| Code check (`merge_intervals` vs 500 random inputs) | 1,518 tokens, doctests pass, **0/500 wrong** |
| Needle in a **117,605-token** prompt | **found**; 365 tok/s prompt, 10.8 tok/s generation at that depth |
| pi one-shot through `start_flint.sh` | correct |

**[pibench](../pibench.md):** **18/18 solved**, 100% of hidden tests, median 4.3 min per run,
median 5,099 output tokens. One of only two perfect scores.

## How it behaves

- **Looks before it writes.** On the web tasks it fetches the pages with curl and greps for the
  classes it needs before writing a single line of scraper, about 2.8 probes per run, more than
  any other model.
- **Audits beyond the bug report.** In the bug-fix task it found the hidden bugs nobody
  reported, all three times, and listed them in its summary.
- **Checks content, not just existence.** It re-reads its CSV with `csv.reader`, counts rows,
  checks the raw bytes with `xxd` for encoding problems, asserts every scraped link looks like a
  link, and deletes its scratch files before it leaves.
- **Thorough to a fault.** The same bookshop task took 6.3 minutes in one run and 12.7 in
  another (40 tool calls, including downloading the site's CSS to understand the star ratings).
  Right both times, just with a very different amount of curiosity.
- **Thinks a lot:** ~11.6K characters of reasoning per run, about 1.6× Qwen3.6.

## Gotchas

- **Text only.** No vision projector exists; use another model for screenshots.
- **Thinking can't be turned off.** `enable_thinking: false` is ignored; give it enough tokens.
  (That's why a 64-token "no thinking" test returns an empty answer.)
- **It needs a llama.cpp from after 2026-10-21** (our fork's `adventures` branch here); the
  stock b9606 build doesn't know the architecture.
- The log line `common_fit_params: failed to fit params ... n_gpu_layers already set by user`
  is harmless: auto-fit is skipped because the script sets `-ngl 99`.
- **RAM matters more than VRAM here.** With less than ~20 GB free RAM, the 18.5 GB of experts
  will fight your page cache. See [troubleshooting.md](../troubleshooting.md).

## Knobs

Env vars for `scripts/start_flint.sh` (or `QMODEL=flint` with `start-qwen.sh` / `run-qwen.sh`).

| Knob | Default | What it does |
|---|---|---|
| `NCMOE` | 42 | layers whose experts stay in RAM; 44 frees VRAM if your desktop needs it |
| `BATCH` / `UBATCH` | 2048 / 2048 | 512 saves VRAM but makes prompts ~3× slower |
| `CTX` | 131072 | context window; smaller helps if you don't need it |
| `KVQUANT` | q8_0 | KV cache type |
| `TEMP` / `TOP_P` / `TOP_K` / `MIN_P` | 1.0 / 0.95 / 20 / 0 | sampling, from the card |
