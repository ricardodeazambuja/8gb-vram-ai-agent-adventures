# 🧭 Qwen3.6-35B-A3B — the baseline everyone else gets measured against

> **TL;DR**
>
> | | |
> |---|---|
> | On disk | 17.7 GB (`UD-IQ4_XS`) + 0.9 GB vision projector |
> | Where it lives | experts of 32 of 40 layers in RAM, everything else on the GPU (7.6 GB VRAM) |
> | Context | **128K** |
> | Generation | 22.8 tok/s |
> | Prompt processing | 221 tok/s (15.6K-token prompt) |
> | [pibench](../pibench.md) | 16/18 solved (96% of hidden tests) |
> | [Vision probe](../vision-probe.md) | 100% overall |
> | Launch | `bash scripts/start-qwen.sh` |

## What is it?

Qwen3.6-35B-A3B is the first open-weight Qwen3.6 model from Alibaba's Qwen team. It is a
**Mixture-of-Experts** model: 35B parameters in total, but only about **3B are active per
token**. It has 40 layers, 256 experts per layer (8 routed plus 1 shared per token), a vision
encoder, and a native context of 262,144 tokens.

The clever bit is the layout: three out of every four layers are **Gated DeltaNet** (a form of
linear attention with a fixed-size state and no KV cache), and only one in four uses full
attention. Fewer layers that remember every token means a small KV cache, which is exactly
what an 8 GB card wants to hear.

It is the **baseline** on the [scoreboard](../scoreboard.md): a strong, well-documented coder
that fits 128K here, so every other model gets compared to it.

## How does it fit in 8 GB?

The file is 17.7 GB, which is a polite way of saying "not on your GPU". But only a handful of
experts run for each token, so the expert weights can sit in system RAM while the GPU keeps
the attention layers, the shared parts and the KV cache. `NCMOE=32` keeps the experts of 32
of the 40 layers in RAM and puts the other 8 on the GPU, for 7.6 GB of VRAM. Because only 10
of the 40 layers keep a KV cache (2 KV heads × 256), 128K of context at q8_0 costs just
1.42 GB. The general idea is explained in [how-it-fits.md](../how-it-fits.md).

## Install & run

You need the basics from [getting-started.md](../getting-started.md) first (mamba, an NVIDIA
driver, pi). Then:

```bash
# 1. The pinned build toolchain (CUDA 12.2, gcc 12) as a conda env
mamba create -y -n llamacpp-cuda --file envs/llamacpp-cuda.linux-64.lock

# 2. Build llama.cpp (release b9606) into vendor/llama.cpp, ~10–20 min
CUDA_VER=12.2 LLAMA_REF=b9606 bash scripts/build-llama-cuda.sh

# 3. A small env with the Hugging Face CLI (the download script insists the `hf` CLI
#    lives in this env, not just somewhere on your PATH)
mamba create -y -n llamacpp -c conda-forge huggingface_hub

# 4. Download + sha256-check the model and its vision projector (~18.6 GB)
bash scripts/download-models.sh qwen36

# 5. Server + pi in one go (qwen36 is start-qwen.sh's default)
bash scripts/start-qwen.sh

# ...with images enabled (projector runs on the CPU by default)
bash scripts/start-qwen.sh --image

# One-shot question, leaves the server up
bash scripts/start-qwen.sh -p "explain @README.md"

# Done for the day
bash scripts/stop-server.sh
```

Prefer your own client? `QMODEL=qwen36 bash scripts/run-qwen.sh` starts just the server, an
OpenAI-compatible API on `http://127.0.0.1:8080/v1`.

## Official numbers

From [Qwen's own model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), same harness for
both columns:

| | Gemma4-26B-A4B | **Qwen3.6-35B-A3B** |
|---|---:|---:|
| SWE-bench Verified | 17.4 | **73.4** |
| SWE-bench Pro | 13.8 | **49.5** |
| Terminal-Bench 2.0 | 34.2 | **51.5** |
| MCPMark | 14.2 | **37.0** |
| LiveCodeBench v6 | 77.1 | **80.4** |
| GPQA (Qwen's card) | 82.3 | **86.0** |
| τ³-bench | 59.0 | **67.2** |

Caveats, because there are always caveats: these are Qwen's numbers, run with Qwen's agent
scaffold at 200K context, on the full-precision model. On Terminal-Bench 2.1, other vendors
put it at 43 (NVIDIA's table) and 44.9 (Recursive's chart). And Gemma 4 26B scores 57.4 on
SWE-bench Verified in NVIDIA's table, so harnesses matter a lot.

## Our numbers

Measured on the RTX 2070 Max-Q (8 GB) with 32 GB RAM, q8_0 KV, flash attention on, sampling
from the card (temp 0.6, top-p 0.95, top-k 20). Single runs; this laptop's run-to-run noise is
about ±13%.

| Config | VRAM | Generation | Prompt (15.6K tokens) | Tool call |
|---|---:|---:|---:|---|
| `--cpu-moe` (all experts in RAM), 128K | 4.5 GB | 15.7 tok/s ¹ | — | ✓ |
| **`NCMOE=32`, 128K (default)** | **7.6 GB** | **22.8 tok/s** | **221 tok/s** | ✓ |

¹ measured while a 50 GB download was hogging the machine.

**Code check** (`merge_intervals`, tested against 500 random inputs): correct, 0/500 wrong,
but it took **3,558 tokens** to get there, and one of its own five doctests had a wrong
expected value. It thinks. A lot.

**[pibench](../pibench.md):** 16/18 solved, 96% of hidden tests passed, median 3.4 min per
run, median 3,480 output tokens. Both misses are described below.

**[Vision probe](../vision-probe.md):** 100% overall: perfect OCR, perfect understanding and
CAD reading, and boxes that land right on target, just not in the format you asked for (7% in
pixels as requested, 100% in its own 0–1000 format). Median 44 s per image, most of it the
projector chewing on the CPU.

## How it behaves

- **Fast and direct.** About 14 tool calls and 3.8K output tokens per pibench run, fewer than
  Flint. On code tasks it reads, edits, writes tests, runs pytest, done.
- **Writes scrapers from memory.** On the JavaScript quotes page it never looked at the HTML,
  guessed the author selector (`span.small.text` instead of `small.author`), and every author
  came out as an empty string. Its check confirmed there were 30 items with the right keys,
  which there were, and it declared victory after 74 seconds. Confident and shallow.
- **Fixes what you report, not always what's hiding.** In one of three bug-fix runs it fixed
  both reported bugs and missed the hidden one (leftover text after the last unit).
- **Touches things you didn't ask it to.** Once it overwrote the provided `sample.txt` with its
  own text before testing. Harmless here; in a real repo, an unrequested change.
- **The rest was clean:** rounds 2 and 3 had only that one bug-fix miss.

## Gotchas

- It thinks for a long time. Short `max_tokens` limits end in `finish=length` with no answer.
- `NCMOE=32` leaves only ~0.5 GB of VRAM headroom. If your desktop is greedy, use `NCMOE=34`.
- Box coordinates come back as `bbox_2d` on a 0–1000 grid, whatever you ask for. Convert.
- "Killed" right after a clean load is the host-RAM OOM killer, not VRAM. See
  [troubleshooting.md](../troubleshooting.md).

## Knobs

All are env vars for `scripts/start-qwen.sh` / `scripts/run-qwen.sh` (`-h` lists everything).

| Knob | Default | What it does |
|---|---|---|
| `NCMOE` | 32 | layers whose experts stay in RAM; lower = faster, more VRAM; empty = all in RAM (~4.5 GB VRAM) |
| `CTX` | 131072 | context window |
| `KVQUANT` | q8_0 | KV cache type |
| `TEMP` / `TOP_P` / `TOP_K` | 0.6 / 0.95 / 20 | sampling, from the card |
| `--image` | off | load the vision projector |
| `MMPROJ_GPU` | 0 | 1 = projector on the GPU (needs VRAM you don't really have) |
