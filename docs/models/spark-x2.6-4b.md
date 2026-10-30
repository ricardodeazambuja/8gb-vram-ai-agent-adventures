# ⚡ Spark-X2.6-4B — small, fast, and very sure of itself

> **TL;DR**
>
> | | |
> |---|---|
> | Size on disk | 4.07 GiB (`Q8_0`) |
> | Where it lives | everything on the GPU (~7.4 GB used) |
> | Context | **108K** (110,592 tokens) |
> | Generation | **~48–50 tok/s**, second only to the Nemotron Bolt E4B (52) |
> | Prompt processing | ~1,450–1,620 tok/s |
> | [pibench](../pibench.md) | 14/18 solved, 3.3 min median per run |
> | [Vision probe](../vision-probe.md) | not run (text only) |
> | Launch | `bash scripts/start_spark.sh` |

## What is it?

[XHToken's Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B) is a dense 4.1B text
model (Apache-2.0) built for on-device agents. It's a post-training refresh of Spark-X2.5-4B,
same architecture (`Spark2_5ForCausalLM`):

- 36 layers, 16 query heads and 4 KV heads of 256 dimensions, a tied 131K vocabulary.
- Three sliding-window layers (512 tokens) for every full-attention layer, so **only 9 of the
  36 layers keep a long KV cache**.
- A per-head sigmoid gate on the attention output.
- Thinking on by default, with a `reasoning_effort` of `high` or `low`. The card claims a 1M
  context.

## How does it fit in 8 GB?

Easily, which is the fun part. The 4 GB of Q8_0 weights go entirely on the GPU, and because
only 9 layers keep a long KV cache, a q8_0 cache costs about 19 KiB per token. That's what lets a
tiny model carry **108K of context** on an 8 GB card.

Why 108K and not 128K? With a quantized KV cache, flash attention also keeps an f16 copy of one
layer's K and V in its compute buffer, about 5 KiB more per token. At 128K the total is ~7.37 GB,
more than the ~7.2 GB free next to the desktop, so it runs out of memory by ~165 MiB. 108K is
the measured fit, with ~0.3 GB to spare. See [How it fits](../how-it-fits.md).

## Install & run

Spark's architecture is in upstream llama.cpp (since PR #27868, 2026-09-06). The repo runs it on
our llama.cpp fork's `adventures` branch (upstream of 2026-10-28 plus two PRs; see
[getting-started](../getting-started.md#3-build-llamacpp-once-per-flavour)), the same build as
[Dory](dory1.0-4b.md). If you already built it, skip step 1. (First measured on upstream of
2026-10-15; the fork build gives 49.8 tok/s and 1,641 tok/s prompt, 2026-10-30.)

```bash
# 1. Build our llama.cpp (fork, adventures branch, pinned) next to this repo (~20 min)
CUDA_VER=12.2 LLAMA_REPO=https://github.com/ricardodeazambuja/llama.cpp \
  LLAMA_REF=f0456c64d7b891ada1a0a924716ce3244122e92b SRC_DIR=$PWD/../llama.cpp \
  bash scripts/build-llama-cuda.sh

# 2. Download the official Q8_0 GGUF, pinned revision, sha256-checked
bash scripts/download-models.sh spark

# 3. Serve it and open pi
bash scripts/start_spark.sh

# Smaller context, more VRAM headroom:
CTX=65536 bash scripts/start_spark.sh

# One question, no chat:
bash scripts/start_spark.sh -p "explain @README.md" --no-session </dev/null

# Done? Free the GPU:
bash scripts/stop-server.sh
```

**Thinking level:** `start_spark.sh` runs pi with `--thinking high`. pi sends its thinking level
as the request's `reasoning_effort`, and Spark's chat template accepts only `high` or `low`. pi's
default, `medium`, gets every request rejected with HTTP 400 ("Unexpected reasoning effort
medium"). If you launch pi yourself, pass `--thinking high` or `--thinking low`.

## Official numbers

From [XHToken's card](https://huggingface.co/XHToken/Spark-X2.6-4B), measured by XHToken in
thinking mode with the thinking level set to high, **not re-measured here**:

| Benchmark | Spark-X2.6-4B |
|---|---:|
| SWE-bench Verified | 62.8 |
| SWE-bench Pro | 54.3 |
| GPQA Diamond | 77.1 |
| AIME 2026 | 91.2 |
| τ²-bench | 86.9 |
| BFCL v4 | 71.8 |
| IFEval | 93.6 |

Big claims for 4B. Note that its SWE-bench Verified score has no Qwen3.6 number in the same
table, so the comparison on the [scoreboard](../scoreboard.md) is cross-source.

## Our numbers

| Test | Result |
|---|---|
| Generation, short coding prompt | **50.4 tok/s** (re-checked a day later: 48.1) |
| Prompt processing, 16.8K tokens | 1,619 tok/s (re-check: 1,454) |
| Tool call | ✓ |
| Needle at 25% depth in a 104,905-token prompt | **found**, at 791 tok/s prompt / 19.1 tok/s generation |
| Code check (`merge_intervals`, 500 random tests) | correct, 0/500 wrong, but 4,693 tokens of reasoning |
| VRAM | 7.55 GB loaded, 7.66 GB after the tests |

**[pibench](../pibench.md): 14/18 solved, 96% of hidden tests passed, median 3.3 min per run.**
Bug fix 3/3 and login 3/3; the CLI, the feature, the book scraper and the JavaScript page 2/3
each.

## How it behaves

- **The busiest model in the room.** The most tool calls (~19 per run), the most output
  (~9K tokens per run) and by far the most reasoning (~17K thinking characters per run). At
  ~48 tok/s it still finishes a run in a median 3.3 minutes.
- **It checks often, but it checks the wrong thing.** It inspected its own output in more runs
  than any other model, yet all four of its failures were wrong values that looked plausible:
  - **Every book got five stars.** It counted the star icons on each page, but every book shows
    five; the real rating is in a class name (`star-rating Two`) and CSS decides which are lit.
    It didn't find a shop full of five-star books suspicious.
  - **One quote, ten times.** A loop bug (`extract_quote(page)` instead of
    `extract_quote(quote)`) repeated each page's first quote. 30 entries, mostly Albert
    Einstein, done in 67 seconds without a second look.
  - **A spec detail:** it made `--ignore` take one word per flag, although the spec (which it
    copied into its own docstring) says `--ignore WORD [WORD ...]`.
  - **Another:** a restock report not sorted by quantity, as asked.
- **Great at reading code.** On the bug-fix task it found every hidden bug in all three runs.
  Its misses are in specs and data, not in debugging.

## Gotchas

- **The thinking-level trap** (above): `medium` → HTTP 400. Use `start_spark.sh`, or
  `--thinking high`/`low`.
- **128K doesn't fit** next to Q8_0 weights; 108K is the default. `UBATCH=256` only saves
  92 MiB, not enough for 128K. The official Q4_K_M (3.3 GiB) would free ~0.8 GB for full 128K,
  but nobody has tried it here yet.
- **Leftover servers.** If a run is killed or times out, an orphaned llama-server can keep the
  GPU; `bash scripts/stop-server.sh` (or `kill -9` as a last resort).
- **Harmless log noise:** `added <｜end▁of▁sentence｜> logit bias = -inf` concerns the warm-up
  request only.

## Knobs

| Variable | Default | What it does |
|---|---|---|
| `CTX` | 110592 | Context window. 131072 runs out of memory. |
| `KVQUANT` | `q8_0` | KV-cache type. |
| `NGL` | 99 | Layers on the GPU (all of them). |
| `TEMP` / `TOP_P` / `TOP_K` | 1.0 / 0.95 / 0 (off) | Sampling, from the card. |
| `--thinking` (pi flag) | `high` | Must be `high` or `low`. |
