# 💧 LFM3-1.7B — a 1.7B model that holds its own in pibench

> **TL;DR**
>
> | | |
> |---|---|
> | On disk | 1.85 GB (`Q8_0`) |
> | Where it lives | entirely on the GPU, 128K context (3.7 GB VRAM) |
> | Context | **128K** here (the model does 256K) |
> | Generation | **106 tok/s**, the fastest model here |
> | Prompt processing | **4,800 tok/s** (15K-token prompt) |
> | Code check | correct twice, 0/500 wrong, but needs **5.9K–6.9K tokens** (fails at the usual 6K cap) |
> | [pibench](../pibench.md) | **12/18 solved** (88% of hidden tests), 2.2 min median per run |
> | [Security audit](../security.md) | clean |
> | Launch | `QMODEL=lfm3 bash scripts/start-qwen.sh` (needs the newer llama.cpp build, below) |

## What is it?

The first text model of **Liquid AI**'s LFM3 family (Oct 2026, LFM Open License v1.0, the same
licence as [LFM2.5](lfm2.5-24b-a2b.md): commercial use only under $10M revenue). A new
"tri-hybrid" architecture: 24 layers, 13 gated short convolutions, 5 sliding-window attention and
6 global attention, 256K context. Like LFM2.5 it was post-trained inside agent harnesses,
**pi among them**, and it always thinks first. The card claims it beats Qwen3.5 9B on average
(49.4 vs 47.2), with SWE-bench Verified 43.4.

## How does it fit in 8 GB?

Easily: the whole model is 1.85 GB at `Q8_0`, and a 128K q8_0 KV cache is small because only 6 of
its 24 layers keep full-length attention. 3.7 GB in all, so there's room left for a second model
(a speech model, a judge) on the same GPU.

## Install & run

After [getting-started.md](../getting-started.md):

```bash
# 1. A llama.cpp from after 2026-10-27 (LFM3 support: PR #31252): our fork's adventures
#    branch, upstream 8ca4060 (where these numbers were measured) plus two model PRs
CUDA_VER=12.2 LLAMA_REPO=https://github.com/ricardodeazambuja/llama.cpp \
  LLAMA_REF=f0456c64d7b891ada1a0a924716ce3244122e92b SRC_DIR=$PWD/../llama.cpp \
  bash scripts/build-llama-cuda.sh

# 2. Download + sha256-check (1.85 GB), then audit it
bash scripts/download-models.sh lfm3
bash utils/audit-gguf.sh models/lfm3-1.7b/LFM3-1.7B-Q8_0.gguf

# 3. Server + pi
QMODEL=lfm3 bash scripts/start-qwen.sh

# Stop it
bash scripts/stop-server.sh
```

Server only, for any OpenAI-compatible client: `QMODEL=lfm3 bash scripts/run-qwen.sh`.

## Our numbers

RTX 2070 Max-Q (8 GB), q8_0 KV, flash attention on, 128K context, the card's sampling (temp 0.1,
top-k 50, repeat-penalty 1.1). `scripts/qwen-bench.py`, single run.

| Config | VRAM | Generation | Prompt (15K tokens) | Tool call |
|---|---:|---:|---:|---|
| all on the GPU, batch 2048 | 3.7 GB | **105.7 tok/s** | **4,802 tok/s** | ✓ |

Twice [Spark](spark-x2.6-4b.md) (48–50) and [E4B](nemotron-bolt-e4b.md) (52), and a 15K-token
prompt is read in about 3 seconds.

**Code check** (`merge_intervals` vs 500 random inputs, run with `--sandbox`): at the usual
6,000-token cap, **no answer, twice**: both runs were still reasoning when the cap hit. With
`--max-tokens 32768` it was **correct twice**, own doctests pass, 0/500 wrong, in **5,872 and
6,913 tokens**, the most of any model here (Qwen3.6 3.6K, Ornith 1.4–1.8K). At 105 tok/s that is
still about a minute.

**[pibench](../pibench.md): 12/18 solved**, 88% of hidden tests, median **2.2 min** per run. That
is more than its 24B sibling LFM2.5 (10/18) and the 8B Nemotron E4B (7/18), from a model a tenth
of the size. The CLI, the feature and the login were 3/3, the bug fix 2/3 (once it fixed only the
reported bugs and still accepted `5x` and `abc`). Where it fell down was the data it scraped:

- 📚 **books, 0/3.** One run wrote the CSV by hand without quoting, so "1,000 Places to See
  Before You Die" split at the comma; the others got the order wrong or wrote duplicate rows
  (23 and 63 rows for 11 books).
- 🪄 **JavaScript page, 1/3.** It kept the curly quotes it was told to drop, and once left `é` as
  the literal text `é`.

It is also the chattiest model here: a median of 12,160 output tokens per attempt (Gemma 7.1K,
Ornith 2.7K) and up to 113 tool calls in one attempt. At 106 tok/s that still finishes quickly.

## Security audit

`utils/audit-gguf.py --ref` the source repo's chat template, before it was ever loaded
([security.md](../security.md)): sha256 matches the pinned revision, header clean, chat template
identical to the source repo's, no hidden characters, no Jinja internals.

## Gotchas

- Needs a llama.cpp build from after 2026-10-27; older builds don't know the `lfm3` architecture.
- It always thinks, and at length: give it a generous `max_tokens`, or you get reasoning and no
  answer (see the code check: it needs ~7K tokens for a small function).
- Check its data. It runs its code but, like most models here, doesn't read what comes out.

## Knobs

Env vars for `scripts/start-qwen.sh` / `scripts/run-qwen.sh` with `QMODEL=lfm3`.

| Knob | Default | What it does |
|---|---|---|
| `CTX` | 131072 | context window (the model does 256K) |
| `KVQUANT` | q8_0 | KV cache type |
| `TEMP` / `TOP_K` / `REPEAT_PENALTY` | 0.1 / 50 / 1.1 | sampling, from the card |
