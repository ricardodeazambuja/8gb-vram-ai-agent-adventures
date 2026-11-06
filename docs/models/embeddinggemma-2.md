# 🧲 EmbeddingGemma 2 — local search for code and docs

> **TL;DR**
>
> | | |
> |---|---|
> | On disk | 0.31 GB (`Q8_0`), plus a 0.56 GB vision and audio projector we haven't used |
> | Where it lives | entirely on the GPU (1.5 GB), or on the CPU |
> | Output | 768-d vectors, which can be cut to 512, 256 or 128 |
> | Speed | 270–280 short passages/s; 49 doc chunks/s |
> | Repo search | 80% top hit on docs; 93% top hit on code when fused with BM25 |
> | [Security audit](../security.md) | clean |
> | Launch | `scripts/llm eg2` · [`examples/search.py`](../../examples/search.py) |

## What is it?

[`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2), GGUF:
[`ggml-org/embeddinggemma-2-GGUF`](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF)
(Q8_0 310 MB + 555 MB vision/audio projector).

Not a chat model: an **embedding** model. Text (incl. code), images, video and audio all land in
one 768-d vector space (Matryoshka: truncate to 512/256/128). 740M parameters, of which the text
part is 270M; 8K context; task prefixes (`task: search result | query: ...`). Google reports MTEB
code 78.7 vs 68.8 for the first EmbeddingGemma.

Why here: it would give pi a **local code / doc search** next to the chat model, and it's small
enough to share the GPU (or run on the CPU). llama.cpp: arch `gemma-embedding2`; our fork has it.

## Install & run

Build our llama.cpp once: [getting-started.md](../getting-started.md). Then:

```bash
bash scripts/download-models.sh eg2
scripts/llm eg2                         # server on port 8080, 4 slots (it only serves; no pi)
curl -s localhost:8080/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input":["task: search result | query: how do I stop the server?"]}'
```

Use the task prefixes: `task: search result | query: ...` for questions and
`title: none | text: ...` for documents. `python3 bench/retrieval/run_retrieval.py embed KEY` scores
repo search, and `bash bench/smoke/run.sh eg2` is the quick check.
[`examples/search.py`](../../examples/search.py) searches with it. (The building block: `QMODEL=eg2 bash scripts/run-qwen.sh`.)

## Our numbers

Reruns of the quick check are in [`bench/smoke/`](../../bench/smoke/README.md).

**Smoke test (2026-10-28, `vendor/llama.cpp-lfm3`, Q8_0, all on the GPU, 1.5 GB):** 768-d vectors;
5 of 6 toy queries found the right passage among 8 (the miss was a near tie, 0.678 vs 0.674);
256 short passages embedded in 0.9 s (280/s).

**Repo search (2026-10-29, [`bench/retrieval/`](../../bench/retrieval/)):** 30 questions a user
would ask about this repo ("why did my whole terminal window close while a big model was
loading?"), each with the doc(s) that answer it; the corpus is `docs/` split at `## `, 236
chunks. A hit@k means a right file among the k best chunks.

| Search | hit@1 | hit@3 | hit@5 | Speed |
|---|---:|---:|---:|---:|
| BM25 keywords (no model) | 67% | 90% | 97% | — |
| EmbeddingGemma 2, 768 dims | **80%** | 90% | 97% | 49 chunks/s GPU (1.5 GB), 33/s CPU |
| EmbeddingGemma 2, 256 dims | **80%** | 93% | 97% | same vectors, cut |
| EmbeddingGemma 2, 128 dims | 67% | 83% | 87% | |
| 768 dims + BM25, fused (RRF) | **80%** | **100%** | **100%** | |

It finds the meaning where keywords fail ("the agent benchmark" → pibench), and keywords find
what it misses (the literal `<unused32>` token: rank 9 by meaning, 2 fused). Fused, every
question has a right doc in the top 3. Cutting to 256 dims (Matryoshka) costs nothing; 128 does.
The CPU alone is only 1.5× slower than the GPU here (2,000-character chunks), so it can run next
to a big model without taking any VRAM.

**Code search (same probe, `--corpus code`):** 30 questions about the scripts ("which iptables
rules keep the sandbox away from the LAN?", "where is the expert split the user picked
remembered?"), over `scripts/`, `utils/` and `bench/` in 60-line chunks (163), with Gemma's
`code retrieval` query prefix.

| Search | hit@1 | hit@3 | hit@5 |
|---|---:|---:|---:|
| BM25 keywords | 87% | **100%** | **100%** |
| EmbeddingGemma 2, 768 dims | 77% | 90% | 93% |
| EmbeddingGemma 2, 256 dims | 83% | 87% | 93% |
| 256 dims + BM25, fused | **93%** | 97% | **100%** |

Here keywords win alone: code is full of exact names. Its worst miss: "which script runs the
model's code inside a container with no network?" ranked `qwen-codecheck.py` 57th, because that
file says `--network none`, not "no network". Fused, it's the best of both again. Code chunks
embed at 13/s (longer in tokens than prose). A trap we fell into first: the probe's own script
holds the questions, so it matched them word for word; it is left out of the corpus. Still to
do: a bigger, harder question set.

## Gotchas

- Keep the 4 slots. With 1 slot, throughput halves (132 vs 270 passages/s).
- The corpus in `bench/retrieval/` is this repo's live docs and code, so it grows over time and the
  scores above aren't an exact rerun any more. See [Known gaps](../reproduce.md#known-gaps).
