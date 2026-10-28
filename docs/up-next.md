# 🔭 Up next

Models and chores waiting their turn. Each model goes through the [security audit](security.md)
before it runs, then gets a [model page](models/) and moves into the README lineup (or the
[graveyard](graveyard.md)).

## In progress

| What | Status |
|---|---|
| [pibench](pibench.md) for [Groot](models/groot-19b-a3b.md) and [LFM2.5](models/lfm2.5-24b-a2b.md) | running (both at `NCMOE=27`, for VRAM headroom) |
| [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) | downloaded, audited clean; its experts need ~18 GB of RAM, so it waits for a session with the browser closed |

## Models to try

### 🧲 EmbeddingGemma 2 (Google, Apache 2.0)

[`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2), GGUF:
[`ggml-org/embeddinggemma-2-GGUF`](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF)
(Q8_0 310 MB + 555 MB vision/audio projector).

Not a chat model: an **embedding** model. Text (incl. code), images, video and audio all land in
one 768-d vector space (Matryoshka: truncate to 512/256/128). 740M parameters, of which the text
part is 270M; 8K context; task prefixes (`task: search result | query: ...`). Google reports MTEB
code 78.7 vs 68.8 for the first EmbeddingGemma.

Why here: it would give pi a **local code / doc search** next to the chat model, and it's small
enough to share the GPU (or run on the CPU). llama.cpp: arch `gemma-embedding2`, already in our
`vendor/flint-llama.cpp` build.

To measure: embeddings/s on CPU and GPU while a big model is loaded; retrieval quality on this
repo (find the file that answers a question) and on the vision-probe images; does Q8_0 match BF16.

### ⚖️ Clef / Clef-Flash (Cloudflare, Apache 2.0)

[`Cloudflare/clef-flash`](https://huggingface.co/Cloudflare/clef-flash) (Qwen3.5-9B backbone) and
[`Cloudflare/clef`](https://huggingface.co/Cloudflare/clef) (Qwen3.8-27B backbone). GGUFs:
[`ggml-org/Clef-Flash-GGUF`](https://huggingface.co/ggml-org/Clef-Flash-GGUF) (Q4_K_M 6.7 GB,
fits on the GPU) and [`ggml-org/Clef-GGUF`](https://huggingface.co/ggml-org/Clef-GGUF)
(Q4_K_M 19.5 GB). A third one, Clef-Omni (Qwen3-Omni-30B-A3B MoE, audio too), has no GGUF yet.

Not a chat model either: a **decision model**. You send a state (text, optionally images) and a
schema of typed questions (pick one of N, a score, yes/no); a small head on top of the backbone
returns a probability for every allowed answer, no text generation. Served by llama-server at
`/v1/systemone` (llama.cpp PR #29831, merged 2026-10-03, text only; in our build). The original
repo needs its own Python (`joint_schema_model.py`); the GGUF route avoids running that.

Why here: it reads the input once and answers, so it's **prompt-speed bound**, not
generation-bound. The dense 27B, hopeless as a chat model here (2.5 tok/s, see the
[graveyard](graveyard.md)), might be usable as a judge. Candidate uses: grading pibench
transcripts, routing a request to the right local model, "is this tool call safe?".

To measure: decisions/s and latency for both sizes; agreement with our own pibench grading.

## Chores

- **Rebuild `vendor/flint-llama.cpp`**: its `llama-server` has an RPATH into the
  `GitStuff/Gemma4` checkout this repo was copied from. The libraries there are byte-identical
  today, but deleting or rebuilding that checkout would break or silently change this one.
- **Groot's default `NCMOE=25`** leaves ~0.45 GB of VRAM; with a browser open it failed to load once
  and ran out of memory on a first request at 26. Consider 26–27 as the default.
