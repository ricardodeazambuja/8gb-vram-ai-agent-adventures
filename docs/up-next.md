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

### 🧢 ThinkingCap fine-tunes (bottlecap.ai, Apache 2.0)

[bottlecapai](https://huggingface.co/bottlecapai) post-trains popular models to **think less
for the same answers**. On their Gemma 4 26B card: 39% fewer thinking tokens on average (14–69%
by benchmark), accuracy 75.0 vs the base's 75.2, and far fewer runaway traces (truncated
4.35% → 0.72%, looping 3.64% → 0.76%). At 20–40 tok/s, a third fewer tokens is a third less
waiting. Weights only: every other file matches the base model.

| Model | GGUF that fits | Fit here | Why test it |
|---|---|---|---|
| [ThinkingCap gemma-4-26B-A4B-it](https://huggingface.co/bottlecapai/ThinkingCap-gemma-4-26B-A4B-it-GGUF) | IQ4_XS 13.6 GB (Q4_K_M 17.0) + mmproj 1.2 | like our [Gemma 4 26B](models/gemma4-26b-a4b.md) | **First pick.** A clean A/B against a model we know well (18/18 on pibench): same score, shorter jobs? Ours is Google's QAT build (UD-Q4_K_XL); theirs starts from the regular one, so compare IQ4_XS vs QAT and keep that in mind. |
| [ThinkingCap Nemotron-3.5-Lightning-30B-A3B](https://huggingface.co/bottlecapai/ThinkingCap-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) | Q4_K_M 25.4 GB (only that or f16) | experts in RAM; bigger than Ornith, so browser closed | A new family for the lineup; LFM2.5's card benchmarks against its base. |
| [ThinkingCap Qwen3.8-27B RCO-4.2bpw](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B-GGUF-RCO-4.2bpw) | 14.3 GB + mmproj 0.9 | dense 27B, mostly on the CPU | Low priority: the plain 27B ran at 2.5 tok/s ([graveyard](graveyard.md)); shorter thinking won't rescue that. Check that llama.cpp reads RCO first. |
| ThinkingCap Qwen3.8-Flash-Next | ≥ 100 GB at IQ4_XS | no | Same reason as the base in the graveyard. |

To measure (Gemma first): thinking tokens and wall time per pibench job against the base, plus
the code check's token count; same sampling for both.

### 💧 LFM3 family (Liquid AI, LFM Open License v1.0)

The [LFM3 collection](https://huggingface.co/collections/LiquidAI/lfm3) (Oct 2026): small
models (1.7–2.2B) on a new "tri-hybrid" architecture (24 layers: 13 gated short convolutions,
5 sliding-window attention, 6 global attention), official GGUFs for all of them, ~1.1 GB at
Q4_K_M. Same license as [LFM2.5](models/lfm2.5-24b-a2b.md): commercial use only below $10M
annual revenue.

**Needs a new llama.cpp build:** support was merged on 2026-10-27
([#31252](https://github.com/ggml-org/llama.cpp/pull/31252) LFM3-1.7B + VL,
[#31257](https://github.com/ggml-org/llama.cpp/pull/31257) Audio + Omni); our
`vendor/flint-llama.cpp` (Oct 21) has no `lfm3` arch. Video input is still an open PR (#31283).
One rebuild covers this and the RPATH chore below.

| Model | What it is | Why test it |
|---|---|---|
| [**LFM3-1.7B**](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF) | text, 256K context, post-trained in agent harnesses **including pi** | **First pick.** Its card claims a 1.7B beats Qwen3.5 9B on average (49.4 vs 47.2), with SWE-bench Verified 43.4 and LiveClaw 45.6. Entirely on the GPU with room to spare, so likely the fastest model here; does it get anywhere on pibench, where E4B got 7/18 and Spark 14/18? Also a `QAD-Q4_0` file, like LFM2.5's. |
| [LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF) | vision (images; video not in llama.cpp yet), thinks first, object grounding | The [vision probe](vision-probe.md) at 1.1 GB + 0.2 GB projector: how close to Qwen3-VL-8B's 89%? |
| [LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF) / [LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF) | speech in and out (18 languages), tool calls from voice; Omni adds images and video | Outside what we measure today; a voice front-end for pi would be a new chore. Later. |
| [d1-2B](https://huggingface.co/LiquidAI/d1-2B-GGUF) / [d1-Omni-2B](https://huggingface.co/LiquidAI/d1-Omni-2B-GGUF) | **decision models** like Clef: calibrated answers in one pass, plus spans, boxes and quantity intervals | A 1.1 GB alternative to Clef-Flash; test them side by side. The card doesn't say how llama.cpp serves it (`/v1/systemone` like Clef?): check before downloading. |

## Chores

- **Rebuild `vendor/flint-llama.cpp` at a commit after 2026-10-27** (LFM3 support), then re-run
  the speed tests of the models it serves to check nothing moved. Also fixes: its `llama-server` has an RPATH into the
  `GitStuff/Gemma4` checkout this repo was copied from. The libraries there are byte-identical
  today, but deleting or rebuilding that checkout would break or silently change this one.
- **Groot's default `NCMOE=25`** leaves ~0.45 GB of VRAM; with a browser open it failed to load once
  and ran out of memory on a first request at 26. Consider 26–27 as the default.
