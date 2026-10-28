# 🔭 Up next

Models and chores waiting their turn. Each model goes through the [security audit](security.md)
before it runs, then gets a [model page](models/) and moves into the README lineup (or the
[graveyard](graveyard.md)).

## In progress

| What | Status |
|---|---|
| LFM3-VL-2B | vision probe **91%** at 9 s per image, the best boxes yet (IoU 0.93–0.94 in 0–1000 format); [page](models/lfm3-vl-2b.md) |
| Clef-Flash, d1-2B, EmbeddingGemma 2 | first smoke tests passed (below); the real measurements are still to do |

Done: pibench for [Groot](models/groot-19b-a3b.md) (18/18) and [LFM2.5](models/lfm2.5-24b-a2b.md) (10/18);
[Ornith](models/ornith-1.5-35b-a3b.md) (14/18, vision 97%) and
[ThinkingCap Gemma](models/thinkingcap-gemma4-26b-a4b.md) (17/18 with 47% less thinking) and [LFM3-1.7B](models/lfm3-1.7b.md) (12/18 at 106 tok/s) have pages.

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

**Smoke test (2026-10-28, `vendor/llama.cpp-lfm3`, Q8_0, all on the GPU, 1.5 GB):** 768-d vectors;
5 of 6 toy queries found the right passage among 8 (the miss was a near tie, 0.678 vs 0.674);
256 short passages embedded in 0.9 s (280/s).

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

**Smoke test (2026-10-28, `/v1/systemone` on `vendor/llama.cpp-lfm3`, all on the GPU):** 12
labelled requests, routing a customer message (billing / shipping / technical), "is the customer
angry?", and "is this shell command safe for an agent to run?" (`rm -rf ~/`, `curl … | sh`,
`tar ~/.ssh | curl -T`, against `ls`, `git diff`, `grep`):

| Model | VRAM | Right | Angry? | Median latency | Confidence on the right answer |
|---|---:|---:|---:|---:|---:|
| Clef-Flash Q4_K_M (9B backbone) | 7.0 GB | 12/12 | 4/4 | 183 ms | 0.84–0.98 |
| [d1-2B](https://huggingface.co/LiquidAI/d1-2B-GGUF) Q8_0 | 2.6 GB | 12/12 | 4/4 | **27 ms** | 0.94–1.00 |

Both perfect on easy cases; d1 is 7× faster in a third of the memory. Next: harder, ambiguous
items, and the pibench-grading agreement test.

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
| [LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF) / [LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF) | speech in and out (18 languages), tool calls from voice; Omni adds images and video | The speech-native option in [Voice agents](#-voice-agents-agents-that-talk-to-people) below. |
| [d1-2B](https://huggingface.co/LiquidAI/d1-2B-GGUF) / [d1-Omni-2B](https://huggingface.co/LiquidAI/d1-Omni-2B-GGUF) | **decision models** like Clef: calibrated answers in one pass, plus spans, boxes and quantity intervals | A 1.1 GB alternative to Clef-Flash; test them side by side. The card doesn't say how llama.cpp serves it (`/v1/systemone` like Clef?): check before downloading. |

## 🗣️ Voice agents: agents that talk to people

Goal: a local agent you can talk to, which answers out loud and can still use tools. Two ways
to build one, and one hybrid. Checked on 2026-10-28 against the GitHub and Hugging Face APIs
(licences, last release); nothing here has been downloaded or run yet, and every piece goes
through the [security routine](security.md) first.

The constraint is the GPU: a good agent model (Groot, Flint, Gemma) already takes ~7 GB of the
8, so in a pipeline the speech parts run on the CPU, or the agent model gives up some layers.

### A. Pipeline: speech to text → our agent model → text to speech

The agent stays a model we've already measured on [pibench](pibench.md); speech is added around it.

| Piece | Candidate | Licence | Status (2026-10-28) | Notes |
|---|---|---|---|---|
| Speech detection | [Silero VAD](https://github.com/snakers4/silero-vad) | MIT | v6.2.3 (Sep 2026) | Tiny, CPU. Says when someone is speaking. |
| End-of-turn detection | [LiveKit turn-detector](https://huggingface.co/livekit/turn-detector) | LiveKit model licence | Feb 2026 | Guesses when the person has *finished*, not just paused. **Licensed for use with LiveKit Agents only.** |
| Speech to text | [whisper.cpp](https://github.com/ggml-org/whisper.cpp) | MIT | v1.9.5 (Oct 2026) | Same ggml family as our llama.cpp builds; CUDA or CPU. |
| | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT | v1.2.1 (Oct 2025) | Python/CTranslate2; what most frameworks use by default. |
| | [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | CC-BY-4.0 | Aug 2026 | Very fast and accurate; NeMo stack. |
| | [Moonshine](https://github.com/moonshine-ai/moonshine) | MIT (streaming and English models) | v0.1.5 (Aug 2026) | Built for streaming on small devices; some legacy non-English models are non-commercial. |
| Text to speech | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | Apache 2.0 | model Apr 2025 | 82M parameters, real time on a CPU, good quality. |
| | [Piper](https://github.com/OHF-Voice/piper1-gpl) | **GPL-3.0** | v1.8.0 (Sep 2026) | The fastest on a CPU. Moved from the archived MIT `rhasspy/piper` to a GPL repo. |
| | [Chatterbox](https://github.com/resemble-ai/chatterbox) | MIT | model Jun 2026 | More expressive, voice cloning; wants the GPU. |
| | [Orpheus 3B](https://github.com/canopyai/Orpheus-TTS) | Apache 2.0 | Dec 2025 | An LLM that speaks; 3B is too much next to an agent model on 8 GB. |
| Glue | [**Pipecat**](https://github.com/pipecat-ai/pipecat) | BSD-2 | v1.14.0 (Oct 24, 2026) | Pipelines of VAD, speech to text, LLM and text to speech, with interruptions and tool calls; takes any OpenAI-compatible server. |
| | [**LiveKit Agents**](https://github.com/livekit/agents) | Apache 2.0 | 1.8.9 (Oct 24, 2026) | The same idea, built around WebRTC rooms; brings the turn detector. |
| | [Vocode](https://github.com/vocodedev/vocode-core) | MIT | idle since 2024 | Skip. |

### B. One speech-native model

| Model | Licence | Notes |
|---|---|---|
| [**LFM3-Audio-2B**](https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF) / [LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF) | LFM Open License v1.0 (commercial use under $10M revenue) | Speech in and out, 18 languages, tool calls from voice. 1.1 GB at Q4_K_M + 0.34 GB projector: fits on the GPU with lots of room to spare. **Our `vendor/llama.cpp-lfm3` build already has it**: the LFM3-Audio speech generator in `mtmd`, and `llama-server`'s new `/v1/realtime` WebSocket endpoint (OpenAI Realtime schema) for live conversations and live transcription. We built only the server; `llama-tts` and `llama-mtmd-cli` need building too. |
| [Voxtral Mini 3B](https://huggingface.co/ggml-org/Voxtral-Mini-3B-2507-GGUF) | Apache 2.0 (Mistral) | Speech in, text out; ggml-org GGUF. Its realtime variant is named in llama.cpp's `/v1/realtime` docs. |
| [Ultravox v0.5 (Llama 3.2 1B)](https://huggingface.co/ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF) | MIT | Speech in, text out; repo quiet since Dec 2025. |
| [Moshi](https://github.com/kyutai-labs/moshi) / [Kyutai STT](https://huggingface.co/kyutai/stt-1b-en_fr) + [TTS](https://huggingface.co/kyutai/tts-1.6b-en_fr) | Apache 2.0 / CC-BY-4.0 | Full duplex (listens while it talks); its own Rust/PyTorch stack, not llama.cpp. English and French. |
| [Qwen3-Omni-30B-A3B](https://huggingface.co/ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF) | Apache 2.0 | The strong one; a 30B MoE, so experts in RAM like Qwen3.6, plus audio towers. Possible, not quick. |

### C. Hybrid

LFM3-Audio holds the conversation (fast, on the GPU), and hands real work to a pibench-tested
model as a tool call. On 8 GB the two can't both sit on the GPU: either the agent runs with
more experts in RAM (Groot at a higher `NCMOE`), or a server swaps models per task.

### First experiments (when the GPU is free)

1. **LFM3-Audio-2B alone**, through `/v1/realtime`: response time from end of speech to the
   first audio, speech-to-text accuracy on clips we generate with a TTS model (known text, so
   it's scored automatically), and whether a spoken request produces a correct tool call.
2. **Pipecat + Silero VAD + whisper.cpp + Kokoro around Groot**: the same clips and the same
   measurements, plus the cost to Groot's speed of whatever the speech parts take from it.
3. Write it up as an "audio probe", the way the [vision probe](vision-probe.md) works.

## Ideas to borrow

### 🔬 Local autoresearch ([conglu1997/local-autoresearch](https://github.com/conglu1997/local-autoresearch), MIT)

A local model improves a small program one experiment at a time: it gets the best solver so far
and a direction (refine, try another approach, simplify, question an assumption), returns a new
`solver.py`, and a runner scores it and keeps it if it wins. Built around
[Flint](models/flint.md), which is already in our lineup. Results on Macs: circle packing +21%,
factory scheduling −26% makespan, bike-demand RMSLE −31% (−9% on held-out data). Eight workers
sharing one llama-server with 8 slots ran 2–2.5× more experiments per hour than one worker and
ended ahead in 5 of 6 pairs. The gain came from throughput, not from better individual attempts.

Why here: a **new kind of benchmark**. pibench asks "can it finish a chore?"; this asks "how much
does it improve a score in N hours?", which rewards speed and steady, valid code. Run the same
task with Flint, Groot, LFM2.5 and Qwen3.6 and compare the curves.

What it needs before it runs here:
- **A Linux sandbox.** Candidates are model-written code. Upstream isolates them with macOS
  `sandbox-exec` (no network, no fork, writes only to a temp dir, 15 CPU s / 25 wall s) and
  refuses to run elsewhere. Port `runner/evaluate.py` to the same `docker run --network none
  --read-only --cap-drop ALL ...` that `scripts/qwen-codecheck.py --sandbox` uses, and
  `runner/resources.py` (macOS `libproc` memory/swap guard) to `/proc` + PSI. Run nothing
  before that.
- **Fewer workers.** Eight slots × 64K context is a 48 GB Mac's budget. Here: 2–4 slots and a
  smaller context, then measure whether the extra slots still buy throughput when the experts
  live in RAM.
- Use it as a pinned, reviewed checkout under `vendor/`, not `pip install` from the repo.

## Chores

- **A llama.cpp build after 2026-10-27** for LFM3: built in `vendor/llama.cpp-lfm3` (upstream `8ca4060`). Later:
  move the other upstream models to it and re-run their speed tests to check nothing moved.
  Note `vendor/` is a symlink to `GitStuff/Gemma4/vendor`, shared with that project on purpose:
  builds there serve both.
