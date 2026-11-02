# 🔭 Up next

Models and chores waiting their turn. Each model goes through the [security audit](security.md)
before it runs, then gets a [model page](models/) and moves into the README lineup (or the
[graveyard](graveyard.md)).

## Next: a task that needs an agent and eyes

[The LFM3-VL tests](fine-tuning.md#can-lfm3-vl-do-more-than-look-2026-11-01) are done: temperature 0.7
ends its doom loops but not its failures, and neither a LoRA nor a new projector made one model do
both. Gemma 4 26B already does (18/18, glance 30/30), and LFM3-1.7B driving with VL as eyes gets 57/60 in
5 GB. No test yet needs both skills at once, so that's next: a pibench task with pictures, on those two.
Then the larger glance sample for Gemma, and a Gemma 12B fit check, which has image input with no
vision encoder, if a projector for it exists.

## In progress

| What | Status |
|---|---|
| [Clef-Flash](models/clef-flash.md), [d1-2B](models/d1-2b.md), [EmbeddingGemma 2](models/embeddinggemma-2.md) | measured; only a bigger, harder question set is left |

## Models to try

### 🧲 EmbeddingGemma 2 (Google, Apache 2.0)

Measured, and it stays useful: see its [model page](models/embeddinggemma-2.md). A bigger, harder
question set is the only test left.

**Test TIPS alongside it.** [TIPS](https://arxiv.org/abs/2410.16512) and
[TIPSv2](https://arxiv.org/abs/2604.12012) (Google DeepMind,
[code](https://github.com/google-deepmind/tips), Apache 2.0, other materials CC-BY 4.0) also
embed images and text, and add one vector per image patch. Not measured here yet. Compare it
with EmbeddingGemma 2's vision side on two things: an automatic style score for the image
experiments, and text queries on outdoor scenes for WildSeed. Don't expect it to help with clicks:
at 448×448 it works on a 32×32 grid, too coarse for small page targets.

It needs Python and `.npz` weights, which breaks the `.gguf`-only rule, so it needs the user's
approval first and then runs only in `bench/finetune/sandbox.sh`. Start with a small model (S/14
or B/14). A copy of the weights is in `../tips/pytorch/checkpoints`; check that the v2 files can
still be downloaded before deleting them.

### ⚖️ Clef / Clef-Flash (Cloudflare, Apache 2.0)

Measured against d1-2B: [Clef-Flash](models/clef-flash.md) and [d1-2B](models/d1-2b.md). The dense
[Clef](https://huggingface.co/Cloudflare/clef) (19.5 GB) is untested.

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
| [d1-2B](https://huggingface.co/LiquidAI/d1-2B-GGUF) / [d1-Omni-2B](https://huggingface.co/LiquidAI/d1-Omni-2B-GGUF) | **decision models** like Clef: calibrated answers in one pass, plus spans, boxes and quantity intervals | [d1-2B is measured](models/d1-2b.md) and served by `/v1/systemone`. d1-Omni-2B is untested. |

### 📍 LocateAnything-3B (NVIDIA, non-commercial research license)

Measured: see its [model page](models/locateanything-3b.md).

### 🧠 VibeThinker 3B as a helper

Brought in from [`vibethinker_tests`](https://github.com/ricardodeazambuja/vibethinker_tests): see its
[model page](models/vibethinker.md). To try: a bigger model plans and checks, and the 3B writes the code in one shot.

## 🗣️ Voice agents: agents that talk to people

Goal: a local agent you can talk to, which answers out loud and can still use tools. Two ways
to build one, and one hybrid. Checked on 2026-10-28 against the GitHub and Hugging Face APIs
(licences, last release). Only LFM3-Audio has been downloaded and run so far (results below);
every piece goes through the [security routine](security.md) first.

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
| [**LFM3-Audio-2B**](https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF) / [LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF) | LFM Open License v1.0 (commercial use under $10M revenue) | Speech in and out, 18 languages, tool calls from voice. 1.1 GB at Q4_K_M + 0.34 GB projector: fits on the GPU with lots of room to spare. **Our fork's `adventures` branch already has it**: the LFM3-Audio speech generator in `mtmd`, and `llama-server`'s new `/v1/realtime` WebSocket endpoint (OpenAI Realtime schema) for live conversations and live transcription. We built only the server; `llama-tts` and `llama-mtmd-cli` need building too. |
| [Voxtral Mini 3B](https://huggingface.co/ggml-org/Voxtral-Mini-3B-2507-GGUF) | Apache 2.0 (Mistral) | Speech in, text out; ggml-org GGUF. Its realtime variant is named in llama.cpp's `/v1/realtime` docs. |
| [Ultravox v0.5 (Llama 3.2 1B)](https://huggingface.co/ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF) | MIT | Speech in, text out; repo quiet since Dec 2025. |
| [Moshi](https://github.com/kyutai-labs/moshi) / [Kyutai STT](https://huggingface.co/kyutai/stt-1b-en_fr) + [TTS](https://huggingface.co/kyutai/tts-1.6b-en_fr) | Apache 2.0 / CC-BY-4.0 | Full duplex (listens while it talks); its own Rust/PyTorch stack, not llama.cpp. English and French. |
| [Qwen3-Omni-30B-A3B](https://huggingface.co/ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF) | Apache 2.0 | The strong one; a 30B MoE, so experts in RAM like Qwen3.6, plus audio towers. Possible, not quick. |

The first tests are on [LFM3-Audio-2B's page](models/lfm3-audio-2b.md): speech out and in, spoken tool
calls and a live conversation all work, with first audio in under 0.2 s. Next: tools over
`/v1/realtime`, recorded human speech instead of its own voice, and the hybrid below.

### C. Hybrid

LFM3-Audio holds the conversation (fast, on the GPU), and hands real work to a pibench-tested
model as a tool call. On 8 GB the two can't both sit on the GPU: either the agent runs with
more experts in RAM (Groot at a higher `NCMOE`), or a server swaps models per task.

### First experiments (when the GPU is free)

1. **LFM3-Audio-2B, the open parts:** tools over `/v1/realtime`, then recorded human speech. Speed,
   generated speech and tool calls over the chat API are done (above).
2. **Pipecat + Silero VAD + whisper.cpp + Kokoro around Groot**: the same clips and the same
   measurements, plus the cost to Groot's speed of whatever the speech parts take from it.
3. Write it up as an "audio probe", the way the [vision probe](vision-probe.md) works.

## 🎛️ Fine-tuning the small models

The first rounds are done, and [Fine-tuning](fine-tuning.md) has the results: norm-only, IA3 and LoRA on
LFM2.5-VL and LFM3-VL, the grid robot, and serving from `llama-server`. Still open from that plan:

- **[Liquid's cookbook](https://github.com/Liquid4All/cookbook/tree/main/finetuning):** GRPO with the
  [visual control probe](visual-control.md) as its reward, Antidoom (FTPO) for doom loops, and SFT for
  d1-3B on Banking77.
- **FiLM, conditioned:** one model, several tasks switched by a code.
- **Another family:** Qwen3-VL-2B or 4B, to tell "fine-tuning works" from "it works on Liquid's models".
  Confirm its sizes and licence on Hugging Face first.
- **LFM3-Omni-2B,** which adds audio and ties in with the voice agents.

Same rules as before: safetensors weights, a pinned revision, no `trust_remote_code` outside the
sandbox, its own mamba environment, and notebooks read before they run. Score on held-out seeds and
changed layouts, then rerun the [vision probe](vision-probe.md).

### Faster and better fine-tuning

Written and reviewed against the goals above on 2026-10-31. The limit is less often training speed
than wasted runs: one VAE scaling bug cost about 3 hours. Time savings are guesses until measured.

**Status, 2026-11-01:** done: steps 1, 2 and 3 (`train_style.py preflight`, `--ckpts`, `eval_sdcli.py`,
`bake_sdxl.py`), step 6 and steering tests 1, 2, 4 and part of 6. Results are in
[Making pictures](image-generation.md#teaching-sdxl-a-style-or-a-person) and
[Fine-tuning](fine-tuning.md#steering-vs-training-2026-11-01). Not started: steps 4, 5 and 7 to 13, and
steering tests 3 and 5. Next for images: regularisation pictures on the 3000-step style LoRA, and why
style words turn green (not the starting word).

**Do first, it helps everything:**
1. **Preflight before a long run:** colour round trip, overfit one image in 2 minutes, untrained
   control. For the vision models the same idea: a one-step fit test and a small scored sample.
2. **Test through what we ship:** export to stable-diffusion.cpp or llama.cpp and score there. SDXL
   norm-only needs a bake step first, and our LoRA file's format still has to be tried in `sd-cli`.
   That also keeps the "everything in ggml" standard.
3. **Checkpoints at 250, 500 and 1000 steps:** one run gives the learning curve, and fewer steps may do.
4. **Automatic scores:** a style number and a likeness number, so a queue can drop bad runs. The
   likeness model is a new download, so safetensors only, with its licence checked; many face
   models aren't free to use.
5. **Several short seeds, keep the best:** norm-only seeds ranged from 115 to 139. Choose on a separate
   validation set, never on the held-out test, or the reported score flatters us.

**Then finish the image goals:**
6. **Person test:** norm-only surprised us here, so add DoRA, LoRA + norm and Klein `-r` as a no-training
   baseline, then stack a person with a style. Both person adapters already leak into "a man", so check
   `no-trigger` pictures. A stronger class word or regularisation pictures may help.
7. **A style with no famous name:** the prompt beat 1000 steps of training on all three famous styles,
   though 3000 steps caught it on Van Gogh. The stated prize is a tiny file per style, which needs a style the prompt can't reach.
8. **LoRA on Klein base 4B:** still the plan, and whether it fits in 8 GB is unknown. Everything fast
   about SDXL so far doesn't transfer, so this one needs its own fit test.
9. **Captions from our own vision models,** only if style results stay weak.

**Then the vision goals, which matter more than making them train faster:**
10. **Real camera pictures,** the open question since 2026-10-30.
11. **Qwen3-VL-2B or 4B,** to tell "fine-tuning works" from "it works on Liquid's models".
12. **A system prompt test on LFM3-1.7B,** our first plain text fine-tune. (VL's doom loops are
    [dealt with](fine-tuning.md#can-lfm3-vl-do-more-than-look-2026-11-01): temperature 0.7 ends them, and an agent LoRA
    didn't help.) The question: can training replace part of the prompt? Measure the agent's
    system prompt first and skip this if it's small next to the context. Move habits such as tone,
    format and thinking first into training; keep tool schemas, facts and safety rules in the prompt.
    Compare the full prompt, a short stub, and the stub plus the tuned model on pibench. It saves
    context and first-request time, but RAM only if `-c` also goes down, and each prompt change means
    retraining.
13. **Train on mistakes,** in place of the GRPO step we skipped: collect misses on training seeds only,
    never held-out ones. It's a cheaper version of the same idea. Random recovery data didn't
    help before, so it needs testing.

**Try steering before training, on both kinds of model:** tests 1, 2 and 4 below are done, see
[steering vs training](fine-tuning.md#steering-vs-training-2026-11-01): it works, in a narrow strength
window, and didn't fix doom loops (0/6). Test 6 is done on the person. It can push a behaviour or look
but can't add facts.
- **Text and vision models:** [steering vectors](https://arxiv.org/abs/2312.06681v2) need no gradient
  training. Our llama.cpp fork already has `--control-vector` and `cvector-generator`, and `lfm2.cpp`
  and `lfm3.cpp` apply the vectors. Check that the generator works on them, then add two tries to step
  12: a fourth arm in the system prompt test, and a "try something else" vector for doom loops. Each
  vector needs a small strength sweep.
- **Image models:** `sd-cli` has no such flag. The closest options are
  [Concept Sliders](https://arxiv.org/abs/2311.12092v1), which are small LoRAs trained to move one
  concept with an adjustable strength, and [Textual Inversion](https://textual-inversion.github.io),
  which learns one new word and leaves the model alone. `sd-cli` has `--lora-model-dir` and `--embd-dir`
  for these, but neither is tried yet. Inversion is the cheap test for steps 6 and 7: can a word
  replace the 2.5 MB file, and does it leak less into plain prompts? On SDXL only, since Klein has a
  different text encoder. [h-space](https://arxiv.org/abs/2210.10960v2) is the true equivalent of a
  steering vector, but it was shown on older pixel models and would need new code, so it's only
  worth a look if these fall short.

**Steering tests,** each with fixed prompts and seeds, against the same model with no vector:
1. **Does it work here?** Build one simple vector (a tone or format habit) with `cvector-generator` on
   LFM3-1.7B and LFM3-VL. Stop and write down why if the generator fails on them.
2. **Strength sweep:** a few strengths, positive and negative, to find where it starts to help and where
   the output breaks. Record VRAM and speed too, and the time and size of each vector.
3. **System prompt arm:** the stub plus a vector on pibench, beside the full prompt, the stub alone and
   the tuned model.
4. **Doom loops:** a "try something else" vector on pibench, counting repeated commands and solved tasks.
5. **Vision control:** the [visual control probe](visual-control.md) on LFM3-VL with a vector, beside
   the norm-only result, to see whether steering touches decisions or only style.
6. **Images:** Textual Inversion on the person and the nameless style, through `sd-cli`, scored with the
   same prompts as the LoRA and norm-only runs, including `no-trigger` leaks. A Concept Slider follows
   only if the inversion file works and a strength control is still needed.

**Steering docs:** a repeatable script and saved results under `bench/`, then a short "steering vs
training" section in [Fine-tuning](fine-tuning.md) with what worked and failed, the image results in
[Making pictures](image-generation.md), and commands in [Reproducing our results](reproduce.md). Failed tests
get written up too.

**Only if timing says so:** caching the vision tower's output. Our runs see each scene once, so it
saves nothing on a first run, only on repeats over the same data, and it rules out the
vision-side-only method. Measure the vision tower's share of a step first. SDXL with gradient checkpointing
off is a one-line try, but it only helps SDXL.

**Dropped:** caching lower layers. Norm-only and IA3 train every layer, so it would only suit a
LoRA on the top layers, a method we aren't using.

**Always try the cheap thing first,** in this order: prompt, reference image, a harness rule (skipping
blocked moves added 2 or more solved mazes to LoRA and IA3), norm-only or IA3, then LoRA or DoRA.
That doesn't replace the tests above, since they're how we learn where each one stops working.

## 🖼️ Image generation

Researched and first measured 2026-10-31. Stable Diffusion's own repos are still in `GitStuff/` from
years ago, but it's no longer the best fit for this laptop. Sizes in the candidates table come from
Hugging Face, and its speed and VRAM figures are vendor or blog claims.

Measured so far: **[Making pictures on 8 GB](image-generation.md)**. FLUX.2 Klein 4B is the everyday pick
(23 s, 3.5 GB). The rest of this section is the plan.

**Runner:** [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), the llama.cpp of
image models: ggml, CUDA, GGUF, and every model below is on its list. That keeps our "everything in
ggml" standard. It applies LoRAs at run time on quantized models, and takes `.safetensors` LoRAs.

| Model | Licence | Size | Why try it |
|---|---|---|---|
| [**Iris-3B**](https://huggingface.co/speridlabs/iris-3b) + 4-step Turbo (Oct 2026) | Apache 2.0 | [Q4 1.7 GB](https://huggingface.co/realrebelai/iris-3b_GGUFs), Turbo 6 GB | Paints pixels directly, so no VAE. Fine-tuned, it also does depth and upscaling, which suits our vision work. The GGUFs are a third party's. |
| [**FLUX.2 Klein 4B**](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (Jan 2026) | Apache 2.0 | [Q4_K_M about 2.6 GB](https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF) (blog figure) | The safest 8 GB pick. A [base version](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) exists for fine-tuning. |
| [Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | Apache 2.0 | 6B parameters, GGUF said to need 5-6 GB | Well liked, about 8 steps. Blogs disagree on whether 8 GB is comfortable. |
| [FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) (Oct 2026) | FLUX Kommunity, free under $5M revenue, repo gated | 5.3-5.8 GB + 4.9 GB text encoder | Likely the best quality that fits. NVIDIA claims 8 GB with INT8, but it isn't stated whether that includes the text encoder. No GGUF seen yet. |
| [Qwen-Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) (Sep 2026) | Non-commercial | 7B parameters | One blog says about 6 GB at Q4; Alibaba gives no figure. Licence is the catch. |
| SDXL | Open RAIL++ | 2.6B parameters | The fallback that's known to fit. |

**Memory:** the text encoder is often as big as the image model. It only runs once per prompt, so
the plan is encoder on the CPU or in RAM, then the image model alone on the GPU.

**Fine-tuning:** the two apply differently from our vision models.
- **LoRA** works on all of these. FLUX.2 and FLUX 3 ship a base model for it, and Z-Image has 8 GB
  guides, so far seen only as search snippets (512 px, low rank, offloading), from a vendor selling
  cloud GPUs. Klein 4B has no confirmed 8 GB training run.
- **Norm-only and IA3** are untested on image models. IA3 would bake in as before. Norm-only may
  have little to train: these models often steer their norms from the prompt and timestep, not from
  stored gains, so check each model first.
- **Training memory** is the real question. Text embeddings and, for latent models, images can be
  computed once up front, which frees a lot. Iris-3B works in pixels, so that second saving doesn't
  apply and it may need small crops.

**First SDXL results (2026-10-31):** at 1000 steps, naming the style in the prompt beat both LoRA and
norm-only, and norm-only did better on a person than on styles, the reverse of the guess below. By
2026-11-01 an 8 KB learned word had beaten both on the person, and 3000 steps of LoRA had caught the
prompt on Van Gogh. The text below is the original plan.

**Styles are the fine-tuning test we most want.** Style is the classic LoRA job, and it may suit the
tiny methods unusually well: style-transfer research (AdaIN) found that much of an image's style
sits in the mean and spread of its features, which is exactly what norm gains and IA3 scales
control. SDXL's U-Net has ordinary stored norm gains, so it's the best place to try norm-only, and a
tiny file per style would be a real win over a whole model. Measure three things per method: does a
small set (about 20 images) move the style, does it hold on subjects outside the training set, and
what does it cost in VRAM and minutes. Score with fixed prompts and seeds, side by side with the
untouched model, plus a style-similarity number so it isn't only our eyes. First check how far prompts
alone go, as the baseline. Train on our own pictures or public-domain art, not living artists' work.

**Your own subject in any style or situation.** Train on your own photos (a person, pet, object or
place) so the model learns the subject under a trigger word, then ask for it anywhere: "<subject> on
a beach", "<subject> as a watercolour". This is a harder test than style. A style changes how
everything looks, but a subject needs new detail that the model doesn't have, so expect LoRA to be
needed and the tiny methods to struggle; that's worth measuring, not assuming. The real prize is
stacking a subject LoRA with a style one at run time, which stable-diffusion.cpp's prompt tags allow.
Measure with about 20 photos held to a few new poses and places: is it recognisable in new
situations, does it survive a style change, and does it leak into pictures that don't ask for it?
Keep some photos out of training to compare against. Photos and adapters stay on this machine under
the git-ignored `models/`, never in `bench/` or a commit, and only of you or people who agreed.

**Order:** the first generation runs and the [SDXL style and person tests](image-generation.md#teaching-sdxl-a-style-or-a-person)
are done. What follows is in [Faster and better fine-tuning](#faster-and-better-fine-tuning).

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
  --read-only --cap-drop ALL ...` that `scripts/qwen-codecheck.py` uses, and
  `runner/resources.py` (macOS `libproc` memory/swap guard) to `/proc` + PSI. Run nothing
  before that.
- **Fewer workers.** Eight slots × 64K context is a 48 GB Mac's budget. Here: 2–4 slots and a
  smaller context, then measure whether the extra slots still buy throughput when the experts
  live in RAM.
- Use it as a pinned, reviewed checkout under `vendor/`, not `pip install` from the repo.

## Chores

- **Our llama.cpp (done 2026-10-30).** A fork,
  [`ricardodeazambuja/llama.cpp`](https://github.com/ricardodeazambuja/llama.cpp), branch
  [`adventures`](https://github.com/ricardodeazambuja/llama.cpp/tree/adventures) (pinned at
  `3cb2992`, then `7ff495f` with PrismML's fork merged on 2026-11-02): upstream `8ca4060` (2026-10-28) + LocateAnything PR #24749 (ported) + Dory PR #30306
  (at `43e865ec`) + our LFM2.5-VL template detection fix. Cloned next to this repo (`../llama.cpp`; `ADV_LLAMA` overrides), built in
  place. Dory, Spark, Flint, Bolt 74B and E4B, Groot, Ornith, LFM2.5-24B, LFM3, LFM3-VL,
  Clef-Flash and LocateAnything run on it, rechecked against their documented speeds: all within
  ~3% or faster (Bolt 15.5 vs 14.0 tok/s), except Groot at -5%, and an old/new/old A/B on the same
  afternoon (37.7 / 37.5 / 37.2 tok/s) shows that is the laptop that day, not the build. The stock-`b9606`
  models moved over the same night after an old/new/old A/B, generation in tok/s: Qwen3.6 23.9 /
  23.3 / 23.1, the distill 32.9 / 33.2 / 33.3, Qwen3-VL 41.7 / 39.9 / 41.4, Gemma 21.4 / 19.9-21.1
  / 19.4, prefill within 3%, every tool-call check passed. The distill uses 0.8 GB less VRAM on
  the fork (5.98 vs 6.82 GB, repeated twice; cause not traced). Newer llama.cpp has no `--no-mmap`
  (it's `--load-mode none`), which `run-server.sh` now handles. The Gemma 12B coder and agentic
  followed on 2026-11-01, and Bonsai 2's `PTQ1_0` on 2026-11-02, when PrismML's fork was merged in
  (38 tok/s against 28.5 on PrismML's own build, and a q8_0 KV gives it 32K instead of 16K). A full
  recheck after the merge found every other model within run-to-run noise
  ([`bench/recheck/`](../bench/recheck/README.md)).
  Note `vendor/` is a symlink to `GitStuff/Gemma4/vendor`, shared with that project on purpose:
  builds there serve both.

## ⚡ Parallel Language Models (PLM): watch, and borrow one idea

[Parallel Language Models via Variational Decoding](https://arxiv.org/abs/2610.26693)
([project page](https://www.justuswill.com/plm/), Oct 2026). A PLM proposes several tokens in one
forward pass and checks the previous batch in the same pass. One transformer does both jobs, with
two input embeddings. Rejected drafts never enter the KV cache, so there's no rollback.

**Paper's numbers** (RTX 6000 Ada 48 GB, fp16, one prompt at a time, vLLM). Pretrained, normal /
exact mode / relaxed mode: 350M 746 / 1,323 / 2,001 t/s, 1.3B 310 / 579 / 856, 3B 140 / 285 / 430.
Post-trained: 2.3–5.7× faster, 1.3B about 1,090 t/s, only ~10% more time per call at 7 proposed
tokens. The 1.3B matches or beats an identically trained normal model on math, coding and
instructions. Before post-training, exact mode keeps up and relaxed loses about 80 Elo. The
post-training comparison rests on one seed.

**Why not now (checked 2026-10-30):**
- `github.com/mandt-lab/plm` is empty. The paper promises code "upon publication" and doesn't
  mention weights.
- Training from scratch only. The authors tried converting their own normal models and concluded
  it doesn't work, so nothing we already run can become a PLM.
- About 1,884 GPU-hours for the 1.3B, ~2.5× a normal run. That figure came from a partial extract
  (the PDF wouldn't read), so check it before quoting it.
- vLLM with a custom attention matrix, no sign of llama.cpp support: it would need a port.
- No memory figures, and nothing on tool calling or JSON. Grammar constraints with relaxed mode are
  an open question.

**Why watch it:** a 1.3B model fits this card easily, and LFM3-1.7B runs at 107 t/s here, so
several hundred t/s on 8 GB could be possible. That's extrapolation, not a measurement. Recheck the
repo for code and weights.

**Idea to try without PLM: relaxed acceptance.** Keep a draft token if the main model finds it
likely, not only if it matches exactly. The paper reports 46.5% more accepted tokens per call at
comparable quality. It might work with llama.cpp's speculative decoding or our MTP setup, as a small
experiment on our fork. The output is no longer identical to normal decoding, so measure quality
and check that tool calls still parse.

**Tried it (2026-10-31): faster, but not worth keeping.** On Gemma 4 26B with its MTP head, n-max 2,
temperature 1.0, on the fork's experiment branch
[`relaxed-accept`](https://github.com/ricardodeazambuja/llama.cpp/tree/relaxed-accept) ([harness](../bench/relaxed/run_relaxed.py)).
The rule keeps a drafted token when the target gives it at least τ times the probability of its top
choice, and turns off once a grammar is active.

| Mode | tok/s | Drafts kept | Ran out of 1500 tokens (of 70) | Tool calls |
|---|---:|---:|---:|---:|
| no draft | 20.8 / 22.1 | — | 4 | 30/30 |
| MTP, exact match | 23.6 / 26.2 | 71% | 7 | 30/30 |
| MTP, `--spec-draft-sampling probabilistic` | 27.0 | 76% | 8 | 30/30 |
| relaxed τ 0.5 | 28.3 | 82% | 13 | 30/30 |
| relaxed τ 0.2 | 28.8 | 87% | 13 | 30/30 |
| relaxed τ 0.05 | 29.3 | 89% | 8 | 30/30 |

Two runs where a mode was repeated. The arithmetic scores looked like a disaster at first (19/30
against 29/30), but every miss in every mode was the thinking running out of tokens, not a wrong
answer. With a 4000-token budget and new tasks no finished arithmetic answer was wrong in any mode:
no draft and probabilistic 30/30 over two rounds, τ 0.2 14/15 over one (the other hit the limit; its
second round timed out). So relaxed acceptance doesn't break answers or tool calls here, but it
makes the model ramble, which eats much of the extra speed, and the 7% over probabilistic is within
this laptop's noise. Probabilistic drafting already ships in llama.cpp, needs no patch and
keeps the model's own distribution, so that's the one to use: +26% over no draft in one run, same
scores as no draft on the new tasks, though 2 of 50 hit the limit against none. It isn't the
default in `run-server-mtp.sh` yet because that's a single speed measurement. The temperature 0.6
runs were dropped for time, and these are small, easy tasks.

## 🔥 Backburner: a draft head for a small model

[Soft-MTP](https://github.com/JaydenTeoh/Soft-MTP) ([paper](https://arxiv.org/abs/2610.38331),
Oct 2026) changes how multi-token prediction heads are trained, so each head learns the model's own
next-token prediction. By itself it isn't worth using here: about 1.5% more speed over standard MTP
(1.76× vs 1.74× at 1.2B), well below this laptop's noise, and it needs training from scratch.

**The experiment:** train a small draft head on a frozen, already-trained model such as LFM3-1.7B,
leaving the model itself untouched, then use it for speculative decoding. Gemma's head gives us
+19–31% at greedy, so the question is whether a head added afterwards can get near that.

**Open before starting:**
- The paper leaves this case open: a model never trained to predict further ahead may not hold what
  a head needs.
- llama.cpp would have to run an MTP head for LFM3; we haven't checked.
- Gemma's gain vanishes at temperature 1.0, so measure at the temperatures we actually use.
