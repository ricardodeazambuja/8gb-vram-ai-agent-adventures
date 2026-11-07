# 🔭 Up next

What's waiting its turn: the next tests, models to try, and ideas worth borrowing. Each new model goes through
the [security audit](security.md) before it runs, then gets a [model page](models/) and moves into the README
lineup (or the [graveyard](graveyard.md)). Finished work leaves this page; [the last section](#done-and-where-its-written-up)
says where it went.

## Next

1. **A task that needs an agent and eyes.** [The LFM3-VL tests](fine-tuning.md#can-lfm3-vl-do-more-than-look-2026-11-01)
   are done: temperature 0.7 ends its doom loops but not its failures, and neither a LoRA nor a new projector made
   one model do both. Gemma 4 26B already does (18/18, glance 30/30), and LFM3-1.7B driving with VL as eyes gets
   57/60 in 5 GB. No test yet needs both skills at once, so: a pibench task with pictures, on those two. Then the
   larger glance sample for Gemma.
2. **TwIL-8B's first pibench run**, in the sandbox. It's downloaded, audited and served
   ([page](models/twil-8b.md)); `scripts/llm twil` stays server-only until this run, because pi gives a model a
   shell. Then the speed test and the code check, with a quiet desktop (it needs ~0.7 GB of VRAM free beyond its own).
3. **LFM3-Audio's open parts:** tool calls over `/v1/realtime`, and recorded human speech instead of its own
   voice ([voice agents](#️-voice-agents-agents-that-talk-to-people) below).

### When the GPU is quiet (browser closed, or at night)

`scripts/llm` and `scripts/imagine` (2026-11-06) were tested live on the small models, the helpers and Klein, but
the desktop held 2.5 GB of VRAM, so these still need a run:

- [ ] **The big models through `scripts/llm`:** `qwen36`, `distill`, `flint` and `ornith` with the expert cache,
      `gemma` (cache + template fix) and `thinkingcap`. With a busy desktop they correctly refused; check that each
      loads, answers a `-p` one-shot, and that `status`/`stop` work. Watch memory pressure on the first load.
- [ ] **`bonsai2` and `spark`:** that the context really is 32K / 108K now (`scripts/llm status`) and that pi gets
      `--thinking medium` / `high` without an HTTP 400.
- [ ] **The interactive "stop the server too?" question** after a normal pi session.
- [ ] **Rerun `bench/recheck` and the vision probe through `scripts/llm <name> serve`**, so their numbers are the
      launcher defaults (cache, Gemma's q8_0 KV and template fix). Today both use the backend scripts with the
      static split, and their pages say so.
- [ ] **Groot and LFM2.5-24B with the cache:** measured faster (+16% and +42%, [Groot](models/groot-19b-a3b.md),
      [LFM2.5](models/lfm2.5-24b-a2b.md)) but not the launcher default, because `MOE_CACHE=auto` has no measured size
      for them that leaves room for a desktop. Measure each one's peak next to a ~0.5 and a ~1.7 GB desktop, give
      them a size and a desktop allowance in `run-qwen.sh`, then add them to `scripts/llm`'s cache list.

## In progress

| What | Status |
|---|---|
| [Clef-Flash](models/clef-flash.md), [d1-2B](models/d1-2b.md), [EmbeddingGemma 2](models/embeddinggemma-2.md) | measured; only a bigger, harder question set is left |
| Fine-tuning | open steps 4, 5 and 7–13 and steering tests 3 and 5: [the list](#faster-and-better-fine-tuning) |
| Pruning experts | parked: [below](#️-pruning-experts-from-the-moe-models-reap) |

## Models to try

| Model | Fit here | Why test it |
|---|---|---|
| [TIPS / TIPSv2](https://github.com/google-deepmind/tips) (Google DeepMind, Apache 2.0, other materials CC-BY 4.0; [paper](https://arxiv.org/abs/2410.16512), [v2](https://arxiv.org/abs/2604.12012)) | small (S/14 or B/14 first) | Embeds images and text, one vector per image patch. Compare with EmbeddingGemma 2's vision side: an automatic style score for the image experiments, and text queries on outdoor scenes for WildSeed. Not for clicks: at 448×448 it works on a 32×32 grid. Needs Python and `.npz` weights, so the user's approval first, then only in `bench/finetune/sandbox.sh`. A copy of the weights is in `../tips/pytorch/checkpoints`; check that the v2 files can still be downloaded before deleting them. |
| [Clef](https://huggingface.co/Cloudflare/clef) (Cloudflare, Apache 2.0) | 19.5 GB, dense | The big sibling of Clef-Flash; untested. |
| [ThinkingCap Nemotron-3.5-Lightning-30B-A3B](https://huggingface.co/bottlecapai/ThinkingCap-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) (bottlecap.ai, Apache 2.0) | Q4_K_M 25.4 GB (only that or f16); experts in RAM, bigger than Ornith, so browser closed | A new family for the lineup, already trained to think less ([ThinkingCap Gemma](models/thinkingcap-gemma4-26b-a4b.md) halved Gemma's thinking for 17/18). LFM2.5's card benchmarks against its base. |
| [ThinkingCap Qwen3.8-27B RCO-4.2bpw](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B-GGUF-RCO-4.2bpw) | 14.3 GB + mmproj 0.9, dense 27B mostly on the CPU | Low priority: the plain 27B ran at 2.5 tok/s ([graveyard](graveyard.md)); shorter thinking won't rescue that. |
| [LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF), [d1-Omni-2B](https://huggingface.co/LiquidAI/d1-Omni-2B-GGUF) (Liquid AI) | small | The Omni versions of [LFM3-Audio](models/lfm3-audio-2b.md) and [d1](models/d1-2b.md): images and video too. Untested. |
| Speech models: [Voxtral Mini 3B](https://huggingface.co/ggml-org/Voxtral-Mini-3B-2507-GGUF), [Ultravox v0.5](https://huggingface.co/ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF), [Qwen3-Omni-30B-A3B](https://huggingface.co/ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF), [Moshi](https://github.com/kyutai-labs/moshi) | see [voice agents](#️-voice-agents-agents-that-talk-to-people) | Alternatives to LFM3-Audio. |
| Image models: Z-Image Turbo, FLUX 3 Image Klein, Qwen-Image 2.1 | see [image generation](#️-image-generation) | The candidates not measured yet. |

## 🗣️ Voice agents: agents that talk to people

Goal: a local agent you can talk to, which answers out loud and can still use tools. Two ways to build one, and
one hybrid. Checked on 2026-10-28 against the GitHub and Hugging Face APIs (licences, last release). Only
LFM3-Audio has been downloaded and run so far ([its page](models/lfm3-audio-2b.md): speech out and in, spoken tool
calls and a live conversation all work, first audio in under 0.2 s; [`examples/listen.py`](../examples/listen.py),
[`speak.py`](../examples/speak.py)). Every piece goes through the [security routine](security.md) first.

The constraint is the GPU: a good agent model (Groot, Flint, Gemma) already takes ~7 GB of the 8, so in a pipeline
the speech parts run on the CPU, or the agent model gives up some layers.

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
| [**LFM3-Audio-2B**](https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF) / [LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF) | LFM Open License v1.0 (commercial use under $10M revenue) | Speech in and out, 18 languages, tool calls from voice; `/v1/realtime` (OpenAI Realtime schema) for live conversations. Measured: [its page](models/lfm3-audio-2b.md). |
| [Voxtral Mini 3B](https://huggingface.co/ggml-org/Voxtral-Mini-3B-2507-GGUF) | Apache 2.0 (Mistral) | Speech in, text out; ggml-org GGUF. Its realtime variant is named in llama.cpp's `/v1/realtime` docs. |
| [Ultravox v0.5 (Llama 3.2 1B)](https://huggingface.co/ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF) | MIT | Speech in, text out; repo quiet since Dec 2025. |
| [Moshi](https://github.com/kyutai-labs/moshi) / [Kyutai STT](https://huggingface.co/kyutai/stt-1b-en_fr) + [TTS](https://huggingface.co/kyutai/tts-1.6b-en_fr) | Apache 2.0 / CC-BY-4.0 | Full duplex (listens while it talks); its own Rust/PyTorch stack, not llama.cpp. English and French. |
| [Qwen3-Omni-30B-A3B](https://huggingface.co/ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF) | Apache 2.0 | The strong one; a 30B MoE, so experts in RAM like Qwen3.6, plus audio towers. Possible, not quick. |

### C. Hybrid

LFM3-Audio holds the conversation (fast, on the GPU), and hands real work to a pibench-tested model as a tool call.
On 8 GB the two can't both sit on the GPU: either the agent runs with more experts in RAM (Groot at a higher
`NCMOE`), or a server swaps models per task.

### Experiments, in order

1. **LFM3-Audio-2B, the open parts:** tools over `/v1/realtime`, then recorded human speech.
2. **Pipecat + Silero VAD + whisper.cpp + Kokoro around Groot**: the same clips and the same measurements, plus the
   cost to Groot's speed of whatever the speech parts take from it.
3. Write it up as an "audio probe", the way the [vision probe](vision-probe.md) works.

## 🎛️ Fine-tuning the small models

The first rounds are done, and [Fine-tuning](fine-tuning.md) has the results: norm-only, IA3 and LoRA on LFM2.5-VL
and LFM3-VL, the grid robot, steering, and serving from `llama-server`; the picture side is in
[Making pictures](image-generation.md#teaching-sdxl-a-style-or-a-person). Same rules for anything new: safetensors
weights, a pinned revision, no `trust_remote_code` outside the sandbox, its own mamba environment, notebooks read
before they run. Score on held-out seeds and changed layouts, then rerun the [vision probe](vision-probe.md).

### Faster and better fine-tuning

The plan of 2026-10-31. The limit is less often training speed than wasted runs: one VAE scaling bug cost about
3 hours. Done: steps 1–3 and 6, and steering tests 1, 2, 4 and part of 6. What's left:

**Tools that help every run:**
4. **Automatic scores:** a style number and a likeness number, so a queue can drop bad runs. The likeness model is
   a new download, so safetensors only, with its licence checked; many face models aren't free to use.
5. **Several short seeds, keep the best:** norm-only seeds ranged from 115 to 139. Choose on a separate validation
   set, never on the held-out test, or the reported score flatters us.

**Images:**
7. **A style with no famous name:** the prompt beat 1000 steps of training on all three famous styles, though 3000
   steps caught it on Van Gogh. The stated prize is a tiny file per style, which needs a style the prompt can't
   reach. Next on the style LoRA: regularisation pictures, and why style words turn green (not the starting word).
8. **LoRA on Klein base 4B:** whether it fits in 8 GB is unknown. Everything fast about SDXL so far doesn't
   transfer, so it needs its own fit test.
9. **Captions from our own vision models,** only if style results stay weak.

**Vision and text, which matter more than training faster:**
10. **Real camera pictures,** the open question since 2026-10-30.
11. **Another family: Qwen3-VL-2B or 4B,** to tell "fine-tuning works" from "it works on Liquid's models". Confirm
    its sizes and licence on Hugging Face first.
12. **A system prompt test on LFM3-1.7B,** our first plain text fine-tune. Can training replace part of the prompt?
    Measure the agent's system prompt first and skip this if it's small next to the context. Move habits such as
    tone, format and thinking first into training; keep tool schemas, facts and safety rules in the prompt. Compare
    the full prompt, a short stub, and the stub plus the tuned model on pibench. It saves context and first-request
    time, but RAM only if `-c` also goes down, and each prompt change means retraining.
13. **Train on mistakes** (or GRPO, below): collect misses on training seeds only, never held-out ones. Random
    recovery data didn't help before, so it needs testing. Related: [Liquid's cookbook](https://github.com/Liquid4All/cookbook/tree/main/finetuning)
    has GRPO with the [visual control probe](visual-control.md) as its reward, Antidoom (FTPO) for doom loops and SFT
    for d1-3B on Banking77; and [evolution strategies](#-evolution-strategies-instead-of-grpo-blog-post-2026-11-04)
    are a gradient-free alternative. Also untried: FiLM (one model, several tasks switched by a code) and
    LFM3-Omni-2B.

**Steering, open tests** ([what's done](fine-tuning.md#steering-vs-training-2026-11-01): it works in a narrow
strength window, didn't fix doom loops, can push a behaviour or look but can't add facts):
- **Test 3, system prompt arm:** the stub plus a vector on pibench, beside the full prompt, the stub alone and the
  tuned model (step 12).
- **Test 5, vision control:** the [visual control probe](visual-control.md) on LFM3-VL with a vector, beside the
  norm-only result, to see whether steering touches decisions or only style.
- **Test 6, the rest:** Textual Inversion on the nameless style (the person is done, and won), scored like the LoRA
  and norm-only runs, including `no-trigger` leaks. A [Concept Slider](https://arxiv.org/abs/2311.12092v1) only if
  a strength control is still needed. [h-space](https://arxiv.org/abs/2210.10960v2) is the true image equivalent of
  a steering vector, but it was shown on older pixel models and would need new code.

**Only if timing says so:** caching the vision tower's output (it only helps repeats over the same data; measure
its share of a step first), and SDXL with gradient checkpointing off (a one-line try).

**Always try the cheap thing first,** in this order: prompt, reference image, a harness rule (skipping blocked moves
added 2 or more solved mazes to LoRA and IA3), norm-only or IA3, then LoRA or DoRA. That doesn't replace the tests
above, since they're how we learn where each one stops working.

## 🖼️ Image generation

Measured: **[Making pictures on 8 GB](image-generation.md)**: FLUX.2 Klein 4B (the everyday pick, 23 s, 3.5 GB),
Iris-3B and SDXL, all through [`scripts/imagine`](../scripts/imagine), plus style and person training on SDXL. The
runner is [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), which keeps the "everything in
ggml" standard. Still untested (sizes from Hugging Face; speed and VRAM figures are vendor or blog claims):

| Model | Licence | Size | Why try it |
|---|---|---|---|
| [Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | Apache 2.0 | 6B parameters, GGUF said to need 5-6 GB | Well liked, about 8 steps. Blogs disagree on whether 8 GB is comfortable. |
| [FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) (Oct 2026) | FLUX Kommunity, free under $5M revenue, repo gated | 5.3-5.8 GB + 4.9 GB text encoder | Likely the best quality that fits. NVIDIA claims 8 GB with INT8, but not whether that includes the text encoder. No GGUF seen yet. |
| [Qwen-Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) (Sep 2026) | Non-commercial | 7B parameters | One blog says about 6 GB at Q4; Alibaba gives no figure. Licence is the catch. |

Also untried: ControlNet, IP-Adapter and PhotoMaker (`sd-cli` has flags for them). Training notes for new models:
norm-only may have little to train on FLUX-style models, which steer their norms from the prompt and timestep
rather than stored gains, so check each model first; Iris-3B works in pixels, so precomputed latents don't save
memory and it may need small crops. Pictures of a person are only of you or people who agreed, and they and their
adapters stay under the git-ignored `models/`, never in `bench/` or a commit.

## ✂️ Pruning experts from the MoE models (REAP)

[REAP](https://arxiv.org/abs/2510.13999) (Cerebras, ICLR 2026; [code](https://github.com/CerebrasResearch/reap),
Apache 2.0) deletes a MoE model's least useful experts in one shot, with no retraining. It scores each expert on a
calibration set by its router gate values and the size of its output, so an expert that is rarely picked, or barely
changes the output when it is, goes first. The paper reports it beating expert merging on generative tasks, and
near-lossless code generation and tool calling at 50% on Qwen3-Coder-480B and Kimi-K2.

Why it fits here: our big MoE models keep every expert in RAM, so 25–40% fewer experts is several GB of RAM back
(Bolt and Ornith are RAM-bound), and the [VRAM expert cache](how-it-fits.md#12-let-a-cache-pick-the-experts-the-default-for-the-big-moes)
holds a larger share of what's left. The catch: REAP works on Hugging Face checkpoints, and a 35B model in BF16 is
~70 GB, more than our 32 GB of RAM (its layer-wise mode may get around that; untested); the other route is scoring in
our llama.cpp fork and writing a GGUF without the dropped experts.

**Tried (2026-11-05), parked:** on 18 real pi sessions Qwen3.6 picks every expert, and frequency-pruned GGUFs
keeping 230 of 256 experts per layer already change the top token 10.5% of the time (KL 0.48); 192 13%, 128 18%
([details](../bench/moe-cache/README.md#how-concentrated-is-expert-use-expert_usagepy-2026-11-05)). Small per-layer
losses compound over 40 layers. Worth another look only if REAP's own score or a healing fine-tune can be tried
cheaply.

Removing whole layers instead ([ShortGPT](https://arxiv.org/abs/2403.03853), [Gromov et al.](https://arxiv.org/abs/2403.17887))
drops up to about half of some models' deeper layers with little loss on question-answering benchmarks, usually
after a short LoRA "healing" run. Multi-step tool use tends to break first, so it's only worth trying on a dense
model that almost fits, to get it fully on the GPU.

## Ideas to borrow

### 🧬 Evolution strategies instead of GRPO ([blog post](https://qiyaoliang.com/blog/es-vs-pg-long-horizon/), 2026-11-04)

Liang et al. (Cognizant AI Lab) compare evolution strategies with policy gradient (GRPO) for agent training. ES adds
Gaussian noise to the weights, scores each noisy copy with the task's pass/fail reward, and moves the weights toward
the copies that scored well: forward passes only, no backprop, no per-token credit. Policy gradient wins on short
tasks; ES should win as episodes get longer. Preliminary: Qwen3.5-4B on Terminal-Bench 2.1 went from 12.1% to 21.1%
with plain ES (population 32, σ 0.0005, 200 iterations on H200s), against 18.2% for a policy gradient checkpoint.
The method: [ES at Scale](https://arxiv.org/abs/2509.24372).

Why it fits here: it needs only inference, and our rewards (pibench's verifiers, the maze harness) are outcome-only
scores. Noise over a few thousand numbers is far easier than over billions, and our norm-only and IA3 adapters are
that size. The cost is evaluations: 32 copies × 16 maze runs × 20 decisions at 0.42 s is ~70 minutes per iteration.
A first test: ES on LFM3-1.7B's norm weights for multi-step maze driving (step 13 above), against the
gradient-trained norm-only adapter.

<a id="strata"></a>

### 🪨 Strata, an engine built for big MoE models on gaming PCs ([Niko1221/Strata](https://github.com/Niko1221/Strata), MIT, read 2026-11-06)

Built with parts of llama.cpp/ggml, around the same idea as our fork's cache: attention and shared layers on the
GPU, an expert cache that fills the rest of VRAM, every expert in RAM computed by the CPU in parallel, plus the
model's own MTP layer drafting 2–3 tokens. Its main model, Qwen3.8-Flash-Next, doesn't fit here
([graveyard](graveyard.md#the-ones-we-never-let-in-the-door)). **Qwen3.6-35B-A3B is worth an A/B:** its
docs measure 64.0 vs 41.1 tok/s against stock llama.cpp (`-ncmoe 28`) on an RTX 3060 12 GB with MTP off, and MTP
on gave another 1.6x on a 32 GB card. Our cache gives 1.58x over static offload, so the engines may be close without
MTP, and MTP is what we lack. Catches: it only runs Unsloth's MTP GGUFs, the smallest being UD-Q4_K_XL (22.9 GB,
setup plans 32 GB of RAM, all we have); its numbers are its own; it's a whole app with a downloaded engine, so: a
source build, the GGUF security routine, and a container with no network.

### 🧱 Megakernels ([Hyperkernel](https://github.com/Hyperstitioncc/Hyperkernel), looked at 2026-11-05)

Fusing a model's GPU kernels into one "megakernel" removes the gaps between kernel launches. Nothing to reuse:
Hyperkernel targets B300 datacenter GPUs only, and its generator isn't published. The idea also targets a bottleneck
we probably don't have: our big MoE models spend decode waiting on the CPU and RAM. The cheap first step is one
Nsight Systems run (`nsys profile`) of ~200 decode tokens on Qwen3.6 with the cache, to see how much of each token
is GPU work, idle time, PCIe copies and launch gaps. Only real GPU gaps or one slow kernel would justify kernel work.

### 🔬 Local autoresearch ([conglu1997/local-autoresearch](https://github.com/conglu1997/local-autoresearch), MIT)

A local model improves a small program one experiment at a time: it gets the best solver so far and a direction,
returns a new `solver.py`, and a runner scores it and keeps it if it wins. Built around [Flint](models/flint.md).
Results on Macs: circle packing +21%, factory scheduling −26% makespan, bike-demand RMSLE −31% (−9% on held-out
data); eight workers sharing one server ran 2–2.5× more experiments per hour. Why here: a **new kind of benchmark**,
"how much does it improve a score in N hours?". Before it runs: port its macOS `sandbox-exec` isolation to the same
`docker run --network none --read-only --cap-drop ALL` that `scripts/qwen-codecheck.py` uses (run nothing before
that), use 2–4 slots instead of 8, and keep it as a pinned, reviewed checkout next to this repo.

### ⚡ Parallel Language Models (PLM): watch

[Parallel Language Models via Variational Decoding](https://arxiv.org/abs/2610.26693) ([project page](https://www.justuswill.com/plm/),
Oct 2026): one forward pass proposes several tokens and checks the previous batch. Post-trained, 2.3–5.7× faster;
the 1.3B matches an identically trained normal model on math, coding and instructions (one seed). Why not now
(2026-10-30): the code repo is empty, it trains from scratch only (converting a normal model doesn't work), ~1,884
GPU-hours for the 1.3B (from a partial extract; check before quoting), vLLM only. Why watch: a 1.3B fits this card
easily, and LFM3-1.7B already runs at 106 tok/s here. Its relaxed-acceptance idea we did try:
[graveyard](graveyard.md#relaxed-draft-acceptance-faster-but-it-rambles).

### 🔥 Backburner: a draft head for a small model

[Soft-MTP](https://github.com/JaydenTeoh/Soft-MTP) ([paper](https://arxiv.org/abs/2610.38331), Oct 2026) trains
multi-token prediction heads to learn the model's own next-token prediction; alone it's ~1.5% over standard MTP, below
our noise. **The experiment:** a small draft head on a frozen LFM3-1.7B, for speculative decoding; Gemma's built-in
head gives +19–31% at greedy, so can a head added afterwards get near that? Open: the paper leaves this case open,
llama.cpp would have to run an MTP head for LFM3, and Gemma's gain vanishes at temperature 1.0, so measure at the
temperatures we use.

## Done, and where it's written up

| What | Now in |
|---|---|
| LFM3-1.7B, LFM3-VL-2B, LFM3-Audio-2B, d1-2B (support merged upstream 2026-10-27) | their [model pages](models/); `scripts/llm lfm3`, `lfm3vl`, `lfm3audio`, `d1` |
| ThinkingCap Gemma 4 26B: half the thinking, 17/18 | [its page](models/thinkingcap-gemma4-26b-a4b.md) |
| EmbeddingGemma 2, Clef-Flash, LocateAnything-3B, VibeThinker | their model pages; [`examples/`](../examples/README.md) |
| Our llama.cpp fork (2026-10-30), PrismML's merge (2026-11-02), the recheck | [getting started](getting-started.md#3-build-llamacpp-once), [`bench/recheck/`](../bench/recheck/README.md) |
| The MoE expert cache, now the default in `scripts/llm` | [how-it-fits §12](how-it-fits.md#12-let-a-cache-pick-the-experts-the-default-for-the-big-moes) |
| VibeThinker's audit (2026-11-07): clean, templates identical to the source | [security](security.md#audits-so-far) |
| Relaxed draft acceptance | [graveyard](graveyard.md#relaxed-draft-acceptance-faster-but-it-rambles) |
| The first image models, style and person training | [Making pictures](image-generation.md) |
| Cactus's Mirage (dropped) | [`bench/cactus/`](../bench/cactus/README.md) |
