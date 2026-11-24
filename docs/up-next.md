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
2. **LFM3-Audio's open parts:** tool calls over `/v1/realtime`, and recorded human speech instead of its own
   voice ([voice agents](#️-voice-agents-agents-that-talk-to-people) below).

## In progress

| What | Status |
|---|---|
| [Clef-Flash](models/clef-flash.md), [d1-2B](models/d1-2b.md), [EmbeddingGemma 2](models/embeddinggemma-2.md) | measured; only a bigger, harder question set is left |
| Fine-tuning | open steps 4, 5 and 7–13 and steering tests 3 and 5: [the list](#faster-and-better-fine-tuning) |
| Pruning experts | parked: [below](#️-pruning-experts-from-the-moe-models-reap) |

## Models to try

| Model | Fit here | Why test it |
|---|---|---|
| [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (Thinking Machines, Apache 2.0, released 2026-11-20; GGUFs by [ggml-org](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF), revision `db61c5b`; looked up 2026-11-21, nothing downloaded) | **Smallest files (sizes from the HF API, 2026-11-22):** [AtomicChat](https://huggingface.co/AtomicChat/Inkling-2-Mini-GGUF) (revision `3095098`, Apache 2.0, a plain quant of the original, not one of the uncensored fine-tunes) is the smallest: `AD-IQ1_S` 6.88 GB, `AD-IQ1_M` 7.26 GB, `AD-IQ2_XXS-Q2` 7.90 GB, `AD-IQ2_M` 8.67 GB, `AD-IQ3_XXS` 12.58 GB, `AD-IQ4_XS` 15.00 GB; mmproj Q8_0 0.35 GB. Their own agreement with their BF16 on the top-1 token (30-language corpus, `llama-perplexity --kl-divergence`, not verified): 79.5%, 81.1%, 84.3%, 85.7%, 93.3%, 95.8%; even their Q8_0 gets only 97.7%. They tested on llama.cpp b12600–b12603; their `Q4_K_M` and `Q5_K_M` still use the PR #32695 layout and don't load there (the `AD-*` files do). [Unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) (revision `45fd087`): `UD-IQ1_S` 7.76 GB, `UD-IQ1_M` 8.02 GB, `UD-IQ2_XXS` 8.49 GB, `UD-Q2_K_XL` 9.14 GB, `UD-IQ3_XXS` 11.66 GB, each also under `Mainline_Support/`; the 0.40/0.65 GB `dspark-*` files are draft models, not Inkling. Others: bartowski's smallest is `IQ2_XXS` 10.32 GB, prithivMLmods/DevQuasar `Q2_K` 10.86 GB. None fits all on the GPU (size + KV + ~1 GB of buffers + desktop VRAM), so the experts go to RAM anyway: try a 3-bit file first (Unsloth `UD-IQ3_XXS` or AtomicChat `AD-IQ3_XXS`), not 1-bit. IQ4_NL 16.4 GiB + mmproj 0.32–0.60 GiB; 26.4B MoE, 3.4B active (8 of 128 experts; the card adds ~1.0B per extra loop pass, so 5.4B effective per token, and compute per token is closer to a 5B-active model), so experts in RAM like our other big MoEs. Estimated from the published dimensions: experts ~24.5B, everything else ~1.9B (~1.9 GiB at Q8). Not measured: speed, and the VRAM of the KV cache, which is kept once per pass | A new family and a new architecture: 24 layers, the middle 12 run three times per token (48 layer evaluations). Text and images, `reasoning_effort` from `none` to `max`. **Blocked on our build:** the fork (2026-11-08 sync) has no Inkling code; it needs upstream with [#32694](https://github.com/ggml-org/llama.cpp/pull/32694) and the fix [#32704](https://github.com/ggml-org/llama.cpp/pull/32704) (merged 2026-11-21; without it, perplexity ~130 instead of 7.65). Before syncing, check that the CUDA issues opened 2026-11-21 are closed: NaN perplexity ([#32710](https://github.com/ggml-org/llama.cpp/issues/32710)), slow prefill and a crash ([#32707](https://github.com/ggml-org/llama.cpp/issues/32707)). Use the ggml-org files or Unsloth's `Mainline_Support/` copies; [Unsloth's](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) root files need their own llama.cpp build. Whether the expert cache works with the looped layers is untested. **Smaller file to try first:** [LookAtMyQuant](https://github.com/ShinMegamiBoson/lookatmyquant) (commit `c0a04a7a`, looked at 2026-11-21) scored Unsloth's GGUFs against the BF16 original (their own evals, not verified: GSM8K `test[:200]` greedy at `reasoning_effort="none"`, 164 HumanEval; original 147 / 66): `UD-IQ3_XXS` 11.66 GB got 148 / 68, `UD-Q2_K_XL` 9.14 GB 132 / 49, `UD-IQ1_S` 7.76 GB 127 / 46. So `UD-IQ3_XXS` may match the original at ~6 GB less than IQ4_NL (16.4 GiB = 17.6 GB); Unsloth revision `ae9327c` has mainline copies of all three as `Mainline_Support/Inkling-2-Mini-UD-*.gguf_file`. LookAtMyQuant's own 5.38 GB and 8.87 GB Inkling files (reported 134 / 60 and 148 / 58) are not GGUF: packed `.npz` for their MLX runtime with Metal kernels, Apple Silicon only, so not usable here. **On a 12 GB card, all on the GPU** ([Kintsugi-27B's head-to-head](https://github.com/professorpalmer/Kintsugi-27B/blob/9e60a64a640ddaa0b98677a2268c44598839f0e8/benchmarks/README.md#head-to-head-inkling-2-mini-on-the-same-12-gb-card), 2026-11-20, their numbers, not verified): Unsloth `UD-Q2_K_XL` on Unsloth's llama.cpp build b12588-mix-c0c3de8, the vendor's sampling (temperature 1.0): AppWorld test_normal 283 / 336, HumanEval 160 / 164 at medium thinking (LookAtMyQuant's 49 was greedy at effort `none`, so the two don't contradict), 24 / 24 needles + 20 / 20 code copies at 60K; f16 KV at 64K tested, out of memory at a 67,584-token window, loads to ~70K only with short answers, and q8_0 KV didn't fit even at 32K (an odd result, unexplained). See [Kintsugi](#kintsugi) below. **On an 8 GB card, experts split by layer** ([@analogalok on X](https://x.com/analogalok/status/2124937493258932336), 2026-11-23, read through the fxtwitter API since x.com refused the fetch; his numbers, not verified; the replies he says hold the GGUF link and an earlier benchmark, and the video, weren't read): RTX 4060 8 GB + 16 GB RAM, Unsloth's new `UD-Q3_K_XL` (not in our 2026-11-22 size list; revision and sha256 not looked up) on stock llama.cpp b12651, no custom build (not checked: our fork clone has no tags past b1269, nothing fetched). Decode went from 17.1 to 22.0 tok/s; on a 10,823-token prompt, prefill 88 tok/s and decode 22.0 tok/s. He says `--n-cpu-moe 12` was slower because it spills the looped layers 6–17 across PCIe (read three times per token). Instead: `-ot` sends only the experts of blocks 0–6 and 18–23 to the CPU, keeping the looped layers on the GPU (his regex includes block 6, which by his own numbering is looped; unexplained), `-ub 64`, and `--load-mode none -t 6 -tb 12` (`-t` = physical P-cores, `-tb` = all threads). The rest of his command: `--ctx-size 18000 --parallel 1`, q8_0 K and V, `-ngl 99 -fit off -fa on --no-op-offload`, `--temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0`. Worth testing when this runs: the split by layer (which experts go to RAM) instead of `--n-cpu-moe`, since the looped layers are read three times per token. **Send the thinking back** ([@net_termina on X](https://x.com/net_termina/status/2124952437522149445), 2026-11-23, via the fxtwitter API): Inkling scored below an unnamed 9B model until their harness sent the earlier turns' thinking back, then "night and day" (no numbers given). They say the official model card warns about it; the card at revision `7a67ae2` (read 2026-11-23) does not, but its `chat_template.jinja` (same revision) renders every earlier assistant turn's `reasoning_content` as a `<|content_thinking|>` block, not only the turns after the last user message, so the model expects its past thinking in the prompt. pi already echoes `reasoning_content` back (see `bench/moe-cache/README.md`); check that llama-server's Inkling parser fills it and that the rendered prompt has those blocks (`/apply-template` on a captured pi request) before judging quality. The card also warns about long reasoning loops (cap output tokens and agent turns) and says the minimum effort doesn't fully disable thinking. **Experts streamed from the SSD** ([@atomic_chat_hq on X](https://x.com/atomic_chat_hq/status/2124953436055564594), 2026-11-23; their numbers, not verified): their [atomic-engine](https://github.com/AtomicBot-ai/atomic-engine/tree/e110381ae2066d4f5018585263b9b302cd2f4516) (MIT, "SSD-based expert streaming inference engine built on llama.cpp"; branch `codex/expert-weight-source` at commit `e110381`, 2026-11-22) with a draft model claims 122 tok/s on an M5 Max against 40 tok/s for llama.cpp, and 50% less RAM on smaller machines by reading experts from the SSD. The README section linked is for Apple Silicon only; whether any of it works on CUDA was not checked. Ideas only (not read yet): how they choose which experts stay in RAM, and whether it beats the OS paging an mmap'd GGUF, which is what our RAM limit and the oomd kills come from. Their [Nov 20 post](https://x.com/atomic_chat_hq/status/2123908975205007590) only announces the AtomicChat GGUFs listed above (it gives 27.7 GB for the largest file and recommends `AD-IQ1_M`, 81% top-1 agreement). |
| [North Mini Code 1.0](https://huggingface.co/CohereLabs/North-Mini-Code-1.0) (Cohere, Apache 2.0, [blog post](https://cohere.com/blog/north-mini-code) dated 2026-06-09; original revision `d11e61a`; GGUFs by [Unsloth](https://huggingface.co/unsloth/North-Mini-Code-1.0-GGUF), revision `e306bb4`, and [bartowski](https://huggingface.co/bartowski/North-Mini-Code-1.0-GGUF), revision `6ff6563`; looked up 2026-11-21, nothing downloaded) | 30B MoE, 3B active (8 of 128 experts, no shared expert), so experts in RAM like our other big MoEs. Unsloth `UD-IQ4_XS` 15.23 GB, `UD-IQ3_XXS` 11.71 GB; bartowski `Q4_K_M` 18.74 GB. 49 layers, only 13 with full attention (the rest use a 4096-token sliding window), 4 KV heads: KV cache ~1.9 GiB at 64K in f16 (estimated from `config.json`, not measured) | Cohere's first coding/agent model, trained for OpenCode; 256K context, 64K max output. A new family at the same size as Qwen3.6-35B-A3B, which Cohere's chart compares against (the benchmark scores are only in an image; the text gives 33.4 on the Artificial Analysis Coding Index). **Our build already supports it:** the arch (`cohere2_moe`, [#24260](https://github.com/ggml-org/llama.cpp/pull/24260)) and its own chat parser ([#24615](https://github.com/ggml-org/llama.cpp/pull/24615), [#26018](https://github.com/ggml-org/llama.cpp/pull/26018)) are in the 2026-11-08 sync. Audit against the CohereLabs `chat_template.jinja`. Untested here: speed, tool calls with pi, and the expert cache. |
| [TIPS / TIPSv2](https://github.com/google-deepmind/tips) (Google DeepMind, Apache 2.0, other materials CC-BY 4.0; [paper](https://arxiv.org/abs/2410.16512), [v2](https://arxiv.org/abs/2604.12012)) | small (S/14 or B/14 first) | Embeds images and text, one vector per image patch. Compare with EmbeddingGemma 2's vision side: an automatic style score for the image experiments, and text queries on outdoor scenes for WildSeed. Not for clicks: at 448×448 it works on a 32×32 grid. Needs Python and `.npz` weights, so the user's approval first, then only in `bench/finetune/sandbox.sh`. A copy of the weights is in `../tips/pytorch/checkpoints`; check that the v2 files can still be downloaded before deleting them. |
| [Clef](https://huggingface.co/Cloudflare/clef) (Cloudflare, Apache 2.0) | 19.5 GB, dense | The big sibling of Clef-Flash; untested. |
| [Nemotron-3.5-Lightning-30B-A3B](https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) (NVIDIA, its own license: the card says "other"; GGUFs by ggml-org, revision `8cd0821`) | Q4_0 18.9 GB, the same tier as our other big MoEs, plus an MTP head (`mtp-…-Q4_0.gguf`, 1.16 GB); hybrid Mamba-2 MoE with 6 attention layers, so a long context is cheap | The official, smaller way into the family ThinkingCap (next row) builds on. A Mac test with stock llama.cpp ([llama-modchip](#modchip) below) measured 57 tok/s decode and 2.8× on copy-heavy code edits with `ngram-mod`. Read the license before downloading. |
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
On 8 GB the two can't both sit on the GPU: either the agent runs with more experts in RAM (Groot with a smaller
`MOE_CACHE`), or a server swaps models per task.

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
   Training ideas to try on it are in [kohya sd-scripts](#sd-scripts) below (Min-SNR, masked loss, validation loss,
   several vectors per word).
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

Not pursued: ControlNet, IP-Adapter and PhotoMaker (`sd-cli` has flags for them): Klein's reference edit, masks and img2img
cover our edits, see [Editing a picture](image-generation.md#️-editing-a-picture-start-here); revisit if a task needs an
exact pose or layout. Training notes for new models:
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

### 🪢 Harness adapters: a LoRA per model and agent harness ([blog post](https://huggingface.co/blog/harness-hub/harness-adapters), 2026-11-06)

Hugging Face trains one small LoRA per model and harness with reinforcement learning (GRPO), inside the
unmodified harness: the published LFM2.5 adapters are for pi, hermes and claude-code; the runs in progress for
pi, opencode and hermes. Their models are three of ours: LFM2.5-24B-A2B, LFM3-1.7B and TwIL-8B.
The reward is the task's verifier (the runs in progress add a small bonus for fewer tool calls; the published pi
adapter's card lists the verifier alone), and a proxy,
[safeagents](https://github.com/huggingface/safeagents), blocks reward-hacking calls (an undeclared tool, a host that
appears nowhere in the conversation). The final runs were still training when it was posted; only LFM2.5's adapters are out
([collection](https://huggingface.co/collections/harness-hub/lfm25-24b-harness-adapters)). The pi one's card
([harness-hub/LFM2.5-24B-A2B-pi](https://huggingface.co/harness-hub/LFM2.5-24B-A2B-pi)) reports, on 100 of their
data-analysis tasks: pass@1 −4.2 points (within their ±5 noise) and **39% fewer tool calls** on the tasks both solve.
It also comes as a 37 MB GGUF for llama.cpp's `--lora`
([harness-hub/LFM2.5-24B-A2B-pi-GGUF](https://huggingface.co/harness-hub/LFM2.5-24B-A2B-pi-GGUF)).

What to take from it, in order:

1. **The pi adapter on LFM2.5 in pibench.** An independent check of their claim on our chores. Baseline: 10/18, a median of
   15 tool calls a job (13.3 on average on the solved ones; [results](../bench/results/results.csv)). Open question: it was trained on the BF16 weights and we run Liquid's QAD `Q4_0`; the card
   says lower-precision bases drift from what it learned, and its suggested `Q8_0` (~25 GB) is too big for 32 GB of
   RAM next to a desktop. Its two clean zeros ([a bug fix stopped early, the books site read as
   Latin-1](models/lfm2.5-24b-a2b.md)) are habits, not
   wasted calls, so expect fewer calls rather than more solves. Before it runs: pin and audit the GGUF, and diff the
   `chat_template.jinja` the adapter repo ships against Liquid's. Then a `scripts/llm lfm25` option for the LoRA, and
   a night run (~2 h: its 18 attempts took 1.8 h last time) into its own `RAW=` folder.
2. **Tool calls compared their way:** on the attempts both versions solved, side by side, in `bench/batch_stats.py`.
3. **safeagents' host rule as a pibench check:** list the hosts every attempt reached and flag any that aren't the
   task's sites. Done by hand for TwIL on 2026-11-07 (only the task sites); `batch_stats.py` could do it for every run.
4. **Later:** their TwIL-8B and LFM3-1.7B adapters, if published (our baselines: [8/18](models/twil-8b.md#our-numbers),
   [12/18](models/lfm3-1.7b.md)); their held-out data-analysis tasks
   ([SmolDataEnvs](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-harbor-train) is the training set) as a
   seventh pibench chore, after a review, since each task is a third-party Docker image.

<a id="modchip"></a>

### 🕹️ llama-modchip: ideas only, none of its code ([synchopate/llama-modchip](https://github.com/synchopate/llama-modchip), Apache 2.0, read 2026-11-07 at `3cba01e`)

A CPU-first llama.cpp patch set (Nemotron Lightning 30B, DeepSeek-V4-Flash 284B in 64 GB). **We reuse ideas,
never its code:** its patches replace whole llama.cpp files, its Hugging Face repos ship prebuilt `llama-server`
binaries next to the GGUFs, its quickstart pipes `curl` into `python3`, and its server reads settings from a
`/dev/shm` file other processes can write. Its own numbers are large and lightly checked: the headline CPU decode
ran with 4 of 6 experts per token (`apple-silicon/README.md`). Worth trying here, all with stock llama.cpp or a
few lines of our own:

1. **`--spec-type ngram-mod`**, upstream llama.cpp's n-gram drafter. The independent Mac test
   (`apple-silicon/results/spec.json`, stock build `5266f24da`, greedy, Nemotron Lightning) measured 55 → 155
   tok/s on a code edit (copy a file with one change, 93% of drafts accepted), and no change on new code or text
   (56 tok/s); upstream's `ngram-simple` defaults gave 90. Our questions: does it hold with experts computed on the
   CPU (every drafted token can pull in its own experts), on pi's snippet edits rather than whole-file copies,
   and at our sampling temperatures (Gemma's MTP only paid off near greedy, see
   [scripts/README](../scripts/README.md#speculative-decoding-mtp))? First test: Qwen3.6, three prompts
   (copy-edit, new code, text), ~15 min; then one pibench batch if it helps.
2. **Software prefetch in the CPU K-quant dot products** (Q4_K/Q5_K/Q6_K, a few lines in
   `ggml-cpu/arch/x86/quants.c`, written ourselves). Their reason (keep a CPU KV cache in L3) doesn't apply, since
   our KV is on the GPU; the one plausible gain is crossing 4 KB page boundaries, where hardware prefetchers stop.
   Low priority, and only with a before/after `srvbench`.
3. **A per-op CPU timer** (an env variable that sums time per op and tensor type for one graph): would show where
   the CPU time goes when it computes cache misses. Diagnostic only, in our fork.

Not for us: 2 MB huge pages (our experts are mmapped from ext4, and this kernel lacks
`CONFIG_READ_ONLY_THP_FOR_FS`, so file pages can't be huge); Fractal-64 (changes the on-disk layout of upstream's
`Q2_0`); RouteCrack/Stockfish drafting (tables trained on Nemotron text); fewer experts per token (`--override-kv`
does it already, and our [pruning study](../bench/moe-cache/README.md#pruned-qwen36-even-10-fewer-experts-changes-one-top-token-in-ten-expertsprune-evalsh-2026-11-05)
says Qwen3.6 is sensitive); the Claude Code gateway and HARP-Cache (pi reprocesses only ~25 tokens a turn).
One aside it led to: in our own pibench transcripts, LFM2.5 starts every new session with nothing cached, while
Qwen3.6, Flint and Groot usually reuse ~1.3–1.9K tokens of pi's prompt. Seconds per session; not urgent.

<a id="inference-optimisation"></a>

### 📈 inference-optimisation skill: speed-tuning method, ideas only ([sybil-solutions/inference-optimisation-skill](https://github.com/sybil-solutions/inference-optimisation-skill), MIT, read 2026-11-21 at `fda968e`)

An agent skill for tuning a server's speed step by step, with results from vLLM, SGLang and EXL3 on Intel Arc,
RTX 3090 and H200. Nothing on llama.cpp with experts in RAM. **Not installed as a skill:** its prompt tells the
agent to open a PR and to store a sudo password in a file, against our rules. Its `bench.py` was not read and
must not be run unread. Its `ceiling.py` assumes the whole model is in GPU or unified memory; our real limit (CPU
expert reads + PCIe) is already modelled in [`bench/moe-cache/estimate.py`](../bench/moe-cache/estimate.py).
All numbers below are theirs, not checked here. Ideas worth taking:

1. **A fixed accept rule for speed changes:** ≥6 decode samples per run and their median; run order control,
   candidate, control; a candidate wins only if it beats *both* neighbouring controls and slows no other metric
   by more than ~3%; repeat any surprising win. We interleave already, but by eye, and several model pages quote
   single runs (Flint's say ±13% noise). Cheapest step: an `--interleave` mode in `srvbench` that prints the
   verdict.
2. **Trim the MTP draft head's vocabulary.** The draft's LM head is read on every draft step; keep only the most
   used tokens (they kept 512 of 1,940 blocks, 98.6% coverage) while the target still checks against the full
   vocabulary, so the output is unchanged. They report +27% (vLLM, Arc Pro B70) and +14% (SGLang, Arc B390) at
   one user. Their worked example is Ornith-1.5-35B-A3B, which we run: its 248K-token LM head is 0.38 GB of the
   1.65 GB read per token (EXL3 sizes; ours differ). In llama.cpp this needs fork work (not checked yet: whether
   the MTP head has its own output tensor or shares the model's), so measure first how much of an MTP step is the
   LM head.
3. **Sweep the draft length per context length.** At 128K, one draft ran 29.6 tok/s vs 15.5 for three, though
   three won at short context. We picked `--spec-draft-n-max 2` once, at short context.
4. **q8_0 KV speed on models where few layers have KV.** They measured 45.4 → 27.0 tok/s in llama.cpp with KV in
   16 of 64 layers. Several of our launchers default to q8_0 KV (Bonsai 2, Groot, Flint, Dory); Dory's page
   compares quality only. One f16-vs-q8_0 decode run per model, at a context where both fit, ~10 min each.
5. **Flag looping output in speed runs.** Their 117 tok/s "record" was a model repeating a numbered list. A
   repeated-shingle check in `srvbench`/`qwen-bench.py` would catch it.

Not for us: the vLLM/SGLang/Intel XPU gotchas, the Docker memory script (`mem.sh`), the run queue (we run one GPU
job at a time anyway) and the publishing flow.

**To revisit:** repo <https://github.com/sybil-solutions/inference-optimisation-skill>; the version read is
[`fda968e`](https://github.com/sybil-solutions/inference-optimisation-skill/tree/fda968e7fc1996aa23c67139c755fe541c9d4be0)
(single commit, 2026-11-16). Worth rereading, under `skills/inference-optimisation/`: `references/levers.md` (every
lever with its measured effect), `references/methodology.md` (measurement traps) and `references/ceilings.md`
(the theoretical-maximum arithmetic, worked on Ornith-1.5). Their raw hill-climb logs are linked from the README
(sybil-solutions/local-ai-registry, trellis-serve, qwen-serve); not read here.

<a id="kintsugi"></a>

### 🏺 Kintsugi-27B: a 2.5-bit Qwen3.8-27B for 12 GB cards, engine ideas only ([professorpalmer/Kintsugi-27B](https://github.com/professorpalmer/Kintsugi-27B), Apache 2.0, engine MIT, read 2026-11-21 at `9e60a64`)

Qwen3.8-27B (dense) at 2.47 bits per weight in EXL3 trellis code ("Escha W2"), a 10.5 GB GGUF that only its own
llama.cpp fork loads, shipped as prebuilt Windows/Linux bundles plus source tarballs. Full 262K context on an RTX
4070 12 GB; their numbers: decode 67 tok/s at 64K, 31 at 245K, HumanEval 161 / 164, AppWorld 151 / 168. The
write-ups set pass lines before each run and published a correction when an AppWorld claim turned out to be
biased (scored on the seeds the bit map was chosen on).

**The model is not for us:** the coded weights alone are 7.51 GB, on an 8 GB card with ~6.3 GB free by
day (the desktop holds ~1.7 GB), before the head, MTP, linear-attention state, KV and compute buffers. Splitting a dense 27B
with the CPU already died at 2.5 tok/s ([graveyard](graveyard.md#qwen38-27b-dense-in-i-quants-25-toks)), compute-bound
on decoding the weights, and trellis code is likely costlier on a CPU. Their engine and binaries fall under "ideas,
not code"; the slot is held by [Bonsai 2 27B](models/bonsai2-27b.md) (5.95 GB, all on the GPU, 38 tok/s). Ideas
worth taking:

1. **A two-stage output head** (`LLAMA_HEAD2=N` in their engine): a small q4_0 copy of the head in VRAM scores
   every token; the exact Q8_0 head in system RAM re-scores the N best. Qwen-family heads have a ~248K-token
   vocabulary, so this frees VRAM on our tight models (Bonsai 2 first, which needs a quiet desktop). Goes with
   draft-head trimming ([inference-optimisation](#inference-optimisation), item 2). Questions: what N keeps
   sampling at temperature 1.0 unchanged (top-k 20 suggests N ≥ 20 for Qwen), and what the RAM re-score costs per
   token over our PCIe. Fork work.
2. **A KV cache split between VRAM and RAM** (`--kv-vram-cells N`): the first N cells of each K/V tensor in VRAM,
   the rest in pinned system RAM in one CUDA address range, the next layer's cold cells copied while the GPU
   computes. Upstream only has all-GPU or all-CPU (`--no-kv-offload`). Past the VRAM line, decode is set by the
   PCIe stream: at 245K they stream 1.87 GB per step at 24 GB/s; we measured 6.45 GB/s
   ([estimate.py](../bench/moe-cache/estimate.py)), so the drop past the line would be much steeper here. Worth it
   only for long-context runs. Their [long-context study](https://github.com/professorpalmer/Kintsugi-27B/blob/9e60a64a640ddaa0b98677a2268c44598839f0e8/research/long-context-decode/README.md)
   documents a real bug in this design (when K isn't a whole number of 2 MiB pages, V spans two host runs and
   the write-back must cover both) and what didn't help (sparse V read, two copy pipelines, 3-bit K, 2-bit head
   stage).
3. **Lowest KL is not best on tasks.** Their own bit maps had up to 9% lower KL than the shipped one but lost
   recall at 117K, AppWorld or long chat ([bit-allocation study](https://github.com/professorpalmer/Kintsugi-27B/blob/9e60a64a640ddaa0b98677a2268c44598839f0e8/research/bit-allocation/README.md)).
   For our [expert pruning](#️-pruning-experts-from-the-moe-models-reap) and quant choices: gate candidates on
   recall and agent tasks, not only KL or perplexity, and choose on one set of seeds, report on fresh ones.

Not for us: batch-invariant kernels, delta slot saves (pi reprocesses only ~25 tokens a turn), their saved-chat
Python layer, and the Qwen3.8 template fix (a system message after the first turn, sent by Claude Code and Codex;
pi doesn't, as far as we know: check before dismissing).

**To revisit:** repo <https://github.com/professorpalmer/Kintsugi-27B>; the version read is
[`9e60a64`](https://github.com/professorpalmer/Kintsugi-27B/tree/9e60a64a640ddaa0b98677a2268c44598839f0e8)
(2026-11-21). Read: the README, `model/`, `engine/`, `benchmarks/`, `research/README.md`, and the first part of
`research/long-context-decode/` and `research/bit-allocation/`. Not read: the bundle scripts and Python layer,
the engine source, `harnesses/`, and the other studies. The engine source is a release asset
(`Kintsugi-27B-v1.1-engine-src-5f393adb5.tar.gz`, sha256 `ded8fd4e…`); if we ever read it, ideas only.

<a id="sd-scripts"></a>

### 🎚️ kohya sd-scripts: training ideas only, none of its code ([kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts), Apache 2.0, read 2026-11-15 at `cb70b9685`, v0.12.3)

The best-known Python trainer for Stable Diffusion models (LoRA, fine-tuning, Textual Inversion). Read through
its README and docs only; nothing was cloned or run. **We reuse ideas, never its code:** it is a big accelerate/torch
stack that would need a new "repo code" exception in [Security](security.md), and it doesn't train FLUX.2 Klein
(its models are SD1/2, SDXL, SD3, FLUX.1, Lumina, HunyuanImage, Anima and MOSS-Image). Its inference scripts, block
swap, ConvRot INT8 and fp8 options are no use to us either (sd.cpp does our inference; they target DiT models).
Docs to check for details: [`docs/train_network_advanced.md`](https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network_advanced.md)
(loss options, LoRA+), [`docs/masked_loss_README.md`](https://github.com/kohya-ss/sd-scripts/blob/main/docs/masked_loss_README.md),
[`docs/validation.md`](https://github.com/kohya-ss/sd-scripts/blob/main/docs/validation.md),
[`docs/train_textual_inversion.md`](https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_textual_inversion.md),
[`docs/train_leco.md`](https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_leco.md),
[`docs/loha_lokr.md`](https://github.com/kohya-ss/sd-scripts/blob/main/docs/loha_lokr.md). Worth trying, each a few
lines in our own [`train_style.py`](../bench/imagegen/train_style.py) / [`invert.py`](../bench/imagegen/invert.py),
judged like our other image runs (two seeds, by eye, plus the no-trigger leak check):

1. **Min-SNR loss weighting** (`--min_snr_gamma 5`). Our loss is plain MSE. The 3000-step Van Gogh LoRA needed 50 min
   to match the prompt; faster convergence would shorten that. Also in the same section: `--debiased_estimation_loss`
   and Huber loss.
2. **Masked loss** (a mask image or the picture's alpha channel limits where the loss counts). The invented man's
   LoRA picked up the blue shirt and glasses from the training pictures. Regularisation pictures already stop that
   leak; training only on the masked subject might do the same without making 20 extra pictures. Needs masks for the
   20 training pictures.
3. **A validation loss** (a held-out set, same noise seed and timesteps every time). Fills the "no automatic score"
   gap with a cheap number for choosing the step count and for spotting overfitting. Not a likeness score.
4. **Multi-resolution noise and input perturbation** (`--multires_noise_iterations 6-10 --multires_noise_discount 0.3`,
   `--ip_noise_gamma 0.1`). We already have `--noise-offset`. Might help detail and the style leak; no evidence either way.
5. **More than one vector per learned word** (`--num_vectors_per_token`, up to 77 tokens shared with the prompt).
   Our word has one vector. More might fix the green habit and Gris's missing structure, at a few KB more.
   Needs the token-expansion code in `invert.py`.
6. **LoRA+** (`loraplus_lr_ratio=16`: the up matrix learns faster than the down one). We already train with
   parameter groups, so it's one more line. Mainly a speed-up: worth trying with item 1 if the 3000-step runs stay
   necessary.
7. **LECO** (`sdxl_train_leco.py`): a LoRA trained from prompts alone, no pictures, to erase, strengthen or slide a
   concept. A candidate for removing the green bias or for a real style-strength knob (our `(word:0.7)` was no
   control). Its docs give no VRAM figure, so it needs a fit test first.
8. **LoKr / LoHa** (from LyCORIS). sd.cpp already maps `lokr_*` and `hada_*` tensors (`src/name_conversion.cpp`), so
   they should load. Low priority: our [fine-tuning notes](fine-tuning.md) cite a 2026 study finding well-tuned plain
   LoRA about as good.

Its LoRA post-processing scripts (`networks/resize_lora.py`, `merge_lora`, `extract_lora_from_models`) now also live in
a smaller companion repo, covered [next](#sd-lora-tools). Whatever we write must keep the check that every tensor loads
in `sd-cli` (ours: 1120 of 1120).

<a id="sd-lora-tools"></a>

### 🧰 kohya sd-lora-tools: LoRA post-processing, ideas only ([kohya-ss/sd-lora-tools](https://github.com/kohya-ss/sd-lora-tools), Apache 2.0, read 2026-11-15 at `cb4e63256`)

Small command-line tools that work on `.safetensors` LoRA files, by the maintainer of
[sd-scripts](#sd-scripts). Young and quiet: v0.1.0, 9 stars, last push 2026-04-26, and its README says APIs may
change without notice. Needs torch. **Ideas only, same rule as above;** each tool is a short script we'd rewrite
against our own LoRAs (the files from [`train_style.py`](../bench/imagegen/train_style.py)). Docs, one page per tool:
[`docs/`](https://github.com/kohya-ss/sd-lora-tools/tree/main/docs) (`resize.md`, `merging.md`, `extract_lora.md`,
`convert_lora.md`, `utilities.md`); the source is in `src/sd_lora_tools/`.

1. **Post-hoc EMA of checkpoints** (`lora_post_hoc_ema.py`): averages the LoRA files saved along one training run,
   with an exponential decay or the power-function EMA from the paper its code cites
   ([arXiv 2312.02696](https://arxiv.org/abs/2312.02696)). `train_style.py --ckpts` saves the checkpoints, but only
   `subject-dora` and `subject-lora+norm` have any so far (250 and 500 steps), so a fair test needs one rerun with
   more of them, such as `--ckpts 250,500,750`. Then: does the average leak less, or look steadier across seeds, than
   the last checkpoint? (LoRA only. A learned word's 8 KB vectors could be averaged the same way, but `invert.py`
   doesn't save intermediate words yet.)
2. **Per-layer scaling when merging** (`svd_merge.py --regex_scales`, "LBW"): scales chosen layers by a regular expression
   as the LoRA is written. The strength control we lacked: `(word:0.7)` on a learned word barely changed anything, and a
   whole-file LoRA weight dims every layer at once. Weakening only some U-Net blocks might keep a style's paint and drop its green
   or its darkening. Test on the Van Gogh 3000-step LoRA, which leaks most.
3. **Rank reduction by singular values** (`resize.py --dynamic_method sv_ratio|sv_fro|sv_cumulative`). Two uses: shrink the
   47 MB rank-8 LoRA, and, more useful, read off how much of rank 8 the training used. Every SDXL run so far was
   rank 8, so we don't know yet whether a lower or higher rank would do.
4. **Blending two LoRAs into one file** (`svd_merge.py --models a b --ratios …`, with optional rank reduction): a
   style LoRA plus a person LoRA as a single file, to compare with our two-learned-word stack
   ([`stack.py`](../bench/imagegen/stack.py)). `sd-cli` can already load LoRAs, so baking into a checkpoint
   (`merge_to_model.py`) isn't needed.
5. **LoRA from the difference of two checkpoints** (`extract_lora.py`: SVD of tuned minus original weights; works on plain
   safetensors for SD, SDXL, FLUX and others). It would turn a published SDXL fine-tune into a small file for `--lora`,
   only from safetensors we've pinned and audited, per the [security routine](security.md); it can't recover what a
   low-rank file can't hold. Not for our norm-only runs: they change only 1-D norm gains and biases, not the weight
   matrices it factors, and those are already a 2.5 MB file.

Small utilities worth reimplementing as one check script: `check_weights.py` (tensor statistics) and
`compare_weights.py` (two files equal within a tolerance), for "did the bake or the resize change anything".

Low priority: `convert_lora.py` between sd-scripts and Diffusers key naming. Our diffusers-format LoRA already loads in
`sd-cli` (1120 of 1120), so conversion only matters if we share files with a tool that wants the other layout.

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
| Groot and LFM2.5 get the expert cache by default (2026-11-07): +20% and +29% decode, and more room for the desktop than their static splits | [`bench/moe-cache`](../bench/moe-cache/README.md#groot-and-lfm25-sizes-for-the-launcher-modelsstep17-groot-lfm25-sizesh-2026-11-07) |
| The vision probe through `scripts/llm` (2026-11-07): same scores within one answer's noise, 5–27% less time per image | [vision probe](vision-probe.md#the-results) |
| Every model through `scripts/llm` (2026-11-07): loads, pi one-shot, status/stop, the "stop the server too?" question (n / y / Enter, and not asked for a server it didn't start), contexts, Spark/Bonsai thinking flags; the recheck at launcher defaults (cached models 28–62% faster); TwIL's first speed and code check | [`bench/recheck/`](../bench/recheck/README.md#through-scriptsllm-2026-11-07) |
| TwIL-8B's first pibench (2026-11-07): 8/18 at 24K, code chores 7 of 9, web chores 1 of 9; the context ran out in 14 of 18 attempts, and 32K solved no more (3/6, twice as slow); `scripts/llm twil` now opens pi | [its page](models/twil-8b.md#our-numbers), [pibench](pibench.md#the-results) |
| VibeThinker's audit (2026-11-07): clean, templates identical to the source | [security](security.md#audits-so-far) |
| Relaxed draft acceptance | [graveyard](graveyard.md#relaxed-draft-acceptance-faster-but-it-rambles) |
| The first image models, style and person training | [Making pictures](image-generation.md) |
| Cactus's Mirage (dropped) | [`bench/cactus/`](../bench/cactus/README.md) |
