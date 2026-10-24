# 🪦 The graveyard (things that didn't make it)

Every working model on [the scoreboard](scoreboard.md) is standing on a small pile of things that
didn't work. They're here so you don't have to dig the same holes. Each one gets a fair
headstone: what we tried, the numbers, and the cause of death.

All measured on the same laptop: RTX 2070 Max-Q (8 GB), 32 GB RAM. A useful number to keep in
mind first: on this rig the *same* configuration, run again, scatters by about **±13%**. Any
"speedup" smaller than that is a rumour.

---

## Qwen3.8-27B, dense, in i-quants: 2.5 tok/s

**Tried:** Qwen's dense 27B as ISTA-DASLab's GSQ-RCO `IQ3_XXS-mtp` file (10.4 GB), split by whole
layers: 30 of 64 on the GPU, the rest on the CPU, 32K context.

**Numbers:** 2.5 tok/s. Tool calls worked; the answer arrived, eventually. Pushing to 36 GPU
layers crashed on the first request (CUDA out of memory in the cuBLAS workspace). A `llama-bench`
sweep fit a line of ~0.110 s + 0.008 s per CPU-resident layer, i.e. ~9 tok/s only if all 64
layers were on the GPU. Hyperthreads made it worse; its built-in MTP head made it worse too
(2.0 tok/s, 57% draft acceptance).

**Cause of death:** the CPU half was **compute-bound, not bandwidth-bound**: 600% CPU (all 6
cores), GPU at 20%, only ~14 GB/s of RAM traffic. The file is mostly IQ1/IQ2/IQ3 i-quants, which
are expensive to decode on a CPU. The pre-test estimate (6–8 tok/s from bandwidth) was wrong by
~3× for exactly that reason.

**Afterlife:** the same model lives on as [Bonsai 2 27B](models/bonsai2-27b.md): ternary weights,
5.95 GB, all on the GPU, ~27 tok/s.

## ik_llama.cpp for that same 27B: slower

**Tried:** the ik_llama.cpp fork, whose README promises better CPU performance and faster
i-quant kernels, built with the same toolchain.

**Numbers:** 1.75 tok/s vs mainline's 2.63 at 30 GPU layers; CPU-only 0.93 vs 1.01.

**Cause of death:** its CPU i-quant kernels were not faster on this file. One fork, zero wins.

## Gemma 4 MTP self-speculation: real, but not at temperature 1.0

**Tried:** Gemma 4's Multi-Token Prediction head (a 0.25 GB draft that ships inside Unsloth's
QAT repo) with `--spec-type draft-mtp`. Speculative decoding is lossless by construction, so the
only question is speed.

**Numbers:** at greedy (temp 0), 79–88% draft acceptance and **+15–31%** decode (≈21 → 27
tok/s). At the default temp 1.0, acceptance falls to 66–74% and the gain is **within noise**. And
`--spec-draft-p-min > 0` is a trap: with sampling it collapsed the output into `fmt-fmt-fmt…`
loops, which look like 85–100% acceptance because degenerate text is very easy to predict.

**Cause of death:** not dead, just unemployed. It's a win for greedy or low-temperature coding,
and buys nothing at the sampling settings these models are meant to run with. If you want it:
`--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0`.

## EAGLE3 for Gemma 4: slower, unstable, crashes

**Tried:** the `RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3` draft, converted to GGUF, on a
llama.cpp build (`88a3927`) with fresh EAGLE3 support.

**Numbers:** ~42% acceptance (vs MTP's ~80%, and not a quantization artefact). Its draft is a
full transformer layer, so it was **slower than no speculation even at greedy** (−19% at n-max 3,
−34% at n-max 5).

**Cause of death:** temp 1.0 degenerated, and the chat-template path (which is how pi talks to the
server) segfaulted. Bonus trap: one "fix confirmed" test was really an empty reply hiding the
crash. Always check that it *generates*, not just that it stays alive.

## DiffusionGemma 26B-A4B: 5.2 tok/s

**Tried:** a block-diffusion model on the Gemma 4 base, run through an open llama.cpp PR's
`llama-diffusion-cli` plus a small shim so pi could talk to it. It worked end to end.

**Numbers:** best config (`NCMOE=27`, flash attention on auto) ≈ **5.2 tok/s**, against ~30 tok/s
for normal decoding of the same split: 5.8× slower. 26 GPU expert layers ran out of memory; forcing
flash attention on made it 2.7× slower. pi's default system prompt didn't even fit in its 4K
context.

**Cause of death:** diffusion spends roughly **66×** the per-token compute of autoregressive
decoding (a 256-token canvas, ~21 denoising steps, 2 blocks for a 162-token answer). With most
experts on the CPU, that multiplier lands on the slowest hardware in the box. Lesson that shaped
everything after: **diffusion only pays when the whole model fits in VRAM.**

## Nemotron-Labs-Diffusion-8B: fast on code, broken tool calls

**Tried:** NVIDIA's tri-mode 8B (autoregressive, block diffusion, or self-speculation where a
diffusion pass drafts and an AR pass verifies), Q4_K_M at 5.2 GB, fully on the GPU, on the
spiritbuun/buun-llama-cpp fork. It fit, as the previous headstone demanded.

**Numbers:** plain AR decoding 41.8 tok/s. Self-speculation with 16-token drafts: **60.7 tok/s on
code (1.45×)**, but at best 27.7 tok/s on prose (0.66×) because most drafts get rejected. Block
diffusion ~16.5 tok/s (0.4×) on everything. The server path hardcodes 4-token drafts: 42.3 on
code, 28.0 on prose.

**Cause of death:** tool calls don't parse. The model writes `<function=…>` without the
`<tool_call>` wrapper its own template expects, and keeps generating past the call. An agent
model that can't call tools is just a very fast essay writer. Not adopted.

## FlashMap drafter: unfinished

**Tried:** porting FlashMap, a ~500M "flow-map" speculative drafter (from a paper reporting
4–6× speedups on an H200), to llama.cpp for the first time, on Qwen3-8B Q4_K_M. It ran end to end
in 7.65 GB of VRAM.

**Numbers:** mean acceptance length ~3.2 tokens on math (paper: 7.2–7.7) and ~1.5 on prose
(paper: 3.1–3.3). Net: math +50% (59–63 vs 39–42 tok/s), code ±0, prose −25–35%; overall
39.0 vs 39.8 tok/s for the target alone.

**Cause of death:** not dead, in a coma. Target quantization, the adaptive draft-depth controller
and the fused single-graph cycle were all ruled out; the prime suspect is a numerical bug in the
drafter-graph port. The next step is one instrumented NumPy-vs-ggml comparison of a single draft
step, not more benchmark sweeps.

## Gemma 4 12B coder at Q2_K: fits, runs, talks nonsense

**Tried:** the smallest quant of the 12B coder fine-tune. It fits entirely on the GPU and runs at
~19 tok/s.

**Numbers:** degenerate output on every path: raw prompts, the chat template at temp 0.7, the
native turn format. Things like `l. l. l.`, CJK noise, or rambling to the token limit.

**Cause of death:** the quant is simply too aggressive for this model. Q4_K_M of the same model
writes clean, correct code. Skip Q2_K; the run script refuses it anyway.

## Vulkan as the default backend: 5× slower

**Tried:** the zero-build path, conda's Vulkan llama.cpp, which works on any driver.

**Numbers:** ~4.9 tok/s on Gemma 4 26B vs **~23.5 tok/s** with a source-built CUDA backend at the
identical CPU/GPU split. Pure CPU is ~1.9.

**Cause of death:** not dead, demoted. Vulkan is still the fallback when you can't build CUDA, but
on this Turing laptop its MoE path is the bottleneck. Build CUDA (see
[getting started](getting-started.md)).

## The ones we never let in the door

Checked and skipped on paper, so you don't have to:

- **Qwen3.8-Flash-Next** (REAP-288 / GSQ-RCO builds): the smallest files are 58–66 GB+, more than
  32 GB RAM + 8 GB VRAM.
- **Qwen3.8-Whittle-16B:** its own card calls it a "non-starter for agent and long multi-turn use:
  repetition looping".
- **Whittle-Qwen-3.8-35B-A3B:** a one-person research preview trained on 4K-token rows whose own
  table doesn't beat stock Qwen3.6-35B-A3B.
- **LLaDA2.2-mini:** a 16B MoE diffusion model, ~9 GB at Q4: experts on the CPU, i.e. the
  DiffusionGemma problem again.
- **Gemma 4 31B (dense):** no experts to park in RAM, so most of it would run on the CPU; the
  estimate was ~1–3 tok/s (not measured). For this card the realistic upgrade is a
  higher-precision quant of the 26B MoE, not a bigger model.

## Moved out (not dead!) 🏠

**VibeThinker 1.5B / 3B**, small math-and-coding reasoners (the 1.5B runs at ~100 tok/s fully on
the GPU at Q8_0), grew up and got its own place:
[vibethinker_tests](https://github.com/ricardodeazambuja/vibethinker_tests).
