# 🧳 How do you fit a 40B model in 8 GB?

Short answer: you don't. Not all at once, anyway. You fit *the part that matters right now* and
let the rest live somewhere cheaper. This page is the packing list: every trick that got a model
on [the scoreboard](scoreboard.md) running on one RTX 2070 Max-Q (8 GB VRAM) with 32 GB of system
RAM.

Two things fight over those 8 GB, and you will meet them on every page of this repo:

```
 8 GB of VRAM
 ┌─────────────────────────────────────────────────────────────┐
 │ model weights (the brain)  │ KV cache (the memory of the   │ compute │
 │                            │ conversation, grows with ctx) │ buffers │
 └─────────────────────────────────────────────────────────────┘
       ↑ shrink with quantization,     ↑ shrink with smarter attention
         or move to RAM                  and KV quantization
```

Everything below is a way to make one of those boxes smaller, or to move it out of the GPU
without making everything slow.

---

## 1. The big one: park the experts in RAM

Most of the big models here are **Mixture-of-Experts** (MoE). Instead of one giant feed-forward
block per layer they have a crowd of small "experts", and a router picks a handful per token.
Gemma 4 26B-A4B, for example, routes each token to **8 of 128** experts per layer. Qwen3.6-35B-A3B
and Flint have 256 experts per layer and pick 8 and 6 of them, plus one shared expert.
Nemotron Bolt 74B picks 9 of 240–432 per layer, which is how 74B parameters become ~9B per token.

So the experts are most of the *parameters* but only a sliver of them is *touched* per token.
llama.cpp exploits that by putting different tensors on different devices:

| What | Where | Why |
|---|---|---|
| Attention, embeddings, router, norms | GPU | small, used by every token |
| KV cache (your context) | GPU | grows with context, needs fast access |
| **Expert weights** | **system RAM** | the bulk of the model, but sparse per token |

The flags:

- `--cpu-moe`: **all** layers' experts go to RAM. Least VRAM, slowest.
- `--n-cpu-moe N`: the experts of the **first N** layers go to RAM, the rest stay on the GPU.
  Lower N = more experts on the GPU = faster, but more VRAM.

In this repo that's the `NCMOE` knob. Leave it empty for "all experts in RAM":

```bash
QMODEL=qwen36 NCMOE= bash scripts/run-qwen.sh   # every expert in RAM: least VRAM
QMODEL=qwen36 bash scripts/run-qwen.sh          # default NCMOE=32: 8 layers' experts on the GPU
```

What it trades, measured on Flint (128K context, q8_0 KV):

| Experts on the GPU | VRAM | Generation |
|---|---:|---:|
| none (`--cpu-moe`) | 4.4 GB | 22.2 tok/s |
| 7 layers (`NCMOE=42`) | 6.6 GB | 24.9 tok/s |
| 9 layers (`NCMOE=40`) | 7.2 GB | 26.1 tok/s |

Notice the gain is modest: the CPU still does most of the expert work. Each layer's experts cost
Flint ~300 MiB of VRAM. The per-model defaults (`NCMOE=32` for Qwen3.6, `34` for the distill,
`42` for Flint) are the measured "fits with ~0.5 GB to spare next to a desktop" values.

**Why isn't RAM bandwidth the bottleneck?** Because only a few experts fire per token. On
Gemma 4 26B at 25 tok/s with 22 of 30 layers' experts in RAM, that works out to roughly 15 GB/s of
RAM reads, comfortably under dual-channel DDR4. Measure before you believe "it must be
memory-bound": an earlier guess here was wrong by exactly that.

## 2. Quantization, in plain words

A model's weights are numbers. At full precision each one takes 16 bits. Quantization stores them
with fewer bits and a little shared scale per small group, trading a bit of accuracy for a lot of
size.

- **Q8_0 / Q6_K / Q5_K / Q4_K_M:** 8, ~6, ~5 and ~4.5 bits per weight. Q4_K_M is the usual "sweet
  spot" (the Gemma 12B fine-tunes, Qwen3-VL-8B). Q8_0 is close to lossless (Spark, 4.4 GB).
- **Unsloth "UD" quants (UD-Q4_K_XL, UD-Q3_K_XL, UD-IQ4_XS):** mixed precision, spending more bits
  on the tensors that matter. Flint runs as UD-Q3_K_XL (18.5 GB), because UD-Q4_K_XL (25 GB) would
  leave the page cache no room in 32 GB of RAM.
- **QAT (quantization-aware training):** the model was trained knowing it would be quantized, so
  the 4-bit file keeps much more quality. That's why Gemma 4 26B-A4B is used as a QAT file.
- **i-quants (IQ1–IQ4):** squeeze harder with lookup tables. Great on a GPU, **expensive to decode
  on a CPU**. That killed the dense Qwen3.8-27B here (see [the graveyard](graveyard.md)).
- **Ternary (PTQ1_0):** every weight is −1, 0 or +1, with one FP16 scale per 128 weights, about
  1.72 bits per weight. Bonsai 2 squeezes the whole Qwen3.8-27B into **5.95 GB** this way, small
  enough for the entire model to sit on the GPU. It needs PrismML's llama.cpp fork.

Rule of thumb from our scars: **the most aggressive quant of a model can simply break it.** The
Gemma 12B coder's Q2_K fits on the GPU and runs at ~19 tok/s, and produces nonsense.

## 3. The KV cache, and why some models get long context for free

The KV cache is the model's memory of the conversation: for every token in the context, every
attention layer keeps a key and a value. So its size is roughly:

```
KV bytes per token = (layers with full attention) × 2 (K and V) × KV heads × head dim × bytes per value
```

The trick is in the first factor. Many new models only give *some* layers a long memory:

- **Sliding-window attention (SWA):** most layers only look at the last few hundred or thousand
  tokens, so their cache stops growing. Gemma 4 (1024-token windows), Spark (3 of every 4 layers
  use a 512-token window), Flint (35 of 49 layers, 513-token window), Dory (1024-token windows).
- **Linear attention / Gated DeltaNet:** the Qwen3.6/3.8 family runs three of every four layers
  with a fixed-size recurrent state and **no KV cache at all**.
- **Mamba layers:** Nemotron Bolt 74B goes furthest. Of its 88 layers, 40 are Mamba-2 (a fixed
  ~160 MB of state, whatever the context), 40 are experts, and only 8 are attention, with 2 KV
  heads each.
- **Shared KV:** the Bolt E4B (a Gemma 4 E4B underneath) uses 512-token windows in most layers,
  and its last 18 layers reuse the caches of earlier ones.

Here is what that buys, per token of context (computed from each model's config):

| Model | Layers with a long KV cache | f16 | q8_0 | 128K at q8_0 |
|---|---|---:|---:|---:|
| Nemotron Bolt 74B | **8 of 88** | 8 KiB | ~4.25 KiB | **0.57 GB** |
| Nemotron Bolt E4B | 4 of 42 | 16 KiB | ~8.5 KiB | 1.14 GB (it runs 64K: 0.57 GB) |
| Qwen3.6-35B-A3B / distill | 10 of 40 | 20 KiB | ~10.6 KiB | 1.42 GB |
| Flint | 14 of 49 | 28 KiB | ~15 KiB | ~1.9 GB |
| Spark-X2.6-4B | 9 of 36 | — | ~19 KiB | see §5 |
| Qwen3.8-27B / Bonsai 2 | 16 of 64 | 64 KiB | 34 KiB | 4.55 GB |
| Qwen3-VL-8B (no tricks) | **36 of 36** | 144 KiB | ~77 KiB | doesn't fit |

Same 8 GB, wildly different context: the 74B Bolt gets 128K for about half a gigabyte and the
Qwen3.6 MoE for 1.42 GB, while Qwen3-VL-8B,
with plain attention in every layer, tops out at 16K (2.42 GB at f16, 1.28 GB at q8_0).

Side effect worth knowing: on Gemma 4 26B, **64K decodes at essentially the same speed as 32K**
(~22–25 tok/s either way). Decode speed is set by where the experts live, not by the context
length.

## 4. Quantize the KV cache too (usually)

`KVQUANT=q8_0` stores the cache at ~8.5 bits instead of 16, roughly halving it. It's a
long-context lever: at 32K Gemma's cache is ~0.6 GB and quantizing it is noise, but at 128K it
frees about 2 GB, enough to move more experts back to the GPU:

| Gemma 4 26B context | f16 KV | q8_0 KV | Gain |
|---|---|---|---|
| 65536 | `NCMOE=22` → 26.1 tok/s | `NCMOE=20` → 30.5 tok/s | +17% |
| 131072 | `NCMOE=27` (only fit) → 19.5 tok/s | `NCMOE=22` → 27.0 tok/s | +38% |

Two catches:

- **It needs flash attention.** llama.cpp won't quantize the V cache without it, so any `KVQUANT`
  turns `-fa` on.
- **It's not always allowed.** PrismML's fork (Bonsai 2) has no Turing (sm_75) code for its q8_0
  flash-attention kernel: the server loads fine, then aborts on the first request with *"no device
  code compatible with CUDA arch 750"*. Bonsai 2 therefore runs an **f16** cache, which is exactly
  why it's stuck at 16K (32K doesn't fit).

## 5. The hidden tax: flash attention's f16 copy

Spark-X2.6-4B looked like a 128K model on paper: Q8_0 weights plus ~19 KiB/token of q8_0 KV. In
practice 128K ran out of memory by about 165 MiB. The culprit: with a *quantized* KV cache, flash
attention works on an **f16 copy of one layer's K and V**, which lives in the compute buffer and
grows with context (~5 KiB per token here). The compute buffer went from 218 MiB at 32K to
697 MiB at 128K.

So a q8_0 cache really cost Spark ~24 KiB/token, not 19. Solved for ~0.3 GB of headroom, that
gives Spark's default of **110592 tokens (108K)**. Lesson: the KV size you calculate is a floor,
not a budget.

## 6. Big batches for big prompts (on RAM-expert MoEs)

Generation is one token at a time; processing your prompt (prefill) is many tokens at once. With
the experts in RAM, a **big batch** lets llama.cpp ship the expert matrix multiplies to the GPU
for the whole batch. On Flint, raising the batch from 512 to 2048:

| Batch / ubatch | Experts on GPU | VRAM | Prefill (14.4K tokens) | Generation |
|---|---|---:|---:|---:|
| 512 | 9 layers (`NCMOE=40`) | 7.2 GB | 192 tok/s | 26.1 tok/s |
| **2048** | **7 layers (`NCMOE=42`)** | **7.25 GB** | **542 tok/s** | 24.3 tok/s |

Prefill nearly triples for about 1 tok/s of generation. The bigger compute buffer costs one layer
of experts, which is why Flint's default is `NCMOE=42` with batch 2048. For an agent like pi, whose
every turn re-reads a big system prompt and file dumps, that's the difference between waiting a
minute and waiting 25 seconds.

Nemotron Bolt 74B shows the same effect with every expert in RAM: 136 → 310 tok/s of prefill from
batch 2048 alone, generation unchanged at 14 tok/s. Putting expert layers on the GPU barely helps
it (13–15 tok/s whatever we tried), because per token it touches ~6.8B always-on parameters (on
the GPU) and only ~1.7B routed-expert ones (in RAM): the GPU side sets the pace.

## 7. Dense models: offload whole layers instead

A dense model has no experts to park; every token goes through every layer. The lever is
`-ngl N`: put the first N layers on the GPU and run the rest on the CPU. The Gemma 4 12B coder
(Q4_K_M, 48 layers) runs best with `NGL=42` (6 layers in RAM) at 32K context: ~17 tok/s.

| 12B coder quant | Best `-ngl` | tok/s |
|---|---|---:|
| Q4_K_M | 42 | ~17 |
| Q6_K | 32 | ~7 |
| Q8_0 | 24 | ~4.5 |

Every layer you push to the CPU is paid for on every token, so this gets slow fast. And don't
borrow the MoE flags: `--cpu-moe` does nothing on a dense model.

## 8. Or just be small: everything on the GPU

The happiest models here don't split at all:

| Model | File | Where | Speed |
|---|---|---|---:|
| Spark-X2.6-4B | Q8_0, 4.4 GB | all on GPU, 108K | ~50 tok/s |
| Qwen3-VL-8B | Q4_K_M 5.03 GB + 0.75 GB projector | all on GPU, 16K | ~38 tok/s |
| Dory1.0-4B | Q5_K_M, 3.67 GB | all on GPU, 64K | 29 tok/s (2 loops) |
| Bonsai 2 27B | ternary, 5.95 GB | all on GPU, 16K | ~27 tok/s |

No RAM traffic, no CPU math: fast, as long as you can live with the context.

### Dory's surprise: `-ngl 999`

Dory is a *looped* model: a block of 20 layers runs 1–4 times per token. The llama.cpp build for
it counts every loop as extra layers, so the usual "all layers" value `-ngl 99` silently leaves
weights on the CPU. At 4 loops that kept 2.9 GB of weights and most of the KV cache on the host:
3.8 tok/s instead of 19. The clue was VRAM going *down* as loops went up. `-ngl 999` fixes it, and
the repo's Dory default already uses it.

## 9. TurboQuant: an even smaller V cache

The Gemma 4 12B agentic fine-tune is dense too, and at Q4_K_M it *almost* fits. The fix was
`turbo3`, a ~3.5-bit V-cache codec (Walsh-Hadamard-rotated polar quantization) from the
[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) fork. It's a runtime
option, not a model format. With `-ctk q8_0 -ctv turbo3` (keys are touchy, values tolerate
aggression), **all 48 layers fit on the GPU**: 7.76 of 8 GB at 16K, ~28.3 tok/s, about 1.7× the
partial-offload speed.

```bash
bash scripts/start-turbo.sh    # = TURBO=1 QUANT=Q4_K_M CTX=16384, then pi
```

It needs `--parallel 1 -b 512` (the run script's default with `TURBO=1`); the default
`n_parallel=4` + `n_batch=2048` OOMs.

## 10. Pick the right backend: CUDA, not Vulkan

Same model (Gemma 4 26B), same split (`NCMOE=22`), same laptop:

| Backend | Generation |
|---|---:|
| CPU only | ~1.9 tok/s |
| Vulkan (stock build) | ~4.9 tok/s |
| **CUDA (built from source)** | **~23.5 tok/s** |

CUDA is about 5× Vulkan here, far outside the usual "CUDA is a bit faster" folklore. The prebuilt
conda CUDA binaries need a newer driver than this laptop's, so `scripts/build-llama-cuda.sh` builds
llama.cpp against the driver's own CUDA version in an isolated environment. Details in
[getting started](getting-started.md).

## 11. VRAM is not the only wall 🧱

The GPU isn't the only thing that can run out. The MoE models keep 15–20 GB of experts in system
RAM, and Linux will happily let you load all of that and then **kill the server** a few seconds
later when something else needs memory. It looks like this: a clean load, pi connects, and the
server log just says `Killed`. That's the kernel's OOM killer, not the GPU.

```bash
journalctl -k | grep -i oom      # confirm it was host RAM
```

Fixes: a smaller context, a smaller quant (that's why Flint uses UD-Q3_K_XL), closing the browser
with 40 tabs, or more swap. Re-tuning `NCMOE` won't help; the GPU was never the problem. More
symptoms like this live in [troubleshooting](troubleshooting.md).
