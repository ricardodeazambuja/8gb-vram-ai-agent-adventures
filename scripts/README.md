# Scripts — launch, build, download, benchmark

Everything under `scripts/`, and the knobs they take. Every script prints its own manual with
`-h` (it's the header comment, so it never drifts from the code).

> The 3-command quick start lives in the [main README](../README.md#quick-start), the full setup
> in [getting started](../docs/getting-started.md). This file is what you reach for once you want
> to change a knob.

## Once

| Script | What it does |
|---|---|
| `build-llama-cuda.sh` | Builds llama.cpp from source against CUDA into `vendor/` (or `SRC_DIR=`). Picks a ref (`LLAMA_REF`), a repo (`LLAMA_REPO`, for forks) and an optional `PATCH`; auto-detects the GPU arch and smoke-tests the result. The per-model commands are in [getting started](../docs/getting-started.md#3-build-llamacpp-once-per-flavour). |
| `setup.sh` | Creates the `llamacpp` conda env (llama.cpp + huggingface_hub) and downloads Gemma 4 26B into `models/`. `BACKEND=cuda` also builds the CUDA backend. Idempotent. |
| `download-models.sh` | Downloads any of `qwen36 distill flint bolt e4b bonsai2 qwen3vl dory spark qwen38 dflash`, pinned to a repo revision and sha256-checked. Needs the `hf` CLI in the `ENV_NAME` env (default `llamacpp`). |
| `configure-pi.sh` | Adds the `llamacpp` provider (every model in `../config/pi-provider.json`) to `~/.pi/agent/models.json`. |

## Every day

| Script | What it does |
|---|---|
| `start.sh` | **Gemma 4 26B**, all-in-one: server + wait + pi, cleans up on exit. `--menu` walks you through every knob; first launch offers to auto-tune. Knobs below. |
| `run-server.sh` | **Gemma 4 26B**, server only (`--cpu-moe`, 32K, port 8080). `--image` loads the vision projector. |
| `start-qwen.sh` / `run-qwen.sh` | **Every other MoE and small model**, picked with `QMODEL=qwen36 \| distill \| flint \| bolt \| e4b \| bonsai2 \| qwen3vl \| dory \| spark \| qwen38`: server + pi, or server only. Per-model defaults (context, offload, KV type, sampling, build) are in `run-qwen.sh -h`. |
| `start_flint.sh`, `start_bolt.sh`, `start_e4b.sh`, `start_bonsai2.sh`, `start_spark.sh`, `start_dory.sh`, `start_qwen3vl.sh` | One-word launchers for those models: `start-qwen.sh` with the right `QMODEL` and pi flags, refusing to start if another model already holds the port. |
| `run-12b-coder.sh` | Gemma 4 12B coder (dense, partial GPU offload). |
| `run-12b-agentic.sh`, `start-agentic.sh`, `start-turbo.sh` | Gemma 4 12B agentic; `start-turbo.sh` is the zero-config TurboQuant setup. |
| `run-pi.sh` | Opens pi against whatever is serving on port 8080. |
| `stop-server.sh` | Stops the server listening on the port (default 8080). |

## Measuring

| Script | What it does |
|---|---|
| `qwen-bench.py` | Quick smoke test of whatever is serving: generation speed, a tool call, a ~15K-token prompt. |
| `qwen-codecheck.py` | Asks for `merge_intervals` and checks it against 500 random inputs. |
| `benchmark-config.sh` | Finds the fastest `NCMOE`/`CTX` for Gemma 4 26B on *your* GPU. |
| `benchmark-mtp.sh`, `run-server-mtp.sh` | Speculative decoding with Gemma's MTP head (it didn't pay off; see the [graveyard](../docs/graveyard.md)). |
| `make-speed-chart.py` | Redraws the backend speed chart. |
| `_banner.sh`, `_tuning.sh` | Shared helpers sourced by the launchers. |

The bigger benchmarks (pibench and the vision probe) live in [`../bench/`](../bench/README.md).

The rest of this page is the knob reference for **Gemma 4 26B** (`start.sh` / `run-server.sh`).
For the other models, run `bash scripts/run-qwen.sh -h` and see their [model pages](../docs/models/).

## Useful overrides

```bash
NCMOE=22    bash scripts/run-server.sh    # FASTEST on 8 GB: 8 expert layers on GPU (default is all-CPU experts)
CTX=65536   bash scripts/run-server.sh    # bigger context (see the ceiling table below)
BACKEND=cuda bash scripts/run-server.sh   # force native CUDA backend (auto-selected once built)
BACKEND=cpu bash scripts/run-server.sh    # no GPU offload — slow, benchmark baseline only
PORT=9000   bash scripts/run-server.sh    # different port
CTX=131072 KVQUANT=q8_0 bash scripts/run-server.sh   # quantize the KV cache to fit long context
bash scripts/run-server.sh --image        # enable image input (CLI flag, not an env var)
```

These env vars work on **`run-server.sh`, `start.sh`, and `build-llama-cuda.sh`** alike (e.g.
`CTX=65536 NCMOE=27 BACKEND=cuda bash scripts/start.sh`).

## A different quant, or your own file

`setup.sh` fetches the QAT 4-bit GGUF by default. The QAT repo ships **only this one 4-bit file**
(QAT is trained for 4-bit). For other precisions, use the **non-QAT** repo
[`unsloth/gemma-4-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/tree/main),
which has the full Q2–Q8 range (Q5 ≈ 21 GB, Q6 ≈ 23 GB, Q8 ≈ 27 GB).

```bash
# a higher-precision quant of the SAME model (better quality, more RAM, SLOWER; non-QAT repo)
MODEL_REPO=unsloth/gemma-4-26B-A4B-it-GGUF \
MODEL_FILE=gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf bash scripts/setup.sh

# already have a GGUF somewhere? skip the download and point the server at it
MODEL=/path/to/model.gguf bash scripts/run-server.sh
```

(A bigger quant scales up the GPU-resident tensors too, so fewer expert layers fit in VRAM and the
RAM-resident ones are heavier — expect *slower* generation, not the same ~23 tok/s. What fits in
your RAM, and whether a bigger Gemma 4 is worth it, is in
[how it fits](../docs/how-it-fits.md).)

## Context size

The context window is `CTX` (**default 32768** = 32K; model max **262144**). The context is the
KV cache, and it lives in **VRAM**, not RAM (only the expert weights are offloaded) — so it shares
the 8 GB with the attention weights and on-GPU experts. **More context means fewer experts on the
GPU, which means slower** — context trades directly against speed:

| Goal | Setting | Max context on 8 GB | Speed |
|---|---|---|---|
| **Full speed** (recommended on 8 GB) | `NCMOE=22` | **~32K** | ~23 tok/s |
| Balance | `NCMOE=27`, etc. | ~64K | medium |
| **Max context** (script default) | `--cpu-moe` (omit `NCMOE`) | **~128K** (≈160K ceiling) | slower (all experts on CPU) |

(The KV cache itself is tiny — ~0.6 GB at 16K — thanks to flash attention and Gemma's
sliding-window layers, and its size is **independent of the quant**. 256K, the model's trained max,
does not fit on 8 GB.) If you change `CTX`, pass the **same** value to `configure-pi.sh`:

```bash
# e.g. a 128K window (slower): all experts to RAM, big context
CTX=131072 bash scripts/start.sh
CTX=131072 bash scripts/configure-pi.sh   # keep pi's contextWindow in sync with the server's -c
```

## KV-cache quantization (`KVQUANT`)

You can quantize the **KV cache itself** to shrink it in VRAM, freeing room for more context or more
on-GPU experts. `KVQUANT=q8_0` (near-lossless) roughly halves it; `q5_1`/`q4_0` go further but cost
quality. It's a **long-context lever** — at 32K the KV cache is already ~0.6 GB so it barely matters,
but past ~64K the freed VRAM fits a faster expert split, so the win is both **bigger usable context**
and **higher tok/s** (e.g. +38% at 128K here).

```bash
CTX=131072 KVQUANT=q8_0 bash scripts/start.sh   # 128K context with a quantized KV cache
```

(Best on CUDA; on Vulkan, flash-attention + KV-quant can be slow — the server warns.) The measured
gains, the full table, and why it forces flash attention on are in
[how it fits](../docs/how-it-fits.md).

## Sampling (temperature & friends)

Set these on the **server** via env vars; they become the defaults for every request. The defaults
follow [unsloth's Gemma 4 recommendation](https://unsloth.ai/docs/models/gemma-4/qat): `TEMP=1.0`,
`TOP_P=0.95`, `TOP_K=64`.

```bash
TEMP=0.7 bash scripts/start.sh                              # more deterministic
TEMP=1.0 TOP_P=0.95 TOP_K=64 bash scripts/start.sh          # the defaults, explicit
EXTRA_ARGS="--min-p 0.01 --repeat-penalty 1.1 --seed 42" bash scripts/start.sh   # anything else
```

(`pi` doesn't expose sampling flags, so the server is where you set them. An OpenAI client that
*does* send `temperature`/`top_p` overrides the server default for that request.)

## Images (vision)

Gemma 4 is natively multimodal, but the text GGUF doesn't carry the vision encoder — llama.cpp
needs a **separate multimodal projector** (`mmproj`). Download it once (~1.2 GB, BF16), then start
the server with `--image`:

```bash
curl -L -o models/gemma4-26b-a4b-qat/mmproj-BF16.gguf \
  https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/mmproj-BF16.gguf
bash scripts/run-server.sh --image        # add NCMOE=22 BACKEND=cuda as usual
bash scripts/start.sh --image             # or all-in-one: --image is forwarded to the server
```

Now send images through the standard OpenAI vision format (`image_url` with a `data:image/...;base64,`
URI). Notes:

- **Vision only.** This projector is `gemma4v` — images, **not** audio. (Gemma 4 *can* do audio, but
  this BF16 file has no audio conformer; that needs a different/unified projector.)
- **Projector runs on the CPU** (`--no-mmproj-offload`): on 8 GB there's no VRAM left for a 1.2 GB
  BF16 tower beside the experts + KV. Image encoding is therefore CPU-bound, but decode stays ~full
  speed. Override the path with `MMPROJ=…` if you keep the file elsewhere.
- Without `--image` the server is **text-only** — the projector is never loaded.

Full multimodal write-up: [the Gemma 4 26B page](../docs/models/gemma4-26b-a4b.md).

## Speculative decoding (MTP)

Gemma 4 ships a **Multi-Token Prediction** head that llama.cpp uses for *lossless* self-speculative
decoding: a draft head proposes the next few tokens and the full model verifies them in one batched
pass — identical output, potentially faster generation. The 0.25 GB QAT head ships in our own model
repo and fits at the same `NCMOE`.

```bash
MTP=1 TEMP=0.7 bash scripts/start.sh           # MTP on (n-max 2), at a coding temperature
bash scripts/run-server-mtp.sh                 # interactive: pick every MTP knob with safe defaults
```

**It only pays off at low temperature** — **+15–30 % at greedy/coding temperature**, but **no
measurable gain at the default `TEMP=1.0`** (the gain falls within this rig's ±13 % run-to-run
noise). So enable it for coding work and keep the temperature low; for chat at `TEMP=1.0` it's not
worth it. The full study, the measurement-noise analysis, and why EAGLE3 doesn't work here are in
[`../docs/graveyard.md`](../docs/graveyard.md).

## Performance & tuning

The model is **30 layers, 128 experts/layer, top-8 routing**. `--n-cpu-moe N` (env `NCMOE=N`) keeps
the **first N** layers' experts on CPU and puts the rest on the GPU — fewer on CPU = faster, more
VRAM. On this 8 GB card `NCMOE=22` (8 layers' experts on the GPU) is the fast default and leaves
~1.2 GB headroom; if you hit out-of-memory, raise it (e.g. 24) or lower `CTX`.

**Build the CUDA backend — it's ~5–6× faster.** Vulkan is only the zero-build fallback; on this RTX
2070 the Vulkan MoE path (not RAM bandwidth) was the bottleneck, and CUDA took generation from
~4.9 → ~23.5 tok/s. Why MoE makes 8 GB feasible at all, plus the full llama-bench tables, are in
[how it fits](../docs/how-it-fits.md).

```bash
bash scripts/build-llama-cuda.sh        # auto-detects your CUDA + GPU arch, ~20 min
NCMOE=22 BACKEND=cuda bash scripts/run-server.sh
```

**Don't want to guess `NCMOE`/`CTX`?** `benchmark-config.sh` probes real configs on *your* GPU and
reports the fastest that fits per context. Easier still: on a first fresh launch `start.sh` offers to
run that measurement once and **remembers** the result, so later launches reuse it instantly.
`bash scripts/start.sh --menu` is a guided front-end to all of it (backend, context, `NCMOE`,
`KVQUANT`, sampling, image). The full method — the median-of-runs noise handling, the auto-tune
cache, the menu steps — is in [the Gemma 4 26B page](../docs/models/gemma4-26b-a4b.md).

**The menu remembers your choices.** Each `--menu` run saves your answers to `.gemma4-menu` and
pre-fills them as the **[defaults]** next time, so you can re-launch with the same setup by pressing
Enter through. Setting a knob back to its hardcoded default drops it (the menu never drifts away from
the defaults). Two separate stores, so two separate resets: menu preferences live in `.gemma4-menu`
(delete it to reset), while the tuned context/`NCMOE`/`KVQUANT` live in `.gemma4-tuning` (re-measure
with `AUTOTUNE=1`). An explicit env var still wins over a saved menu default for that launch.

**Keep pi in sync.** The menu runs `configure-pi.sh` for you. On the env-var path it isn't
automatic — if you launch with e.g. `CTX=131072`, also run `CTX=131072 bash scripts/configure-pi.sh`
once, or pi caps the usable context at its previously configured window (32768 by default).

## Using it without pi

`llama-server` exposes a standard OpenAI API, so any client works:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-26b-a4b-qat",
       "messages":[{"role":"user","content":"What is 17 times 23? Answer briefly."}],
       "max_tokens":600}' | python3 -m json.tool
```

**Reasoning note:** Gemma 4 thinks before answering. llama.cpp returns the chain-of-thought in a
separate `reasoning_content` field and the final answer in `content`. Give it a generous
`max_tokens` — too small and the thinking consumes the whole budget, leaving `content` empty. `pi`
handles this automatically. There's also a built-in web UI at <http://127.0.0.1:8080>.
