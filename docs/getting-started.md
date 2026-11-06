# 🧰 Getting started

From "fresh Linux box" to "a 35B model writing code on an 8 GB laptop GPU" in about an hour,
most of which is your internet connection and a compiler doing their thing while you make
coffee.

## What you need

| | Minimum | What we used |
|---|---|---|
| GPU | NVIDIA, 8 GB VRAM, driver that supports CUDA ≥ 12.2 (`nvidia-smi` → "CUDA Version") | RTX 2070 Max-Q, 8 GB (Turing, `sm_75`) |
| RAM | 16 GB for the small models; **32 GB** for the big MoEs (their experts live in RAM) | 32 GB |
| Disk | ~5 GB per small model, ~15–25 GB per big one, ~4 GB per llama.cpp build | plenty, eventually not |
| OS | Linux | Ubuntu 22.04 |
| Tools | [Miniforge](https://github.com/conda-forge/miniforge) (`mamba`), `git`, Node.js ≥ 22.19 for pi | |

> **Why so much RAM for an "8 GB" project?** Because the trick for the big models is to keep
> the bulky expert weights in system RAM and only the busy parts on the GPU. The 8 GB is the
> VRAM budget; the RAM is where the rest of the elephant sits. See [how it fits](how-it-fits.md).

## 1. Clone

```bash
git clone https://github.com/ricardodeazambuja/8gb-vram-ai-agent-adventures
cd 8gb-vram-ai-agent-adventures
```

## 2. The CUDA build toolchain (once)

All builds use the same pinned toolchain: 220 conda-forge packages (CUDA toolkit 12.2.2,
nvcc 12.2.140, gcc/g++ 12.4), URLs and md5 sums in `envs/llamacpp-cuda.linux-64.lock`.
Nothing system-wide is touched.

```bash
mamba create -y -n llamacpp-cuda --file envs/llamacpp-cuda.linux-64.lock
```

## 3. Build llama.cpp (once per flavour)

Every model runs on one build of our llama.cpp fork, cloned next to this repo. The
fork carries the model PRs we use on top of a pinned upstream commit, and the launch scripts find
it there, or set `ADV_LLAMA=/path`. `build-llama-cuda.sh` builds it by default at the commit
everything here was tested on. Build only what you need
(~10–20 min each; free the GPU first, the build ends with a quick smoke test).

| Build | For | Command |
|---|---|---|
| `../llama.cpp`, **our fork**: [`ricardodeazambuja/llama.cpp`](https://github.com/ricardodeazambuja/llama.cpp), branch [`moe-cache-max-batch`](https://github.com/ricardodeazambuja/llama.cpp/tree/moe-cache-max-batch) at `69ca796dd`: the [`adventures`](https://github.com/ricardodeazambuja/llama.cpp/tree/adventures) branch (upstream 2026-10-28 + LocateAnything PR #24749 + Dory PR #30306 + [PrismML's fork](https://github.com/PrismML-Eng/llama.cpp) for `PTQ1_0`) + the [MoE expert cache](how-it-fits.md#12-let-a-cache-pick-the-experts-opt-in-our-forks-moe-offload-branch) | every model in `scripts/llm list` | `CUDA_VER=12.2 bash scripts/build-llama-cuda.sh` |
| `../llama.cpp`, extra target | text to speech with LFM3-Audio ([`examples/speak.py`](../examples/speak.py)) | `(cd ../llama.cpp && cmake --build build --target llama-tts)` |

The script treats `../llama.cpp` as its own checkout: if you already have a clone there with a branch
checked out, it refuses to touch it. Build that one in place, or pass `SRC_DIR=/somewhere/else`.

`CUDA_VER=12.2` matters: without it the script picks your driver's CUDA version for the
toolkit. The GPU architecture is auto-detected (override with `CUDA_ARCH=86` and friends).

## 4. A small env for downloads

The download scripts use the Hugging Face `hf` command from a conda env called `llamacpp`:

```bash
mamba create -y -n llamacpp -c conda-forge huggingface_hub
```

(`bash scripts/setup.sh` also creates this env, plus a Vulkan llama.cpp as a no-compiler
fallback, and downloads Gemma 4 26B. If a download script complains there's no `hf` in the
env, install a recent `huggingface_hub` into it; any env name works with `ENV_NAME=`.)

## 5. Download a model

Every download is pinned to an exact Hugging Face revision and checked against its sha256, so
you get the very bytes we tested. A file that fails the check is deleted, because a GGUF can
load fine and still be corrupt (we learned this the [hard way](troubleshooting.md)).

```bash
bash scripts/setup.sh                       # Gemma 4 26B-A4B (~14 GB) + the llamacpp env
bash scripts/download-models.sh groot       # any name from `scripts/llm list`; several at once is fine
bash scripts/download-models.sh -h          # the full list, including the picture models
```

`scripts/llm list` shows which models are downloaded, and the download command for the rest.

## 6. Install pi and point it at your laptop

[pi](https://github.com/badlogic/pi-mono) is the coding agent that drives the models: it reads
files, runs commands and edits code, all against a local OpenAI-compatible server.

```bash
npm i -g @earendil-works/pi-coding-agent
bash scripts/configure-pi.sh      # registers the "llamacpp" provider in ~/.pi/agent/models.json
```

The launchers keep pi's context window in sync with the server, so you only do this once.

## 7. Go

One launcher for every model. It starts the server with the model's tested settings, waits for it,
and opens pi (chat models) or prints the server's URL (vision and helper models):

```bash
scripts/llm list                       # what's here and what each one is for
scripts/llm gemma                      # Gemma 4 26B-A4B + pi
scripts/llm groot                      # Groot + pi
scripts/llm lfm3vl                     # a vision model: just the server, on http://127.0.0.1:8080/v1
scripts/llm status                     # what's running
scripts/llm stop                       # stop it
```

One model at a time: they all listen on port 8080 and each one wants the whole GPU, so `scripts/llm`
refuses to start a second one. Want a one-shot answer instead of a chat? Any pi flag passes through:

```bash
scripts/llm flint -p "explain @README.md" --no-session
```

[The cheatsheet](cheatsheet.md) maps tasks to commands ("search my notes", "make a picture", …). The older
launchers (`scripts/start.sh`, `start_flint.sh`, `QMODEL=... start-qwen.sh`) still work; they
are what `scripts/llm` builds on. [`scripts/README.md`](../scripts/README.md) has every knob.

## 8. Pictures (optional)

The image models run on [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), built
next to this repo with the same toolchain: the commands are in
[Making pictures](image-generation.md#run-it-yourself). Then:

```bash
bash scripts/download-models.sh klein kleinte kleinvae      # FLUX.2 Klein, ~5.3 GB
scripts/imagine "a red fox in the snow"                    # prints the path of the picture
```

`scripts/imagine -h` has every option (other models, sizes, seeds, img2img, inpainting, reference edits).

## 9. The helpers from Python

[`examples/`](../examples/README.md) has a short script per helper model: search with EmbeddingGemma,
decisions with d1 or Clef-Flash, pictures in and tool calls out with LFM3-VL, boxes with LocateAnything,
speech in and out with LFM3-Audio. Each starts its model with `scripts/llm` and stops it afterwards.

## Something went sideways?

Check [troubleshooting](troubleshooting.md). It's a list of every way this went wrong for us,
which turned out to be a long list.
