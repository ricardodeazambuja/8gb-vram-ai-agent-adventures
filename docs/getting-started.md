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

Most models run on stock llama.cpp. Newer ones need a newer llama.cpp or an unmerged pull
request: those all run on one build of our llama.cpp fork, cloned next to this repo (it carries
the model PRs we use on top of a pinned upstream commit; `run-qwen.sh` finds it there, or set
`ADV_LLAMA=/path`). The command pins the commit everything here was tested on. The other forks
get their own folder under `vendor/`, so builds never step on each other. Build only what you need
(~10–20 min each; free the GPU first, the build ends with a quick smoke test).

| Build | For | Command |
|---|---|---|
| `vendor/llama.cpp` (stock, release `b9606`) | Gemma 4 26B, Gemma 12B coder, Qwen3.6, the Qwen3.8 distill, Qwen3-VL | `CUDA_VER=12.2 LLAMA_REF=b9606 bash scripts/build-llama-cuda.sh` |
| `../llama.cpp`, **our fork**: [`ricardodeazambuja/llama.cpp`](https://github.com/ricardodeazambuja/llama.cpp), branch [`adventures`](https://github.com/ricardodeazambuja/llama.cpp/tree/adventures) (upstream 2026-10-28 + LocateAnything PR #24749 + Dory PR #30306) | Dory, Spark, Flint, Nemotron Bolt 74B and E4B, Groot, Ornith, LFM2.5-24B, LFM3, LFM3-VL, LocateAnything, Clef-Flash | `CUDA_VER=12.2 LLAMA_REPO=https://github.com/ricardodeazambuja/llama.cpp LLAMA_REF=f0456c64d7b891ada1a0a924716ce3244122e92b SRC_DIR=$PWD/../llama.cpp bash scripts/build-llama-cuda.sh` |
| `vendor/prism-llama.cpp` (PrismML fork) | Bonsai 2 27B | `CUDA_VER=12.2 LLAMA_REPO=https://github.com/PrismML-Eng/llama.cpp LLAMA_REF=prism-b10777-e3de715 SRC_DIR=$PWD/vendor/prism-llama.cpp bash scripts/build-llama-cuda.sh` |
| `vendor/llama-cpp-turboquant` (TurboQuant fork) | Gemma 12B agentic | `SRC_DIR=$PWD/vendor/llama-cpp-turboquant LLAMA_REF=feature/turboquant-kv-cache ENV_NAME=llamacpp-cuda bash scripts/build-llama-cuda.sh` |

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
bash scripts/setup.sh                                  # Gemma 4 26B-A4B (~14 GB) + the llamacpp env
bash scripts/download-models.sh flint            # or any of: qwen36 distill bonsai2 qwen3vl dory spark flint bolt e4b
```

The Gemma 12B models have their own download instructions on their pages.

## 6. Install pi and point it at your laptop

[pi](https://github.com/badlogic/pi-mono) is the coding agent that drives the models: it reads
files, runs commands and edits code, all against a local OpenAI-compatible server.

```bash
npm i -g @earendil-works/pi-coding-agent
bash scripts/configure-pi.sh      # registers the "llamacpp" provider in ~/.pi/agent/models.json
```

The launchers keep pi's context window in sync with the server, so you only do this once.

## 7. Go

Every model has a one-word launcher that starts the server, waits for it, and opens pi:

```bash
BACKEND=cuda bash scripts/start.sh     # Gemma 4 26B-A4B
bash scripts/start_flint.sh            # Flint
QMODEL=qwen36 bash scripts/start-qwen.sh
```

The full list is in the [README](../README.md#the-lineup). One model at a time: they all
listen on port 8080 and each one wants the whole GPU. When you're done:

```bash
bash scripts/stop-server.sh
```

Want a one-shot answer instead of a chat? Any pi flag passes through:

```bash
bash scripts/start_flint.sh -p "explain @README.md" --no-session </dev/null
```

(The `</dev/null` is not decoration: without it `pi -p` waits for input that never comes.)

## Something went sideways?

Check [troubleshooting](troubleshooting.md). It's a list of every way this went wrong for us,
which turned out to be a long list.
