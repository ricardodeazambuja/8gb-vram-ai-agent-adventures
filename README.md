# 8GB VRAM AI Agent Adventures 🎒

**Big local models, a modest laptop GPU, and an AI agent writing code on top. What could go wrong?**
(Plenty. It's all written down.)

This repo started with one question: *can a 26-billion-parameter model run on a laptop with an
8 GB graphics card?* It could. Then we got curious, and twelve models later we have install
scripts, measured speeds, a sandboxed agent benchmark, a vision probe, and a graveyard.

Everything here runs on one machine: an **RTX 2070 Max-Q with 8 GB of VRAM and 32 GB of RAM**,
the kind of GPU you find in a gaming laptop from a few years ago. The models are served by
[llama.cpp](https://github.com/ggml-org/llama.cpp) and driven by [pi](https://github.com/badlogic/pi-mono),
a coding agent that reads files, runs commands and edits code, entirely offline.

## The lineup

| | Model | Size | Context | Speed | [pibench](docs/pibench.md) | [Vision](docs/vision-probe.md) | One-liner |
|---|---|---|---:|---:|---:|---:|---|
| 🪨 | [**Flint**](docs/models/flint.md) | 40B MoE, 2.5B active | 128K | 24 t/s | **18/18** | text only | The careful coder. Looks before it leaps. |
| 💎 | [**Gemma 4 26B-A4B**](docs/models/gemma4-26b-a4b.md) | 26B MoE, 4B active | 32K–128K | 23.5 t/s | **18/18** | **100%** | Where it all started. Quietly aced everything. |
| ⚡ | [**Nemotron 3.5 Bolt 74B**](docs/models/nemotron-bolt-74b.md) | 74B hybrid MoE, 9B active | 128K | 14 t/s | 15/18 | 91% | The biggest one here. 74B in 2 bits, slow and thoughtful. |
| 🐉 | [**Qwen3.6-35B-A3B**](docs/models/qwen3.6-35b-a3b.md) | 35B MoE, 3B active | 128K | 23 t/s | 16/18 | **100%** | The yardstick everyone else is measured against. |
| 🏎️ | [**Qwen3.8 distill**](docs/models/qwen3.8-35b-a3b-distill.md) | 35B MoE, 3B active | 128K | 26 t/s | 16/18 | 88% | Fastest job turnaround of the big ones. Thinks in haiku. |
| 🌱 | [**Groot-19B-A3B**](docs/models/groot-19b-a3b.md) | 19B MoE, 3B active | 128K | **39.5 t/s** | **18/18** | 74% | Qwen3.6 with half the experts. Fastest big model here. |
| 💧 | [**LFM2.5-24B-A2B**](docs/models/lfm2.5-24b-a2b.md) | 24B conv/attention MoE, 2B active | 128K | 31.5 t/s | 10/18 | text only | Not a Qwen. Trained inside agent harnesses, pi among them. |
| 🌳 | [**Bonsai 2 27B**](docs/models/bonsai2-27b.md) | 27B, ternary weights | 16K | 27 t/s | 14/18 | 98% | A 27B model in 6 GB. Big brain, small desk. |
| ✨ | [**Spark-X2.6-4B**](docs/models/spark-x2.6-4b.md) | 4B dense | 108K | 48–50 t/s | 14/18 | text only | Twice as fast as the big ones. Overthinks, then rushes. |
| 🐇 | [**Nemotron 3.5 Bolt E4B**](docs/models/nemotron-bolt-e4b.md) | 8B (4B effective) | 64K | 52 t/s | 7/18 | text only | The fastest here. A voice-assistant model; coding isn't its day job. |
| 👁️ | [**Qwen3-VL-8B**](docs/models/qwen3-vl-8b.md) | 8B + vision | 16K | 38 t/s | — | 89% | Reads an invoice in 6 seconds flat. |
| 🔁 | [**Dory1.0-4B**](docs/models/dory1.0-4b.md) | 4B, looped | 64K | 29 t/s | — | text only | Runs its middle layers twice. Just keeps swimming. |
| 🧑‍💻 | [**Gemma 4 12B coder**](docs/models/gemma4-12b-coder.md) | 12B dense | 32K | 17 t/s | — | — | A community coding fine-tune; Q4 is the sweet spot. |
| 🤖 | [**Gemma 4 12B agentic**](docs/models/gemma4-12b-agentic.md) | 12B dense | 16K | 28 t/s | — | — | Fully on the GPU thanks to a ~3.5-bit V cache. |

**pibench** = six real chores (fix bugs, build a CLI, scrape websites, log in through a form) done
alone, in a sandbox, graded by hidden tests; solved attempts out of 18. **Vision** = twelve
images with known answers: OCR, bounding boxes, charts, counting, engineering drawings. "—" means
we haven't run it (yet). The full numbers, and what the vendors claim, are on the
[**scoreboard**](docs/scoreboard.md).

## Quick start

```bash
git clone https://github.com/ricardodeazambuja/8gb-vram-ai-agent-adventures
cd 8gb-vram-ai-agent-adventures

# 1. the build toolchain and a llama.cpp build (once, ~20 min)
mamba create -y -n llamacpp-cuda --file envs/llamacpp-cuda.linux-64.lock
CUDA_VER=12.2 LLAMA_REF=b9606 bash scripts/build-llama-cuda.sh

# 2. a model (Gemma 4 26B, ~14 GB) and the coding agent
bash scripts/setup.sh
npm i -g @earendil-works/pi-coding-agent && bash scripts/configure-pi.sh

# 3. go
BACKEND=cuda bash scripts/start.sh
```

Other models need their own download and sometimes their own llama.cpp build: the
[getting-started guide](docs/getting-started.md) has the full menu, and every model page has
copy-paste commands.

## Things we learned the fun way

- **A 74B model runs at 14 tokens/s on an 8 GB GPU** (and a 40B one at 24). NVIDIA's Nemotron Bolt is
  trained for 2 bits, and its always-on layers fit on the GPU while 66B of experts sit in RAM.
- **Mixture-of-experts models only use a few billion parameters per token**, so the bulky experts
  can live in system RAM while the GPU does the busy work. [How it fits](docs/how-it-fits.md).
- **One flag tripled prompt speed.** For models whose experts sit in RAM, a bigger batch lets
  llama.cpp ship the expert maths to the GPU: Flint went from 192 to 542 tokens/s reading a prompt.
- **The vendor benchmarks and our laptop disagree.** Gemma 4 26B scores 17.4 on SWE-bench Verified
  in one vendor's table and 57.4 in another's; on our six chores it went 18 for 18.
- **The difference between a pass and a fail is checking your work.** Every model ran its code.
  The ones that failed printed a wrong answer and walked right past it. [pibench](docs/pibench.md).
- **Every model can find shapes in a picture; almost none use the coordinates you ask for.**
  [The vision probe](docs/vision-probe.md).
- **A GGUF can load perfectly and still be garbage.** Verify the sha256 before you blame the
  model, the kernel, or the universe. [Troubleshooting](docs/troubleshooting.md).

## The docs

| | |
|---|---|
| [Getting started](docs/getting-started.md) | Requirements, builds, downloads, pi, first launch |
| [The scoreboard](docs/scoreboard.md) | Every number: ours, the vendors', everyone vs. Qwen3.6; how each one behaves; which one to pick |
| [pibench](docs/pibench.md) | The agent benchmark: the chores, the sandbox, the results, what we saw in the transcripts |
| [The vision probe](docs/vision-probe.md) | OCR, boxes, charts, counting, CAD drawings |
| [How it fits](docs/how-it-fits.md) | The tricks that make big models fit in 8 GB, explained |
| [The graveyard](docs/graveyard.md) | Things we tried that didn't make the cut, and why |
| [Troubleshooting](docs/troubleshooting.md) | Every way this went wrong for us, and the fix |
| [Security](docs/security.md) | What a stranger's GGUF can hide, and the audit every new model gets before it runs |
| [Up next](docs/up-next.md) | Models and chores waiting their turn |
| [Model pages](docs/models/) | One page per model: install, run, official and measured numbers, quirks |
| [`scripts/README.md`](scripts/README.md) | Every script and knob |
| [`bench/README.md`](bench/README.md) | Running the benchmarks yourself |

## What's in the box

```
scripts/     launchers, downloads (pinned + sha256-checked), builds, quick benchmarks
config/      the pi provider definition (all models, context sizes, thinking levels)
envs/        the pinned CUDA build toolchain
patches/     the llama.cpp patch for Dory
bench/       pibench (sandboxed agent benchmark) and the vision probe
docs/        everything you read above
utils/       a GGUF inspector and a GGUF security auditor
models/      where downloads go (not in git)
vendor/      where llama.cpp builds go (not in git)
```

## Fine print

Each model comes with its own license (Apache 2.0, MIT, NVIDIA's, Google's Gemma terms…); check
its page and model card before you build something on it. Numbers are from one laptop and mostly
single runs; your mileage will vary, possibly in your favour. Nothing here is endorsed by the
model makers. They made the models; we just made them fit.
