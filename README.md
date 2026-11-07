# 8GB VRAM AI Agent Adventures 🎒

**Big local models, a modest laptop GPU, and an AI agent writing code on top. What could go wrong?**
(Plenty. It's all written down.)

This repo started with one question: *can a 26-billion-parameter model run on a laptop with an
8 GB graphics card?* It could. Then we got curious, and more than twenty models later we have install
scripts, measured speeds, a sandboxed agent benchmark, a vision probe, and a graveyard.

Everything here runs on one machine: an **RTX 2070 Max-Q with 8 GB of VRAM and 32 GB of RAM**,
the kind of GPU you find in a gaming laptop from a few years ago. The models are served by
[llama.cpp](https://github.com/ggml-org/llama.cpp) and driven by [pi](https://github.com/badlogic/pi-mono),
a coding agent that reads files, runs commands and edits code, entirely offline.

## The lineup

| | Model | Size | Context | Speed | [pibench](docs/pibench.md) | [Vision](docs/vision-probe.md) | One-liner |
|---|---|---|---:|---:|---:|---:|---|
| 🪨 | [**Flint**](docs/models/flint.md)<br>`flint` | 40B MoE, 2.5B active | 128K | 32 tok/s (24) | **18/18** | text only | The careful coder. Looks before it leaps. |
| 💎 | [**Gemma 4 26B-A4B**](docs/models/gemma4-26b-a4b.md)<br>`gemma` | 26B MoE, 4B active | 32K–128K | 30 tok/s (23.5) | **18/18** | **100%** | Where it all started. Quietly aced everything. |
| 🧢 | [**ThinkingCap Gemma 4 26B**](docs/models/thinkingcap-gemma4-26b-a4b.md)<br>`thinkingcap` | 26B MoE, 4B active | 64K | 17.4 tok/s | 17/18 | not run | Gemma told to think less: half the thinking, one fewer solve. |
| ⚡ | [**Nemotron 3.5 Bolt 74B**](docs/models/nemotron-bolt-74b.md)<br>`bolt` | 74B hybrid MoE, 9B active | 128K | 14 tok/s | 15/18 | 91% | The biggest one here. 74B in 2 bits, slow and thoughtful. |
| 🐉 | [**Qwen3.6-35B-A3B**](docs/models/qwen3.6-35b-a3b.md)<br>`qwen36` | 35B MoE, 3B active | 128K | 39 tok/s (23) | 16/18 | **100%** | The yardstick everyone else is measured against. |
| 🏎️ | [**Qwen3.8 distill**](docs/models/qwen3.8-35b-a3b-distill.md)<br>`distill` | 35B MoE, 3B active | 128K | **49 tok/s** (26) | 16/18 | 88% | Quick jobs (2.5 min). Thinks in haiku. |
| 🌱 | [**Groot-19B-A3B**](docs/models/groot-19b-a3b.md)<br>`groot` | 19B MoE, 3B active | 128K | 47 tok/s (39.5) | **18/18** | 74% | Qwen3.6 with half the experts. Among the fastest big models here. |
| 💧 | [**LFM2.5-24B-A2B**](docs/models/lfm2.5-24b-a2b.md)<br>`lfm25` | 24B conv/attention MoE, 2B active | 128K | 43.5 tok/s (31.5) | 10/18 | text only | Not a Qwen. Trained inside agent harnesses, pi among them. |
| 🐦 | [**Ornith-1.5-35B-A3B**](docs/models/ornith-1.5-35b-a3b.md)<br>`ornith` | 35B MoE, 3B active | 128K | 40 tok/s (31) | 14/18 | 97% | Qwen3.6 that trained on its own tasks. The quickest jobs of the big models, and it guesses. |
| 🌳 | [**Bonsai 2 27B**](docs/models/bonsai2-27b.md)<br>`bonsai2` | 27B, ternary weights | 32K | 38 tok/s | 16/18 | 98% | A 27B model in 6 GB. Big brain, small desk. |
| ✨ | [**Spark-X2.6-4B**](docs/models/spark-x2.6-4b.md)<br>`spark` | 4B dense | 108K | 48–50 tok/s | 14/18 | text only | As fast as the quickest big ones, all on the GPU. Overthinks, then rushes. |
| 🚀 | [**LFM3-1.7B**](docs/models/lfm3-1.7b.md)<br>`lfm3` | 1.7B tri-hybrid | 128K | **106 tok/s** | 12/18 | text only | Tiny, as fast as its VL sibling and twice as fast as the rest, and better at chores than models 14× its size. |
| 👀 | [**LFM3-VL-2B**](docs/models/lfm3-vl-2b.md)<br>`lfm3vl` | 2B + vision | 32K | 106 tok/s | 0/18 | 91% | 2.6 GB, 8 s an image, and the tightest boxes of any model here. Good for one picture, one tool call ([60/60](docs/visual-control.md)); in long agent sessions it loops. |
| 🐇 | [**Nemotron 3.5 Bolt E4B**](docs/models/nemotron-bolt-e4b.md)<br>`e4b` | 8B (4B effective) | 64K | 52 tok/s | 7/18 | text only | Fast, all on the GPU. A voice-assistant model; coding isn't its day job. |
| 👁️ | [**Qwen3-VL-8B**](docs/models/qwen3-vl-8b.md)<br>`qwen3vl` | 8B + vision | 16K | 38 tok/s | 1/18 | 89% | Reads an image in 5 seconds flat. Not an agent. |
| 🔁 | [**Dory1.0-4B**](docs/models/dory1.0-4b.md)<br>`dory` | 4B, looped | 64K | 29 tok/s | 14/18 | text only | Runs its middle layers twice. Just keeps swimming, and gets there: Spark's score at 4B. |

**pibench** = six real chores (fix bugs, build a CLI, scrape websites, log in through a form) done
alone, in a sandbox, graded by hidden tests; solved attempts out of 18. **Vision** = twelve
images with known answers: OCR, bounding boxes, charts, counting, engineering drawings. "not run" means
we haven't run it (yet). The full numbers, and what the vendors claim, are on the
[**scoreboard**](docs/scoreboard.md). The name under each model starts it: `scripts/llm <name>`
([below](#using-the-models)). MoE = mixture of experts: only a few billion of the weights work on each token, and
the rest (the experts) can wait in RAM. **Speed** is at `scripts/llm`'s settings next to a quiet desktop. For Qwen3.6,
the Qwen3.8 distill, Flint, Ornith, Groot, LFM2.5 and Gemma that's our [MoE expert cache](docs/how-it-fits.md#12-let-a-cache-pick-the-experts-the-default-for-the-big-moes),
which keeps the busiest experts on the GPU; in brackets, the older static split (`NCMOE` = how many layers'
experts sit in RAM; `scripts/llm <name> --static`), from earlier runs. The two come from different runs; the gain at
the same VRAM is 19–58%, and about 10% less next to a busy desktop ([scoreboard](docs/scoreboard.md) footnote ¹).

## The helpers

Not chat models, so they don't drive pi. `scripts/llm <name>` starts one, and
[`examples/`](examples/README.md) has a short Python script for each: search, decisions, pictures in,
boxes out, speech in and out. The picture makers run through `scripts/imagine`.

| | Model | What it does | Memory | Best result here |
|---|---|---|---:|---|
| 🗣️ | [**LFM3-Audio-2B**](docs/models/lfm3-audio-2b.md)<br>`lfm3audio` | Hears and speaks | 3.3 GB | First audio in under 0.2 s, 3/3 spoken tool calls |
| 🧲 | [**EmbeddingGemma 2**](docs/models/embeddinggemma-2.md)<br>`eg2` | Search over code and docs | 1.5 GB | 93% top hit on code, with BM25 |
| ⚖️ | [**Clef-Flash**](docs/models/clef-flash.md)<br>`clefflash` | Yes/no and pick-one decisions | 7.1 GB | 86% grading pibench, 4/189 false alarms |
| ⚡ | [**d1-2B**](docs/models/d1-2b.md)<br>`d1` | The same, much smaller | 2.6 GB | 27 ms a decision |
| 📍 | [**LocateAnything-3B**](docs/models/locateanything-3b.md)<br>`locate` | Finds things in pictures | 5 GB | 15/15 shapes, mean IoU 0.94 |
| 🧠 | [**VibeThinker 3B and 1.5B**](docs/models/vibethinker.md)<br>`vibe3b`, `vibe15` | One-shot code and math | 5.0 GB, 3.1 GB | 0/500 wrong on the code check, 65 and 108 tok/s; 0/18 on pibench (3B), not an agent |
| 🧮 | [**TwIL-8B**](docs/models/twil-8b.md)<br>`twil` | Math and code reasoner | 5.35 GB | 0/500 wrong on the code check, 38 tok/s; 8/18 on pibench, where its long thinking fills the 24K context. Needs a quiet desktop |
| 🐭 | [**LFM2.5-VL-450M**](docs/models/lfm2.5-vl-450m.md)<br>`lfm25vl` | Tiny vision model for fine-tuning | 1.5 GB | 0.1 s a decision once tuned |
| 🖌️ | [**FLUX.2 Klein 4B**](docs/models/flux2-klein-4b.md)<br>`imagine -m klein` | Pictures, the everyday pick | 3.5 GB | 23 s an image |
| 🔤 | [**Iris-3B Turbo**](docs/models/iris-3b.md)<br>`imagine -m iris` | Pictures with text | 7.3 GB | Cleanest sign text, 33 s |
| 🎨 | [**SDXL**](docs/models/sdxl.md)<br>`imagine -m sdxl` | Pictures, trainable | 5.8 GB | Learned our styles and a person |

## Quick start

```bash
git clone https://github.com/ricardodeazambuja/8gb-vram-ai-agent-adventures
cd 8gb-vram-ai-agent-adventures

# 1. the build toolchain and a llama.cpp build (once, ~20 min)
mamba create -y -n llamacpp-cuda --file envs/llamacpp-cuda.linux-64.lock
CUDA_VER=12.2 bash scripts/build-llama-cuda.sh

# 2. a model (Gemma 4 26B, ~14 GB) and the coding agent
bash scripts/setup.sh
npm i -g @earendil-works/pi-coding-agent && bash scripts/configure-pi.sh

# 3. go
scripts/llm gemma
```

Other models need their own download (`bash scripts/download-models.sh groot`, for example): the
[getting-started guide](docs/getting-started.md) has the full menu, and every model page has
copy-paste commands.

## Using the models

**[The cheatsheet](docs/cheatsheet.md)** has it all on one page: "I want to…, so I run…" for every task, and
every model with its command.

```bash
scripts/llm list                       # every model: what it's for, its size, downloaded or not
scripts/llm groot                      # a coding agent: starts the server, then opens pi
scripts/llm groot -p "explain @README.md"   # one-shot
scripts/llm lfm3vl                     # a vision or helper model: starts the server, prints its URL
scripts/llm status                     # what's running;  scripts/llm stop  stops it

scripts/imagine "a red fox in the snow"                  # a picture with FLUX.2 Klein (~25 s)
scripts/imagine "a shop sign that says OPEN" -m iris     # Iris-3B when the picture needs text

python3 examples/decide.py guard "rm -rf ~/"             # helpers from Python: examples/README.md
```

`scripts/llm` knows each model's tested settings (context, KV cache, sampling, where the experts live)
and runs the server in its own systemd scope, so running out of memory kills the server, not your
terminal. One model at a time: it refuses to replace a different model that is already serving.
Every setting can still be overridden (`CTX=65536 scripts/llm flint`); [`scripts/README.md`](scripts/README.md)
has them all.

## Things we learned the fun way

- **A 74B model runs at 14 tokens/s on an 8 GB GPU** (and a 40B one at 24–32). NVIDIA's Nemotron Bolt is
  trained for 2 bits, and its always-on layers fit on the GPU while 66B of experts sit in RAM.
- **Mixture-of-experts models only use a few billion parameters per token**, so the bulky experts
  can live in system RAM while the GPU does the busy work. [How it fits](docs/how-it-fits.md).
- **One flag tripled prompt speed.** For models whose experts sit in RAM, a bigger batch lets
  llama.cpp ship the expert maths to the GPU: Flint went from 192 to 542 tokens/s reading a prompt.
- **The vendor benchmarks and our laptop disagree.** Gemma 4 26B scores 17.4 on SWE-bench Verified
  in one vendor's table and 57.4 in another's; on our six chores it went 18 for 18.
- **The difference between a pass and a fail is checking your work.** Every model ran its code.
  The ones that failed printed a wrong answer and walked right past it. [pibench](docs/pibench.md).
- **Nearly every model can find shapes in a picture; almost none use the coordinates you ask for.**
  [The vision probe](docs/vision-probe.md).
- **A 2B vision model got better by training 201,472 numbers.** Fine-tuning only its norms took it
  from 109 to 150 of 150 decisions in `llama-server`, and through 5 of 8 mazes where a LoRA 75 times
  bigger managed 1.
  [Fine-tuning](docs/fine-tuning.md).
- **Stable Diffusion is no longer the easy choice on 8 GB.** FLUX.2 Klein draws 1024×1024 in 23 s
  in 3.5 GB and can spell. On SDXL, an 8 KB learned word taught it a person, without leaking into other prompts. [Making pictures](docs/image-generation.md).
- **Two models can be better than one.** LFM3-VL sees but can't work as an agent, and neither a LoRA nor
  a new projector fixed that. Temperature 0.7 did end its doom loops. Letting LFM3-1.7B drive with VL as its
  eyes took 57/60 picture decisions in 5 GB. [The tests](docs/fine-tuning.md#can-lfm3-vl-do-more-than-look-2026-11-01).
- **A GGUF can load perfectly and still be garbage.** Verify the sha256 before you blame the
  model, the kernel, or the universe. [Troubleshooting](docs/troubleshooting.md).

## The docs

| | |
|---|---|
| [**Cheatsheet**](docs/cheatsheet.md) | I want to…, so I run…: every task and every model, one command each |
| [Getting started](docs/getting-started.md) | Requirements, builds, downloads, pi, first launch |
| [The scoreboard](docs/scoreboard.md) | Every number: ours, the vendors', everyone vs. Qwen3.6; how each one behaves; which one to pick |
| [pibench](docs/pibench.md) | The agent benchmark: the chores, the sandbox, the results, what we saw in the transcripts |
| [The vision probe](docs/vision-probe.md) | OCR, boxes, charts, counting, CAD drawings |
| [The visual control probe](docs/visual-control.md) | Look, decide, call a tool: dials, buttons, traffic lights, a grid robot |
| [Fine-tuning](docs/fine-tuning.md) | Teaching vision models to call tools and steer: LoRA vs IA3 vs training only the norms, and serving the result from llama.cpp |
| [Making pictures](docs/image-generation.md) | Image generators on 8 GB: FLUX.2 Klein, Iris-3B and SDXL side by side |
| [How it fits](docs/how-it-fits.md) | The tricks that make big models fit in 8 GB, explained |
| [The graveyard](docs/graveyard.md) | Things we tried that didn't make the cut, and why |
| [Troubleshooting](docs/troubleshooting.md) | Every way this went wrong for us, and the fix |
| [Security](docs/security.md) | What a stranger's GGUF can hide, and the audit every new model gets before it runs |
| [Reproducing our results](docs/reproduce.md) | Which command makes which number, what's pinned, and how close a rerun lands |
| [Up next](docs/up-next.md) | Models and chores waiting their turn |
| [Model pages](docs/models/) | One page per model: install, run, official and measured numbers, quirks |
| [`scripts/README.md`](scripts/README.md) | Every script and knob |
| [`examples/README.md`](examples/README.md) | The helper models from Python: search, decisions, vision, speech |
| [`bench/README.md`](bench/README.md) | Running the benchmarks yourself |
| [`utils/README.md`](utils/README.md) | The GGUF inspector and security auditor |

## What's in the box

```
scripts/     llm (start any model) and imagine (pictures), downloads (pinned + sha256-checked), builds, quick benchmarks
examples/    short Python recipes for the helper models
config/      the pi provider definition (all models, context sizes, thinking levels)
envs/        the pinned CUDA build toolchain and the Python packages for training
patches/     the llama.cpp patch for Dory, a CUDA 12.2 build fix for the MoE cache, and the unfinished FlashMap drafter
bench/       pibench (sandboxed agent benchmark), the vision and visual control probes, fine-tuning, steering, image generation
docs/        everything you read above
utils/       a GGUF inspector and a GGUF security auditor
loop/        the raw overnight commands and journal
models/      where downloads go (not in git)
```

## Fine print

This repo's own work is under the [MIT License](LICENSE). Third-party material keeps its own terms:
the llama.cpp patch, model outputs and generated pictures, and text from the practice websites.

Each model comes with its own license (Apache 2.0, MIT, NVIDIA's, Google's Gemma terms…); check
its page and model card before you build something on it. Numbers are from one laptop and mostly
single runs; your mileage will vary, possibly in your favour. Nothing here is endorsed by the
model makers. They made the models; we just made them fit.
