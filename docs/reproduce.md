# 🔁 Reproducing our results

This page maps results to commands and saved files: what to build, which command makes which
number, and how close a rerun should land. **Not every result has that yet:** see [Known gaps](#known-gaps).

You can start with no GPU at all. This recomputes the headline numbers, and their p-values, from the
committed result files, and it reports any that disagree with the docs:

```bash
python3 bench/check_numbers.py
```

## How close a rerun should land

| Kind of result | Examples | Expect |
|---|---|---|
| Deterministic | weight distances, baked GGUFs, sha256s, the scenes (drawn from seeds) | the same |
| Seeded requests to one server | replay | the same on this machine: a rerun from a clean clone matched at all 13 points, in all three arms |
| Trained | adapters, and the glance scores that depend on them | close, not identical: GPU training isn't bit-exact |
| Agent runs | pibench | the same range: the model, the sandbox and live sites all vary. Small gaps are noise |
| Speed and VRAM | time per decision, GB | this laptop on that day; expect about ±5% here, and different numbers elsewhere |
| Judged by eye | image likeness and style | the same pictures to look at, but the verdict is ours |

## Setup

1. **llama.cpp:** our fork at the pinned commit, as in
   [getting started](getting-started.md#3-build-llamacpp-once). Every LFM3 result, steering included,
   ran on the `adventures` branch at `3cb2992b3`, which the default build contains. For an exact LFM3
   rerun, build that commit into its own folder:
   `LLAMA_REF=3cb2992b3 SRC_DIR=$PWD/../llama.cpp-3cb2992 bash scripts/build-llama-cuda.sh`.
2. **Weights:** first make the [`llamacpp` download env](getting-started.md#4-a-small-env-for-downloads),
   then run `bash scripts/download-models.sh lfm3 lfm3vl groot lfm25vl`. These files are pinned and checked
   by sha256. Gemma has [its own steps](models/gemma4-26b-a4b.md#install--run).
3. **Python for training,** the exact packages we used:

   ```bash
   mamba create -y -n vlft python=3.12 pip
   mamba run -n vlft pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r envs/vlft.txt
   bash scripts/download-hf-weights.sh lfm25vl
   bash scripts/download-hf-weights.sh lfm3vl
   ```

   The second download includes LFM3-VL's own Python. It must only run through
   [`sandbox.sh`](../bench/finetune/sandbox.sh), which needs Docker with GPU access. Anything that draws
   scenes, like `run_control.py`, needs this env's Pillow too, so run it with
   `~/miniforge3/envs/vlft/bin/python`. Plain `python3` is enough for the rest of this page unless shown.
4. **pibench only:** build the image and network and add the firewall, as in
   [`bench/README.md`](../bench/README.md#running-it).
5. **Pictures only:** build `stable-diffusion.cpp` at `0a067120`, as in
   [Making pictures](image-generation.md#run-it-yourself), and download the weights:
   `bash scripts/download-models.sh klein kleinte kleinvae iris iriste sdxl sdxlvae`.

Run one GPU job at a time, with servers in their own scope. Overlapping jobs skewed our timings and
once made a training run time out.

## Can LFM3-VL do more than look?

[`bench/beyond-vision/run.sh`](../bench/beyond-vision/run.sh) `STEP` has every experiment from
[that section](fine-tuning.md#can-lfm3-vl-do-more-than-look-2026-11-01). Run it with no step for the list.

| Result | Step | Writes to `bench/beyond-vision/results/` unless noted |
|---|---|---|
| 286 of 287 tensors differ, cosine about 0.8 | `weights` | prints it |
| Repeats 32/52, 8/52, 0/52 | `replay` | `replay-lfm3vl-think.json`, `replay-lfm3-think.json` |
| Temperature 0.7 on pibench | `pibench-temp` | `pibench-temp07.json` |
| Graft 14/150, 10/48 before the limit | `graft` | `../visual-control/results/glance-graft-think-off.json`, `logs/graft-on.txt`. A run that hits its limit saves no JSON, so the partial score is in that log |
| Agent LoRA, training | `train-agent` | `logs/tr-agent.txt`, adapter in `models/` |
| Agent LoRA, 21/24 to 7/24 and 101/150 | `score-agent` | `replay-agent-lora.json`, `../visual-control/results/glance-agent-lora-think-off.json` |
| Agent LoRA on pibench | `pibench-agent` | `pibench-agent-lora.json` |
| LFM3-1.7B + VL, 57/60 | `eyes` | `eyes.json` |
| Gemma 30/30, Groot 27/30 | `big-models` | `../visual-control/results/glance-*-think-on.json` |
| New projector, 35/150 and constant answers | `projector` | `projector-eval.json`, `projector-train.json` |
| Overfit, loss 0.15 (run `projector` first) | `overfit` | `projector-overfit.json`, `logs/over-proj.txt` |

The inputs are committed in `bench/beyond-vision/data/`:

- `points.json`: the 13 stuck conversations, with the thinking kept in.
- `stuck.json`: LFM3-1.7B's replies at those points.
- `agent.json`: the 177 training examples.

`replay.py find` and `make_agent_data.py` rebuilt these from raw transcripts, which aren't in git.
A new pibench run gives different transcripts, so use the committed files to repeat our numbers.
The `replay` step saves LFM3-1.7B's replies to `models/beyond-vision-logs/stuck.json`, for comparison.

## Steering

[The scripts README](../scripts/README.md#steering-vectors) has the general recipe. Our runs: build the
generator once, make the two vectors on the CPU, sweep their strengths, then run pibench with the vector.

```bash
mamba run -n llamacpp-cuda cmake --build ../llama.cpp/build --target llama-cvector-generator
G=../llama.cpp/build/bin/llama-cvector-generator; S=bench/steering; mkdir -p models/steering
$G -m models/lfm3-1.7b/LFM3-1.7B-Q8_0.gguf -ngl 0 -t 4 --method mean -o models/steering/concise-lfm3.gguf \
  --positive-file $S/concise_pos.txt --negative-file $S/concise_neg.txt
$G -m models/lfm3-vl-2b/LFM3-VL-2B-Q8_0.gguf -ngl 0 -t 4 --method mean -o models/steering/retry-lfm3vl.gguf \
  --positive-file $S/retry_pos.txt --negative-file $S/retry_neg.txt
python3 $S/sweep.py --model models/lfm3-1.7b/LFM3-1.7B-Q8_0.gguf --vector models/steering/concise-lfm3.gguf \
  --name concise-lfm3-fine --scales=-0.3,-0.2,-0.1,-0.05,-0.02,0,0.02,0.05,0.1,0.2,0.3 --ngl 99
python3 $S/sweep.py --model models/lfm3-vl-2b/LFM3-VL-2B-Q8_0.gguf --vector models/steering/retry-lfm3vl.gguf \
  --name retry-lfm3vl --scales=0,0.03,0.06,0.1,0.2 --ngl 99
RAW=$PWD/bench/results/steering-doom MODELS=lfm3vl REPEATS=1 TIME_LIMIT=900 \
  EXTRA_ARGS="--control-vector-scaled $PWD/models/steering/retry-lfm3vl.gguf:0.03" bash bench/run-bench.sh
CTX=65536 MODELS=lfm3vl bash bench/run-bench.sh    # the plain runs to compare with, 18 attempts, up to 6 hours
python3 $S/compare_doom.py
```

## Fine-tuning the vision models

**LFM2.5-VL-450M** is native to transformers, so none of the model repo's Python runs and it needs no
sandbox:

```bash
M=models/lfm2.5-vl-450m-hf; T="$HOME/miniforge3/envs/vlft/bin/python bench/finetune/train_glance.py"
$T train --model $M --out models/lfm2.5-vl-450m-adapters/norm --method norm --lr 1e-3    # norm-only
$T train --model $M --out models/lfm2.5-vl-450m-adapters/lora --method lora --lr 2e-4    # LoRA
$T train --model $M --out models/lfm2.5-vl-450m-adapters/ia3 --method ia3 --lr 1e-2      # IA3 (also 3e-3)
$T train --model $M --out models/lfm2.5-vl-450m-adapters/vnorm --method vnorm --lr 1e-3  # vision side
$T eval  --model $M --adapter models/lfm2.5-vl-450m-adapters/norm --key norm
$T eval  --model $M --adapter models/lfm2.5-vl-450m-adapters/norm --key norm-shift --shift  # degraded
$T eval  --model $M --adapter models/lfm2.5-vl-450m-adapters/norm --key norm-new --new      # new scenes
```

`--key` names the result file, so give every eval its own. Repeat with `--seed 1` and `--seed 2`, each with
its own `--out` and `--key`. For the grid, add `--task grid`, or `--task both` for one
adapter on both jobs, with `--recover 4` for the recovery data and `--mask-blocked` on `eval --task grid`
for the harness rule. Each adapter's `train.json` records its settings, and scores land in
[`bench/finetune/results/`](../bench/finetune/results/).

**LFM3-VL-2B** uses the same flags, inside the sandbox. Use `--lr 2e-4` for LoRA and norm-only, and
`--lr 1e-2` for IA3:

```bash
export OUT=$PWD/models/lfm3-vl-2b-adapters
bash bench/finetune/sandbox.sh train --model /model --out /out/norm --method norm --lr 2e-4
bash bench/finetune/sandbox.sh eval --model /model --adapter /out/norm --key lfm3vl-norm
bash bench/finetune/sandbox.sh eval --model /model --adapter /out/norm --key lfm3vl-norm-new --new
```

Results stay in `OUT`, so check them before copying them out. To serve an adapter,
[bake or convert it](fine-tuning.md#serving-it-through-llamacpp), then score it with
`run_control.py glance KEY --think off`, with `--new` or `--shift` for the other sets (and the vlft Python).

## Pictures

Everything uses `P="mamba run --no-capture-output -n vlft python"`. Runs land in
`models/image-gen/style-runs/`, and their cost in
[`style-runs.json`](../bench/imagegen/results/style-runs.json).

```bash
python3 bench/imagegen/run_gen.py klein iris sdxl                    # the speed table
$P bench/imagegen/run_inputs.py klein iris sdxl                      # picture, mask and reference inputs
python3 bench/imagegen/fetch_commons.py                              # our exact public domain sets
python3 bench/imagegen/make_subject.py                               # the invented person
T="$P bench/imagegen/train_style.py"
$T train --set vangogh --method lora --trigger vgstyl                # also norm
$T train --set gris --method lora --trigger grsstyl                  # also norm
$T train --set atget --method lora --trigger atgstyl                 # also norm
$T train --set vangogh --method lora --trigger vgstyl --steps 3000 --tag=-3k
$T train --set subject --method dora --trigger zxq                   # also lora, norm, lora+norm
$P bench/imagegen/make_reg.py --word man --count 20
$T train --set subject --method lora --trigger zxq --reg reg-man --tag=-reg
$T eval --run subject-lora-reg --trigger zxq
$P bench/imagegen/klein_person.py                                    # Klein -r, no training
$P bench/imagegen/bake_sdxl.py --run subject-norm                    # 10+ minutes when the disk is nearly full
$P bench/imagegen/invert.py --set subject --word zxqtok              # the 8 KB word
$P bench/imagegen/invert.py --set vangogh --word vgword --like painting
$P bench/imagegen/invert.py --set gris --word grword --like painting
$P bench/imagegen/invert.py --set vangogh --word vgart --like art --tag=-art
$P bench/imagegen/eval_sdcli.py --run subject-word --trigger zxqtok  # sd-cli pictures, for any run
$P bench/imagegen/stack.py                                           # person + style words
```

Write `--tag=-3k` with the `=`: on its own, argparse reads `-3k` as a flag.

## The other benchmarks

| Benchmark | How to run it |
|---|---|
| pibench | [`bench/README.md`](../bench/README.md#running-it) |
| pibench, Bonsai 2 at 32K | `RAW=$PWD/bench/results/raw-bonsai2-32k MODELS=bonsai2 bash bench/run-bench.sh`, about 2.3 hours. Numbers: `RAW=… python3 bench/batch_stats.py bonsai2`, committed as [`stats-bonsai2-32k.txt`](../bench/results/stats-bonsai2-32k.txt) |
| pibench, 2026-11-02 batch (Dory, Qwen3-VL, both Gemma 12Bs, VibeThinker 3B) | `MODELS="coder12 agentic12 dory qwen3vl vibe3b" bash bench/run-bench.sh`, about 10 hours, then `python3 bench/summarize.py` (the two Gemma 12B launchers were removed when the models were retired; they're
in git history at `2764962`). The per-model numbers in the write-up: `python3 bench/batch_stats.py dory qwen3vl agentic12 coder12 vibe3b`, committed as [`stats-2026-11-02.txt`](../bench/results/stats-2026-11-02.txt) (it reads the raw transcripts, which stay on this disk) |
| Vision probe | [`bench/vision/README.md`](../bench/vision/README.md) |
| Visual control | [Run it yourself](visual-control.md#run-it-yourself) |
| Decision models and the command guard | Weights: `download-models.sh d1 clefflash`. `bench/grading/run_guard.py` runs from its own prompts; `run_judge.py` also needs the raw pibench transcripts. Each script's header has its usage, and [`bench/grading/results/`](../bench/grading/results/) the scores |
| Retrieval | Weights: `download-models.sh eg2`. `bench/retrieval/run_retrieval.py bm25` needs no server; `embed KEY` needs an embedding server on port 8080. Scores: [`bench/retrieval/results/`](../bench/retrieval/results/) |
| Voice (LFM3-Audio-2B) | [`bench/voice/README.md`](../bench/voice/README.md). Rerun on the fork on 2026-11-01: same scores as the first run |
| Clef-Flash, d1, EmbeddingGemma and LocateAnything quick checks | `bash bench/smoke/run.sh`. Weights: `download-models.sh clefflash d1 eg2 locate`. Rerun on the fork on 2026-11-01: same scores, [saved here](../bench/smoke/results/) |
| Every chat model launches, with speed, tool call and code check | `LAUNCH=llm bash bench/recheck/run.sh`, about 70 minutes: through `scripts/llm` at its defaults (expert cache, Gemma's template fix), with a pi one-shot per chat model (2026-11-07). Plain `bash bench/recheck/run.sh` launches through bare `run-qwen.sh` / `run-server.sh`, the static split (2026-11-01 and 2026-11-02). [Results](../bench/recheck/README.md) |
| Bonsai 2's context fit: 32K default, 48K max, q8_0 KV | `bash bench/bonsai-ctx/run.sh`, about 10 minutes. [Results](../bench/bonsai-ctx/README.md) |
| MoE expert cache (how-it-fits §12) and the Gemma template fix: A/B rows | The default build (`bash scripts/build-llama-cuda.sh`, the fork's `moe-cache-max-batch` branch at `69ca796dd`; the numbers were measured on `moe-offload` `3a72c57be`, which it extends by one flag, [§12](how-it-fits.md#12-let-a-cache-pick-the-experts-the-default-for-the-big-moes)). Each A/B row: `bash bench/moe-cache/srvbench.sh LABEL ARGS` with the flags in [`RESULTS.md`](../bench/moe-cache/RESULTS.md); the agent-loop runs: `bench/moe-cache/agentloop.py`. Raw logs in [`bench/moe-cache/logs/`](../bench/moe-cache/logs/); `bench/check_numbers.py` recomputes the headline numbers from them |
| MoE expert cache: the pibench batch | `MOE_CACHE=auto MODELS="qwen36 flint gemma26b" RAW=$PWD/bench/results/raw-moecache bash bench/run-bench.sh`, then `RAW=… python3 bench/batch_stats.py qwen36 flint gemma26b` ([`stats-moecache.txt`](../bench/results/stats-moecache.txt)) and `python3 bench/moe-cache/pibench_speed.py bench/results/raw bench/results/raw-moecache qwen36 flint gemma26b` ([`pibench-speed.txt`](../bench/moe-cache/logs/pibench-speed.txt)); both read the raw transcripts, which stay on this disk |
| MoE expert cache: long prompts, other models, the pre-pibench fit check, expert usage, pruning, the experimental steps | One script each: `longctx/`, `models/`, `fitcheck.sh`, `expert_usage.py`, `experts/prune-eval.sh`, `experimental/`; what each runs and where its logs are is in [the Files table](../bench/moe-cache/README.md#files). Steps 8 to 11 need the fork's `moe-cache-warm` build (how to rebuild it: [the README](../bench/moe-cache/README.md#two-experiments-in-the-fork-experimental)) |
| Cactus Mirage | `python -I bench/cactus/fetch.py models/cactus` once, then `bash bench/cactus/sandbox.sh -I /bench/cactus/mirage_glance.py 30` (no network). [`bench/cactus/README.md`](../bench/cactus/README.md) |
| Gemma 12B probe (pibench harness vs the models) | Needs the retired 12B launchers from git history; [`bench/gemma12-probe/README.md`](../bench/gemma12-probe/README.md#rerun) |
| [`examples/`](../examples/README.md) | No numbers: recipes. Each script starts its model with `scripts/llm` |
| Relaxed acceptance | Build the fork's `relaxed-accept` branch into its own folder: `LLAMA_REF=relaxed-accept SRC_DIR=$PWD/../llama.cpp-relaxed bash scripts/build-llama-cuda.sh` (`LLAMA_REF` takes a branch, tag or commit; the repo defaults to our fork), then `bench/relaxed/run_relaxed.py` |

## What isn't in git, and why

| Not committed | Size | Why it's OK |
|---|---|---|
| Model weights, adapters, pictures | 284 GB (`du -sh models`, 2026-11-06) | Downloads are pinned, and the commands above rebuild the rest |
| Raw pibench transcripts in `bench/results/` (`raw/` 890 MB, plus `raw-moecache/`, `raw-bonsai2-32k/`, `agent-doom/`, `steering-doom/`, `temp07-doom/`, `raw-gemma-probe/`) | 1.2 GB in all (2026-11-06) | Scores and per-run stats are committed. Later analysis of the raw text, like a new replay, needs a new run |
| The server logs in `loop/logs/` | small | The beyond-vision ones were copied to `bench/beyond-vision/results/logs/`; the others weren't |

## Known gaps

An audit on 2026-11-01 found results these commands can't reproduce yet. `bench/check_numbers.py`
covers only the beyond-vision, glance and steering numbers, and (since 2026-11-04) the MoE-cache ones.

| Result | What's missing |
|---|---|
| The fork speed rechecks | Scripts and raw output aren't committed |
| Thinking stays in the prompt ("token counts prove it") | The script isn't committed, and it needs the raw transcripts |
| Random-walk control, LoRA at scale 0, the green-cast figures | No script or result found |
| Needle tests, and most of the [graveyard](graveyard.md) | No script or raw output |
| Speed, VRAM and code-check tables | Older figures have no raw output. A full rerun of every chat model is in [`bench/recheck/`](../bench/recheck/README.md) |
| Retrieval | It reads the live docs and code, so the corpus has grown (269 and 235 chunks, was 236 and 163) and a rerun is a different test |
| pibench habits and judge scores | They need the raw transcripts, which are only on this disk |
| Detect-all counts, web-page clicks | No saved output; the page captures aren't committed |
| Training time and VRAM, image verdicts | The evidence is in ignored `models/` and `loop/logs/` |
| Results from `vendor/llama.cpp-lfm3` and `vendor/locate-llama.cpp` | Those builds are gone. Voice and the quick checks have been rerun on the fork |
| TwIL-8B pibench | Not run yet; its speed and code check are in the recheck (2026-11-07) |

[`loop/`](../loop/README.md) keeps the night-by-night commands and journal. It's a record of what
happened, so use this page to repeat anything.
