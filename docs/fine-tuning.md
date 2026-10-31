# 🎛️ Fine-tuning a 450M vision model on this laptop

The [visual control probe](visual-control.md) showed a small vision model can act on what it
sees: LFM3-VL-2B gets 144 of 150 one-picture decisions right with thinking on, and 117 without
it. Can a model a quarter of that size learn the same job in a few minutes of training on an
8 GB GPU? And how much of the model do we actually have to train?

Everything is in [`bench/finetune/`](../bench/finetune/): `train_glance.py` (fit test, training,
evaluation) and the results.

## The setup

- **Model:** [LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M) (Liquid AI, LFM
  Open License v1.0), loaded in fp16 with plain transformers 5.22: native support, **no
  `trust_remote_code`**, weights pinned and checked by sha256. [LFM3-VL-2B](models/lfm3-vl-2b.md)
  was the first choice, but transformers has no LFM3 support yet and its repo's own Python has to
  run, so it came later, in a sandbox ([below](#lfm3-vl-2b-in-a-sandbox)).
- **Task:** the five glance scenes (tanks, a dialog's buttons, objects on a table, a traffic
  light, a pressure gauge). Training uses 400 scenes of each kind from seeds the test never uses
  (`train-tanks-17`), so no test picture is ever trained on.
- **Target:** the right tool call, rendered through the model's own chat template
  (`<|tool_call_start|>[refill(tank="B")]<|tool_call_end|>`), or a one-line "no such button"
  when no call is right. The loss covers only the answer.
- **Training:** one pass over 2,000 scenes, batch 8 (gradient accumulation), fp16 autocast with
  fp32 copies of what is trained (the RTX 2070 has no bf16), gradient checkpointing.
- **Scoring:** the probe's own 30 scenes per kind and the probe's own scorers. The answers are
  parsed with Python's `ast` (call names and literal arguments only); nothing the model writes is
  run.

## The results

| What we trained | Trainable parameters | Training time | Peak VRAM | 🛢️ | 🖱️ | 🤏 | 🚦 | ⏲️ | **All** (clean) | Degraded pictures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nothing (the base model) | 0 | — | — | 8 | 6 | 0 | 0 | 0 | **14** | 19 |
| **LoRA** (rank 16, attention + MLP + conv projections), 3 seeds | 6.0M (1.3%) | 7.5 min | 1.36 GB | 30 / 30 / 26 | 24 / 28 / 21 | 22 / 19 / 22 | 30 ×3 | 30 ×3 | **136 / 137 / 129**, mean 134 | 118 / 130 / 118, mean **122** |
| **Norm-only**, language model (its RMSNorm gains), 3 seeds | **34,560 (0.008%)** | 5.6 min | 1.27 GB | 28 / 30 / 29 | 25 / 21 / 23 | 22 / 23 / 22 | 30 ×3 | 30 / 26 / 29 | **135 / 130 / 133**, mean 133 | 121 / 116 / 115, mean 117 |
| **IA3** (a learned scale per channel), lr 1e-2, 3 seeds | 79,872 (0.018%) | 5.8 min | 1.27 GB | 27 / 25 / 18 | 29 / 25 / 22 | 25 / 21 / 20 | 30 / 30 / 30 | 27 / 21 / 19 | **138 / 122 / 109**, mean 123 | 133 / 108 / 92, mean 111 |
| IA3, lr 3e-3 | 79,872 | 5.8 min | 1.27 GB | 24 | 24 | 22 | 30 | 17 | 117 | 102 |
| **Norm-only, vision side** (the image encoder's LayerNorms) | 38,400 (0.009%) | 5.5 min | 1.27 GB | 15 | 0 | 0 | 30 | 28 | 73 | 45 |
| *for comparison:* [LFM3-VL-2B](models/lfm3-vl-2b.md), no training, thinking on / off | — | — | — | 25 / 24 | 30 / 30 | 29 / 26 | 30 / 27 | 30 / 10 | *144 / 117* | — |

Out of 150 (30 scenes of each kind). Measured 2026-10-29, greedy decoding in transformers, under
0.25 s a decision. "Degraded pictures" are the same test scenes shrunk to 70%, darker or
brighter, lower contrast and JPEG-compressed at quality 35, a worse camera. LoRA, the
language-model norms and IA3 at 1e-2 were trained three times (seeds 0, 1, 2: a different order of the training
scenes and a different start for LoRA and IA3); the others once. LFM3-VL's row was measured through
llama.cpp with a forced call, so the harnesses differ.

## What we learned

- **The base model never calls a tool with a picture in front of it.** Its card says function
  calling works "for text-only input", and here it held: 0 calls in 150 scenes. It answered in
  prose ("I'm sorry, I don't have the capability to control a robot arm"), wrote the tool's
  name as plain text (`stop`, `close_valve`), and misread the tanks ("A: 50% (below 30%)"). Its
  14 points are all scenes where *no* call was the right answer.
- **A few minutes of training fixes that, and more.** Around 134/150 after LoRA: well above
  LFM3-VL-2B without thinking (117), close to it with thinking (144), at a quarter of its size
  and without a 3-second think. Every answer is now a well-formed call, multiple tank refills
  included. What's left is mostly clicks and picks a little off target.
- **Training only the language model's norm gains is as good as LoRA on clean pictures** (mean
  133 against 134, and the seeds overlap: 130–135 against 129–137) with 173× fewer trainable
  numbers. On degraded pictures LoRA keeps a little more (mean 122 against 117). Our first,
  single-seed run said the opposite (121 against 118): one run was not enough to tell them apart.
  So the "tune only the normalization" idea works on a transformer for teaching it *how to
  answer* in a narrow domain; it just isn't more robust than LoRA.
- **IA3 is the least predictable.** At 3e-3 it lagged (117, calling `close_valve` on 13 dials
  where something else was right). At 1e-2 its first run was the best of all (138 clean, 133
  degraded), and the next two seeds gave 122 and 109: a mean of 123, with the widest spread.
  LoRA and the norms are steadier.
- **The vision side alone teaches the format, not the pointing.** Training only the image
  encoder's LayerNorms made the model *call* tools (traffic light 30/30, gauge 28/30), so a
  nudge in what the language model sees is enough to switch it into tool-calling mode. But its
  clicks and picks are placeholders, the same (500, 500) or (400, 400) whatever the picture
  (0/30 each), and it refills all three tanks. Where to point is decided in the language model.
- **LoRA's misses on degraded pictures look like over-learning the clean ones:** in its first
  run it refilled tanks that aren't low, often all three, and the tank score fell from 30 to 11.
- **The caveat:** training and test pictures come from the same drawing code, with different
  seeds. This shows the model can learn *our* scenes, not that it learned gauges in general. The
  degraded-picture test is a first step; real photos would be the real test.

## The grid robot: from 0 to 5 of 8 mazes

The same recipe on the probe's [grid task](visual-control.md#the-tests): a 7×7 grid, a blue
robot, a green goal, and in half the episodes 11 wall cells. One fresh request per step (the
picture, the rules and the last three moves as text), one `move` call. Training data: every step
of the shortest walk through 400 grids from their own seeds (2,756 steps, no test grid among
them), the target being a move on a shortest path. Test: the probe's own 16 grids, closed loop.

| Model | Open grids | Mazes | Steps / shortest (solved) | Blocked moves | Time per step |
|---|---:|---:|---:|---:|---:|
| LFM2.5-VL-450M, base | 0/8 | 0/8 | — | — | — |
| LFM2.5-VL-450M + LoRA (9.9 min of training) | **8/8** | 1/8 | 1.00 | 154 | ~0.15 s |
| LFM2.5-VL-450M + norm-only (7.1 min) | 7/8 | **3/8** | 1.00 open, 1.24 mazes | 95 | ~0.15 s |
| LFM2.5-VL-450M + LoRA, **harness skips moves seen blocked** | **8/8** | **5/8** | 1.00 open, 1.35 mazes | 14 | ~0.2 s |
| LFM2.5-VL-450M + norm-only, **harness skips moves seen blocked** | 7/8 | **5/8** | 1.00 open, 1.58 mazes | 16 | ~0.2 s |
| *controls:* base + the same harness / a random walk + the same harness (mean of 200) | *0/8 / 0.7* | *0/8 / 0.4* | — | — | — |
| *[LFM3-VL-2B](models/lfm3-vl-2b.md), thinking, no training* | *8/8* | *3/8* | *1.50* | *6* | *44 s* |
| *[Qwen3-VL-8B](models/qwen3-vl-8b.md), no training* | *8/8* | *1/8* | *1.80* | *67* | *0.7 s* |

On open grids both tuned models walk the shortest path every time they arrive. In mazes they do
what the thinking model does in a few hundred times the time, but they fail differently: the
thinking model rarely bumps a wall and runs out of steps thinking, the tuned ones walk straight
into a wall and keep pressing (one norm-only episode: 28 blocked moves out of 28). They learned
"head for the goal", not "go around": the training walks never show a blocked move, so the
"(blocked)" in the prompt means nothing to them. One seed each.

**Teaching recovery didn't work; a harness rule did.** Adding 4 random off-path states per grid
to the training data, most of them right after a bump ("left (blocked)" in the prompt), made both
models *worse* (LoRA 6/8 and 1/8, norm-only 3/8 and 1/8). In hindsight that data couldn't teach
what we wanted: the shortest-path move from a cell is the same whether or not you just bumped, so
"(blocked)" never changed the right answer and there was nothing to learn from it. The real
failure is one wrong move at one cell: after a bump the picture doesn't change, so a model that
only looks at the picture makes the same wrong move forever. So we moved the fix into the
harness: score all four moves by the model's own probability and take the best one that hasn't
already been blocked from this cell. Both tuned models then solve **5 of 8 mazes**, ahead of
LFM3-VL-2B with thinking (3/8) at a fraction of a second per step. The controls show the harness
isn't doing the work: with the base model it solves nothing, and a random walk under the same
rule averages 0.4 of 8. That's the visual control lesson again: let the harness keep the state
and the rules, and ask the model only what it's good at.

## One adapter for both jobs

Trained on the glance scenes and the grid walks together (4,756 examples, one pass):

| Trained on both | Glance, clean | Glance, degraded | Grid open / mazes | Grid open / mazes, harness skips blocked moves | Training |
|---|---:|---:|---:|---:|---:|
| LoRA, 1 seed | 127 | 124 | 8/8, 0/8 | 8/8, 5/8 | 17 min |
| **Norm-only** (34,560 parameters), 3 seeds | 139 / 115 / 133, mean 129 | 118 / 102 / 135, mean 118 | 8/8 ×3; 0, 1, 2 /8 | **8/8 and 6/8, all three seeds** | 13 min |

**Steering is the solid part:** all three norm-only seeds walk every open grid and, with the
harness rule, solve 6 of 8 mazes, the best maze score in this repo (LFM3-VL-2B with thinking:
3/8; the tuned grid-only models: 5/8). **The glance side gets less steady when it shares the
model:** the three seeds score 139, 115 and 133. The 115 comes from one habit: that model calls
`report` on 20 of 30 gauges, including the ones that needed a valve, the very mistake
LFM3-VL-2B makes without thinking. So the same 34,560 numbers can hold both behaviours, but
which rule survives training is a matter of luck at this size of run. Train each job's adapter
separately, or check every run, before trusting one.

## Serving it: not through llama.cpp yet

The plan was to merge the result, convert it to GGUF and run it through the same visual control
harness as everyone else. The base model's GGUF (pinned, `download-models.sh lfm25vl`, audit
clean) loads fine, but in our llama.cpp build (upstream, 2026-10-27) **any request whose tools
have a string argument fails**: "error parsing grammar: expecting newline or end at ::=
tool-call". Tools with integer arguments or none get through, though one integer call came back
cut short (`{"x":1234`). So until that's fixed, an LFM2.5 model with tools is a transformers job
here (or a raw-prompt job, parsing `<|tool_call_start|>` ourselves).

## Run it yourself

```bash
mamba create -n vlft python=3.12 pip
mamba activate vlft
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.8.0 torchvision==0.23.0
pip install transformers==5.22.0 peft accelerate pillow safetensors
# weights: LiquidAI/LFM2.5-VL-450M at revision fc6221c into models/lfm2.5-vl-450m-hf/
python bench/finetune/train_glance.py fit   --model models/lfm2.5-vl-450m-hf          # one step, peak VRAM
python bench/finetune/train_glance.py train --model models/lfm2.5-vl-450m-hf --out models/lfm2.5-vl-450m-adapters/norm --method norm --lr 1e-3
python bench/finetune/train_glance.py eval  --model models/lfm2.5-vl-450m-hf --adapter models/lfm2.5-vl-450m-adapters/norm --key norm [--shift]
```

(Call the env's `python` directly: `mamba run` takes `--n` for itself.)

## LFM3-VL-2B, in a sandbox

Same scenes, same recipe, thinking off, and this time the model repo's own Python had to run
(`trust_remote_code`), pinned at revision `ba44bd5`, weights checked by sha256.
[`sandbox.sh`](../bench/finetune/sandbox.sh) runs it in Docker with the GPU, no network, a read-only
root and code, not as root, one writable folder. A search of the 3,328 lines of Python found no
network, shell or `eval` calls; they were not read line by line, so the sandbox is what we rely on.
Its Triton kernels need a C compiler at run time, so inside the sandbox it runs on plain torch.

| What we trained | Trainable parameters | Training time | Peak VRAM | 🛢️ | 🖱️ | 🤏 | 🚦 | ⏲️ | **All** (clean) | Degraded pictures | Per decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nothing (the base model, thinking off) | 0 | — | — | 8 | 28 | 29 | 22 | 10 | **97** | — | 0.8 s |
| **LoRA**, 1 seed | 15.4M (0.8%) | 25 min | 4.86 GiB | 30 | 30 | 29 | 30 | 30 | **149** | **149** | 0.6 s |

Out of 150, measured 2026-10-30. The one miss, in both columns, is the same pick scene. That beats
the untouched model with thinking on (144/150 at 2.7 s, through llama.cpp) at under a quarter of the
time per decision, and the gap thinking was buying is gone: the rules of these scenes went into the
weights. It's one seed, and training and test use the same drawing code, so this shows it learned
*our scenes*, not that it got better at seeing in general. The checks below test exactly that.

### Did it hold up? (2026-10-30)

| Check | Untouched, thinking off | Tuned |
|---|---:|---:|
| Three LoRA seeds, glance | 97 | **149 / 150 / 150** |
| Three seeds, degraded pictures | — | **149 / 150 / 150** |
| Three seeds, [scenes never trained on](../bench/finetune/scenes_new.py) (new layouts, palettes, button names) | 105 | **150 / 150 / 149** |
| [Vision probe](vision-probe.md), one seed | 68% | **78%** (91% for the untouched model with thinking on) |

So it isn't memorising our drawing code, and it forgot nothing. One adapter trained on glance and
the grid together (47 min, 4.86 GiB): glance 150/150, new scenes 148/150, grid 8/8 open and 3/8
with walls, 5/8 with the harness rule that never repeats a blocked move.

**In llama.cpp.** `convert_lora_to_gguf.py` from our fork makes a 31 MB adapter, and it runs on the
stock model with no sandbox: 149/150, 0.42 s a decision, 3.2 GB of VRAM. With the adapter's scale
set to 0, a 30-scene sample fell to 21/30, so it really is applied.

    QMODEL=lfm3vl EXTRA_ARGS="--lora models/lfm3-vl-2b-adapters/lora.gguf" bash scripts/run-qwen.sh --image

Ask with thinking off (`chat_template_kwargs: {"enable_thinking": false}`). The adapters aren't in
the repo; `sandbox.sh train` rebuilds one in about 25 minutes.

