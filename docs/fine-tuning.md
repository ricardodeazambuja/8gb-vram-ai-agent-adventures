# 🎛️ Fine-tuning small vision models on this laptop

The [visual control probe](visual-control.md) showed a small vision model can act on what it
sees: LFM3-VL-2B gets 144 of 150 one-picture decisions right with thinking on, and 117 without
it. Can a few minutes of training on an 8 GB GPU close that gap, or even let a model a quarter of
that size do the job? And how much of the model do we actually have to train?

Everything is in [`bench/finetune/`](../bench/finetune/): `train_glance.py` (fit test, training,
evaluation) and the results. Every run on this page has its commands in
[Reproducing our results](reproduce.md).

## In short

- **Tuned LFM3-VL-2B is the result to use.** With thinking off, baked into a GGUF and served by
  `llama-server`, it gets 150/150 on the glance scenes at 0.3 s a decision in 3.1 GB of VRAM. The
  untouched model served the same way gets 109; with thinking on (forced call), 144 at 2.7 s.
- **It holds up.** 149 and 148 of 150 on drawings it never trained on (new layouts, palettes and
  button names), 150 and 149 on degraded pictures, and its [vision probe](vision-probe.md) score
  went up, not down (78% against 68%, thinking off).
- **The tiny methods are enough.** Norm-only and IA3 train about 200K values (under 1 MB), about 75
  times fewer than LoRA, and land within three scenes of it. Trained on the grid robot too, IA3
  plus a harness rule solves 7 of 8 mazes, the best maze score in this repo.
- **The 450M LFM2.5-VL learns our drawings, not the job.** Training takes it from 14/150 to around
  133, but its served norm-only adapter falls to 77 on never-seen drawings.
- **Training gave LFM3-VL a visual skill, not an agent's.** No fine-tune or steering vector made it
  pass pibench; temperature 0.7 ends its doom loops ([below](#can-lfm3-vl-do-more-than-look-2026-11-01)).

**Use it** (after [baking the adapter](#serving-it-through-llamacpp); the adapters aren't in the repo):

```bash
MODEL=models/lfm3-vl-2b-baked/ia3.gguf scripts/llm lfm3vl
python3 examples/look.py --fast        # one picture, one tool call, thinking off
```

## The tiny methods vs LoRA

| | Norm-only (BN, LN, RMSNorm gains) | IA3 | LoRA |
|---|---|---|---|
| What it trains | the gains already in the norm layers | one new scale per row or column of a layer | a small low-rank matrix added to a layer's weights |
| What that can do | turn existing features up or down | turn existing features up or down | build new mixes of features |
| Size | smallest | small | biggest of the three |
| Good for | changing behaviour the model already has in it | the same | adding things the model doesn't know |
| On disk | [baked into a whole model](#serving-it-through-llamacpp) | baked into a whole model | a separate file you can dial, switch off or stack |

Think of a mixing desk. Norm-only and IA3 only move the faders, so if the sound you want is already
on a track, that's all you need, and it's why they did so well here: LFM3-VL could already see the
scenes and only had to change how it acted on them. LoRA adds new tracks. That costs more, but it's
what you need for knowledge the model doesn't have, like one person's face.

None of this is new. Style carried in norm parameters goes back to
[Dumoulin et al.](https://arxiv.org/abs/1610.07629) and [AdaIN](https://arxiv.org/abs/1703.06868)
(Huang & Belongie). [Training only BatchNorm](https://arxiv.org/abs/2003.00152) is from Frankle et
al., [LayerNorm tuning](https://arxiv.org/abs/2312.11420) for language models from Zhao et al., and
[IA3](https://arxiv.org/abs/2205.05638) from Liu et al.'s T-Few. Variants of LoRA exist (DoRA, OFT,
LoKr), but a [2026 study](https://arxiv.org/abs/2601.22708) reports that a well-tuned plain LoRA
matches or beats most of them; we've only seen its summary, not read it through.

## The setup

- Models: [LFM3-VL-2B](models/lfm3-vl-2b.md) and [LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M)
  (Liquid AI, LFM Open License v1.0), both pinned and checked by sha256
  (`scripts/download-hf-weights.sh lfm3vl` / `lfm25vl`). LFM2.5-VL loads in fp16 with plain
  transformers 5.22: native support, **no `trust_remote_code`**. Transformers 5.22 has no LFM3
  support, so LFM3-VL's repo's own Python (revision `ba44bd5`) has to run, and it runs only in the
  sandbox below.
- Task: the five glance scenes (tanks, a dialog's buttons, objects on a table, a traffic
  light, a pressure gauge). Training uses 400 scenes of each kind from seeds the test never uses
  (`train-tanks-17`), so no test picture is ever trained on.
- Target: the right tool call, rendered through the model's own chat template
  (`<|tool_call_start|>[refill(tank="B")]<|tool_call_end|>`), or a one-line "no such button"
  when no call is right. The loss covers only the answer.
- Training: one pass over 2,000 scenes, batch 8 (gradient accumulation), fp16 autocast with
  fp32 copies of what is trained (the RTX 2070 has no bf16), gradient checkpointing.
- Scoring: the probe's own 30 scenes per kind and the probe's own scorers. The answers are
  parsed with Python's `ast` (call names and literal arguments only); nothing the model writes is
  run. "Degraded pictures" are the same test scenes shrunk to 70%, darker or brighter, lower
  contrast and JPEG-compressed at quality 35, a worse camera.
- Sandbox: [`sandbox.sh`](../bench/finetune/sandbox.sh) runs LFM3-VL in Docker with the GPU, no network, a read-only
  root and code, not as root, one writable folder. A search of the 3,328 lines of Python found no
  network, shell or `eval` calls; they were not read line by line, so the sandbox is what we rely on.
  Its Triton kernels need a C compiler at run time, so inside the sandbox it runs on plain torch.
- Baselines: the untouched LFM3-VL scores differently in each harness, so every comparison
  names its own. Thinking off, it gets 97/150 in transformers, 109 served by `llama-server`, and 117
  through llama.cpp with a forced call ([visual control](visual-control.md#the-results)); its 144
  with thinking on is also the forced-call harness.

## LFM3-VL-2B

Same scenes, same recipe, thinking off, trained and scored in the sandbox (transformers).

| What we trained | Trainable parameters | Training time | Peak VRAM | 🛢️ | 🖱️ | 🤏 | 🚦 | ⏲️ | **All** (clean) | Degraded pictures | Per decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nothing (the base model, thinking off) | 0 | — | — | 8 | 28 | 29 | 22 | 10 | **97** | — | 0.8 s |
| **LoRA**, 1 seed | 15.4M (0.8%) | 25 min | 4.86 GiB | 30 | 30 | 29 | 30 | 30 | **149** | **149** | 0.6 s |

Out of 150, measured 2026-10-30. The one miss, in both columns, is the same pick scene. That beats
the untouched model with thinking on (144/150 at 2.7 s, through llama.cpp with a forced call) at under
a quarter of the time per decision, and the gap thinking was buying is gone: the rules of these scenes went into the
weights. It's one seed, and training and test use the same drawing code, so this shows it learned
*our scenes*, not that it got better at seeing in general. The checks below test exactly that.

### Did it hold up? (2026-10-30)

| Check | Untouched, thinking off (transformers) | Tuned |
|---|---:|---:|
| Three LoRA seeds, glance | 97 | **149 / 150 / 150** |
| Three seeds, degraded pictures | — | **149 / 150 / 150** |
| Three seeds, [scenes never trained on](../bench/finetune/scenes_new.py) (new layouts, palettes, button names) | 105 | **150 / 150 / 149** |
| [Vision probe](vision-probe.md), one seed | 68% | **78%** (91% for the untouched model with thinking on) |
| Norm-only (language model, 201K values, 0.4 MB, 20 min, 4.69 GiB, lr 2e-4), one seed | 97 clean, 105 new | **150** clean, **149** new, **150** degraded |
| IA3 (196K values, 0.8 MB, 21 min, 4.63 GiB, lr 1e-2), one seed | 97 clean, 105 new | **150** clean, **147** new, **149** degraded |
| *LFM2.5-VL-450M, adapters trained on glance + grid, on new scenes* | *14* | *LoRA 110, norm-only 98 (127 and 139 on the drawings they trained on)* |

So it isn't memorising our drawing code, and it forgot nothing. The tiny methods do the job on the
2B model as well: about 75 times fewer values than LoRA and within three scenes of it. The misses
on new scenes are all tanks: IA3's three refill both A and B. They run in llama.cpp too,
[baked into the weights](#serving-it-through-llamacpp). The 450M model tells a different story: its
adapters do far better than its 14/150 base, but they lose 17 and 41 points on drawings they never
saw, the norm-only one the most, so its near-perfect clean scores overstated how much it had
learned ([LFM2.5-VL-450M](#lfm25-vl-450m-the-same-recipe-at-a-quarter-the-size)). Those LFM2.5 adapters
shared their training with the grid, so they aren't a like-for-like match for LFM3-VL's glance-only
ones.

### One adapter for glance and the grid

Trained on the glance scenes and the [grid robot](#the-grid-robot-from-0-to-5-of-8-mazes)'s walks
together, one adapter does both jobs (glance and new scenes in transformers, untouched baseline 97 and 105):

| LFM3-VL-2B, trained on both (one seed) | Glance | New scenes | Grid open | Mazes | Mazes, harness skips blocked moves | Mazes in `llama-server` | Training | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 97 | 105 | 3/8 (served) | — | — | 0/8 | | |
| LoRA | 150 | 148 | 8/8 | 3/8 | 5/8 | 1/8 | 47 min | 4.86 GiB |
| **Norm-only** | 150 | 148 | 8/8 | **6/8** | 6/8 | **5/8** | 44 min | 4.69 GiB |
| **IA3** | 150 | 149 | 8/8 | 5/8 | **7/8** | 4/8 | 41 min | 4.63 GiB |

The smallest adapters beat LoRA at steering and match it at the glance, the same result as on
LFM2.5 ([below](#one-adapter-for-both-jobs)) but without its unsteady glance. Before the harness rule norm-only hits a wall 32 times
over the 16 grids, against LoRA's 96. IA3 with the rule, 7/8, is the best maze score in this repo.
Steps stay close to the shortest path on what they solve (1.03 to 1.13 times it). That's one seed
each over 8 mazes, so the order of the three is a rough reading, and the LFM2.5 seeds below varied
by more than these gaps on the glance.

Baked into a GGUF and served, the glance holds (norm-only 150 and 148, IA3 150 and 149) but mazes
slip: norm-only and IA3 lose one, LoRA two (`--lora`, on stock). The server samples rather than taking
the top move, so this looks like sampling, though no run here separates that from the
format change. Open grids stay 8/8, at 0.3 s a step. The harness rule isn't in the served runs.

## LFM2.5-VL-450M: the same recipe at a quarter the size

Can a model a quarter of LFM3-VL's size learn the same job? On the drawings it trained on, yes.

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
0.25 s a decision. LoRA, the
language-model norms and IA3 at 1e-2 were trained three times (seeds 0, 1, 2: a different order of the training
scenes and a different start for LoRA and IA3); the others once. LFM3-VL's row was measured through
llama.cpp with a forced call, so the harnesses differ.

### What we learned

- The base model never calls a tool with a picture in front of it. Its card says function
  calling works "for text-only input", and here it held: 0 calls in 150 scenes. It answered in
  prose ("I'm sorry, I don't have the capability to control a robot arm"), wrote the tool's
  name as plain text (`stop`, `close_valve`), and misread the tanks ("A: 50% (below 30%)"). Its
  14 points are all scenes where *no* call was the right answer.
- A few minutes of training fixes that, and more. Around 134/150 after LoRA: well above
  LFM3-VL-2B without thinking (117, forced call), close to it with thinking (144), at a quarter of its
  size and without a 3-second think. After training every answer is a well-formed call, multiple tank
  refills included. What's left is mostly clicks and picks a little off target.
- Training only the language model's norm gains is as good as LoRA on the clean test pictures (mean
  133 against 134, and the seeds overlap: 130–135 against 129–137) with 173× fewer trainable
  numbers. On degraded pictures LoRA keeps a little more (mean 122 against 117). Our first,
  single-seed run said the opposite (121 against 118): one run was not enough to tell them apart.
  So the "tune only the normalization" idea works on a transformer for teaching it *how to
  answer* in a narrow domain; it just isn't more robust than LoRA.
- IA3 is the least predictable. At 3e-3 it lagged (117, calling `close_valve` on 13 dials
  where something else was right). At 1e-2 its first run was the best of all (138 clean, 133
  degraded), and the next two seeds gave 122 and 109: a mean of 123, with the widest spread.
  LoRA and the norms are steadier.
- The vision side alone teaches the format, not the pointing. Training only the image
  encoder's LayerNorms made the model *call* tools (traffic light 30/30, gauge 28/30), so a
  nudge in what the language model sees is enough to switch it into tool-calling mode. But its
  clicks and picks are placeholders, the same (500, 500) or (400, 400) whatever the picture
  (0/30 each), and it refills all three tanks. Where to point is decided in the language model.
- LoRA's misses on degraded pictures look like over-learning the clean ones: in its first
  run it refilled tanks that aren't low, often all three, and the tank score fell from 30 to 11.
- The caveat: training and test pictures come from the same drawing code, with different
  seeds. This shows the model can learn *our* scenes, not that it learned gauges in general. Drawings
  it never saw settled it: served, the norm-only adapter falls from 133 to 77
  ([Serving](#serving-it-through-llamacpp)), and the glance + grid adapters lose 17 and 41 points
  ([Did it hold up?](#did-it-hold-up-2026-10-30)). LFM3-VL's adapters held.

### The grid robot: from 0 to 5 of 8 mazes

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

### One adapter for both jobs

Trained on the glance scenes and the grid walks together (4,756 examples, one pass):

| Trained on both | Glance, clean | Glance, degraded | Grid open / mazes | Grid open / mazes, harness skips blocked moves | Training |
|---|---:|---:|---:|---:|---:|
| LoRA, 1 seed | 127 | 124 | 8/8, 0/8 | 8/8, 5/8 | 17 min |
| **Norm-only** (34,560 parameters), 3 seeds | 139 / 115 / 133, mean 129 | 118 / 102 / 135, mean 118 | 8/8 ×3; 0, 1, 2 /8 | **8/8 and 6/8, all three seeds** | 13 min |

**Steering is the solid part:** all three norm-only seeds walk every open grid and, with the
harness rule, solve 6 of 8 mazes (LFM3-VL-2B with thinking:
3/8; the tuned grid-only models: 5/8; tuned LFM3-VL's IA3 reached
[7/8](#one-adapter-for-glance-and-the-grid)). **The glance side gets less steady when it shares the
model:** the three seeds score 139, 115 and 133. The 115 comes from one habit: that model calls
`report` on 20 of 30 gauges, including the ones that needed a valve, the very mistake
LFM3-VL-2B makes without thinking. So the same 34,560 numbers can hold both behaviours, but
which rule survives training is a matter of luck at this size of run. Train each job's adapter
separately, or check every run, before trusting one.

## Serving it through llama.cpp

LFM2.5 and LFM3-VL both run tuned in `llama-server`. A LoRA converts to an adapter GGUF
(tried on LFM3-VL only, [below](#a-lora-as-an-adapter-file)). Norm-only and IA3 have no adapter format, but they don't need one:
[`bake_gguf.sh`](../bench/finetune/bake_gguf.sh) writes them into a normal GGUF. Norm-only replaces
the gains. IA3 multiplies each layer's weight by its scales, along the rows for an output scale and
the columns for an input one, which is the same sum the live scale does. It's plain arithmetic on
`model.safetensors`, so none of the model repo's code runs and no sandbox is needed.

```bash
bash bench/finetune/bake_gguf.sh models/lfm3-vl-2b-hf models/lfm3-vl-2b-adapters/ia3 models/lfm3-vl-2b-baked/ia3.gguf Q8_0
MODEL=models/lfm3-vl-2b-baked/ia3.gguf scripts/llm lfm3vl
```

A baked file is a whole model (1.8 GB for LFM3-VL), not a small add-on, and you can't dial it with
`/lora-adapters`. Run with no adapter, the pipeline rebuilt both official GGUFs tensor for tensor,
byte-identical; baking then changed exactly the tensors it should (119 and 46 on LFM3-VL), and
`audit-gguf` came back clean.

Scores out of 150, clean / never-seen scenes / degraded pictures, thinking off. The server samples
with each model's tested settings; the transformers runs are greedy:

| Model | Adapter | In `llama-server` | In transformers |
|---|---|---:|---:|
| LFM3-VL-2B Q8_0 | none | 109 / 110 / 109 | 97 / 105 / — |
| | norm-only | **150 / 149 / 150** | 150 / 149 / 150 |
| | IA3 | **150 / 148 / 149** | 150 / 147 / 149 |
| | LoRA (`--lora`) | 149 / — / — | 149 / 150 / 149 |
| LFM2.5-VL-450M F16 | none | 14 / 14 / 17 | 14 / 14 / 19 |
| | norm-only | 133 / 77 / 121 | 135 / — / 121 |
| | IA3 | 139 / 111 / 134 | 138 / — / 133 |
| | norm-only, glance + grid | 138 / 101 / 118 | 139 / 98 / 118 |

So nothing is lost in the move, and baked LFM3-VL (norm-only or IA3) decides in 0.3 s in 3.1 GB. LFM2.5's norm-only
adapter barely generalizes (77), which the transformers runs never tested; its IA3 does best (111).
LFM2.5 served at 1.5 GB and 0.1 s a decision. The untuned LFM3-VL gained 12 points in the server,
which uses its own sampling, not greedy decoding.

On stock llama.cpp, any LFM2.5-VL tool with a string argument fails with "failed to parse grammar":
llama.cpp looks for the text `List of tools: [`, but LFM2.5-VL's template writes
`"List of tools: " + (tools | tojson)`, so it falls through to a generic analyzer that builds a broken
grammar. Our fork has a one-line fix (`3cb2992`). Start a tuned LFM2.5-VL with
`MODEL=<gguf> scripts/llm lfm25vl`, which uses the sampling it was tested with, the same as LFM3-VL's.

### A LoRA as an adapter file

`convert_lora_to_gguf.py` from our fork makes a 31 MB adapter, and it runs on the stock model with no
sandbox: 149/150, 0.42 s a decision, 3.2 GB of VRAM (a little slower and bigger than a baked file). With the adapter's scale
set to 0, a 30-scene sample fell to 21/30, so it really is applied.

```bash
EXTRA_ARGS="--lora models/lfm3-vl-2b-adapters/lora.gguf" scripts/llm lfm3vl
```

Ask with thinking off (`chat_template_kwargs: {"enable_thinking": false}`). The adapters aren't in
the repo; `sandbox.sh train` rebuilds one in about 25 minutes.

## Run it yourself

Make the `vlft` environment and download the weights as in [Reproducing our results](reproduce.md#setup),
then, for LFM2.5-VL (no sandbox needed):

```bash
PY=~/miniforge3/envs/vlft/bin/python
$PY bench/finetune/train_glance.py fit   --model models/lfm2.5-vl-450m-hf          # one step, peak VRAM
$PY bench/finetune/train_glance.py train --model models/lfm2.5-vl-450m-hf --out models/lfm2.5-vl-450m-adapters/norm --method norm --lr 1e-3
$PY bench/finetune/train_glance.py eval  --model models/lfm2.5-vl-450m-hf --adapter models/lfm2.5-vl-450m-adapters/norm --key norm [--shift]
```

Call the env's `python` directly: `mamba run` takes `--n` for itself. LFM3-VL trains the same way
through `bench/finetune/sandbox.sh`; [Reproducing our results](reproduce.md#fine-tuning-the-vision-models)
has its commands, the grid and the rest.

## Related experiments

Two cheaper ideas tried after the adapters: a steering vector instead of training, and whether LFM3-VL
can be more than a pair of eyes.

## Steering vs training (2026-11-01)

A [steering vector](https://arxiv.org/abs/2312.06681v2) is a small nudge added to every layer's output.
It's built from pairs of contrasting prompts, with no gradient training: under a minute on the CPU, and
190 KB. We tried it because it's far cheaper than the 25 minutes an adapter takes. It works on LFM3, but the safe range is narrow, and it
didn't fix the doom loops.

**It works here.** `llama-cvector-generator` from our fork runs on both LFM3-1.7B and LFM3-VL-2B, and the
server applies the result with `--control-vector-scaled FILE:STRENGTH`. Speed didn't change: 105 tokens
a second with the vector on the GPU.

| Vector, model | Strength | Mean tokens per answer | Facts right |
|---|---:|---:|---:|
| Concise, LFM3-1.7B | 0 | 217 | 2/2 |
| | +0.05 | **151** | 2/2 |
| | +0.1 | answers start to break | |
| | -0.05 | 283 | 1/2 |
| | ±0.5 and beyond | blank or word salad | 0/2 |
| Try something else, LFM3-VL-2B | 0 | 122 | 2/2 |
| | +0.03 | **90** | 2/2 |
| | +0.06 | answers start to break | |
| | +0.1 | broken | |

That's 8 questions per strength, so a coarse check. One git answer at +0.05 also read weaker, and the
two fact questions wouldn't catch that. The usable strength is far below the 0.8 in
[the tool's example](https://github.com/ggml-org/llama.cpp/tree/master/tools/cvector-generator), so sweep from small numbers. The second vector cut answer length as well, so these
vectors move more than the one habit they were built from.

**It didn't fix doom loops.** The "try something else" vector at +0.03 on pibench, against plain
LFM3-VL:

| | Plain | With the vector |
|---|---:|---:|
| Solved | 0/18 | **0/6** |
| Calls that copy an earlier call (median) | 93% | 80% |
| Longest streak of one call, worst run | 769 | 129 |
| Longest streak of one call, median run | 3 | 8 |
| Timeouts | 14/18 at 20 min | 4/6 at 15 min |
| Context | 64K | 32K |

The agent still spends most of its calls repeating itself. The drop in repeats is a weak signal: there
are 6 runs against 18, and a shorter time limit alone caps the worst streak, so read the 769 vs 129 with
care. It solved nothing either way, and its best task, feature-inventory, stayed at 7 of 8 tests.

**Where steering stands against training:** training taught LFM3-VL a visual task it couldn't do, and
a vector can't add a skill like that. A vector is a quick way to shift a habit, at a strength you have
to find by hand, and worth trying when that's all you need. The results are in
[`bench/steering/`](../bench/steering/), the commands in [Reproducing our results](reproduce.md#steering).

## Can LFM3-VL do more than look? (2026-11-01)

LFM3-VL reads pictures well and scores 0/18 on pibench; LFM3-1.7B scores 12/18 and has no eyes. We tried
to get one setup that does both, cheapest test first. Temperature ended the doom loops but didn't make
LFM3-VL an agent, and no fine-tune did either. Pairing the two models works, and Gemma 4 26B already does
both by itself. Every test here is a pilot, so read the ranges, not the
points.

**Why LFM3-VL loops.** We replayed 13 real stuck conversations, 4 samples each, and counted exact repeats of
the stuck call. The two models share 287 tensor shapes, but 286 tensors differ: mean cosine about 0.8
and relative distance about 0.6, or 0.46 and 1.0 for the token embeddings. So they're close relatives
trained apart.

| Same 13 stuck points | Repeats the call | Points where it repeats |
|---|---:|---:|
| LFM3-VL, temperature 0.1 | 32/52 | 11/13 |
| LFM3-VL, temperature 0.7 | 8/52 | 5/13 |
| LFM3-1.7B, temperature 0.1 | **0/52** | **0/13** |

So the cause is in the weights. One trap: our first replay left the thinking out of the history and
showed only 15/52. Real prompts include it, and recorded token counts prove that. So reasoning
that stays in the history feeds the loops, and any replay has to include it
([lessons](#lessons-from-the-training-runs)).

**LFM3-VL on pibench.** All at 6 runs, against plain LFM3-VL's 18. The plain runs had 64K of context, and the other
three 32K, which we only noticed later. No run in those three passed 28K, so that shouldn't have changed them, but it's
a second difference besides the time limit:

| Setup | Solved | Tests passed | Calls that copy an earlier one | Worst streak | Timeouts |
|---|---:|---:|---:|---:|---:|
| Plain, temperature 0.1, 20 min | 0/18 | 76/162 | 93% | 769 | 14/18 |
| [Steering vector](#steering-vs-training-2026-11-01), 15 min | 0/6 | 7/28 | 80% | 129 | 4/6 |
| **Temperature 0.7, 15 min** | 0/6 | 30/54 | **24%** | **3** | 3/6 |
| Temperature 0.7 + agent LoRA, 15 min | 0/6 | 27/54 | 48% | 34 | 3/6 |

- Temperature is the doom loop fix, and it costs nothing: `TEMP=0.7 scripts/llm lfm3vl`. Scrape-books also passed
  tests for the first time (2/5, always 0/5 before). Even so, the shorter limit and smaller context mean the streak
  numbers aren't a fair match.
- It's still not an agent. 0/6 only says the true rate is likely under 46%, but it's
  clearly below LFM3-1.7B's 12/18 (Fisher p = 0.01). It now fails on quality, by stopping early or
  checking badly, instead of looping.
- The agent LoRA didn't pay off. We trained LFM3-VL for 24 minutes on 177 turns from LFM3-1.7B's solved
  runs, plus examples of getting out of a loop. On held-out stuck points, repeats fell from 21/24 to
  7/24, and vision held (101/150 against 110 for the untouched model served on the same 150 scenes, p = 0.31;
  our re-exported copy of it scored 109 there). On pibench it was no better than temperature
  alone. Loss fell from 0.84 to 0.25, so it likely memorised 177 examples rather than learning a habit.

**Giving LFM3-1.7B eyes.**

| Test | Score | Meaning |
|---|---:|---|
| LFM3-VL's projector on LFM3-1.7B, as it is | 14/150 | Blind: answers restate the prompt |
| A new projector, 15 min on 1500 scenes | 35/150 | Still blind: one identical answer to all 30 tanks, signals and gauges |
| Overfit that projector on 40 scenes, 30 passes | loss 0.15 | Still falling slowly, not clearly learning |

The 35/150 isn't progress: 20/30 on signals is what always answering `stop` gets
([lessons](#lessons-from-the-training-runs)). The
projector alone didn't teach 1.7B to read pictures. Whether more data would is still open, but
[LLaVA](https://arxiv.org/abs/2304.08485) reportedly needed hundreds of thousands of examples, so
it's not a cheap test. Loading 1.7B into LFM3-VL's code was the easy part: mapping the GGUF back
reproduced all 286 of LFM3-VL's text tensors.

**What does work.** One-picture decisions with thinking on, against pibench:

| Setup | Glance | pibench | VRAM | Time per decision |
|---|---:|---:|---:|---:|
| **LFM3-1.7B drives, LFM3-VL reports in words** | **57/60** | 12/18 for the driver | **5.0 GB** both | 16 s |
| LFM3-VL alone | 60/60 | 0/18 | 2.6 GB | |

Gemma 4 26B and Groot-19B do both by themselves: 18/18 on pibench, and 30/30 and 27/30 on a smaller
glance sample ([visual control](visual-control.md#the-results)), at 6.2 GB (projector on the CPU, 32K
context) and 7.2 GB. They take about the same 16 to 17 s a decision as the pair, so the pair's edge is
memory, not speed. The pair's driver never sees a picture, only LFM3-VL's report, and all 3 of its
misses were tanks. The pibench score for the pair is LFM3-1.7B's alone: nobody has run a task that needs both yet.

Every test here can be repeated: see [Reproducing our results](reproduce.md#can-lfm3-vl-do-more-than-look).

## Lessons from the training runs

What these runs taught us about checking a result before believing it:

- **Check the adapter exists before trusting its score.** A DoRA run for
  [pictures](image-generation.md#teaching-sdxl-a-style-or-a-person) hit its time limit, and the eval that
  followed quietly ran the base model, so it "scored" like the untouched model. `train_style.py eval`
  refuses a run with no weights; after any timeout, check the adapter is there first.
- **Check that answers change with the picture.** A small gain can still be blind: the new projector's
  35/150, up from 14, came from one identical answer to all 30 tanks, signals and gauges
  ([above](#can-lfm3-vl-do-more-than-look-2026-11-01)).
- **A replay has to match the real prompt.** Leaving the thinking out of replayed conversations
  understated the loops (15/52 against 32/52). Compare the replay's prompt token count with the real
  run's. Tool-call arguments cut off by `max_tokens` also come back as invalid JSON, which looks like
  the model's fault.
- **One seed isn't enough.** LFM2.5's first single-seed run said norm-only was more robust than LoRA;
  three seeds said the opposite, and IA3's three seeds ranged from 109 to 138.
- **Test on drawings the model never saw.** Same drawing code with new seeds overstated what LFM2.5
  learned; [`scenes_new.py`](../bench/finetune/scenes_new.py) caught it.
