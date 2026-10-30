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
  would have been the first choice, but transformers has no LFM3 support yet and its repo's own
  Python would have to run; that waits for a human decision.
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

| What we trained | Trainable parameters | Training time | Peak VRAM | 🛢️ | 🖱️ | 🤏 | 🚦 | ⏲️ | **All** | Degraded pictures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nothing (the base model) | 0 | — | — | 8 | 6 | 0 | 0 | 0 | **14/150** | 19/150 |
| **LoRA** (rank 16, attention + MLP + conv projections) | 6.0M (1.3%) | 7.5 min | 1.36 GB | 30 | 24 | 22 | 30 | 30 | **136/150** | 118/150 |
| **Norm-only** (the RMSNorm gains) | **34,560 (0.008%)** | 5.6 min | 1.27 GB | 28 | 25 | 22 | 30 | 30 | **135/150** | **121/150** |
| **IA3** (a learned scale per channel) | 79,872 (0.018%) | 5.8 min | 1.27 GB | 24 | 24 | 22 | 30 | 17 | **117/150** | 102/150 |
| *for comparison:* [LFM3-VL-2B](models/lfm3-vl-2b.md), no training, thinking on / off | — | — | — | 25 / 24 | 30 / 30 | 29 / 26 | 30 / 27 | 30 / 10 | *144 / 117* | — |

Measured 2026-10-29, one training run each, greedy decoding in transformers, under 0.25 s a
decision. "Degraded pictures" are the same test scenes shrunk to 70%, darker or brighter, lower
contrast and JPEG-compressed at quality 35, a worse camera. (LFM3-VL's row was measured through
llama.cpp with a forced call, so the harnesses differ.)

## What we learned

- **The base model never calls a tool with a picture in front of it.** Its card says function
  calling works "for text-only input", and here it held: 0 calls in 150 scenes. It answered in
  prose ("I'm sorry, I don't have the capability to control a robot arm"), wrote the tool's
  name as plain text (`stop`, `close_valve`), and misread the tanks ("A: 50% (below 30%)"). Its
  14 points are all scenes where *no* call was the right answer.
- **A few minutes of training fixes that, and more.** 136/150 after LoRA: above LFM3-VL-2B
  without thinking (117), close to it with thinking (144), at a quarter of its size and without
  a 3-second think. Every answer is now a well-formed call, multiple tank refills included. All
  14 remaining misses are clicks and picks a little off target.
- **Training only the norm gains is as good as LoRA, and holds up better.** 34,560 numbers, 173×
  fewer than LoRA, give 135/150, and on degraded pictures they keep more (121 against 118). LoRA
  learned the clean tank pictures too well: on the degraded ones it refills tanks that aren't
  low, often all three, and its tank score falls from 30 to 11 (the norm-only model's from 28 to
  21). So for teaching a model *how to answer* in a
  narrow domain, the "tune only the normalization" idea works on a transformer as well as it
  does on a CNN's BatchNorm.
- **IA3 lagged** (117), mostly on the gauge (17/30): it called `close_valve` on 13 dials where
  opening the valve or a report was right. One run at one learning rate, so it may just be under-tuned.
- **The caveat:** training and test pictures come from the same drawing code, with different
  seeds. This shows the model can learn *our* scenes, not that it learned gauges in general. The
  degraded-picture test is a first step; real photos would be the real test.

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
