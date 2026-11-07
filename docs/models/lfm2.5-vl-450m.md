# 🐭 LFM2.5-VL-450M — our fine-tuning subject

> **TL;DR**
>
> | | |
> |---|---|
> | On disk | 0.71 GB (`F16`) + 0.19 GB vision projector |
> | Where it lives | entirely on the GPU (1.5 GB VRAM at 32K context) |
> | Per decision | 0.1 s once tuned |
> | [Security audit](../security.md) | clean |
> | Launch | `scripts/llm lfm25vl` |

## What is it?

[LiquidAI's tiny vision model](https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF). We use it to find out
what fine-tuning can do on this card: it's small enough to train here and to test quickly. Its results
are on the [fine-tuning page](../fine-tuning.md).

## Install & run

```bash
bash scripts/download-models.sh lfm25vl
bash utils/audit-gguf.sh models/lfm2.5-vl-450m/LFM2.5-VL-450M-F16.gguf   # then audit it (and the mmproj)
scripts/llm lfm25vl          # the server, with its projector; `scripts/llm lfm25vl pi` to chat
```

(The building block: `QMODEL=lfm25vl bash scripts/run-qwen.sh --image`.)

It uses our [llama.cpp fork](../reproduce.md#setup), which has the one-line fix that makes tools work.
To run a tuned model, point `MODEL` at the baked file:

```bash
QMODEL=lfm25vl MODEL=models/lfm2.5-vl-450m-baked/norm.gguf bash scripts/run-qwen.sh --image
```

Training needs the Hugging Face weights and a Python env, as in
[Run it yourself](../fine-tuning.md#run-it-yourself).

## Our numbers

All on the [fine-tuning page](../fine-tuning.md#lfm25-vl-450m-the-same-recipe-at-a-quarter-the-size). In short: untuned, 14/150 one-picture decisions
(thinking off); tuned and served by `llama-server`, 133 (norm-only) and 139 (IA3), at 0.1 s a decision in 1.5 GB
([Serving it through llama.cpp](../fine-tuning.md#serving-it-through-llamacpp)). As a grid robot, tuned, it walks
8/8 open grids and up to 5/8 mazes ([the grid robot](../fine-tuning.md#the-grid-robot-from-0-to-5-of-8-mazes)).

## Gotchas

- Tools failed with "failed to parse grammar" on stock llama.cpp. The fix is on our fork, `3cb2992`.
