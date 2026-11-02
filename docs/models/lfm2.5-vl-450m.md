# 🐭 LFM2.5-VL-450M — our fine-tuning subject

> **TL;DR**
>
> | | |
> |---|---|
> | On disk | 0.71 GB (`F16`) + 0.19 GB vision projector |
> | Where it lives | entirely on the GPU (1.5 GB VRAM at 32K context) |
> | Per decision | 0.1 s once tuned |
> | [Security audit](../security.md) | clean |
> | Launch | `QMODEL=lfm25vl bash scripts/run-qwen.sh --image` |

## What is it?

[LiquidAI's tiny vision model](https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF). We use it to find out
what fine-tuning can do on this card: it's small enough to train here and to test quickly. Its results
are on the [fine-tuning page](../fine-tuning.md).

## Install & run

```bash
bash scripts/download-models.sh lfm25vl
QMODEL=lfm25vl bash scripts/run-qwen.sh --image
```

It uses our [llama.cpp fork](../reproduce.md#setup), which has the one-line fix that makes tools work.
To run a tuned model, point `MODEL` at the baked file:

```bash
QMODEL=lfm25vl MODEL=models/lfm2.5-vl-450m-baked/norm.gguf bash scripts/run-qwen.sh --image
```

Training needs the Hugging Face weights and a Python env, as in
[Run it yourself](../fine-tuning.md#run-it-yourself).

## Gotchas

- Tools failed with "failed to parse grammar" on stock llama.cpp. The fix is on our fork, `3cb2992`.
