# 🎨 SDXL — old, slower, and the one we can train

> **TL;DR**
>
> | | |
> |---|---|
> | Time per image | 47 s at 1024×1024, 25 steps |
> | Peak VRAM | 5.8 GB |
> | Inputs | prompt, picture, mask |
> | [Security](../security.md#audits-so-far) | plain `.safetensors`, pinned only (no chat template to audit) |
> | Licence | Open RAIL++ |
> | Launch | `scripts/imagine "..." -m sdxl` |

## What is it?

Stable Diffusion XL base, the older latent model: two text encoders and a VAE, 25 steps for a 1024×1024 picture.
What it has over the newer two is training: styles and a person, on this laptop.

The slowest of our three, with garbled sign text. It's still here because it's the one we taught styles
and a person: [Teaching SDXL](../image-generation.md#teaching-sdxl-a-style-or-a-person). Full comparison:
[Making pictures on 8 GB](../image-generation.md).

## Install & run

It runs on [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), not llama.cpp. Build
it as in [Run it yourself](../image-generation.md#run-it-yourself), then:

```bash
bash scripts/download-models.sh sdxl sdxlvae
scripts/imagine "a lighthouse at dusk" -m sdxl
```

(The building block: `python3 bench/imagegen/run_gen.py sdxl --quick`.)

Training uses the `vlft` env and `bench/imagegen/train_style.py`, as in the section linked above.

## Gotchas

- `-r` runs and reports no error, but the picture is identical to one without it.
- A LoRA for `sd-cli` loads only when all its tensors match (ours: 1120/1120), so check the log.
