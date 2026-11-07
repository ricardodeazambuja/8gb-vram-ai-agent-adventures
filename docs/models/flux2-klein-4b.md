# 🖌️ FLUX.2 Klein 4B — the everyday image model

> **TL;DR**
>
> | | |
> |---|---|
> | Time per image | **23 s** at 1024×1024, 4 steps |
> | Peak VRAM | **3.5 GB** |
> | Inputs | prompt, picture, mask and reference edit |
> | [Security audit](../security.md) | clean (model and text encoder) |
> | Licence | Apache 2.0 |
> | Launch | `scripts/imagine "..." -m klein` |

## What is it?

A 4B text-to-image model in a `Q4_0` GGUF, with Qwen3-4B as its text encoder (run on the CPU) and a VAE. It
draws 1024×1024 in 4 steps, and it's the only one of our three whose reference edits (`-r`) work.

The one to pick: the fastest of our three, half the VRAM, good hands, and reference edits that work
with no training. Full comparison: [Making pictures on 8 GB](../image-generation.md).

## Install & run

It runs on [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), not llama.cpp. Build
it as in [Run it yourself](../image-generation.md#run-it-yourself), then:

```bash
bash scripts/download-models.sh klein kleinte kleinvae        # ~5.3 GB
bash utils/audit-gguf.sh models/image-gen/flux2-klein/flux-2-klein-4b-Q4_0.gguf   # then audit it (and the text encoder)
scripts/imagine "a red fox in the snow" -m klein               # one picture
```

Pictures land in `models/image-gen/out/`. `scripts/imagine --dry-run` shows the full `sd-cli` command. For a
reference edit (Klein only), `scripts/imagine "..." -m klein -r in.png`; to start from a picture, `-i in.png`; and
for one area, `-i in.png --mask mask.png`. (The building block: `python3 bench/imagegen/run_gen.py klein --quick`.)

## Gotchas

- The text encoder runs on the CPU (`--backend te=cpu`) and VAE tiling is on.
- The first run of a model is 8 to 15 s slower while files load.
- A reference edit keeps the scene, but the face can shift a little.
- LoRA training on Klein is untested here.
