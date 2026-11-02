# 🖌️ FLUX.2 Klein 4B — the everyday image model

> **TL;DR**
>
> | | |
> |---|---|
> | Time per image | **23 s** at 1024×1024, 4 steps |
> | Peak VRAM | **3.5 GB** |
> | Inputs | prompt, picture, mask and reference edit |
> | Licence | Apache 2.0 |
> | Launch | `python3 bench/imagegen/run_gen.py klein --quick` |

The one to pick: the fastest of our three, half the VRAM, good hands, and reference edits that work
with no training. Full comparison: [Making pictures on 8 GB](../image-generation.md).

## Install & run

It runs on [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), not llama.cpp. Build
it as in [Run it yourself](../image-generation.md#run-it-yourself), then:

```bash
bash scripts/download-models.sh klein kleinte kleinvae        # ~5.3 GB
python3 bench/imagegen/run_gen.py klein --quick               # one picture
```

Pictures land in `models/image-gen/out/`. `run_gen.py` shows the full `sd-cli` flags. For a reference
edit, add `-r in.png`, and for one area, `-i in.png --mask mask.png`.

## Gotchas

- The text encoder runs on the CPU (`--backend te=cpu`) and VAE tiling is on.
- The first run of a model is 8 to 15 s slower while files load.
- A reference edit keeps the scene, but the face can shift a little.
- LoRA training on Klein is untested here.
