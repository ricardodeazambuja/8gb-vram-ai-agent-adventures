# 🔤 Iris-3B Turbo — the one for text in pictures

> **TL;DR**
>
> | | |
> |---|---|
> | Time per image | 33 s at 1024×1024, 4 steps |
> | Peak VRAM | 7.3 GB of 7.8 |
> | Inputs | prompt, picture; masks paste a hard square |
> | Licence | Apache 2.0 |
> | Launch | `scripts/imagine "..." -m iris` |

It has the cleanest sign text of our three, but leaves nothing spare on the card, and it ignored a request
for watercolour. It paints pixels directly, so there's no VAE. Full comparison:
[Making pictures on 8 GB](../image-generation.md).

## Install & run

It runs on [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), not llama.cpp. Build
it as in [Run it yourself](../image-generation.md#run-it-yourself), then:

```bash
bash scripts/download-models.sh iris iriste
scripts/imagine "a shop sign that says OPEN" -m iris
```

(The building block: `python3 bench/imagegen/run_gen.py iris --quick`.)

## Gotchas

- `-r` runs and reports no error, but the picture is identical to one without it.
- Close everything else that uses the GPU first.
- A Q4 GGUF should use less memory, which we haven't tried.
