# 🖼️ Making pictures on 8 GB

Text models aren't the only thing this laptop can run. A current image model draws a 1024×1024
picture in 23 seconds in 3.5 GB of VRAM, and it can spell. Stable Diffusion was what we'd have
reached for a few years ago; it's now the slowest of the three models we tried.

![The same four prompts on all three models](image-generation-samples.jpg)

*Left to right: FLUX.2 Klein 4B, Iris-3B Turbo, SDXL. Seed 1, shrunk to fit; nothing retouched.*

## The results

Four prompts × two seeds at 1024×1024, measured 2026-10-31:

| Model | Time per image | Peak VRAM | Steps | Hands | Sign text | Watercolour asked for |
|---|---:|---:|---:|---|---|---|
| **FLUX.2 Klein 4B** Q4_0 | **23 s** | **3.5 GB** | 4 | good | right on both, one doubled | yes |
| **Iris-3B Turbo** (6 GB file) | 33 s | 7.3 GB | 4 | good | right on both, cleanest | no, looks like digital cartoon |
| SDXL base | 47 s | 5.8 GB | 25 | passable, messy laces | garbled on one, spacing off on the other | yes |

Times are medians; the first run of each model was 8 to 15 s slower while files loaded. The
quality columns are our own look at the pictures, not a score.

**Pick Klein.** It's the fastest, uses half the VRAM and leaves room for a LoRA or another model.
**Iris-3B Turbo is the one for text**, but at 7.3 of 7.8 GB it leaves nothing spare. Its Q4 GGUF
should use less, which we haven't tried.

The text encoder runs on the CPU (`--backend te=cpu`), since it's needed once per prompt, and VAE
tiling is on. Iris-3B paints pixels directly, so it has no VAE at all.

## Run it yourself

The runner is [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), the llama.cpp
of image models. We used commit `0a067120`, cloned next to this repo:

```bash
git clone --recursive https://github.com/leejet/stable-diffusion.cpp ../stable-diffusion.cpp
git -C ../stable-diffusion.cpp checkout 0a067120b388bfefb9a87b96960cb548b97e9157
git -C ../stable-diffusion.cpp submodule update --init --recursive
mamba run -n llamacpp-cuda bash -c 'cd ../stable-diffusion.cpp && \
  cmake -B build -DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release && \
  cmake --build build -j6'

bash scripts/download-models.sh klein kleinte kleinvae        # ~5.3 GB
python3 bench/imagegen/run_gen.py klein --quick               # one picture
python3 bench/imagegen/run_gen.py klein iris sdxl             # the table above
```

Pictures land in `models/image-gen/out/`, timings in [`bench/imagegen/results/`](../bench/imagegen/results/).
All the downloads are pinned and sha256-checked, and their audits are in
[Security](security.md#audits-so-far).

## What's next

Fine-tuning: teaching these models a style, and then your own photos, so you can ask for your dog
as a watercolour. That's planned but not started; see [Up next](up-next.md#️-image-generation).
