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

## Giving it a picture as well

Prompts aren't the only input. One test of each on our own 1024×1024 fisherman, 2026-10-31:

| Input | Flag | FLUX.2 Klein 4B | Iris-3B Turbo | SDXL |
|---|---|---|---|---|
| Picture + prompt, redraw it | `-i in.png --strength 0.7` | works | works | works |
| Picture + mask + prompt, redraw one area | `-i in.png --mask mask.png` | **works, blends best** | pasted a hard square | works, slight seam |
| Reference + prompt, edit by instruction | `-r in.png` | **works** | ignored | ignored |

**Klein's reference edit is the one to use:** "wearing a red cap instead of his hat" changed the hat
and kept the man, the net and the sunset, though the face shifted a little. Iris-3B and SDXL run
with `-r` and report no error, but they only followed the prompt, so an exit code of 0 proves
nothing there.

White in the mask is the area to redraw. Everything outside should stay put, and the average change
per colour value (out of 255) well outside the mask was 0.0 on Iris-3B, 3.0 on Klein and 7.8 on SDXL;
the two latent models go through a VAE, which we think explains that. At strength 0.7 the
"watercolour" request came out as a painted-looking photo, not a real restyle; lower strengths
keep more, and we only tried this one. ControlNet, IP-Adapter and PhotoMaker also have flags and
are untested.

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
