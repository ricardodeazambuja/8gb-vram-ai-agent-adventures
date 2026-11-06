# 📍 LocateAnything-3B — finds things, and only that

> **TL;DR**
>
> | | |
> |---|---|
> | On disk | 3.62 GB (`Q8_0`) + 0.87 GB vision projector |
> | Where it lives | entirely on the GPU (about 5 GB VRAM at 8K context) |
> | Shapes | 15/15 found, mean IoU 0.94, 0.2 s each |
> | [Visual control](../visual-control.md#locateanything-3b-boxes-instead-of-tool-calls) | pick 30/30, click 24/24, real web pages 42/48 |
> | [Security audit](../security.md) | clean |
> | Licence | academic and non-profit research only |
> | Launch | `scripts/llm locate` · [`examples/locate.py`](../../examples/locate.py) |

## What is it?

[`nvidia/LocateAnything-3B`](https://huggingface.co/nvidia/LocateAnything-3B) (May 2026): a
**grounding specialist**, not a chat model. MoonViT-SO-400M vision encoder + Qwen2.5-3B (3.8B
parameters in all), trained on 12M images and 785M boxes. You give it an image and a phrase,
and it returns boxes or points in 0–1000 coordinates: object detection, "the person in the red
shirt", GUI elements ("Point to: the search button"), text and document layout. Its trick is
**Parallel Box Decoding**: a whole box in one step instead of token by token, up to 2.5× faster.
NVIDIA says it went into Nemotron 3 Nano Omni's grounding and computer-use features.

Why here: it's the **"where"** to LFM3-VL's **"what to do"**. LFM3-VL-2B already has the best
boxes in the [vision probe](../vision-probe.md) (IoU 0.93–0.94) and 60/60 on
[visual control](../visual-control.md); a model trained only to locate things might beat it on
crowded scenes and screenshots, which is what a computer-use agent needs.

## Install & run

It needs our [llama.cpp fork](../reproduce.md#setup), which carries the projector support.

```bash
bash scripts/download-models.sh locate
scripts/llm locate                                 # server on port 8080, projector and --special on
```

[`examples/locate.py`](../../examples/locate.py) asks it for boxes. (The building block:
`QMODEL=locate bash scripts/run-qwen.sh --image`.)

Send an image and one of two prompts to `/v1/chat/completions`:

| Prompt | Use |
|---|---|
| `Locate a single instance that matches the following description: square.` | Always returns a box, even when the thing isn't there |
| `Locate all the instances that matches the following description: circle</c>square.` | Can answer `<box>None</box>`, so ask this first |

Answers look like `<ref>square</ref><box><23><594><195><823></box>`: `x1 y1 x2 y2` on a 0–1000 scale, so
scale them yourself. `bash bench/smoke/run.sh locate` is the quick check.

## Our numbers

### History

**Status (2026-10-29):** Q8_0 + projector pinned (`download-models.sh locate`), downloaded and
audited: clean, chat template identical to NVIDIA's. **Blocked on the runtime:** the projector
declares `clip.projector_type = locateanything`, which upstream llama.cpp (our Oct 27 build, and
master today) doesn't know. Support is in an open upstream PR,
[#24749](https://github.com/ggml-org/llama.cpp/pull/24749) (June 2026, +244 lines, ~130 of them
C++ in `tools/mtmd`), and the GGUF port was made for its author's fork
(`yuuko-eth/llama.cpp@mtmd-grounders`).

**Status (2026-10-30):** PR reviewed and applied in `vendor/locate-llama.cpp`, a worktree of
our Oct 28 build (`8ca4060`) with the PR's two commits on top. Every change runs only for the
`locateanything` projector, plus one resize option that is off by default. Upstream had moved:
the PR's `RESIZE_ALGO_BICUBIC_PILLOW` is now plain `RESIZE_ALGO_BICUBIC` (Pillow-compatible), and
the round-up fix was ported onto the new `calc_size_opt` struct. The GGUF matches what the patch
loads without compiling anything: `mm.input_norm`, `mm.1`, `mm.2`, a `[1152, 64, 64]` position
table, `image_min/max_pixels` set, text model plain `qwen2`. It has separate `attn_q/k/v`
where the PR's converter writes a fused `attn_qkv`; the loader takes either. Still unknown:
whether that port reordered Q/K for split RoPE the way the PR's converter does. If it didn't,
it will load and then point at the wrong places, so the image test decides.

**First test (2026-10-30): it works.** Built (`llama-server`, `llama-mtmd-cli`; CUDA, sm75) and
run with `--special` (without it llama-server drops the `<ref>`/`<box>`/`<NNN>` tokens), greedy,
5.1 GB of VRAM at `-c 8192`. On the vision probe's three shape images (1024x768, 15 shapes,
pixel ground truth), one "Locate a single instance … {color} {shape}." query per shape: 15/15 at
IoU >= 0.5, mean IoU 0.94, 0.2 s each (2.4 s for the first query on a new image, the encode).
The detect-all prompt (`circle</c>square</c>triangle</c>star`) found 15/15, 0.6–0.8 s per image,
and answers `<box>None</box>` for a class that isn't there. Output is 0–1000 normalised
`x1 y1 x2 y2`. So the GGUF port's Q/K order is right. The port now lives in our llama.cpp fork (see
Chores), and `scripts/llm locate` starts it (with `--special`).

**Click, pick and real pages (2026-10-30):** results in the
[visual control probe](../visual-control.md#locateanything-3b-boxes-instead-of-tool-calls). Finds what
is there (pick 30/30, click 24/24, real web pages 42/48 at ~0.3 s), but the single-instance prompt
always returns a box, so it can't say "not there"; the detect-all prompt mostly can.


Rerun on our fork (2026-11-01): 15/15, mean IoU 0.94, 0.22 s, 4.8 GB.

## Gotchas

- Without `--special`, the server drops the boxes. `run-qwen.sh` sets it.
- The single-instance prompt can't say "not there", so an agent could click the wrong thing.
- Small, dense targets on real pages are where it misses.
- Don't load `training_args.bin` from the original repo: it's a pickle.
