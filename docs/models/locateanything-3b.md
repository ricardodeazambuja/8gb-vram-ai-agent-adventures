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
bash utils/audit-gguf.sh models/locateanything-3b/LocateAnything-3B-Q8_0.gguf   # then audit it (and the mmproj)
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

Our default build carries the projector support (upstream PR [#24749](https://github.com/ggml-org/llama.cpp/pull/24749), applied to our fork). On the vision probe's
three shape images, rerun on the fork (2026-11-01, [log](../../bench/smoke/results/2026-11-01-fork-3cb2992/locate.txt)): 15/15 at IoU >= 0.5, mean IoU 0.94, 0.22 s per query, 4.8 GB
of VRAM. Clicks, picks and real pages: [visual control](../visual-control.md#locateanything-3b-boxes-instead-of-tool-calls).

### History

The projector declares `clip.projector_type = locateanything`, which upstream llama.cpp didn't know (2026-10-29).
Support is in the open upstream PR [#24749](https://github.com/ggml-org/llama.cpp/pull/24749); we reviewed it and
applied it to our fork (2026-10-30). The GGUF port came from another fork, so whether its Q/K order was right for
the PR could only be settled by a test: the first one (2026-10-30) found 15/15 shapes at mean IoU 0.94, so it was.
Clicks, picks and real pages followed the same day ([visual control](../visual-control.md#locateanything-3b-boxes-instead-of-tool-calls)).

## Gotchas

- Without `--special`, the server drops the boxes. `run-qwen.sh` sets it.
- The single-instance prompt can't say "not there", so an agent could click the wrong thing.
- Small, dense targets on real pages are where it misses.
- Don't load `training_args.bin` from the original repo: it's a pickle.
