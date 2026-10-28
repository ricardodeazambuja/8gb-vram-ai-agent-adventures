# 👀 LFM3-VL-2B — a 2B vision model that draws the tightest boxes here

> **TL;DR**
>
> | | |
> |---|---|
> | On disk | 1.85 GB (`Q8_0`) + vision projector |
> | Where it lives | entirely on the GPU, projector included (2.6 GB VRAM at 32K context) |
> | Context | 32K here |
> | Generation | **106 tok/s** |
> | Prompt processing | **4,875 tok/s** (15K-token prompt) |
> | [Vision probe](../vision-probe.md) | **91%** overall, **9 s** per image; boxes IoU 0.93–0.94 |
> | [pibench](../pibench.md) | **0/18** (47% of hidden tests): loops on tool calls, unlike LFM3-1.7B (12/18) |
> | [Security audit](../security.md) | clean (model and projector) |
> | Launch | `QMODEL=lfm3vl bash scripts/start-qwen.sh --image` (needs the newer llama.cpp build) |

## What is it?

The vision member of **Liquid AI**'s LFM3 family (Oct 2026, LFM Open License v1.0: commercial use
only under $10M revenue). [LFM3-1.7B](lfm3-1.7b.md)'s tri-hybrid text model plus an image encoder;
it thinks before answering and is trained for object grounding (boxes). Video input exists in the
model but isn't in llama.cpp yet.

## Install & run

After [getting-started.md](../getting-started.md) and the llama.cpp build from
[LFM3-1.7B's page](lfm3-1.7b.md#install--run):

```bash
bash scripts/download-models.sh lfm3vl
bash utils/audit-gguf.sh models/lfm3-vl-2b/LFM3-VL-2B-Q8_0.gguf
bash utils/audit-gguf.sh models/lfm3-vl-2b/mmproj-LFM3-VL-2B-F16.gguf

QMODEL=lfm3vl bash scripts/start-qwen.sh --image
```

## Our numbers

RTX 2070 Max-Q (8 GB), q8_0 KV, 32K context, projector on the GPU, the card's sampling (temp 0.1,
top-k 50, repeat-penalty 1.1). `scripts/qwen-bench.py`: **106.2 tok/s**, a 15K-token prompt at
**4,875 tok/s**, tool call ✓, 2.6 GB of VRAM.

**[Vision probe](../vision-probe.md): 91%**, at a median of 9 s per image (only
[Qwen3-VL-8B](qwen3-vl-8b.md) is quicker, at 6 s, and it scores 89%).

- 🧾 **OCR 100%**: every field of all three invoices, the blurred, rotated "scan" included, in
  under 4 s each.
- 🟥 **Boxes: the most accurate of any model so far.** Asked for pixels, it answers in a 0–1000
  scale like the Qwens (so 7% "as asked"), but read that way all three images are perfect, with a
  mean IoU of **0.93–0.94**. Qwen3.6, Gemma and Qwen3-VL also get 100% in their own formats, but
  none matches these overlaps.
- 📊 **Understanding 69%**: the chart right; it counted the blue dots but not the red ones (so the
  total was wrong too), and got one of four spatial relations wrong.
- 📐 **CAD 93%**: two of 27 dimensions wrong (a hole's position, a hole's diameter).

For 2.6 GB, it's a good little eye: a pi session on a big model could hand it screenshots.

**[pibench](../pibench.md): 0/18**, 47% of hidden tests, at 64K context. Don't let it drive.
It is [LFM3-1.7B](lfm3-1.7b.md) (12/18) plus a vision encoder, and its chat template is
byte-for-byte the same, yet 14 of 18 attempts ran into the 20-minute timeout, each stuck
running one command over and over: the books scraper re-ran the same `curl | grep` 419 times,
and one feature attempt repeated a command 769 times out of 772 calls. LFM3-1.7B never repeated a
command more than 25 times. The four attempts that finished did it badly too: the bug fix
patched one comparison, ran a test file that defines tests but never calls them, saw no output
and declared "the tests pass" (17/27 hidden tests). The vision training seems to have cost it
the agent skills.

## Security audit

`utils/audit-gguf.py --ref` the source repo's chat template, before it was ever loaded
([security.md](../security.md)): sha256s match the pinned revision, headers clean, chat template
identical to the source repo's, no hidden characters, no Jinja internals; the projector is clean.

## Gotchas

- Needs a llama.cpp build from after 2026-10-27.
- Boxes come back in 0–1000 coordinates whatever you ask for; scale them yourself.
- Not an agent: with tools it falls into loops. Use [LFM3-1.7B](lfm3-1.7b.md) for that.
