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
> | [Vision probe](../vision-probe.md) | **91%** overall, **8 s** per image; boxes IoU 0.93–0.94 |
> | [pibench](../pibench.md) | **0/18** (47% of hidden tests): loops on tool calls, unlike LFM3-1.7B (12/18) |
> | [Visual control](../visual-control.md) | **60/60** one-picture decisions with thinking (45/60 without); grid mazes 3/8 |
> | [Security audit](../security.md) | clean (model and projector) |
> | Launch | `scripts/llm lfm3vl` · [`examples/look.py`](../../examples/look.py) |

## What is it?

The vision member of **Liquid AI**'s LFM3 family (Oct 2026, LFM Open License v1.0: commercial use
only under $10M revenue). [LFM3-1.7B](lfm3-1.7b.md)'s tri-hybrid text model plus an image encoder;
it thinks before answering and is trained for object grounding (boxes). Video input exists in the
model but isn't in llama.cpp yet. The [card](https://huggingface.co/LiquidAI/LFM3-VL-2B) is clear about its scope: on-device visual tasks, and
"not recommended for coding, multi-turn conversations, or complex reasoning" (its one agentic
benchmark, MM-ToolSandbox, scores 42.5).

## Install & run

Build our llama.cpp once: [getting-started.md](../getting-started.md). Then:

```bash
bash scripts/download-models.sh lfm3vl
bash utils/audit-gguf.sh models/lfm3-vl-2b/LFM3-VL-2B-Q8_0.gguf
bash utils/audit-gguf.sh models/lfm3-vl-2b/mmproj-LFM3-VL-2B-F16.gguf

scripts/llm lfm3vl          # the server, with its projector; prints the URL
scripts/llm lfm3vl pi       # or chat with it in pi
```

[`examples/look.py`](../../examples/look.py) shows it a picture and lets it pick one tool call. (The building block:
`QMODEL=lfm3vl bash scripts/run-qwen.sh --image`; its knobs still apply, e.g. `CTX=16384 scripts/llm lfm3vl`.)

## Our numbers

RTX 2070 Max-Q (8 GB), q8_0 KV, 32K context, projector on the GPU, the card's sampling (temp 0.1,
top-k 50, repeat-penalty 1.1). `scripts/qwen-bench.py`: **106.2 tok/s**, a 15K-token prompt at
**4,875 tok/s**, tool call ✓, 2.6 GB of VRAM.

**[Vision probe](../vision-probe.md): 91%**, at a median of 8 s per image (only
[Qwen3-VL-8B](qwen3-vl-8b.md) is quicker, at 5 s, and it scores 89%).

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
It has the same architecture as [LFM3-1.7B](lfm3-1.7b.md) (12/18), plus a vision encoder, but its
weights differ, and that's the cause (see [the replay](../fine-tuning.md#can-lfm3-vl-do-more-than-look-2026-11-01)). The chat template is
byte-for-byte the same, yet 14 of 18 attempts ran into the 20-minute timeout, each stuck
running one command over and over: the books scraper re-ran the same `curl | grep` 419 times,
and one feature attempt repeated a command 769 times out of 772 calls. LFM3-1.7B never repeated a
command more than 25 times. The four attempts that finished did it badly too: the bug fix
patched one comparison, ran a test file that defines tests but never calls them, saw no output
and declared "the tests pass" (17/27 hidden tests). That's what the card warns about (see
above). We checked that the harness isn't to blame: the server ran it text-only, the same
llama.cpp build and chat template as LFM3-1.7B, identical GGUF settings apart from the stated
context length, nothing truncated, and the loops start long before pi first compacts the
conversation (feature run 2: call 3, at ~3K tokens).

**[Visual control](../visual-control.md): where it does fit.** One picture, one tool call: read
a gauge and open a valve, find a button and click it, stop at a red light. With thinking it got
**60/60** (144/150 on a set of 30 scenes per kind) at under 3 s a decision; without thinking it
still sees well but stops applying the rules (45/60). Steering a robot step by step works on open
grids (8/8) and mostly not in mazes (3/8), and only when every step is a fresh request: given the
conversation history, it starts copying its own previous answers.

## As a pair of eyes

Let [LFM3-1.7B](lfm3-1.7b.md) drive and LFM3-VL report what it sees: 57/60 in 5.0 GB for both
([details](../fine-tuning.md#can-lfm3-vl-do-more-than-look-2026-11-01); `bash bench/beyond-vision/run.sh eyes`).

To run a [fine-tuned copy](../fine-tuning.md#serving-it-through-llamacpp), point `MODEL` at the baked file:

```bash
QMODEL=lfm3vl MODEL=models/lfm3-vl-2b-baked/ia3.gguf bash scripts/run-qwen.sh --image
```

## Security audit

`utils/audit-gguf.py --ref` the source repo's chat template, before it was ever loaded
([security.md](../security.md)): sha256s match the pinned revision, headers clean, chat template
identical to the source repo's, no hidden characters, no Jinja internals; the projector is clean.

## Gotchas

- Needs a llama.cpp build from after 2026-10-27.
- Boxes come back in 0–1000 coordinates whatever you ask for; scale them yourself.
- Thinking off, it's weak at one-picture decisions (117/150 in the [visual control](../visual-control.md) probe,
  109/150 served by `llama-server` in the fine-tuning check), but fine-tuning fixes that: 150/150 in `llama-server`,
  and 7/8 mazes. See [Fine-tuning](../fine-tuning.md#did-it-hold-up-2026-10-30).
- Not an agent: in long conversations with tools it falls into loops. `TEMP=0.7` stops the loops, but it still solved 0/6. Use [LFM3-1.7B](lfm3-1.7b.md)
  for that, or give it one picture and one decision at a time ([visual control](../visual-control.md)).
