# 📇 Model pages

One page per model: what it is, how it fits in 8 GB, how to run it, the vendor's numbers and ours, and its quirks.
The [README](../../README.md) has the lineup; [`scripts/llm list`](../../scripts/README.md) shows what is downloaded.
The two Gemma 4 12B pages are retired ([graveyard](../graveyard.md)) and keep their own short form.

## The template for a new page

```markdown
# <emoji> <Name> — <one line: what makes it different>

> **TL;DR**
>
> | | |
> |---|---|
> | On disk | <size> (`<quant>`) [+ projector] |
> | Where it lives | <all on the GPU / experts of N of M layers in RAM / ...> (<VRAM> VRAM) |
> | Context | <tokens> |
> | Generation | <tok/s> |
> | Prompt processing | <tok/s> (<prompt size>) |
> | [pibench](../pibench.md) | <n>/18 ... (chat/agent models; helpers put their own task's result here) |
> | [Vision probe](../vision-probe.md) | <score> or "text only" |
> | [Security audit](../security.md) | clean / <verdict> |
> | Licence | <licence> (only if the card states it) |
> | Launch | `scripts/llm <name>` |

## What is it?
## How does it fit in 8 GB?        (big or MoE models; small helpers can skip it)
## Install & run                   (download -> audit -> scripts/llm <name>; the knobs that matter)
## Official numbers
## Our numbers                     (speed and VRAM, pibench, vision, anything model-specific)
## How it behaves                  (what we saw in transcripts)
## Security audit                  (or just the TL;DR row when the audit was clean and short)
## Gotchas
## Knobs                           (Variable | Default | What it does)
```

Conventions: speeds in **tok/s**; "not run" for tests we haven't done; dates as YYYY-MM-DD; every number
traceable to a log or script ([reproduce.md](../reproduce.md)). The launcher names are the ones in
`scripts/llm list` (the same as run-qwen.sh's `QMODEL`, plus `gemma` and `thinkingcap`).
