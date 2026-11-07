# 🧮 TwIL-8B — a small math and code reasoner

> **TL;DR**
>
> | | |
> |---|---|
> | On disk | 5.35 GB (`Q4_K_M`, the verbose variant) |
> | Where it lives | entirely on the GPU (5.0 GiB of weights + the KV cache) |
> | Context | 24K with a q4_0 KV cache (the card wants 40K for hard problems) |
> | Speed | 38.0 tok/s; a 17K-token prompt at 644 tok/s; 6.6 GB of VRAM in use with the desktop ([recheck](../../bench/recheck/results/2026-11-07-llm/twil.txt)) |
> | [pibench](../pibench.md) | 8/18 at 24K: code chores 7 of 9, web chores 1 of 9; the context ran out in 14 of 18 attempts (32K: 3/6, twice as slow) |
> | [Security audit](../security.md) | clean |
> | Licence | webAI Non-Commercial 1.0: non-commercial research and education only |
> | Launch | `scripts/llm twil` (server + pi) |

## What is it?

[webAI's TwIL-8B](https://huggingface.co/webAI-Official/TwIL-8B): IBM's Granite 4.2 8B (dense, 40 layers,
8 KV heads, 131K context), post-trained with full-parameter SFT and GRPO/DAPO reinforcement learning for **math,
code and instruction following**. It thinks (`<think>…</think>`) before answering. Two variants share the
architecture: *balanced* (shorter reasoning) and *verbose* (longer); we run **verbose** Q4_K_M.

The card's claims, on its own harness and the bf16 weights, not the GGUF: MATH-500 98.2, AIME 2025 88.3,
LiveCodeBench 81.6 and IFBench 74.4 (both measured on verbose), against Qwen3.5-9B's 65.6 and 64.5. It is weak on
knowledge (GPQA Diamond 70.2) and has **no tool-use training**; tool use and agentic tasks were not evaluated.
It writes long: ~19.5K tokens per AIME 2026 problem on balanced, more on verbose.

Where it could fit here: the code writer in the "bigger model plans, small model writes" idea that
[VibeThinker 3B](vibethinker.md) was brought in for ([up-next](../up-next.md)).

## Install & run

Build our llama.cpp once: [getting-started.md](../getting-started.md). Then:

```bash
bash scripts/download-models.sh twil      # pinned revision 58d49f0, sha256 checked
bash utils/audit-gguf.sh models/twil-8b/TwIL-8B-Q4_K_M-verbose.gguf   # then audit it
scripts/llm twil                          # server + pi; port 8080, 24K, q4_0 KV, temp 1.0, top-p 0.95
scripts/llm twil serve                    # the server only
```

(The building block: `QMODEL=twil bash scripts/run-qwen.sh`; its knobs still apply. Until its pibench run,
`scripts/llm twil` served it without pi: per [security.md](../security.md) a new model's first shell access is in
pibench's sandbox. There it only reached the task sites.)

**Memory.** The KV cache costs 160 KiB a token in f16, 85 in q8_0 and 45 in q4_0 (40 layers x 8 KV heads x
128). Next to 5.0 GiB of weights, 24K at q4_0 (~1.1 GB) leaves room for a ~0.7 GB desktop by estimate;
32K would need the desktop under ~350 MiB, and q8_0 at 32K (2.7 GB) doesn't fit. These are estimates until the
first run measures the compute buffer. The first run (2026-11-07, night): 6645 MiB in use at load with a
432 MiB desktop, so the estimate holds. 32K also ran that night with the same desktop, 7.0 GB in use (one reading
mid-run, not a measured peak), so the ~350 MiB above was too cautious; it still needs a near-empty desktop. With a daytime desktop (~1.8 GB with Chrome open) it does not fit on the
GPU at all.

## Our numbers

First run 2026-11-07, in the [recheck](../../bench/recheck/README.md#through-scriptsllm-2026-11-07) (one sample,
`scripts/llm twil serve`, 24K, q4_0 KV):

| Test | Result |
|---|---|
| Generation | 38.0 tok/s (1,500 tokens, all of it thinking: it hadn't finished at the cap) |
| A 17K-token prompt | 644 tok/s |
| A tool call (`read`), though it has no tool-use training | ✓ |
| Code check (`merge_intervals`, 16,000-token budget, in the no-network container) | ✓ 0/500 wrong, its own doctests passed; 7,114 tokens, 3.6 min at 32.7 tok/s |

**[pibench](../pibench.md), 2026-11-07, 24K with a q4_0 KV cache: 8/18 solved, 73% of hidden tests passed, median
8.4 min per attempt**, no timeouts ([stats](../../bench/results/stats-twil.txt)). Per task: bug fix 1/3, word count 3/3,
feature 3/3, login 0/3, books 0/3, JS quotes 1/3.

- **It keeps running out of room.** 14 of 18 attempts ended because the conversation filled the 24K window
  (22.6K tokens at most); six of those had already passed every test. It thinks a lot, 26K characters a job,
  and the web pages it fetched with `curl` (50–100K characters a job on the two failing web chores) share the
  same room. It never wrote the books scraper at all.
- **Tools work without tool-use training.** 237 calls (194 of them `bash`). The 35 that failed were the
  commands' own errors (shell quoting, Python tracebacks), `edit`s whose old text didn't match, and four cut off
  when the context ran out. Twice it wrote a call inside its thinking, where pi can't see it, and that attempt
  ended there.
- **The bug fix:** twice it fixed the reported bugs but not the hidden case (22/27): once it declared itself done,
  once the context ran out first.
- In the sandbox it only reached the task sites (`quotes.toscrape.com`, `books.toscrape.com`, `goodreads.com`).

**More context doesn't help.** One repeat at 32K the same night (`CTX=32768`, 7.0 GB of VRAM in use with the
desktop): 3/6 solved (bug fix ✗, word count ✓, feature ✓, login ✓, books ✗ at the 20-minute timeout, JS quotes ✗),
the same rate as at 24K, and twice as slow per job (median 16.7 min). With the extra room it thinks longer (43K
characters a job instead of 26K, peak context 28.7K tokens) and still ran out in two attempts
([stats](../../bench/results/stats-twil-32k.txt)). The launcher stays at 24K.

So the context is a symptom, not the cause: it fills whatever room it gets. It's a code writer, not an agent: give it a short, self-contained job, not a web page to explore.
