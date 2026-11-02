# 🧠 VibeThinker 1.5B and 3B — one-shot reasoners

> **TL;DR**
>
> | | |
> |---|---|
> | On disk | 3B: 3.3 GB (`Q8_0`). 1.5B: 1.9 GB (`Q8_0`) |
> | Where it lives | entirely on the GPU: 3B 5.4 GB, 1.5B 3.7 GB, at 40K context |
> | Generation | 3B **65 tok/s**, 1.5B **108 tok/s** (on our fork, 2026-11-01) |
> | Best at | One-shot code (3B) and math (1.5B) |
> | [pibench](../pibench.md) | not run |
> | [Security audit](../security.md) | clean |
> | Launch | `bash scripts/start_vibethinker.sh`, or `QMODEL=vibe3b bash scripts/run-qwen.sh` for the server alone |

## What is it?

WeiboAI's reasoning models. The 3B is built on Qwen2.5-Coder-3B (MIT) and the 1.5B on Qwen2.5-Math-1.5B. They
think long, then answer once. They were never trained to emit OpenAI tool calls, so on their own they aren't
agents. The best use is as a one-shot code or math engine, or as a helper under a bigger model that plans,
drives the tools and checks the output.

The original work, with the full measurements, is in
[`vibethinker_tests`](https://github.com/ricardodeazambuja/vibethinker_tests) (June 2026).

## Install & run

```bash
bash scripts/download-models.sh vibe3b vibe15
bash scripts/start_vibethinker.sh                    # server + shim + pi
QMODEL=vibe3b bash scripts/run-qwen.sh               # or the plain server, for one-shot work
```

For one-shot work, skip pi and call `/v1/chat/completions` directly: pi always adds its tools and manual, and the
model can spend the whole turn deliberating over them. Give it a big token budget, because one easy math problem
used 4,809 tokens of thinking. `run-qwen.sh` sets the card's sampling (temp 0.6, top-p 0.95, top-k off), and the
card warns against temp 0.

`scripts/vibethinker-toolshim.py` is a small proxy between pi and the server. It turns the model's own formats
(`<function name=...>` XML, JSON in fences) into real tool calls, and rejects tool names that weren't offered.

## Our numbers

From June, on the same laptop, on an older llama.cpp build:

| Test | 3B | 1.5B |
|---|---|---|
| Suffix automaton, one shot | **Correct:** all 509 oracle cases and the 100,000-character check | Buggy, and no fix in 3 self-check rounds |
| Math (`7^2025 mod 100`, least n with n! ÷ 2024) | not recorded | Both right |
| Native tool calls | None | None |
| Tool calls through the shim, one session | 34 recovered, 3 missed | not recorded |
| Used a real tool result (random token test) | 2/2 without pi's extensions, 0/2 with them | not recorded |

As an agent, the 3B's `write`, `edit` and `read` calls are reliable. `bash` is not: it's called about 3 times in 4 in
a fresh session, falling to about 1 in 3 as the session grows, and on a miss the model makes up the output instead
of running the command.

## Gotchas

- **The shim runs what the model writes.** It turns text into tool calls, so treat the agent mode like any
  untested agent: only on work you'd let it run, and in [pibench](../pibench.md) for anything that matters.
- Keep agent sessions short, and check any shell work yourself.
- It's a weak self-debugger: given a failing case, it revises the code but doesn't reliably find the fault.
- The 3B mangles dotted filenames (`secret.txt` became `secretxt`).
- The agent mode needs a reasoning budget, or the model thinks past the token limit and never calls a tool.
  `start_vibethinker.sh` sets 600.
