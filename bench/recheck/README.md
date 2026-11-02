# 🔁 Launch recheck

`bash bench/recheck/run.sh` starts every chat model with its documented launch command, then runs
`scripts/qwen-bench.py` (speed, a tool call, a long prompt) and `scripts/qwen-codecheck.py` (`merge_intervals`
against 500 random inputs, 16,000-token budget, in a no-network container). `run.sh flint groot` runs just those.

Each run saves one file per model and `summary.txt`. It's one sample per model: a code check result says what
happened that time, and speed varies by about ±13% on this laptop. VRAM is read when the server first answers,
so it can be low for models that load lazily (Gemma 4 26B showed 3.7 GB).

Run on 2026-11-01 (results in [`results/2026-11-01/`](results/2026-11-01/)):

| Model | tok/s | Long prompt, tok/s | Tool call | Code check |
|---|---:|---:|---|---|
| Flint | 21.6 | 557 | ✓ | ✓ 0/500 wrong |
| Gemma 4 26B | 20.8 | 239 | ✓ | ✓ |
| ThinkingCap Gemma 4 26B | 13.6 | 303 | ✓ | ✓ |
| Nemotron Bolt 74B | 15.4 | 308 | ✓ | ✓ |
| Qwen3.6-35B | 23.2 | 242 | ✓ | ✓ |
| Qwen3.8 distill | 33.9 | 208 | ✓ | ✓ |
| Groot-19B | 37.1 | 874 | ✓ | ✓ |
| LFM2.5-24B | 31.8 | 948 | ✓ | ✓ |
| Ornith-1.5-35B | 31.4 | 500 | ✓ | ✓ |
| Bonsai 2 27B (PrismML build) | 28.4 | 287 | ✓ | ✓ |
| Spark-X2.6-4B | 49.1 | 1,626 | ✓ | ✗ thought past 16,000 tokens, no code |
| LFM3-1.7B | 105.4 | 4,654 | ✓ | ✓ |
| LFM3-VL-2B | 105.6 | 4,663 | ✓ | ✓ |
| Nemotron Bolt E4B | 51.5 | 1,543 | ✓ | ✗ answered with no code block |
| Qwen3-VL-8B | 41.2 | 879 | ✓ | ✗ 119/500 wrong |
| Dory1.0-4B | 29.0 | 851 | ✓ | ✓ |
| Gemma 4 12B coder | 13.7 | 443 | ✓ | ✓ |
| Gemma 4 12B agentic | 13.4 | 445 | ✓ | ✗ its own test had a syntax error |
| VibeThinker 3B | 65.3 | 2,097 | ✗ expected, see its page | ✓ 0/500, though its own example test failed |
| VibeThinker 1.5B | 108.1 | 3,614 | ✗ expected | ✓ |

All on our fork except Bonsai 2.
