# 🔁 Launch recheck

`bash bench/recheck/run.sh` starts every chat model with the backend script under `scripts/llm` (`run-qwen.sh`, or
`run-server.sh` for the Gemmas) at its defaults, which means the static `NCMOE` split for the MoEs, not the expert
cache `scripts/llm` turns on for five of them. Then it runs
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
| Bonsai 2 27B `Q2_0` (our fork) | 14.3 | 297 | ✓ | ✓ |
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

All on our fork except the `PTQ1_0` Bonsai row.

Run again on 2026-11-02, after [PrismML's fork](https://github.com/PrismML-Eng/llama.cpp) was merged into ours
(`7ff495f`), to check the merge broke nothing (results in
[`results/2026-11-02-prism-merge/`](results/2026-11-02-prism-merge/)). Same launches, so Bonsai `PTQ1_0` still ran
f16 KV at 16K here:

| Model | tok/s | Long prompt, tok/s | Tool call | Code check |
|---|---:|---:|---|---|
| Flint | 21.5 | 515 | ✓ | ✓ 0/500 wrong |
| Gemma 4 26B | 21.5 | 240 | ✓ | ✓ |
| ThinkingCap Gemma 4 26B | 15.4 | 305 | ✓ | ✓ |
| Nemotron Bolt 74B | 15.0 | 310 | ✓ | ✓ |
| Qwen3.6-35B | 22.3 | 236 | ✓ | ✓ 0/500, though its own example test failed |
| Qwen3.8 distill | 34.6 | 204 | ✓ | ✓ |
| Groot-19B | 39.0 | 778 | ✓ | ✓ |
| LFM2.5-24B | 32.6 | 955 | ✓ | ✓ 0/500, though its own example tests failed |
| Ornith-1.5-35B | 32.0 | 470 | ✓ | ✓ |
| **Bonsai 2 27B** (now on our fork) | **37.5** | 302 | ✓ | ✓ |
| Bonsai 2 27B `Q2_0` | 14.3 | 242 | ✓ | ✓ |
| Spark-X2.6-4B | 49.5 | 1,347 | ✓ | ✓ |
| LFM3-1.7B | 106.2 | 4,236 | ✓ | ✓ |
| LFM3-VL-2B | 106.1 | 4,242 | ✓ | ✓ |
| Nemotron Bolt E4B | 52.6 | 1,503 | ✓ | ✗ answered with no code block |
| Qwen3-VL-8B | 42.1 | 696 | ✓ | ✗ 119/500 wrong |
| Dory1.0-4B | 29.1 | 723 | ✓ | ✓ |
| Gemma 4 12B coder | 13.9 | 405 | ✓ | ✓ 0/500, though its own example test failed |
| Gemma 4 12B agentic | 13.4 | 406 | ✓ | ✗ named a variable `sorted`, then called `sorted()`: crash |
| VibeThinker 3B | 66.1 | 2,100 | ✗ expected | ✓ |
| VibeThinker 1.5B | 113.5 | 3,619 | ✗ expected | ✗ an assert in its own answer fails on import |

Every speed is within the ±13% noise of the day before, except Bonsai. The code checks that flipped (Spark and
VibeThinker 3B now pass, VibeThinker 1.5B now fails) are one sample each, at the models' own temperatures.

With the defaults changed to a q8_0 KV and 32K ([context fit](../bonsai-ctx/)), Bonsai did 38.0 tok/s, 277 tok/s on
the long prompt, 7.14 GB, tool call ✓, 0/500 wrong ([`results/2026-11-02/bonsai2.txt`](results/2026-11-02/bonsai2.txt)).
The same test back to back on PrismML's own build (`e3de715`) and on the merged fork, both f16 at 16K:
28.5 and 38.5 tok/s ([`results/2026-11-02-prism-ab/`](results/2026-11-02-prism-ab/); the launch line reads
the same because the build was switched with `ADV_LLAMA`).
