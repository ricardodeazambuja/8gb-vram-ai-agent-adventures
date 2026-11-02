# 🪨 Bonsai 2 context fit

How much context Bonsai 2 27B (`PTQ1_0`) gets on an 8 GB card once its KV cache can be **q8_0**. PrismML's own
build couldn't do that on Turing, so it stopped at 16K with an f16 cache. With PrismML's code merged into our
fork, q8_0 works.

`bash bench/bonsai-ctx/run.sh` starts the server with its documented launch (`QMODEL=bonsai2`) at 16K, 32K,
48K and 64K. For each, it fills all but 4,000 tokens of the context with this repo's docs (`longprompt.py`) and
asks which model page mentions "kryptonite" (the Bonsai page does, about 75K characters into the docs).

Run on 2026-11-02, fork `7ff495f` (results in [`results/2026-11-02/`](results/2026-11-02/)):

| Context | VRAM | Prompt | Prefill | Generation, at that fill | Answer |
|---|---:|---:|---:|---:|---|
| 16K | 6.58 GB | 12,638 tokens | 291 tok/s | 28.7 tok/s | "none": right, the page isn't in a prompt this short |
| **32K (default)** | **7.14 GB** | 28,448 | 241 | 22.4 | ✓ |
| 48K | 7.70 GB | 44,829 | 207 | 17.9 | ✓ |
| 64K | — | — | — | — | doesn't start: no room left on the GPU |

The generation speed here is with the context nearly full. With a short prompt it does 38.0 tok/s at the 32K default
([recheck](../recheck/results/2026-11-02/bonsai2.txt)). 48K works but leaves ~0.3 GB free on the card, too little for a
default, so the default is 32K (`CTX=49152` if you want the rest).
