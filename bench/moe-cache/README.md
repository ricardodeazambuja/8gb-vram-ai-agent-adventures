# 🧠 MoE expert cache and prompt-cache reuse (2026-11-04)

The measurements behind [How it fits §12](../../docs/how-it-fits.md#12-let-a-cache-pick-the-experts-opt-in-our-forks-moe-offload-branch)
and the Gemma 4 template fix on [its model page](../../docs/models/gemma4-26b-a4b.md). They started from three ideas in
[FreeToken](https://github.com/FlashML-org/FreeToken), which we read but did not run:

1. a VRAM cache of recently used experts instead of fixed layers on the GPU;
2. computing cache misses on the CPU instead of copying them over PCIe;
3. keeping the prompt cache valid across an agent's turns.

Ideas 1 and 2 already existed in llama.cpp: upstream PR #29887 (merged) and the open PR #30966, cherry-picked onto
our fork's [`moe-offload`](https://github.com/ricardodeazambuja/llama.cpp/tree/moe-offload) branch at `3a72c57be`.
Idea 3 isn't needed. Qwen3.6 reprocesses about 25 tokens per turn. Gemma 4 wasted 743 and 1,147 tokens on long
`write` turns because its chat template adds a `\n` the model never writes;
[`config/gemma4-26b-nonl.jinja`](../../config/gemma4-26b-nonl.jinja) removes it, which brings both turns to 14
tokens.

**[`RESULTS.md`](RESULTS.md) has every number, command line and build commit.** Its console output is verbatim,
because several results were only ever printed to the console. The work was done in a separate session working in
the llama.cpp fork and copied here afterwards. The paths in the scripts were then made relative. The logic
didn't change: the offline scripts give identical output ([`logs/offline-sizes.txt`](logs/offline-sizes.txt)),
and the agent-loop helpers import the same tools and messages they used to `exec`.

## Files

| File | What |
|---|---|
| `bw/bw.cu` | PCIe host-to-device bandwidth (pinned and pageable), RAM read bandwidth for 1–12 threads, and both at once. `nvcc -O3 -arch=sm_75 -o bw bw.cu -lpthread && ./bw` (env `llamacpp-cuda`) |
| `expert_bytes.py MODEL...` | expert count, top-k, layers and per-expert bytes from a GGUF |
| `gpu_exp_bytes.py MODEL N` | MiB of experts that `--n-cpu-moe N` leaves on the GPU: how the cache sizes were matched |
| `estimate.py` | fits the `llama-bench -ncmoe` sweep and predicts static vs cache+copy vs cache+CPU speed (measured inputs hard-coded) |
| `srvbench.sh LABEL ARGS...` | starts llama-server in its own systemd scope on port 8090 with launcher-like flags plus ARGS, records VRAM, runs `srvbench.py`, stops it. Env: `MODEL`, `CTX`, `BATCH`, `RUNS`, `AFTER`, `ADV_LLAMA`, `OUT` |
| `srvbench.py` | one ~6K-token prefill (the fork's `src/llama-moe-cache.cpp` as text) and one 400-token greedy decode, no prompt cache |
| `agentloop.py URL TURNS` | a pi-like loop: echoes `reasoning_content` and `tool_calls` the way pi does, answers with fake tool results, prints cached and reprocessed prompt tokens per turn. Env: `TASK`, `NEWUSER_AT`, `MAXTOK`, `PI_CAPTURE` (pi's real system prompt and tools, e.g. `../gemma12-probe/capture/agentic12/0001-req.json`) |
| `toolsdiff.py` | decode speed of the same chat request with and without tools |
| `longgen.py` | a 2,500-token streamed answer, speed per 500-token window |
| `divergence.py`, `toklcp.py` | where the cached text (and tokens) first differ from the template's re-render of the parsed reply |
| `gemma4-template.jinja` | Gemma 4 26B's stock template, from the GGUF, for diffing against `config/gemma4-26b-nonl.jinja` |
| `thinkingcap-template.jinja` | ThinkingCap Gemma 4 26B's template, from its GGUF: different file (a `thinking_gate`), same `'\n<channel|>'` after the thinking. [`config/thinkingcap-gemma4-26b-nonl.jinja`](../../config/thinkingcap-gemma4-26b-nonl.jinja) drops it; whether ThinkingCap also writes no newline there isn't checked yet |
| `fitcheck.sh NAME LAUNCHER...` | starts one of our launchers on port 8090, sends a ~30K-token prompt plus 200 tokens, reports VRAM; the pibench pre-check below |
| `pibench_speed.py RAW... MODEL...` | prefill and decode speed inside real pibench sessions, fitted from pi's per-call tokens and `durationMs`; output in `logs/pibench-speed.txt` |
| `tune/` | upstream's `scripts/moe-cache-tune.py` (from PR #30966) on Qwen3.6, Gemma 26B (32K, f16 KV) and Flint: tables, recommendations, per-run JSON, logs and routing traces (`.xz`). Single runs, so ±13% noise |
| `logs/` | llama-server stderr for every run (`srv-LABEL.log`), `cache_run.log` (the 77% hit rate), `offline-sizes.txt` |

`srv-ncmoe32.log` was overwritten by a rerun with the same label: earlier attempts ran out of memory without
`--parallel 1 -b 512`, and only the good runs are in it.

## Not fixed: a needless restore on short SWA prompts

On sliding-window models, llama-server restores a checkpoint even when the whole cache matches, if the prompt
is shorter than the window (~1K tokens for Gemma). Real pi prompts are longer, so this doesn't affect our runs.
Details in RESULTS.md §9.

## Before pibench: does it fit, and does it hold with a long prompt? (fitcheck.sh)

A ~30K-token prompt, then 200 tokens, through the real launchers at pibench's settings (2026-11-04, single runs):

| Model | Layers on the GPU (`NCMOE`) | Cache, share 0.95 |
|---|---|---|
| Qwen3.6, 128K | 20.0 tok/s, 7.70 GB | 20.6 tok/s (2900 MiB), 7.75 GB |
| Flint, 128K | 20.2 tok/s, 7.44 GB | 16.5 tok/s (2168 MiB), 7.45 GB |
| Gemma 26B, 64K q8_0 | out of memory at load (`NCMOE=20`) | 4084 MiB: out of memory; **3500 MiB: 25.3 tok/s, 7.51 GB** |

After a long prompt the short-prompt gains are gone: 200 tokens are too few to tell apart, or the cache
needs warming that a long prefill doesn't do. The pibench batch is the real test. The tuner's recommendations
(`tune/*.txt`): Qwen3.6 2624 MiB / share 0.8, Gemma 3008 / 0.8, Flint 2048 / 0.8, its Flint numbers lower
than both our A/B and the `NCMOE=42` baseline.

Why, from the code (read-only, not yet measured): batches over 32 tokens skip the cache
(`src/llama-moe-cache.cpp`, `max_batch = 32`). They copy every layer's experts per ubatch, 40 layers with
`--cpu-moe` against 32 with `NCMOE=32`, so prefill runs ~10% slower. They also never insert into the cache, so
decode after a long prompt starts nearly empty, and until the cache fills, every miss is copied, none computed on
the CPU. A 200-token answer is mostly that warm-up. pi's turns are short additions to a cached prompt, the case
the cache is good at; pibench decides.

**pibench decided (2026-11-04):** scores held and decode inside the sessions was faster for all three
(Qwen3.6 +59%, Gemma +26%, Flint +23%); Flint's prefill was 17% slower.
[How it fits §12](../../docs/how-it-fits.md#12-let-a-cache-pick-the-experts-opt-in-our-forks-moe-offload-branch)
has the table.
