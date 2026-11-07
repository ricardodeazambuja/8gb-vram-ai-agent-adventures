# 🧠 MoE expert cache and prompt-cache reuse (2026-11-04 to 2026-11-06)

The measurements behind [How it fits §12](../../docs/how-it-fits.md#12-let-a-cache-pick-the-experts-the-default-for-the-big-moes)
and the Gemma 4 template fix on [its model page](../../docs/models/gemma4-26b-a4b.md). They started from three ideas in
[FreeToken](https://github.com/FlashML-org/FreeToken), which we read but did not run:

1. a VRAM cache of recently used experts instead of fixed layers on the GPU;
2. computing cache misses on the CPU instead of copying them over PCIe;
3. keeping the prompt cache valid across an agent's turns.

Ideas 1 and 2 already existed in llama.cpp: upstream PR [#29887](https://github.com/ggml-org/llama.cpp/pull/29887) (merged) and the open PR [#30966](https://github.com/ggml-org/llama.cpp/pull/30966), cherry-picked onto
our fork's [`moe-offload`](https://github.com/ricardodeazambuja/llama.cpp/tree/moe-offload) branch at `3a72c57be`.
Our default build is now the [`moe-cache-max-batch`](https://github.com/ricardodeazambuja/llama.cpp/tree/moe-cache-max-batch)
branch (`69ca796dd`): `moe-offload` plus the `--moe-cache-max-batch` flag ([below](#two-experiments-in-the-fork-experimental)).
Idea 3 isn't needed. Qwen3.6 reprocesses about 25 tokens per turn. Gemma 4 wasted 743 and 1,147 tokens on long
`write` turns because its chat template adds a `\n` the model never writes;
[`config/gemma4-26b-nonl.jinja`](../../config/gemma4-26b-nonl.jinja) removes it, which brings both turns to 14
tokens.

**[`RESULTS.md`](RESULTS.md) has every number, command line and build commit.** Rerun: see
[docs/reproduce.md](../../docs/reproduce.md#the-other-benchmarks). Its console output is verbatim,
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
| `thinkingcap-template.jinja` | ThinkingCap Gemma 4 26B's template, from its GGUF: different file (a `thinking_gate`), same `'\n<channel|>'` after the thinking. [`config/thinkingcap-gemma4-26b-nonl.jinja`](../../config/thinkingcap-gemma4-26b-nonl.jinja) drops it and also renders empty thoughts (below) |
| `divergence2.py` | like `divergence.py`, with thinking off and on; found ThinkingCap's empty-thought bug |
| `empty_thoughts.py FOLDER...` | tool-call turns without thinking in pibench transcripts; output in `logs/empty-thoughts.txt` |
| `expert_usage.py TRACE...` | how concentrated the router's expert picks are per layer, and what keeping the top K% would miss on held-out tokens; output in `logs/expert-usage.txt` |
| `experts/` | expert use on real agent sessions: `trace-qwen36.sh` (the run), `replay_sessions.py` (in the folder above), `agent_experts.py` (the analysis), `prune-eval.sh` with `make_rank.py` and `prune_gguf.py` (pruned GGUFs, KL divergence), logs |
| `gemma-empty.sh` | base Gemma 26B with the fixed `config/gemma4-26b-nonl.jinja`: `divergence2.py` and the agent loop, thinking off and on (uses `longctx/srv.sh`) |
| `fitcheck.sh NAME LAUNCHER...` | starts one of our launchers on port 8090, sends a ~30K-token prompt plus 200 tokens, reports VRAM; the pibench pre-check below |
| `pibench_speed.py RAW... MODEL...` | prefill and decode speed inside real pibench sessions, fitted from pi's per-call tokens and `durationMs`; output in `logs/pibench-speed.txt` |
| `longctx/` | the long-prompt test and the warm-up check: `step1-longctx.sh` (via `srv.sh` and `longctx.py`), `step1b-warmcheck.sh`, `hits.sh`; the prompts and every log in `longctx/logs/` |
| `longctx/memguard.py`, `longctx/memcheck.py` | the out-of-memory guards (below, after the five-model table) |
| `models/` | the same A/B on Groot, LFM2.5, the Qwen3.8 distill, Ornith and Bolt: `step3-models.sh` (uses `longctx/srv.sh`), reruns at matched VRAM `step5-qwen38.sh` and `step12-ornith.sh`, the launcher sizes for Groot and LFM2.5 `step17-groot-lfm25-size.sh`, logs and tuner output |
| `experimental/` | the fork's `moe-cache-warm` build: softer warm-up, `LLAMA_MOE_CACHE_MAX_BATCH`, and a perplexity check (`step8-exp.sh`, `step9-ppl.sh`); the batch-limit runs (`step10`, `step11`, `step14`, `step15`) and the daytime cache sizes (`step16-daytime-size.sh`, `step16b-daytime-static.sh`); logs |
| `tune/` | upstream's `scripts/moe-cache-tune.py` (from PR #30966) on Qwen3.6, Gemma 26B (32K, f16 KV) and Flint: tables, recommendations, per-run JSON, logs and routing traces (`.xz`). Single runs, so ±13% noise |
| `logs/` | llama-server stderr for every run (`srv-LABEL.log`), `cache_run.log` (the 77% hit rate), `offline-sizes.txt` |

`srv-ncmoe32.log` was overwritten by a rerun with the same label: earlier attempts ran out of memory without
`--parallel 1 -b 512`, and only the good runs are in it.

## ThinkingCap: the empty thought block (2026-11-05)

Removing the newline alone changed nothing for ThinkingCap: stock and newline-free templates reprocessed
1416 / 443 / 616 / 48 / 48 tokens over five pi-like turns (`logs/tcap-step2.log`, `tcap-step2b.log` with
thinking on). On tool-call turns it usually writes an *empty* thought, `<|channel>thought\n<channel|>`:
written by the model with thinking on, put in the generation prompt with thinking off. The parser returns
reasoning `""`, and the template renders the block only `if thinking_text`. So the history loses it, the
re-render differs at the very start of the previous turn, and the server reprocesses all of it
(`divergence2.py`, `logs/tcap-step2c.log`). The config template now renders the block for every tool-call
turn, empty or not:

```jinja
{%- if thinking_gate and (thinking_text or message.get('tool_calls')) -%}
    {{- '<|channel>thought\n' + (thinking_text or '') + '<channel|>' -}}
```

Re-render then equals the generation exactly, and reprocessing drops to 2 / 17 / 17 / 28 / 28 (thinking off)
and 3 / 17 / 17 / 28 / 28 (on) (`logs/tcap-step2d.log`; the write turns generated 425–661 tokens each, which
pi had been prefilling again). One task, seeds 1–5, temperature 0.6; the divergence check is temperature 0.
Run on the `moe-offload` build with the cache (`-cmoe --moe-cache-mib 3000`), which doesn't affect prompt
reuse. How much it costs in real sessions (`empty_thoughts.py`, `logs/empty-thoughts.txt`, from the pibench
transcripts): ThinkingCap made 33 of its 197 tool calls (17%) without thinking, each one a whole previous
turn reprocessed. The base Gemma 26B template has the same `if thinking_text` guard, but the base model
thought before 435 of its 436 tool calls across two batches, so with pi (thinking on) it hardly ever hits it.
With thinking *off* it hits it on every turn: the generation prompt ends with the empty block and the history
drops it (`divergence2.py` on base Gemma, `logs/gemma-step2e.log`). The same two-line change is now in
`config/gemma4-26b-nonl.jinja` (lines 239–240), tested with `gemma-empty.sh` ([`logs/gemma-step4.log`](logs/gemma-step4.log)):
reprocessing is 5 / 14 / 14 / 25 / 25 with thinking off and the same with it on. In that run's divergence
check both modes first differ 182–190 characters into the generation, inside the code or the reasoning,
where the raw and the chat call wrote different text. (In step 2e's thinking-on half, the difference 182 characters in is inside the reasoning: the raw and
the chat call wrote different text at temperature 0, which says nothing about the template.) Use the fix with
`EXTRA_ARGS="--chat-template-file config/thinkingcap-gemma4-26b-nonl.jinja"`.

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

In this check the short-prompt gains looked gone after a long prompt. The longer test below doesn't
reproduce that, and why this check differs isn't known. Untested differences: the launchers' flags against
the test's, an older build, the CPU's temperature, and the 200-token answer. The tuner's recommendations
(`tune/*.txt`): Qwen3.6 2624 MiB / share 0.8, Gemma 3008 / 0.8, Flint 2048 / 0.8, its Flint numbers lower
than both our A/B and the `NCMOE=42` baseline.

Why, from the code: batches over 32 tokens skip the cache (`src/llama-moe-cache.cpp`, `max_batch = 32`; since
made a flag, `--moe-cache-max-batch`, [below](#two-experiments-in-the-fork-experimental)) and
copy the used experts of every MoE layer per ubatch (40 on Qwen3.6, 47 on Flint), so prefill runs a few
percent slower. They also never insert into the cache, so decode after a long prompt starts with an empty
cache, and until it fills, every miss is copied rather than computed on the CPU.

## After a long prompt: 600 tokens, and a warm-up that didn't help (`longctx/`, 2026-11-05)

`longctx/step1-longctx.sh`: a ~30K-token prompt from our docs, then 600 greedy tokens twice (the second
request reuses the cached prompt), through llama-server at the launchers' settings; old split, cache, cache
plus warm-up, old split again. Decode in tok/s, request 1 / 2, prompt speed in brackets
([`longctx/logs/step1.log`](longctx/logs/step1.log)):

| | Old split, run 1 | Cache, share 0.95 | Cache + warm-up | Old split, run 2 |
|---|---|---|---|---|
| Qwen3.6 (`NCMOE=32` / 2900 MiB) | 22.8 / 22.2 [230] | **28.3 / 28.7** [215] | 28.1 / 29.3 [226] | 21.7 / 21.9 [228] |
| Flint (`NCMOE=42` / 2168 MiB) | 19.8 / 19.5 [469] | **21.5 / 21.7** [450] | 21.5 / 21.7 [458] | 19.4 / 19.3 [467] |

The cache still wins after a long prompt: Qwen3.6 +29%, Flint +11% against the mean of the old-split runs,
with prompt speed 6% and 4% lower. Smaller gains than at short prompts (+58%, +19%), probably because attention
over 30K tokens takes a larger share of each token (not measured). The first 100 tokens are the slowest but
still ahead: Qwen3.6 26.1 vs 23.3 tok/s, then 28–30 with the cache; Flint 21.6 vs 20.3.

The warm-up was a local experiment in the fork (branch [`moe-cache-warm`](https://github.com/ricardodeazambuja/llama.cpp/tree/moe-cache-warm), `d2503edb4`): large
batches put their most-used experts into free cache slots, so decode wouldn't start cold. It works as built
and makes things slightly worse. In `longctx/step1b-warmcheck.sh` (Qwen3.6, 13.7K-token prompt, 300 tokens,
with the cache statistics on, [`longctx/logs/step1b.log`](longctx/logs/step1b.log)), the decode hit rate fell
from 58.1% to 50.6% and decode from 33.2 to 31.3 tok/s. The cache starts full of the experts the *prompt*
used, so the 95% CPU share applies at once and the experts decode needs get in slowly. Its prompt speed gain
(210 to 237 tok/s) is within single-run noise. The `llama-completion` hit-rate runs at 30K (`hits-*.log`)
printed no hit rates (that needs `-lv 4`), and on Flint they ran out of memory (1760 MiB compute buffer)
where llama-server didn't.

## Five more MoE models (`models/`, 2026-11-05)

`models/step3-models.sh NAME`: the tuner (`--cpu-moe`, 128K, q8_0 KV), then old split / cache at the
VRAM-matched size, share 0.95 / old split again, two 400-token `srvbench` runs each, plus the 30K-token
long-prompt check for the first two. All logs in [`models/logs/`](models/logs/), tuner output in `models/tune/`.

| Model (batch) | Old split | Cache | VRAM old / cache | Decode | 30K prompt, 600 tokens | Prompt speed |
|---|---|---|---|---|---|---|
| Groot 19B-A3B (2048) | `NCMOE=27` | 3321 MiB | 7205 / 7333 MiB | 41.0 → **47.6** (+16%) | 30.9 → 32.7 (+6%) | 790 → 741 (−6%) |
| LFM2.5 24B-A2B (2048) | `NCMOE=27` | 4212 MiB | 7521 / 7512 MiB | 34.0 → **48.5** (+42%) | 29.2 → 37.9 (+30%) | 992 → 920 (−7%) |
| Qwen3.8 distill (512) | `NCMOE=31` | 3661 MiB | 7230 / 7227 MiB | 39.3 → **52.7** (+34%) | 30.8 → 34.0 (+10%) | 220 → 212 (−4%) |
| Ornith 1.5 35B (2048) | `NCMOE=34` | 2983 MiB | 7219 / 7223 MiB | 33.9 → **40.4** (+19%) | 28.5 → 29.7 (+4%) | 461 → 457 (−1%) |
| Bolt 74B-A9B (2048) | `-cmoe` | (600 MiB) | | not run: the tuner ran out of VRAM; then (`step13-bolt.sh`, daytime) `-cmoe` itself didn't load | | |

Old-split numbers are the mean of both old runs (the long check: the first only). The Qwen3.8 row is `models/step5-qwen38.sh`
([`step5.log`](models/logs/step5.log)), which replaced a first try that compared `NCMOE=34` (5926 MiB) with a
cache that had 827 MiB more. It compares the best of each at the same VRAM. The cache needs ~3.5 GB besides
itself (`-cmoe` + cache C loads at ~3489 + C MiB), against ~5.9 GB at `NCMOE=34` plus 408 MiB per layer fewer.
Share 0.8 was slower (46.4 tok/s) despite a higher hit rate (75% vs 65%).

Bolt (`models/step13-bolt.sh`, [`step13.log`](models/logs/step13.log), 2026-11-05 daytime, RAM margin 3 GiB with the
user's OK): with 1538 MiB of VRAM already in use, the launcher's own placement (every expert in RAM, batch 2048,
128K) was projected at 6192 MiB (model 4183, context 708, compute 1299) against 6195 free, and ran out of CUDA
memory on its warm-up decode. The cache runs never started. By the VRAM rule, with 334 MiB in use (night) there
would be ~860 MiB for a cache: ~5% of the 16.8 GB of weights it keeps in RAM. Since only ~1.7B of the ~8.5B
parameters per token are routed experts (its model page), the gain would be small; not worth a run until the
desktop is light.

The Ornith row is `models/step12-ornith.sh` ([`step12.log`](models/logs/step12.log)). Its first try in step 3
gave the cache 3486 MiB, which loaded at 7722 MiB and ran out of VRAM on the first request; 2983 MiB matches
`NCMOE=34`. Peak VRAM (1-s samples) was 7287 MiB for `NCMOE=34` and 7277 for the cache, with 334 in use by the
desktop: by the rule below both leave room for a desktop of ~430 MiB. Hit rate 52%. `QMODEL=ornith MOE_CACHE=auto`
now uses 2983 MiB.

These sizes were measured with 0.33–0.41 GB of VRAM already in use by the desktop. Qwen3.8's model page measured
`NCMOE=34` at 7.2–7.3 GB and `NCMOE=32` out of memory, likely with a heavier desktop; so `NCMOE=31` and a
3661 MiB cache only fit when the desktop is light. The launcher keeps `NCMOE=34` for now (since changed:
`scripts/llm distill` uses the cache, 2770 MiB at batch 2048, below).

A rule for any of these sizes: the server's own VRAM is the loaded total minus "vram before" in the logs. The
card gives CUDA about 7687 MiB (the tuner saw 7353 MiB free with 334 MiB in use), and ~300 MiB should stay free
for the cuBLAS workspace and first-request buffers (Ornith's cache run failed on its first request). So a
setting is safe while server + desktop peak + 300 ≤ 7687 MiB. For Qwen3.8 at batch 512, `NCMOE=31` and the
3661 MiB cache both take ~6820 MiB, so the desktop must stay under ~570 MiB. `NCMOE=34` takes ~5500–5600 and
leaves ~1.8 GB. Each 100 MiB less cache buys 100 MiB of desktop. `MOE_CACHE=auto` in `run-qwen.sh` starts
from the measured size and, since 2026-11-05, shrinks it by however much more VRAM is in use at launch than that
size leaves room for (`MOE_CACHE_ADAPT=0` keeps the measured size; below 1024 MiB it refuses to start). A
number in MiB is used as given. Groot's cache run had
128 MiB more VRAM, so its gain is slightly flattering. The tuner's picks: Groot 3264 MiB / share 0.9, LFM2.5
3968 / 0.95.

### Groot and LFM2.5: sizes for the launcher (`models/step17-groot-lfm25-size.sh`, 2026-11-07)

Step 3 measured them faster but gave them no `MOE_CACHE=auto` size, so `scripts/llm` kept the static split. Night,
desktop 432–435 MiB, batch 2048, 128K, q8_0 KV, share 0.95, cache-path limit 32: `NCMOE=27` / the cache at step 3's
size / 500 MiB less / `NCMOE=27` again, each with two 400-token `srvbench` runs and the turn test
([`step17.log`](models/logs/step17.log)). Desktop room is the rule above: 7687 − 300 − the server's own peak.

| Model | Setting | Own peak | Desktop room | Decode | Turns of 48 / 200 / 600 tokens |
|---|---|---:|---:|---:|---|
| Groot | `NCMOE=27` (both runs) | 7029–7031 MiB | ~360 MiB | 39.5 | 0.77 / 1.19 / 1.57 s |
| Groot | cache 3321 MiB | 7159 | ~230 | 51.4 (+30%) | 0.66 / 1.13 / 1.56 s |
| Groot | **cache 2821 MiB** | **6657** | **~730** | **47.4 (+20%)** | 0.73 / 1.23 / 1.62 s |
| LFM2.5 | `NCMOE=27` (both runs) | 7274–7278 | ~110 | 33.8 | 1.08 / 1.45 / 1.67 s |
| LFM2.5 | cache 4212 MiB | 7296 | ~90 | 46.8 (+38%) | 1.01 / 1.36 / 1.60 s |
| LFM2.5 | **cache 3712 MiB** | **6784** | **~600** | **43.5 (+29%)** | 1.06 / 1.43 / 1.69 s |

The static splits these two have used all along leave little room for the desktop: ~360 MiB for Groot (it ran
out of memory in pibench with a browser open at `NCMOE=25`) and ~110 MiB for LFM2.5. The cache needs ~3.84 GB
(Groot) and ~3.08 GB (LFM2.5) besides itself, so a smaller cache is both faster than the split and gives the
desktop more room. `MOE_CACHE=auto` now uses the bold rows (Groot 2821 MiB, room for 730; LFM2.5 3712, room for
600), and `scripts/llm groot` and `lfm25` turn it on. With a ~1.7 GB daytime desktop the launcher shrinks them to
~1850 and ~2615 MiB. Turns cost the same within noise. The 6K-token prompt read 10–15% slower with the cache on
each first `srvbench` run, which starts cold, and within 5% of the split on the second.

### Qwen3.8: batch 512 or 2048? (`models/step6-qwen38-ubatch.sh`, `step7-turns.sh`)

A bigger batch needs more VRAM besides the cache (`-cmoe`, 128K q8_0 KV: 3489 MiB at 512, 3677 at 1024,
4053 at 2048, from `srv-q38-probe-*.log`), so at the same VRAM the cache shrinks
([`step6.log`](models/logs/step6.log), order 512 / 2048 / 1024 / 512):

| Batch | Cache | Decode | Prompt, 6K / 34K tokens | 30K prompt, then decode | Hit rate |
|---|---|---|---|---|---|
| 512 | 3661 MiB | 52.9 tok/s (two runs: 53.2, 52.6) | 211 / 216 tok/s | 34.2 tok/s | 64.5% |
| 1024 | 3550 MiB | 52.6 | 338 / 343 | 34.6 | 67.0% |
| 2048 | 3174 MiB | 51.5 (−2.6%, within noise) | 517 / 495 (2.3×) | 34.0 | 63.1% |

Prompt time per agent turn, a cached 2,000-token prompt plus N new tokens, median of 3
([`step7.log`](models/logs/step7.log), `turnbench.py`), in ms at batch 512 / 2048:

| New tokens | 16 | 48 | 100 | 200 | 400 | 800 | 1500 | 2048 |
|---|---|---|---|---|---|---|---|---|
| ms | 239 / 247 | 737 / 781 | 1085 / 1132 | 1567 / 1645 | 2035 / 2106 | 3850 / 2637 | 6532 / 3147 | 8755 / 3527 |

Up to 512 new tokens the two are the same; above that batch 2048 wins. With turn time = prompt(N) + D /
decode, batch 2048 costs 3% on turns under 512 new tokens and saves 20% / 11% / 3% on an 800-token turn
(D = 100 / 300 / 1,000 tokens written), and 49% / 35% / 17% on a 2,048-token one. An agent's first turn
(system prompt and tools) and every file it reads are such turns, so batch 2048 is the better default
for Qwen3.8 with the cache.

But size the cache for the desktop: the batch-2048 server grew from 6893 MiB of its own at load to 7069 MiB
after the turn test, which leaves ~320 MiB for the desktop under the rule above. That's less than the
desktop's 334 MiB tonight. It ran, inside the 300 MiB margin. A smaller cache (2900 MiB, say) would leave
room for a normal desktop; not measured yet.

### Two experiments in the fork (`experimental/`)

Built from the fork's branch [`moe-cache-warm`](https://github.com/ricardodeazambuja/llama.cpp/tree/moe-cache-warm) at `ba50735ac` (on top of `moe-offload` 3a72c57be; pushed 2026-11-06) into
`../llama.cpp-warm/build-warm` (`WARM_LLAMA`). To rebuild it: `git -C ../llama.cpp worktree add ../llama.cpp-warm moe-cache-warm`,
then a CMake build into `build-warm/` there. It adds two environment switches. Qwen3.8, batch 2048,
3174 MiB cache, share 0.95 ([`step8.log`](experimental/logs/step8.log)):

- **A softer warm-up** (prompt experts go into free slots without counting as full): decode 48.5 tok/s
  against 51.6 without it, hit rate 60.8% vs 62.2%. The second warm-up that made things worse; dropped.
- **`LLAMA_MOE_CACHE_MAX_BATCH`**, the largest ubatch that goes through the cache and the CPU share instead
  of copying every used expert (upstream: 32). At 512, short turns' prompts got faster: 16 / 48 / 100 / 200
  / 400 new tokens in 258 / 433 / 658 / 1116 / 1955 ms against 328 / 858 / 1219 / 1668 / 2144 (−21%, −50%,
  −46%, −33%, −9%); 800 tokens and up unchanged. Decode 50.2 tok/s (within noise of 51.6).

Does the cache path compute the same thing? Perplexity on 8 chunks of 2,048 tokens of the long prompt, ubatch
512 ([`step9.log`](experimental/logs/step9.log)): `NCMOE=31` 6.8356 ± 0.189, cache 6.8329, cache with
`MAX_BATCH=512` 6.8264. Yes. But full 512-token ubatches through the cache are *slower* (200 vs 233 tok/s,
`NCMOE=31` 266): the hit rate drops to 1.25% and 137 GB get copied. So the threshold has to sit below the
size where copying everything wins.

The sweep, with the VRAM-safe 2900 MiB cache at batch 2048 (`step10-maxbatch.sh`,
[`step10.log`](experimental/logs/step10.log); `srv.sh` now also samples VRAM every second and prints the peak).
Prompt time in ms for N new tokens after a cached 2,000-token prompt:

| Limit | 16 | 48 | 100 | 200 | 300 | 400 | 600 | 800 |
|---|---|---|---|---|---|---|---|---|
| 32 (upstream) | 276 | 815 | 1227 | 1669 | 1948 | 2162 | 2456 | 2695 |
| 256 | 278 | 475 | 692 | 1175 | 1918 | 2137 | 2459 | 2694 |
| 128 | 277 | 479 | 693 | 1646 | 1914 | 2142 | 2456 | 2689 |
| 32 again | 274 | 811 | 1198 | 1649 | 1929 | 2162 | 2453 | 2695 |

256 is the pick: 48–200-token turns take 30–44% less time, nothing gets slower, and the decode hit rate is
unchanged (62.0 / 61.8 / 61.6 / 62.0%), so the bigger batches don't push decode's experts out. Decode varied
from 44.5 to 50.7 tok/s, and the two identical runs with limit 32 differed by 8%, so that's noise. Peak VRAM of
the server itself was 6767–6817 MiB, which leaves room for a desktop of ~570 MiB; `MOE_CACHE=auto` uses 2770
MiB for Qwen3.8 to leave ~700.

The other session has since made this a proper option, `--moe-cache-max-batch N`, on the fork's branch
`moe-cache-max-batch` at `69ca796dd` (`moe-offload` plus one commit; default 32, so nothing changes without the
flag). Pushed on 2026-11-05 as `origin/moe-cache-max-batch`, and the fork's main `build/` now runs it.
`QMODEL=distill MOE_CACHE=auto` passes `--moe-cache-max-batch 256`; the other models keep 32 until 128 is
measured (since done: `run-qwen.sh` now passes 128 for Qwen3.6 and Flint, steps 14–15 below).

With that flag (`step11-flag.sh`, [`step11.log`](experimental/logs/step11.log), built from `69ca796dd` into
`../llama.cpp-warm/build-warm`), 32 against 256 on three models. Change in prompt time for 16 / 48 / 100 / 200 /
300 / 400 / 600 new tokens:

| Model (batch, cache) | 16 | 48 | 100 | 200 | 300–600 | Decode, 32 → 256 | Decode hit rate |
|---|---|---|---|---|---|---|---|
| Qwen3.8 (2048, 2900 MiB) | −1% | **−41%** | **−43%** | **−30%** | 0 to −1% | 46.7 → 44.7 tok/s | 62% |
| Qwen3.6 (512, 2900 MiB) | −1% | −15% | −12% | +6% | −1 to −3% | 40.0 → 39.0 | 63% |
| Flint (2048, 2168 MiB) | +4% | −17% | −7% | **+15%** | 0 to +1% | 33.0 → 31.9 | 33% |

So 256 suits Qwen3.8 only. For Qwen3.6 and Flint, 200-token turns get slower, so the break-even is nearer 100–150
tokens. 128 would probably keep their gains at 48 and 100 tokens without the loss at 200; not tested. Flint
gains least: with a 33% hit rate, more of each turn falls to the CPU. Like the cache size, the limit is per
model.

**128 (step 14, 2026-11-05 night).** `experimental/step14-mb128.sh` ([`step14.log`](experimental/logs/step14.log)) ran
limits 32 / 128 / 256 / 32 again on Qwen3.6 (cache 2800 MiB, batch 512) and Flint (2168, batch 2048) on the main
build, desktop 356–394 MiB. Change in prompt time against the mean of the two runs at 32, which repeat to within 1%
(3–4% at 16 tokens):

| Model, limit | 16 | 48 | 100 | 128 | 150 | 200 | 300–600 | Decode vs 32 |
|---|---|---|---|---|---|---|---|---|
| Qwen3.6, 128 | −6% | **−17%** | **−13%** | **−11%** | −1% | −1% | −2 to −4% | −2% |
| Qwen3.6, 256 | −5% | −16% | −13% | −10% | 0% | **+6%** | −3 to −4% | −8% |
| Flint, 128 | −2% | **−20%** | **−8%** | −1% | −1% | −1% | 0% | −8% |
| Flint, 256 | −2% | −20% | −7% | −1% | **+7%** | **+16%** | 0% | −1% |

128 keeps the short-turn gain and drops the loss at 150–200 tokens on both. Decode is the open question: the four
runs at 32 alone spread over 36.9–39.9 tok/s (Qwen3.6) and 30.8–32.9 (Flint); Qwen3.6 at 128 is inside that
spread, but Flint at 128 and Qwen3.6 at 256 are below it, in opposite settings, and the limit shouldn't touch
decode (one-token batches; srvbench's 6K prompt mostly bypasses the cache, all but its last ~100 tokens). So
`QMODEL=qwen36 MOE_CACHE=auto` now passes `--moe-cache-max-batch 128`.

Flint's decode, settled (`experimental/step15-flint-decode.sh`, [`step15.log`](experimental/logs/step15.log)): limits
32 / 128 / 32 / 128, four 400-token runs each. Means 32.90 / 31.42 / 31.23 / 31.03 tok/s: the machine slowed 5%
between the two runs at 32, and in the second pair 128 is 0.6% behind, within the runs' spread (0.3–0.7 tok/s).
The decode hit rate is higher at 128 (34.8% vs 33.6%, the same in both repeats), so the cache isn't costing
anything; step 14's dip was the same drift, since 128 always ran after 32. Flint's `MOE_CACHE=auto` now passes 128
too.

The peak VRAM in these runs also says how much desktop the launchers' `MOE_CACHE=auto` sizes leave room for,
by the rule above (7687 − 300 − the server's own peak): Qwen3.6 at 2900 MiB peaked at 6973–7005 MiB of its own,
which leaves ~380 MiB; Flint at 2168, ~6883, ~500; Qwen3.8 at 2900 (batch 2048), ~6833, ~550, hence its 2770.
The cache sizes match the VRAM of each model's `NCMOE` default (except the distill's 2770, above, and Groot and LFM2.5,
sized for the desktop in step 17), and those defaults are as tight (Qwen3.6's
`NCMOE=32` ended the long-prompt test at 7335 MiB with 331 already in use, ~7000 of its own). With a busier
desktop, give a smaller `MOE_CACHE` in MiB, or a larger `NCMOE`.

That busier desktop is the daytime one: base Gemma's template run (`gemma-empty.sh`) started with 1737 MiB
already in use, against 334 at night, and peaked at 7608 MiB. So `run-qwen.sh` now does the arithmetic at
launch: `MOE_CACHE=auto` subtracts the excess from its size (with 1650 MiB in use, Qwen3.6 gets 1632 MiB and
Qwen3.8 1822), and any other MoE launch warns when more than 600 MiB is in use.

What the smaller caches cost (`experimental/step16-daytime-size.sh`, [`step16.log`](experimental/logs/step16.log),
2026-11-06): each model's `MOE_CACHE=auto` size against what the launcher shrinks it to next to a ~1.7 GB desktop,
run night / day / night. Both ran with a ~0.5 GB desktop, so this is the smaller cache alone, not GPU contention.
srvbench decode (400 tokens, mean of the runs) and turnbench (ms for N new tokens after a cached ~2K base):

| Model | cache night → day (MiB) | decode tok/s | 48 | 100 | 200 | 600 | decode hit rate |
|---|---|---|---|---|---|---|---|
| Qwen3.6 | 2900 → 1580 | 38.7 → 35.3 (−9%) | +21% | +24% | +14% | +16% | 62.7% → 50.7% |
| Qwen3.8 | 2770 → 1770 | 49.0 → 43.1 (−12%) | +10% | +14% | +11% | +7% | 59.4% → 49.6% |
| Flint | 2168 → 1024 | 32.6 → 29.0 (−11%) | +23% | +14% | +8% | +5% | 35.8% → 30.4% |
| Ornith | 2983 → 1713 | 44.5 → 38.5 (−13%) | +7% | +6% | +6% | +5% | 53.2% → 42.3% |

The two night runs agreed within 1–4% on decode and turns. Flint's day size would be 968 MiB, under the
launcher's 1024 floor, so with a 1.7 GB desktop the launcher refuses and Flint needs the browser closed; it ran
at 1024 here, and its two decode runs split (32.1, 26.0), so its −11% is the least certain. Ornith's night decode,
44.5 tok/s, is 10% above step 12's 40.4 at the same size (that ran the moe-offload build); not looked into.

Is the shrunk cache still worth it, or should a busy day fall back to static offload? Qwen3.6 at the same VRAM
(`step16b-daytime-static.sh`, [`step16b.log`](experimental/logs/step16b.log)): `--n-cpu-moe 36` (1572 MiB of experts
on the GPU, 5905 MiB loaded) / cache 1580 (5919 MiB) / `--n-cpu-moe 36`. Decode 21.4 vs **34.8 tok/s (+62%)**,
turns −19% / −8% / −7% / −5% at 48 / 100 / 200 / 600 tokens with the cache. So the cache keeps nearly all of its
lead over static offload at daytime sizes; a busy desktop costs ~10% of decode, not the cache's gain.

**This queue made systemd-oomd kill the terminal** (03:17, 03:30, 03:31). The tuner reads the whole GGUF into
the page cache on top of the previous model's pages; with distill, Bolt (20.5 GiB) and Ornith back to back,
memory pressure passed oomd's 50% for 20 s. Plain llama-server runs of the same models were fine. Distill and
Ornith's tuner runs were killed, which is why they have no tuner pick.

`oomctl` shows the limits: 50% memory pressure on `user@1000.service` for 20 s (swap 90%), and oomd kills
the descendant cgroup with the most reclaim, which isn't necessarily the one causing it. Every run since has
two guards, in `longctx/`: `memguard.py` (started once, in a scope not named `moe*`) kills our `moe*.scope`
bench units when pressure stays over 20% for 3 s, and `srv.sh` calls `memcheck.py` before each launch, which
drops other models' cached pages and refuses to start without the model's size + 4 GiB available. `srv.sh`
also runs the server with `-lv 4` now, so each log ends with the cache's hit rates.

**pibench (2026-11-04):** scores held and decode inside the sessions was faster for all three
(Qwen3.6 +59%, Gemma +26%, Flint +23%); Flint's prompt speed was 17% lower there, 4% in the test above.
[How it fits §12](../../docs/how-it-fits.md#12-let-a-cache-pick-the-experts-the-default-for-the-big-moes)
has the table.

## How concentrated is expert use? (`expert_usage.py`, 2026-11-05)

A first look at whether [REAP](https://arxiv.org/abs/2510.13999)-style expert pruning
([up next](../../docs/up-next.md)) could pay off here, from the traces
the tuner already wrote (`tune/out-*/run-*.trace.xz`: every expert the router picked, 1034 tokens of one story
prompt). Output in [`logs/expert-usage.txt`](logs/expert-usage.txt). Per layer, median over layers:

| Model | Never picked | Experts covering 90% / 99% of picks | Keep the top 75%: held-out picks lost | Keep 50% |
|---|---|---|---|---|
| Qwen3.6 35B-A3B (256 experts, 8 per token) | 28% | 23% / 52% | 1.9% (worst layer 8.7%) | 4.4% (worst 25%) |
| Gemma 4 26B-A4B (128, 8) | 14% | 43% / 70% | 7.4% (worst 21%) | 25% (worst 54%) |
| Flint 40B-A2.5B (256, 6) | 1.6% | 63% / 89% | 13% (worst 19%) | 32% (worst 40%) |

"Held-out": experts ranked on the first half of the tokens, picks counted on the second half. Qwen3.6 leans on
a small set of experts, Flint spreads its picks almost evenly, and the cache hit rates agree (Qwen3.6 63% and
Flint 33% in step 11). On this trace alone, Qwen3.6 looked like the candidate and Flint not; the agent
sessions below overturn the Qwen3.6 half of that.

What this doesn't show: one story prompt of ~1K tokens (about 32 picks per expert per layer, so "never picked"
is partly chance), and both halves come from the same text, so it says nothing about an agent session yet.
Frequency only: REAP also weighs the gate value and how much each expert changes the output, and a lost pick
goes to the router's next choice rather than to nothing. The next step is a trace of real agent sessions:
any cache run writes one with `LLAMA_MOE_CACHE_TRACE=FILE`.

### On real agent sessions: Qwen3.6 uses all its experts (`experts/`, 2026-11-05)

`experts/trace-qwen36.sh` replays Qwen3.6's 18 pibench sessions (6 tasks × 3 runs, 183,358 tokens) as prompts
on a cache run with `LLAMA_MOE_CACHE_TRACE` and `--moe-cache-max-batch 512`, so prompt batches go through
the traced cache path too (`replay_sessions.py` turns pi's `pi.jsonl` into chat messages). The router picks an
expert from the token and the context before it, so reading a session back routes it as it was routed when the
model wrote it, except the thinking of earlier turns, which Qwen3.6's template leaves out of the history.
`experts/agent_experts.py` splits the trace into sessions, saves the counts
([`qwen36-agent-counts.npz`](experts/logs/qwen36-agent-counts.npz), so the analysis reruns without the 217 MB
trace) and writes [`qwen36-agent.txt`](experts/logs/qwen36-agent.txt). Three layers are left out: the last one
only runs the tokens whose output is needed, and layers 34 and 38 saw just 72 tokens, so their large prompt
batches took the untraced large-batch path anyway (why only those two, we don't know).

| Qwen3.6, 37 layers | Story prompt (above) | Agent sessions |
|---|---|---|
| Experts never picked (median per layer) | 28% | **0%** |
| Experts covering 90% / 99% of picks | 23% / 52% | **68% / 93%** |
| Keep the top 75%: held-out picks lost | 1.9% (second half of the same text) | **7.6%** (a task it wasn't ranked on; 6.3–8.0% by task) |
| Keep the top 90% | 0.8% | 2.0% |
| Ranked on the story, keep 75%, scored on agent sessions | | 23% |

The story trace was too short and too uniform: on agent work every expert gets used, and a ranking from the
story is no better than chance (its top 25% of experts share 22% with the agent sessions' top 25%). A ranking
from other agent tasks does carry over: dropping a tenth of each layer's experts would miss 2% of the picks on
a task it never saw, a quarter 7.6%, half 24%. So frequency-based pruning of Qwen3.6 is worth at most ~10–25%
of the experts (1.4–3.5 GiB of the 14.1 GiB they take in RAM), and only if the router's second choices are nearly
as good, which only a pruned model can answer. REAP's own claims are for models with more experts, and its
score weighs each expert's output, not just how often it's picked.

### Pruned Qwen3.6: even 10% fewer experts changes one top token in ten (`experts/prune-eval.sh`, 2026-11-05)

`experts/prune-eval.sh` ranks each layer's experts on five tasks (the replay counts; layers 34, 38 and 39 from a
`llama-perplexity` pass with 16-token batches, which their 142-slot cache group traces), writes pruned GGUFs with
`prune_gguf.py` (each layer's experts and router rows dropped, `expert_count` set), and compares each one's
next-token distributions with the full model's on the held-out sixth task, feature-inventory (3 sessions,
5 × 4096-token chunks, ~10K scored tokens; `llama-perplexity --kl-divergence`). Logs in [`experts/logs/`](experts/logs/).

| Experts kept per layer | File | Mean KL divergence | Same top token | Picks lost per layer on an unseen task (mean of 6) |
|---|---|---|---|---|
| 256 (the writer, nothing dropped) | 17.7 GB | 0.000 | 100% | 0 |
| 230 (−10%), ranked | 16.2 GB | 0.48 | 89.5% | 2.0% |
| 230 (−10%), at random | 16.2 GB | 0.93 | 83.6% | |
| 192 (−25%), ranked | 13.9 GB | 0.70 | 86.8% | 7.6% |
| 128 (−50%), ranked | 10.1 GB | 1.06 | 82.2% | 24% |

The ranking halves the damage of a random cut, but a 10% cut already changes the top token one time in ten:
a model that wrote different code. The reason is compounding: losing 2% of picks per layer is 6.5 of each
token's 320 picks (8 per layer × 40 layers) going to a second choice, so only 0.14% of tokens are untouched.
The 8 chosen experts' weights are renormalized to sum to 1 (`qwen35moe.cpp` passes `norm_w = true`, as Qwen's
`norm_topk_prob` does), so the experts that are gone don't skew the weights of the ones still picked; the
damage is the rerouting itself. Perplexity is no help here: `llama-perplexity` compares against the stored, rounded base
log-probs and reads 1.037 for the identical model, and it moves the wrong way (keep 128 reads 0.89).

So frequency pruning without healing is a dead end for Qwen3.6 on this machine. What might change that: REAP's
score (gate value × output size, which needs the expert outputs, not just the trace), or a short fine-tune
after pruning (the paper's results are one-shot, on larger
models with more experts). None is cheap here.

