# Quick checks: does every model still work on a new build?

Not a benchmark. [`check.sh`](check.sh) starts each model through `scripts/llm` with its normal settings (the
expert cache where the launcher uses it), on port 8091, and asks:

- **agent models:** "What is 17 * 23?" (greedy; passes if the answer has 391) and one tool-call request (passes
  if it calls `get_weather` with Paris; the call is printed, never run). VibeThinker gets only the first: it
  has no native tool calls;
- **Qwen3-VL and LFM2.5-VL:** the dot-counting picture from [bench/vision](../vision/README.md) (passes if the
  answer is a count; the truth is 7 red, 11 blue);
- **LFM3-VL, LocateAnything, EmbeddingGemma 2, d1, Clef-Flash, LFM3-Audio:** their [example scripts](../../examples/README.md)
  with built-in or fixed inputs. The helpers' own scored checks are in [bench/smoke](../smoke/README.md).

Run it on both builds and compare the logs: `CUDA_BIN=/path/to/llama-server bash bench/buildcheck/check.sh LABEL [NAME...]`.

## 2026-11-08: the merged expert cache (`3f2b2d60a`) against the build before it (`84137489b`)

The default build moved to the fork's `moe-cache-max-batch-merged` branch, the expert cache as merged upstream
([why](../moe-cache/README.md#upstreams-merged-version-faster-once-one-threshold-is-raised-experimentalstep18-2026-11-07-night)).
All 25 models in `scripts/llm list` ran on both, each with the launcher's settings:

- **Everything passes on both builds**, with the same answers: every arithmetic answer and tool call, the same
  picture counts (Qwen3-VL 7/13, LFM2.5-VL 9/10 against a truth of 7/11; small vision models miscount on both),
  the same decisions and probabilities from d1 and Clef-Flash, the same search hit and transcript, and the same
  correct "nothing found" from LocateAnything (there is no circle in that picture).
- **VibeThinker 1.5B** used all 4,096 tokens thinking without answering, on both builds.
- **Answer lengths differ on the expert-cache models** (Groot, Gemma, Qwen3.6, Ornith; Qwen3.6's arithmetic
  answer went from 255 to 378 tokens; the same answer either way): the new eviction policy sends different experts to the CPU, so the arithmetic differs slightly.
- **Decode speeds** differ by up to ~25% on these short answers (Dory 24.4 → 18.5 tok/s with Chrome open, LFM3
  91.7 → 101.7); they are single runs, not a benchmark ([the A/B](../moe-cache/README.md#upstreams-merged-version-faster-once-one-threshold-is-raised-experimentalstep18-2026-11-07-night)
  measures it).

Logs: [`new-small.txt`](logs/new-small.txt) and [`old-small.txt`](logs/old-small.txt) (Chrome open, ~1.7 GB
desktop: Spark, TwIL, Bonsai 2, Qwen3-VL and Clef-Flash ran out of VRAM at load on the new build, and Spark on
the old one too, where Chrome was closed partway through; the `*-serve.txt` files have the errors, Clef-Flash's are inline), then
[`new-big.txt`](logs/new-big.txt) and [`old-big.txt`](logs/old-big.txt) with Chrome closed (~0.5 GB desktop):
the MoEs plus those five, which all loaded. These four logs came from an earlier version of `check.sh` (header
`### smoke`, no `>> starting` lines); the answers it records are the same.

## 2026-11-08: the full upstream sync (`1f6e660c4`)

The fork's `adventures-sync-2026-11-08` branch, upstream master merged into the default build
([speed and agent-loop checks](../moe-cache/README.md#the-full-upstream-sync-experimentalstep19-2026-11-08)).
[`sync.txt`](logs/sync.txt), Chrome closed: **all 25 pass, with the same answers, tool calls, picture counts and
decisions.** What moved: answer lengths on Groot, Gemma, Qwen3.6, Ornith, Dory, TwIL and LFM2.5-VL (the same
answers in a different number of tokens), Flint's tool call took 68 tokens instead of 49, and Clef-Flash's
"unsafe" probability went from 0.920 to 0.915. Upstream changed attention and the 4- and 5-bit matrix kernels'
rounding, so small numeric shifts are expected.

After the sync became the default build (main `../llama.cpp` rebuilt at `1f6e660c4`), a short run through the
launcher, [`default-sync.txt`](logs/default-sync.txt) (LFM3, Bonsai 2, Qwen3.6, Qwen3.8, Gemma, d1): all pass, the
same token counts as `sync.txt`. Not chased: Qwen3.8's speed on these 63-token answers right after loading is
34–38 tok/s on the sync build against 40–42 on the two builds before it; over 400-token answers there is no gap
([step 19c](../moe-cache/README.md#the-full-upstream-sync-experimentalstep19-2026-11-08)).
