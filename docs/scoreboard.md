# 🏆 The scoreboard

Everything we measured, everything the vendors claim, side by side. All of it from one
laptop: RTX 2070 Max-Q (8 GB VRAM), 32 GB RAM, CUDA builds of llama.cpp, 2026-10.

Two health warnings before the numbers:

- **Speeds are single runs.** On this laptop the same run can wobble by about ±13 % (thermals,
  background processes, the phase of the moon). Don't crown a winner over a 5 % gap.
- **Context is what fits**, at each launcher's defaults. Most of these models can do more on a
  bigger GPU; this is what they do on ours.

## Everything at a glance

| Model | Shape | File | Context | Speed | Prompt speed ¹ | pibench | Vision | Launcher |
|---|---|---|---:|---:|---:|---:|---:|---|
| [Gemma 4 26B-A4B](models/gemma4-26b-a4b.md) | MoE, 4B active | UD-Q4_K_XL, ~14 GB | 32K (128K fits) | 23.5 t/s ² | — | **18/18** | **100%** | `scripts/start.sh` |
| [Qwen3.6-35B-A3B](models/qwen3.6-35b-a3b.md) | MoE, 3B active | UD-IQ4_XS, 17.7 GB | 128K | 22.8 t/s | 221 t/s | 16/18 | **100%** | `QMODEL=qwen36 scripts/start-qwen.sh` |
| [Qwen3.8 distill](models/qwen3.8-35b-a3b-distill.md) | MoE, 3B active | IQ4_XS, 19.6 GB | 128K | 26.2 t/s | 167 t/s | 16/18 | 88% | `QMODEL=distill scripts/start-qwen.sh` |
| [Flint](models/flint.md) | MoE, 2.5B active | UD-Q3_K_XL, 18.5 GB | 128K | 24.3 t/s | 542 t/s | **18/18** | text only | `scripts/start_flint.sh` |
| [Groot-19B-A3B](models/groot-19b-a3b.md) | MoE, 3B active | Q4_K_M, 11.7 GB | 128K | **39.5 t/s** | **887 t/s** | **18/18** | 74% | `QMODEL=groot scripts/start-qwen.sh` |
| [LFM2.5-24B-A2B](models/lfm2.5-24b-a2b.md) | conv/attention MoE, 2B active | Q4_0 (QAD), 13.6 GB | 128K | 31.5 t/s | ~940 t/s | 10/18 | text only | `QMODEL=lfm25 scripts/start-qwen.sh` |
| [Nemotron 3.5 Bolt 74B](models/nemotron-bolt-74b.md) | hybrid MoE, 9B active | IQ2 (QAD), 22.0 GB | 128K | 14.0 t/s | 310 t/s | 15/18 | 91% | `scripts/start_bolt.sh` |
| [Bonsai 2 27B](models/bonsai2-27b.md) | dense 27B, ternary | PTQ1_0, 5.95 GB | 16K | 27.0 t/s | 269 t/s | 14/18 | 98% | `scripts/start_bonsai2.sh` |
| [Gemma 4 12B agentic](models/gemma4-12b-agentic.md) | dense 12B | Q4_K_M + turbo3 KV | 16K | 28.3 t/s | — | not run | not run | `scripts/start-turbo.sh` |
| [Gemma 4 12B coder](models/gemma4-12b-coder.md) | dense 12B | Q4_K_M | 32K | ~17 t/s | — | not run | not run | `scripts/run-12b-coder.sh` |
| [Qwen3-VL-8B](models/qwen3-vl-8b.md) | dense 8B + vision | Q4_K_M + projector | 16K | ~38 t/s | 830–900 t/s | not run | 89% | `scripts/start_qwen3vl.sh` |
| [Dory1.0-4B](models/dory1.0-4b.md) | dense 4B, looped | Q5_K_M | 64K (128K ³) | 29.3 t/s | 852 t/s | not run | text only | `scripts/start_dory.sh` |
| [Spark-X2.6-4B](models/spark-x2.6-4b.md) | dense 4B | Q8_0, 4.4 GB | 108K | 48–50 t/s | 1,450–1,620 t/s | 14/18 | text only | `scripts/start_spark.sh` |
| [Nemotron 3.5 Bolt E4B](models/nemotron-bolt-e4b.md) | dense 8B (4B eff.) | Q4_K_M, 6.2 GB | 64K | **52 t/s** | 1,531 t/s | 7/18 | text only | `scripts/start_e4b.sh` |

¹ Processing a ~14–17K-token prompt, which is what a coding agent's first turn costs. "—" = not
measured the same way.
² 27 t/s at 128K with a q8_0 KV cache ([how it fits](how-it-fits.md)).
³ With `CTX=131072 LOOP_KV=last_loop`: 30.1 t/s.

**pibench** = our six-chore agentic benchmark, solved attempts out of 18
([details](pibench.md)). **Vision** = our 12-image probe, overall score
([details](vision-probe.md)).

## What the vendors say

Numbers from the model cards, **not measured by us**, collected 2026-10-23. Three grains of
salt, to taste:

1. **They're for the full-precision models.** We run 1.7–5-bit versions. Only Bonsai 2 publishes
   what its quant keeps: 98.2 % of FP16 on a 14-benchmark average.
2. **Every vendor uses its own test harness, and it matters a lot.** Gemma 4 26B scores **17.4**
   on SWE-bench Verified in Qwen's table and **57.4** in NVIDIA's. Same model, same benchmark,
   40 points apart. Trust a gap of a few points not at all.
3. A score marked *(base)* is for the model it was derived from, not the exact file we run.

| Model | SWE-bench Verified | Terminal-Bench 2.1 | LiveCodeBench v6 | GPQA Diamond | Tool use (benchmark varies) | Source |
|---|---:|---:|---:|---:|---|---|
| Gemma 4 26B-A4B | 17.4 ᵠ / 57.4 ⁿ | 35 ⁿ | 77.1 | 82.3 | τ²-bench 68.2 | [Google card](https://huggingface.co/google/gemma-4-26B-A4B-it); Qwen ᵠ and NVIDIA ⁿ cards |
| Qwen3.6-35B-A3B | 73.4 | 43–45 ⁿʳ | 80.4 | 86.0 ᵍ | τ³-bench 67.2, MCPMark 37.0 | [Qwen card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| Qwen3.8 distill | — | — | — | — | — | [card](https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill): only MMLU 83.4, ARC-C 58.2 (≈ its Qwen3.6 base) |
| Flint | **74.2** | **54.9** | 80.3 | — | IFBench 89.2 | [Recursive card](https://huggingface.co/RecursiveSI/flint) (+ SWE-bench Pro v2 89.7) |
| Nemotron 3.5 Bolt 74B | — | **51.0** ᴮ | — | 86.4 ᴮ | MCP-Atlas 52.2 ᴮ | [NVIDIA card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF) (this exact 2-bit file; Qwen3.6 scores 44.4 / 83.4 in the same table) |
| Nemotron 3.5 Bolt E4B | — | — | — | — | BFCL v3 67.3, τ²-bench 52.3 | [NVIDIA card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB) (Q4_K_M; no coding benchmarks) |
| Bonsai 2 27B | 75.6 ʳ *(base)* | 73.0 *(base)* | **90.1** | 89.2 *(base)* | BFCL v3 74.9 | [PrismML card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) (LCB and BFCL on the ternary model); base = [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Gemma 4 12B agentic | 44.2 *(base)* | — | 72.0 *(base)* | 78.8 *(base)* | τ²-bench telecom ~55 % (base ~15 %, author's own harness) | [author card](https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF) |
| Gemma 4 12B coder | 44.2 *(base)* | — | 72.0 *(base)* | 78.8 *(base)* | — | [author card](https://huggingface.co/yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF): no benchmarks of its own |
| Qwen3-VL-8B | — | — | 39.3 ˢ | — | BFCL v3 66.3 | [Qwen3-VL card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) (non-thinking; MMLU-Pro 71.6) |
| Dory1.0-4B (2 loops) | 59.5 | 18.1 | — | 72.4 | τ³-bench banking 12.9 | [NVIDIA card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16) (4 loops: 61.8 / 22.2) |
| Spark-X2.6-4B | 62.8 | — | — | 77.1 | τ²-bench 86.9, BFCL v4 71.8 | [XHToken card](https://huggingface.co/XHToken/Spark-X2.6-4B) (thinking high; + SWE-bench Pro 54.3) |

ᵠ Qwen's table. ⁿ NVIDIA's Dory table. ʳ Recursive's Flint chart (measured by Recursive).
ⁿʳ 43 in NVIDIA's table, 44.9 in Recursive's. ᵍ Qwen's card says "GPQA".
ˢ An older LCB v6 window (2025-02 to 05), non-thinking; not comparable with the other LCB numbers.
ᴮ NVIDIA's table for the 2-bit file, against BF16 rivals.
The Gemma 4 12B base numbers are Google's (SWE-bench Verified 44.2 as reported in XHToken's table).

**On paper:** for agentic coding at 128K it's Flint ≥ Qwen3.6 ≫ Gemma 4 26B. Bonsai 2's base is
the strongest model here, but at 16K context. The 4B models claim surprisingly high SWE-bench
Verified scores (Spark 62.8, Dory 59.5); Dory's Terminal-Bench (18–22) hints at how they cope
with long multi-step jobs.

**In practice (pibench):** Gemma 4 26B, the one the paper says is far behind, solved every
chore we gave it. Benchmarks measure the model; your laptop measures the model plus quantization
plus agent plus your tasks. Both are worth knowing.

## Everyone vs. Qwen3.6

Qwen3.6-35B-A3B is our yardstick: a strong, well-documented coder that fits 128K here. Speeds
and context are **ratios** to Qwen3.6, measured on this laptop (above 1 is better; for
"code-check tokens", below 1 means it reasoned less to reach the same correct answer). Published
benchmarks are **points** above (+) or below (−) Qwen3.6, taken from the **same source table**
where one exists so the harness cancels out; † marks a comparison across tables, so take it
lighter.

| Model | Speed | Prompt speed | Context | Code-check tokens | SWE-bench Verified | Terminal-Bench 2.1 | LiveCodeBench v6 | GPQA Diamond |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Qwen3.6-35B-A3B** (baseline) | 22.8 t/s | 221 t/s | 128K | 3,558 | 73.4 | 43–45 | 80.4 | 86.0 |
| Gemma 4 26B-A4B | 1.03× | — | 0.25× ¹ | — | −56.0 (Qwen) / −16.0 (NVIDIA) | −8.0 | −3.3 | −3.7 |
| Qwen3.8 distill | 1.15× | 0.76× | 1.00× | **0.16×** | — | — | — | — |
| **Flint** | 1.07× | **2.45×** | 1.00× | 0.43× | **+0.8** | **+10.0** | −0.1 | — |
| Nemotron Bolt 74B | 0.61× | 1.40× | 1.00× | 0.73× | — | **+6.6** ᴮ | — | +3.0 ᴮ |
| Bonsai 2 27B | 1.18× | 1.22× | 0.12× | 0.45× | +2.2 *(base)* | +28.1 *(base)* | +9.7 † | +3.2 † *(base)* |
| Gemma 12B agentic | 1.24× | — | 0.12× | — | −29.2 † *(base)* | — | −8.4 † *(base)* | −7.2 † *(base)* |
| Gemma 12B coder | 0.75× | — | 0.25× | — | −29.2 † *(base)* | — | −8.4 † *(base)* | −7.2 † *(base)* |
| Qwen3-VL-8B | 1.67× | 3.76× | 0.12× | — | — | — | — | — |
| Dory 4B (2 loops) | 1.29× | 3.86× | 0.50× | — | −13.9 | −24.9 | — | −12.9 |
| Spark 4B | 2.11× | 6.58× | 0.84× | 1.32× | −10.6 † | — | — | −8.9 † |
| Nemotron Bolt E4B | **2.28×** | **6.93×** | 0.50× | — | — | — | — | — |

¹ At the launcher's 32K default; Gemma 4 26B also runs 128K (27 t/s with a q8_0 KV cache).
ᴮ From NVIDIA's own table, which includes Qwen3.6 (BF16) next to the 2-bit Bolt.

"Code-check tokens" comes from a small sanity test (write `merge_intervals`, check it against
500 random inputs) that every model passed. It can't rank them, but it does show how chatty each
one's thinking is: 557–784 tokens for the distill, about 1.5K for Flint and Bonsai, ~3.5K for
Qwen3.6, ~4.7K for Spark.

## How they behave

Numbers say how often; this says how. Drawn from reading every pibench transcript (eight models,
144 attempts), the vision answers, and our spot checks for the four models that didn't run pibench.
The last column says how much we watched each one, so you know how far to trust the line.

| Model | In a word | Good at | Watch out for | Based on |
|---|---|---|---|---|
| [Flint](models/flint.md) | **careful** | Probes pages before scraping (~3 times per attempt), audits beyond the bug report, checks raw bytes with `xxd`, cleans up after itself | Wanders: the same chore took 6 min once and 13 the next. Can't see images | pibench, 18 attempts |
| [Gemma 4 26B-A4B](models/gemma4-26b-a4b.md) | **thorough** | Thinks long, acts in few steps, chases odd data until it understands it (found the missing charset with `curl -I`) | Fumbled pi's tool format (8 calls without a file path) and recovered every time. Boxes in its own 0–1000 y-first format | pibench, vision |
| [Nemotron Bolt 74B](models/nemotron-bolt-74b.md) | **deliberate** | Fewest tool calls and tool errors of all; tests edge cases nobody asked for; perfect OCR, charts and drawings | Slow (5–10 min a job). Chased a symptom until it timed out; draws boxes about twice as tall as they are | pibench, vision |
| [Qwen3.6-35B-A3B](models/qwen3.6-35b-a3b.md) | **direct** | Read, edit, test, done; little thinking for a big model; perfect vision | Shallow checks (30 quotes with empty authors, "done"); fixes what you report, not always what's hiding; once overwrote a provided file | pibench, vision |
| [Qwen3.8 distill](models/qwen3.8-35b-a3b-distill.md) | **quick** | Fastest big model per job; test-driven on code; the tersest thinker on the code check (under 800 tokens) | Writes scrapers from memory without looking (0.1 probes per attempt), so it's wrong in the ways memory is wrong | pibench, vision |
| [Bonsai 2 27B](models/bonsai2-27b.md) | **cramped** | Sound reasoning; every code chore solved; best at boxes in pixels, as asked | 16K context: cut off mid-thought in two thirds of its attempts | pibench, vision |
| [Spark-X2.6-4B](models/spark-x2.6-4b.md) | **busy** | Fast; checks its output more often than anyone (94 % of attempts) | Thinks the most and still waves through plausible wrong values (five stars for every book, one quote ten times) | pibench |
| [Nemotron Bolt E4B](models/nemotron-bolt-e4b.md) | **hasty** | The fastest model here; good for short assistant turns | Declares victory on empty output (`{}`, a header-only CSV); skims specs; 5.5 tool errors per attempt | pibench |
| [Qwen3-VL-8B](models/qwen3-vl-8b.md) | **snappy** | Flawless OCR in 6 seconds per image; boxes right in its own format | Doesn't think, so counting slips (12 dots for 11). 16K context | vision, spot checks |
| [Dory1.0-4B](models/dory1.0-4b.md) | **tunable** | More loops for more depth, fewer for more speed; tool calls work | No character study yet; its llama.cpp support is an unmerged PR | smoke tests only |
| [Gemma 4 12B coder](models/gemma4-12b-coder.md) | **neat** | Careful small-scale code (edge cases, sensible data structures); tool calls work | Occasionally drops a line (an LRU `put` lost its last insert); review what it writes | spot checks only |
| [Gemma 4 12B agentic](models/gemma4-12b-agentic.md) | **brisk** | 28 tok/s fully on the GPU; native tool calls | Can loop on `0000…` without its repetition penalty; 16K context | spot checks only |

What separates the top from the middle isn't how much a model thinks or how often it looks: it's
whether it *reads* what it looked at. The full story, with examples, is in
[pibench: what actually happened](pibench.md#what-actually-happened).

## So which one should I use?

| If you want… | Pick | Why |
|---|---|---|
| Long agent sessions (128K) | [**Flint**](models/flint.md) | 18/18 on pibench, the best published agentic scores here, 128K, prompts 2.5× faster than Qwen3.6. Can't see images. |
| A dependable all-rounder that sees | [**Gemma 4 26B-A4B**](models/gemma4-26b-a4b.md) | 18/18 on pibench and 100% on the vision probe. Slower per job than the Qwens. |
| Quick turns | [**Qwen3.8 distill**](models/qwen3.8-35b-a3b-distill.md) | Fastest per job of the big models (2.5 min median), 16/18, 128K. Double-check its scraped data. |
| Careful image reading at 128K | [**Qwen3.6-35B-A3B**](models/qwen3.6-35b-a3b.md) | 100% on the vision probe, 16/18 on pibench. |
| Fast image reading | [**Qwen3-VL-8B**](models/qwen3-vl-8b.md) | 6 s per image, perfect OCR, all on the GPU. Weaker at counting. |
| Raw speed on small jobs | [**Spark-X2.6-4B**](models/spark-x2.6-4b.md) | 48–50 tok/s, 108K, 14/18; careless with details, but twice the E4B's score at nearly its speed. |
| The biggest brain that fits | [**Nemotron Bolt 74B**](models/nemotron-bolt-74b.md) | 74B in 2 bits at 14 tok/s, 128K, images and video; 15/18 and 91% vision. Patience required: 5–10 min per agent job. |
| A snappy local assistant (not a coder) | [**Nemotron Bolt E4B**](models/nemotron-bolt-e4b.md) | 52 tok/s, all on the GPU; 7/18 on pibench, so keep it away from unsupervised coding. |
| The strongest model on paper | [**Bonsai 2 27B**](models/bonsai2-27b.md) | Its base tops every published column; its 16K context cuts most agent jobs short. |
| A looped model, for science | [**Dory1.0-4B**](models/dory1.0-4b.md) | Trade speed for depth by changing the loop count. Genuinely fun to poke at. |
