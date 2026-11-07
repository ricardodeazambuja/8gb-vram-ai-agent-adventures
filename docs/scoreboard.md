# 🏆 The scoreboard

Everything we measured, everything the vendors claim, side by side. All of it from one
laptop: RTX 2070 Max-Q (8 GB VRAM), 32 GB RAM, CUDA builds of llama.cpp, 2026-10 and 2026-11. Every model's
command is in [the cheatsheet](cheatsheet.md).

Two health warnings before the numbers:

- **Speeds are single runs.** On this laptop the same run can wobble by about ±13% (thermals,
  background processes, the phase of the moon). Don't crown a winner over a 5% gap.
- **Context is what fits**, at each launcher's defaults. Most of these models can do more on a
  bigger GPU; this is what they do on ours.

## Everything at a glance

| Model | Shape | File | Context | Speed | Prompt speed ¹ | pibench | Vision | Launcher |
|---|---|---|---:|---:|---:|---:|---:|---|
| [Gemma 4 26B-A4B](models/gemma4-26b-a4b.md) | MoE, 4B active | UD-Q4_K_XL, ~14 GB | 32K (128K fits) | 23.5 tok/s ² | — | **18/18** | **100%** | `scripts/llm gemma` |
| [ThinkingCap Gemma 4 26B](models/thinkingcap-gemma4-26b-a4b.md) | MoE, 4B active | IQ4_XS, 13.6 GB | 64K | 17.4 tok/s | 352 tok/s | 17/18 | not run | `scripts/llm thinkingcap` |
| [Qwen3.6-35B-A3B](models/qwen3.6-35b-a3b.md) | MoE, 3B active | UD-IQ4_XS, 17.7 GB | 128K | 22.8 tok/s | 221 tok/s | 16/18 | **100%** | `scripts/llm qwen36` |
| [Qwen3.8 distill](models/qwen3.8-35b-a3b-distill.md) | MoE, 3B active | IQ4_XS, 19.6 GB | 128K | 26.2 tok/s | 167 tok/s | 16/18 | 88% | `scripts/llm distill` |
| [Flint](models/flint.md) | MoE, 2.5B active | UD-Q3_K_XL, 18.5 GB | 128K | 24.3 tok/s | 542 tok/s | **18/18** | text only | `scripts/llm flint` |
| [Groot-19B-A3B](models/groot-19b-a3b.md) | MoE, 3B active | Q4_K_M, 11.7 GB | 128K | **39.5 tok/s** | **887 tok/s** | **18/18** | 74% | `scripts/llm groot` |
| [LFM2.5-24B-A2B](models/lfm2.5-24b-a2b.md) | conv/attention MoE, 2B active | Q4_0 (QAD), 13.6 GB | 128K | 31.5 tok/s | ~940 tok/s | 10/18 | text only | `scripts/llm lfm25` |
| [Ornith-1.5-35B-A3B](models/ornith-1.5-35b-a3b.md) | MoE, 3B active | Q4_K_M, 21.7 GB | 128K | ~31 tok/s | ~490 tok/s | 14/18 | 97% | `scripts/llm ornith` |
| [Nemotron 3.5 Bolt 74B](models/nemotron-bolt-74b.md) | hybrid MoE, 9B active | IQ2 (QAD), 22.0 GB | 128K | 14.0 tok/s | 310 tok/s | 15/18 | 91% | `scripts/llm bolt` |
| [Bonsai 2 27B](models/bonsai2-27b.md) | dense 27B, ternary | PTQ1_0, 5.95 GB | 32K | 38.0 tok/s | 277 tok/s | 16/18 | 98% | `scripts/llm bonsai2` |
| [Qwen3-VL-8B](models/qwen3-vl-8b.md) | dense 8B + vision | Q4_K_M + projector | 16K | ~38 tok/s | 830–900 tok/s | 1/18 | 89% | `scripts/llm qwen3vl` |
| [Dory1.0-4B](models/dory1.0-4b.md) | dense 4B, looped | Q5_K_M | 64K (128K ³) | 29.3 tok/s | 852 tok/s | 14/18 | text only | `scripts/llm dory` |
| [Spark-X2.6-4B](models/spark-x2.6-4b.md) | dense 4B | Q8_0, 4.4 GB | 108K | 48–50 tok/s | 1,450–1,620 tok/s | 14/18 | text only | `scripts/llm spark` |
| [LFM3-1.7B](models/lfm3-1.7b.md) | dense 1.7B, conv + attention | Q8_0, 1.85 GB | 128K | **106 tok/s** | **4,802 tok/s** | 12/18 | text only | `scripts/llm lfm3` |
| [LFM3-VL-2B](models/lfm3-vl-2b.md) | dense 2B + vision | Q8_0, 1.85 GB + projector | 32K | 106 tok/s | 4,875 tok/s | 0/18 | 91% | `scripts/llm lfm3vl` |
| [Nemotron 3.5 Bolt E4B](models/nemotron-bolt-e4b.md) | dense 8B (4B eff.) | Q4_K_M, 6.2 GB | 64K | 52 tok/s | 1,531 tok/s | 7/18 | text only | `scripts/llm e4b` |
| [TwIL-8B](models/twil-8b.md) | dense 8B | Q4_K_M (verbose), 5.35 GB | 24K | 38.0 tok/s | 644 tok/s | 8/18 | text only | `scripts/llm twil` |

¹ Processing a ~14–17K-token prompt, which is what a coding agent's first turn costs. "—" = not
measured the same way. Speeds are with the experts split by layer (`NCMOE`); `scripts/llm` runs Qwen3.6, the
distill, Flint, Ornith, Groot, LFM2.5 and Gemma with the [MoE expert cache](how-it-fits.md#12-let-a-cache-pick-the-experts-the-default-for-the-big-moes)
instead: 19–58% faster decode at the same VRAM or less next to a quiet desktop (Qwen3.6 39.4, Gemma 30.0,
Flint 32.1, Ornith 40.4, Groot 47.4, LFM2.5 43.5 tok/s; the distill +34%), about 10% less than that next to a busy one (9–13%).
² 27 tok/s at 128K with a q8_0 KV cache ([how it fits](how-it-fits.md)).
³ With `CTX=131072 LOOP_KV=last_loop`: 30.1 tok/s.

**pibench** = our six-chore agentic benchmark, solved attempts out of 18
([details](pibench.md)). **Vision** = our 12-image probe, overall score
([details](vision-probe.md)).

## Seeing and acting

The [visual control probe](visual-control.md): look at a picture, decide, call a tool. **Decisions** = one picture, one tool call (tanks,
buttons, objects, traffic lights, gauges; 60 scenes, "30" = 150 scenes). **Grid** = steer a robot
to a goal one move at a time, open grids and mazes, 8 episodes each.

| Model | Thinking | Decisions | Per decision | Grid, fresh request per step | Grid, full history | Per grid step |
|---|---|---:|---:|---:|---:|---:|
| [LFM3-VL-2B](models/lfm3-vl-2b.md) | on | **60/60** (30: 144/150 ¹) | 2.9 s | **8/8** open, **3/8** mazes | 6/8, 2/8 | ~43 s |
| LFM3-VL-2B | off | 45/60 (30: 117/150 ¹) | **0.4 s** | 3/8, 0/8 (6/8, 0/8 ¹) | 5/8, 0/8 | **0.7 s** |
| [Qwen3-VL-8B](models/qwen3-vl-8b.md) | (none) | 51/60 (30: 124/150 ¹) | 1.0 s | **8/8**, 1/8 | 4/8, 2/8 | 0.7 s |
| [Gemma 4 26B](models/gemma4-26b-a4b.md) | on | **30/30** (6 per kind) | 17 s | — | — | — |
| [Groot-19B](models/groot-19b-a3b.md) | on | 27/30 (6 per kind) | 16 s | — | — | — |

¹ With a forced tool call (`tool_choice: required`, plus a `no_action` tool to decline).

## What the vendors say

Numbers from the model cards, **not measured by us**, collected 2026-10-23. Three grains of
salt, to taste:

1. **They're for the full-precision models.** We run 1.7–5-bit versions. Only Bonsai 2 publishes
   what its quant keeps: 98.2% of FP16 on a 14-benchmark average.
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
| Qwen3-VL-8B | — | — | 39.3 ˢ | — | BFCL v3 66.3 | [Qwen3-VL card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) (non-thinking; MMLU-Pro 71.6) |
| Dory1.0-4B (2 loops) | 59.5 | 18.1 | — | 72.4 | τ³-bench banking 12.9 | [NVIDIA card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16) (4 loops: 61.8 / 22.2) |
| Spark-X2.6-4B | 62.8 | — | — | 77.1 | τ²-bench 86.9, BFCL v4 71.8 | [XHToken card](https://huggingface.co/XHToken/Spark-X2.6-4B) (thinking high; + SWE-bench Pro 54.3) |

ᵠ Qwen's table. ⁿ NVIDIA's Dory table. ʳ Recursive's Flint chart (measured by Recursive).
ⁿʳ 43 in NVIDIA's table, 44.9 in Recursive's. ᵍ Qwen's card says "GPQA".
ˢ An older LCB v6 window (2025-02 to 05), non-thinking; not comparable with the other LCB numbers.
ᴮ NVIDIA's table for the 2-bit file, against BF16 rivals.

**On paper:** for agentic coding at 128K it's Flint ≥ Qwen3.6 ≫ Gemma 4 26B. Bonsai 2's base is
the strongest model here, but at 32K context. The 4B models claim surprisingly high SWE-bench
Verified scores (Spark 62.8, Dory 59.5); Dory's Terminal-Bench (18–22) hints at how they cope
with long multi-step jobs.

**In practice (pibench):** Gemma 4 26B, the one the paper says is far behind, solved every
chore we gave it. The two 4B models with the high SWE-bench claims, Spark and Dory, both
solved 14/18, and the [Gemma 12B fine-tunes](models/gemma4-12b-agentic.md), sold on coding and agent work, solved none
(both retired since).
Benchmarks measure the model; your laptop measures the model plus quantization
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
| **Qwen3.6-35B-A3B** (baseline) | 22.8 tok/s | 221 tok/s | 128K | 3,558 | 73.4 | 43–45 | 80.4 | 86.0 |
| Gemma 4 26B-A4B | 1.03× | — | 0.25× ¹ | — | −56.0 (Qwen) / −16.0 (NVIDIA) | −8.0 | −3.3 | −3.7 |
| Qwen3.8 distill | 1.15× | 0.76× | 1.00× | **0.16×** | — | — | — | — |
| **Flint** | 1.07× | **2.45×** | 1.00× | 0.43× | **+0.8** | **+10.0** | −0.1 | — |
| Nemotron Bolt 74B | 0.61× | 1.40× | 1.00× | 0.73× | — | **+6.6** ᴮ | — | +3.0 ᴮ |
| Bonsai 2 27B | 1.18× | 1.22× | 0.12× | 0.45× | +2.2 *(base)* | +28.1 *(base)* | +9.7 † | +3.2 † *(base)* |
| Qwen3-VL-8B | 1.67× | 3.76× | 0.12× | — | — | — | — | — |
| Dory 4B (2 loops) | 1.29× | 3.86× | 0.50× | — | −13.9 | −24.9 | — | −12.9 |
| Spark 4B | 2.11× | 6.58× | 0.84× | 1.32× | −10.6 † | — | — | −8.9 † |
| Nemotron Bolt E4B | **2.28×** | **6.93×** | 0.50× | — | — | — | — | — |

¹ At the launcher's 32K default; Gemma 4 26B also runs 128K (27 tok/s with a q8_0 KV cache).
ᴮ From NVIDIA's own table, which includes Qwen3.6 (BF16) next to the 2-bit Bolt.

"Code-check tokens" comes from a small sanity test (write `merge_intervals`, check it against
500 random inputs) that every model passed. It can't rank them, but it does show how chatty each
one's thinking is: 557–784 tokens for the distill, about 1.5K for Flint and Bonsai, ~3.5K for
Qwen3.6, ~4.7K for Spark.

## How they behave

Numbers say how often; this says how. Drawn from reading the pibench transcripts and vision answers of the models
below; the ones added since (Groot, Ornith, LFM2.5-24B, LFM3, LFM3-VL, ThinkingCap) are described on their own
pages, under "How it behaves".
The last column says how much we watched each one, so you know how far to trust the line.

| Model | In a word | Good at | Watch out for | Based on |
|---|---|---|---|---|
| [Flint](models/flint.md) | **careful** | Probes pages before scraping (~3 times per attempt), audits beyond the bug report, checks raw bytes with `xxd`, cleans up after itself | Wanders: the same chore took 6 min once and 13 the next. Can't see images | pibench, 18 attempts |
| [Gemma 4 26B-A4B](models/gemma4-26b-a4b.md) | **thorough** | Thinks long, acts in few steps, chases odd data until it understands it (found the missing charset with `curl -I`) | Fumbled pi's tool format (8 calls without a file path) and recovered every time. Boxes in its own 0–1000 y-first format | pibench, vision |
| [Nemotron Bolt 74B](models/nemotron-bolt-74b.md) | **deliberate** | Very few tool calls and tool errors; tests edge cases nobody asked for; perfect OCR, charts and drawings | Slow (5–10 min a job). Chased a symptom until it timed out; draws boxes about twice as tall as they are | pibench, vision |
| [Qwen3.6-35B-A3B](models/qwen3.6-35b-a3b.md) | **direct** | Read, edit, test, done; little thinking for a big model; perfect vision | Shallow checks (30 quotes with empty authors, "done"); fixes what you report, not always what's hiding; once overwrote a provided file | pibench, vision |
| [Qwen3.8 distill](models/qwen3.8-35b-a3b-distill.md) | **quick** | Among the quickest big models per job; test-driven on code; the tersest thinker on the code check (under 800 tokens) | Writes scrapers from memory without looking (0.1 probes per attempt), so it's wrong in the ways memory is wrong | pibench, vision |
| [Bonsai 2 27B](models/bonsai2-27b.md) | **roomier** | Sound reasoning; 16/18 at 32K, 97% of hidden tests; best at boxes in pixels, as asked | At 16K it was cut off mid-thought in two thirds of its attempts (14/18); even at 32K, long scraping jobs run close to the limit | pibench, vision |
| [Spark-X2.6-4B](models/spark-x2.6-4b.md) | **busy** | Fast; checks its output in 94% of attempts | Thinks a lot and still waves through plausible wrong values (five stars for every book, one quote ten times) | pibench |
| [Nemotron Bolt E4B](models/nemotron-bolt-e4b.md) | **hasty** | Fast (52 tok/s, second only to LFM3); good for short assistant turns | Declares victory on empty output (`{}`, a header-only CSV); skims specs; 5.5 tool errors per attempt | pibench |
| [Qwen3-VL-8B](models/qwen3-vl-8b.md) | **snappy** | Flawless OCR, 5 seconds per image; boxes right in its own format | Doesn't think, so counting slips (12 dots for 11). 16K context. As an agent, 1/18: re-sends the same edit that changes nothing | vision, pibench |
| [Dory1.0-4B](models/dory1.0-4b.md) | **steady** | 14/18 on pibench from 4B, 95% of hidden tests; thinks a lot and runs its tests. More loops for more depth | Slow per job (7.2 min); never solved the books chore (star ratings, character set); its llama.cpp support is an unmerged PR | pibench |
| [TwIL-8B](models/twil-8b.md) | **cramped** | Short code jobs: 7 of 9 code chores, 0/500 wrong on the code check | Thinks so long that its 24K context ran out in 14 of 18 attempts, and 32K didn't help (3/6, twice as slow); 1 of 9 web chores; only fits with a quiet desktop | pibench (one night) |

What separates the top from the middle isn't how much a model thinks or how often it looks: it's
whether it *reads* what it looked at. The full story, with examples, is in
[pibench: what actually happened](pibench.md#what-actually-happened).

## So which one should I use?

| If you want… | Pick | Why |
|---|---|---|
| **An everyday coding agent** | [**Groot-19B-A3B**](models/groot-19b-a3b.md) | 18/18 on pibench, the fastest of the big models with the static split (39.5 tok/s, 887 tok/s prompts), 128K, 3.1 min median per job. `scripts/llm groot` |
| Long agent sessions (128K) | [**Flint**](models/flint.md) | 18/18 on pibench, the best published agentic scores here, 128K, prompts 2.5× faster than Qwen3.6. Can't see images. |
| A dependable all-rounder that sees | [**Gemma 4 26B-A4B**](models/gemma4-26b-a4b.md) | 18/18 on pibench and 100% on the vision probe. Slower per job than the Qwens. |
| Quick turns | [**Qwen3.8 distill**](models/qwen3.8-35b-a3b-distill.md) | One of the quickest big models per job (2.5 min median; Ornith 2.2), 16/18, 128K; with `scripts/llm`'s expert cache it decodes fastest of the big ones. Double-check its scraped data. |
| Careful image reading at 128K | [**Qwen3.6-35B-A3B**](models/qwen3.6-35b-a3b.md) | 100% on the vision probe, 16/18 on pibench. |
| Fast image reading | [**Qwen3-VL-8B**](models/qwen3-vl-8b.md) | 5 s per image, perfect OCR, all on the GPU. Weaker at counting. |
| A small vision model that acts on what it sees | [**LFM3-VL-2B**](models/lfm3-vl-2b.md) | 60/60 one-picture tool decisions with thinking, ~3 s each, 2.6 GB. One fresh question per step; not an agent ([visual control](visual-control.md)). |
| Raw speed on small jobs | [**LFM3-1.7B**](models/lfm3-1.7b.md) | 106 tok/s and 4,800 tok/s prompts in 1.85 GB, 128K, 12/18: chores, not hard bugs. |
| A small model that gets more right | [**Spark-X2.6-4B**](models/spark-x2.6-4b.md) | 48–50 tok/s, 108K, 14/18; careless with details, but twice the E4B's score at nearly its speed. |
| The biggest brain that fits | [**Nemotron Bolt 74B**](models/nemotron-bolt-74b.md) | 74B in 2 bits at 14 tok/s, 128K, images and video; 15/18 and 91% vision. Patience required: 5–10 min per agent job. |
| A snappy local assistant (not a coder) | [**Nemotron Bolt E4B**](models/nemotron-bolt-e4b.md) | 52 tok/s, all on the GPU; 7/18 on pibench, so keep it away from unsupervised coding. |
| The strongest model on paper | [**Bonsai 2 27B**](models/bonsai2-27b.md) | Its base tops every published column. 16/18 on pibench at 32K, all in 7.1 GB on the GPU at 38 tok/s. |
| A small agent with a dial | [**Dory1.0-4B**](models/dory1.0-4b.md) | 14/18 on pibench from 4B, all on the GPU at 64K. Trade speed for depth by changing the loop count. |
