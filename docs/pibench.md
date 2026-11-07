# 🏋️ pibench — six chores, nineteen models, one very warm laptop

Published benchmarks are run by the people who made the model, on big GPUs, at full precision,
with their own scaffolding. We wanted to know something smaller and more personal: **on this
laptop, at the quantization we actually run, through the agent we actually use, can the model
get a job done on its own?**

So we gave every model the same six chores, let it work alone, and graded the results with
tests it never saw.

## The chores

| Task | The job | Graded by |
|---|---|---|
| 🐛 **bug fix** | Two bug reports for a duration parser (`"2h5m"` → seconds). There are more bugs than reported. | 27 hidden tests, including the unreported bugs |
| 🧩 **feature** | Add error handling, a restock report and a CSV round trip to a small `Inventory` class | 8 tests, with commas, quotes and accents in the CSV |
| 🛠️ **CLI** | Write a word-frequency command-line tool from a spec (options, stdin, JSON out, exit codes) | 6 tests against a reference implementation |
| 📚 **books** | Scrape one category of [books.toscrape.com](https://books.toscrape.com), visiting every book's page for stock and UPC | 5 tests: titles, prices, ratings, stock, UPCs |
| 🪄 **JS page** | Scrape three pages of a site whose quotes are drawn by JavaScript | 5 tests: 30 quotes, authors, texts, tags, order |
| 🔐 **login** | Log in through a form with a CSRF token, then grab links that only logged-in users see | 3 tests: 15 authors and their exact links |

The web chores use [toscrape.com](https://toscrape.com), sandbox sites built for scraping
practice, so the answers don't change from one day to the next. Every task was checked both
ways before any model touched it: a reference solution passes every hidden test, and the
untouched starter files fail them.

**The rules:** plain [pi](https://github.com/badlogic/pi-mono) 1.3.0 with no extensions, each
model at its launcher's defaults unless its page says otherwise (Gemma and LFM3-VL ran at 64K, the Gemma 12B agentic
at 32K on our fork), 20 minutes per attempt, **3 attempts per task**. An attempt
counts as **solved only if every hidden test passes**. Close doesn't count; it's that kind of
gym.

## The sandbox (because we let robots run shell commands)

Each attempt runs in a fresh Docker container that holds pi, Python, Playwright and a copy of
that task's files, and nothing else from the laptop. The model server stays on the host GPU,
reachable through a private Docker bridge.

```
host:       llama-server on the GPU, bound to the private bridge only (not the LAN)
container:  pi + Python + Playwright/Chromium · non-root · no capabilities · 4 GB RAM · 4 CPUs
network:    internet ✔   model server ✔   your LAN, router, other host ports ✘
```

A firewall rule blocks the LAN, and the runner refuses to start until a probe from inside the
sandbox confirms it can't reach the laptop's other addresses. Grading happens in a second
container with no network at all. The nuts and bolts (and how to run it yourself) are in
[`bench/README.md`](../bench/README.md).

## The results

| Model | Solved | Hidden tests passed | 🐛 | 🧩 | 🛠️ | 🔐 | 📚 | 🪄 | Median time per attempt | Ran out of context |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 🥇 [**Flint**](models/flint.md) | **18/18** | 100% | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 4.3 min | 0 |
| 🥇 [**Gemma 4 26B-A4B**](models/gemma4-26b-a4b.md) | **18/18** | 100% | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 4.8 min | 0 |
| 🥇 [**Groot-19B-A3B**](models/groot-19b-a3b.md) ² | **18/18** | 100% | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3.1 min | 0 |
| [ThinkingCap Gemma 4 26B](models/thinkingcap-gemma4-26b-a4b.md) ² | 17/18 | 98% | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 | 5.5 min | 0 |
| [Qwen3.8 distill](models/qwen3.8-35b-a3b-distill.md) | 16/18 | 98% | 3/3 | 3/3 | 3/3 | 2/3 | 2/3 | 3/3 | 2.5 min | 0 |
| [Qwen3.6-35B-A3B](models/qwen3.6-35b-a3b.md) | 16/18 | 96% | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 | 3.4 min | 0 |
| [Bonsai 2 27B](models/bonsai2-27b.md) at 32K ⁶ | 16/18 | 97% | 3/3 | 3/3 | 2/3 | 3/3 | 2/3 | 3/3 | 6.0 min | 2 of 18 ⁷ |
| [Nemotron 3.5 Bolt 74B](models/nemotron-bolt-74b.md) ¹ | 15/18 | 93% | 2/3 | 3/3 | 3/3 | 3/3 | 1/3 | 3/3 | 6.4 min | 0 |
| [Ornith-1.5-35B-A3B](models/ornith-1.5-35b-a3b.md) ² | 14/18 | 94% | 3/3 | 2/3 | 3/3 | 2/3 | 1/3 | 3/3 | 2.2 min | 0 ³ |
| [Spark-X2.6-4B](models/spark-x2.6-4b.md) | 14/18 | 96% | 3/3 | 2/3 | 2/3 | 3/3 | 2/3 | 2/3 | 3.3 min | 0 |
| [Bonsai 2 27B](models/bonsai2-27b.md) at 16K | 14/18 | 90% | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 2/3 | 4.9 min | 13 of 18 attempts |
| [Dory1.0-4B](models/dory1.0-4b.md) ⁵ | 14/18 | 95% | 3/3 | 3/3 | 2/3 | 3/3 | 0/3 | 3/3 | 7.2 min | 0 |
| [LFM3-1.7B](models/lfm3-1.7b.md) ² | 12/18 | 88% | 2/3 | 3/3 | 3/3 | 3/3 | 0/3 | 1/3 | 2.2 min | 0 |
| [LFM2.5-24B-A2B](models/lfm2.5-24b-a2b.md) ² | 10/18 | 86% | 0/3 | 3/3 | 2/3 | 2/3 | 0/3 | 3/3 | 4.2 min | 0 |
| [TwIL-8B](models/twil-8b.md) ⁸ | 8/18 | 73% | 1/3 | 3/3 | 3/3 | 0/3 | 0/3 | 1/3 | 8.4 min | 14 of 18 ⁹ |
| [Nemotron 3.5 Bolt E4B](models/nemotron-bolt-e4b.md) ¹ | 7/18 | 78% | 1/3 | 3/3 | 0/3 | 1/3 | 0/3 | 2/3 | **2.0 min** | 0 |
| [Qwen3-VL-8B](models/qwen3-vl-8b.md) ⁵ | 1/18 | 52% | 0/3 | 0/3 | 0/3 | 1/3 | 0/3 | 0/3 | 3.9 min | 0 |
| [Gemma 4 12B agentic](models/gemma4-12b-agentic.md) ⁵ (retired) | 0/18 | 48% | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 20.0 min | 1 |
| [LFM3-VL-2B](models/lfm3-vl-2b.md) ² | 0/18 | 47% | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 20.0 min | 0 ⁴ |
| [VibeThinker 3B](models/vibethinker.md) ⁵ | 0/18 | 29% | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0.4 min | 0 |
| [Gemma 4 12B coder](models/gemma4-12b-coder.md) ⁵ (retired) | 0/18 | 28% | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0.9 min | 0 |

Measured 2026-10-23: 108 attempts, about eight hours of a laptop fan doing its best. ¹ The two
Nemotrons ran the same night, same harness, another 36 attempts (each had one timeout). ² Groot,
LFM2.5, Ornith, ThinkingCap Gemma, LFM3 and LFM3-VL ran on 2026-10-27/28, same harness, 108 more attempts (LFM2.5 shared the laptop
with a llama.cpp build, so its times may be a little high). ³ One timeout: a scraper that paged
forever. ⁴ 14 of 18 attempts hit the 20-minute timeout, every one in a loop of the same command. ⁵ Dory, Qwen3-VL,
both Gemma 12B fine-tunes and VibeThinker 3B ran on 2026-11-02, on our fork after the PrismML merge (`7ff495f`), same
harness, 90 more attempts. VibeThinker went through its [tool-call proxy](models/vibethinker.md). ⁶ The same day,
same fork: Bonsai with the q8_0 KV cache that lets it [fit 32K](../bench/bonsai-ctx/README.md); the 16K row ran on
PrismML's own build. ⁷ Both had already passed every test. ⁸ 2026-11-07, same fork (`69ca796`), same harness, 18 more
attempts, at its launcher's 24K with a q4_0 KV cache. ⁹ Six of them had already passed every test.

"Hidden tests passed" has a floor: the untouched starter files already pass 12 of the bug fix's 27 tests and 3 of
the feature's 8. The Gemma 12B coder never changed a file, so its 28% is that floor.

**Read with care.** Three attempts per task means a single lucky or unlucky attempt moves a cell
by a whole point, and these chores take minutes, not hours. This tells you who is dependable
on small, self-contained jobs. It doesn't tell you who will refactor your monorepo.

## Each model in one breath

- **Flint** — the careful one. Probes, tests, audits beyond the bug report, cleans up its scratch
  files. Its one vice is wandering: the same scraping chore took 6 minutes once and 13 the next.
- **Gemma 4 26B** — the surprise. Weak published agentic scores, perfect here. Thinks long, acts
  in few steps, chases odd data. The only one of the first batch to fumble pi's tool format (8 calls without a
  file path), and it recovered every time.
- **Qwen3.8 distill** — the sprinter. Among the quickest big models per attempt, test-driven on code, writes scrapers
  from memory. Double-check its data.
- **Qwen3.6** — direct and quick, checks shallowly. Once fixed only the reported bugs and missed
  the hidden one; once overwrote a provided input file without asking.
- **Spark** — the overthinker. Busiest, chattiest, checks its output most often, and still waves
  through plausible-looking wrong values. Not slow either (3.3 min a job), so its mistakes at
  least arrive quickly.
- **Nemotron Bolt 74B** — the slow thinker. 14 tok/s and long reasoning, so 5–10 minutes a job.
  Probes pages, tests edge cases nobody asked for, and still fell into the charset trap twice
  (once chasing the symptom until it timed out).
- **Nemotron Bolt E4B** — the speedster who doesn't check. 52 tok/s, two minutes a job, and the
  only model to announce "task complete" over an empty `{}`. NVIDIA built it for voice assistants;
  it shows.
- **Bonsai 2** — big brain, small desk. At 16K the context cut it off before it could finish
  thinking (14/18); at 32K it solved 16/18, level with Qwen3.6 and the distill.
- **Groot** — the third perfect score, and the quickest of the three (3.1 min a job). Qwen3.6
  with half its experts pruned, and on these chores it lost nothing.
- **LFM2.5** — the literalist. Fixed exactly the two reported bugs and stopped (0/3), and read
  the books site's UTF-8 as Latin-1 every time (`Noahâ\x80\x99s Ark`), printing the mangled
  titles and moving on (0/3). Everything else, near perfect.
- **Ornith** — the quick-draw. The fastest median job of the big models (2.2 min) and the fewest tokens, but it
  guesses: once it invented the books site's URL scheme and paged forever. Its card's agentic
  scores beat Qwen3.6 by ten points; here it finished two attempts behind.
- **ThinkingCap Gemma** — Gemma on a diet. Half the thinking of the base Gemma for one attempt
  fewer solved; slower per job only because its GGUF decodes at 17 tok/s here.
- **LFM3-1.7B** — the pocket rocket. 106 tok/s, entirely on the GPU, and 12/18 from 1.7B
  parameters: more than its 24B sibling. Talks a lot (12K tokens a job) and is sloppy with data
  (an unquoted CSV, literal `\u00e9`), but the code chores are nearly all there.
- **LFM3-VL-2B** — the broken record. The same backbone plus vision, the same chat template, and
  0/18. In 14 attempts it got stuck running one command over and over until the 20-minute
  timeout (769 times out of 772 calls, once); LFM3-1.7B never repeated one more than 25 times.
  Its card says as much: not recommended for coding or multi-turn conversations. Use it as an
  eye, not as an agent.
- **Dory1.0-4B** — the slow, steady small one. 14/18 from 4B parameters, the same as Spark, taking
  twice as long per job (7.2 min vs 3.3). Thinks a lot, tests its code, and never solved the books chore.
- **Qwen3-VL-8B** — the eye that edits in circles. One perfect one-shot scraper, then the same
  no-op edit again and again.
- **Gemma 4 12B agentic** — all effort, no finish. Busy, slow on this card at 32K, and sloppy with
  tool arguments.
- **Gemma 4 12B coder** — a chat coder. Writes the answer, doesn't use the tools.
- **TwIL-8B** — a mathematician in a cupboard. It thinks at length (26K characters a job) and ran out of its 24K
  context in 14 of 18 attempts, six of them after the work was done. The code chores mostly went fine (7 of 9); the
  web chores didn't (1 of 9): it fetched raw pages and thought about each at length until the room ran out, and
  never even wrote the books scraper. No tool-use training, yet 237 tool calls went through; twice it wrote one inside
  its thinking, where pi can't see it, and the attempt ended.
- **VibeThinker 3B** — a one-shot solver, as advertised. Not an agent, even through the proxy.

The run-by-run notes, with transcript paths and quotes, are in
[`bench/results/notes.md`](../bench/results/notes.md).

## What actually happened

This section is the first batch (2026-10-23): Flint, Gemma 4 26B, the distill, Qwen3.6, Bonsai 2 at
16K and Spark, plus the two Nemotrons the same night. We read every one of the first six's transcripts
(108 of them, so you don't have to). The interesting part isn't who won, it's *how* they won and lost.
The later models are in [Each model in one breath](#each-model-in-one-breath) and
[Five more](#five-more-and-only-one-of-them-an-agent).

### The one rule that explains almost everything

Every model ran its own code. Every single time. What separated a pass from a fail was
whether the model **looked at the values it produced**, or just checked that a file existed.

The failures, almost without exception, were results the model printed and then strolled past:

- Qwen3.6 scraped 30 quotes, checked that each had an `author` key, and declared victory. Every
  author was an empty string.
- The distill collected 15 "(about)" links instead of 15 Goodreads links, printed them, counted
  them, and moved on. Its final message called them "(about) / Goodreads page links".
- Spark gave every book five stars (it counted the star icons, and every book has five icons;
  the real rating hides in a CSS class name), and in another run returned one quote ten times
  per page thanks to a loop that read `page` instead of `quote`.
- The distill and Bonsai both let `requests` mangle "Noah's" into "Noahâs", because the site
  doesn't declare its character set.

The two perfect scorers reacted to what they saw. Flint checked its CSV's raw bytes with `xxd`.
Gemma spotted the mangled title in its own output, grepped the live page for it, found the
missing charset with `curl -I` and fixed the decoding. Same trap, opposite ending.

### How they work, side by side (first batch)

Averages per attempt for the eight first-batch models, from the transcripts (`python3 bench/trace.py --stats`):

| Model | Solved | Tool calls | Thinking (chars) | `curl` probes | Test runs (`pytest`) | Tool errors | Looked at its output |
|---|---:|---:|---:|---:|---:|---:|---:|
| Flint | **100%** | 16.3 | 11.6K | **2.8** | 1.1 | 1.4 | 83% |
| Gemma 4 26B | **100%** | 12.2 | 12.1K | 1.4 | 0.9 | 1.4 | 67% |
| Qwen3.8 distill | 89% | 13.0 | 4.4K | 0.1 | **1.5** | 1.8 | 50% |
| Qwen3.6 | 89% | 13.8 | **3.2K** | 1.3 | 1.1 | 1.8 | 72% |
| Nemotron Bolt 74B | 83% | **11.0** | 7.0K | 1.1 ¹ | 0.7 | **1.1** | 72% |
| Spark 4B | 78% | 18.9 | 17.2K | 1.4 | 1.3 | 2.8 | **94%** |
| Bonsai 2 27B | 78% | 12.2 | 13.6K | 1.8 | 0.3 | 1.8 | 67% |
| Nemotron Bolt E4B | 39% | 15.4 | 15.4K | 0.0 ¹ | 1.4 | 5.5 | 67% |

¹ Both Nemotrons probe pages with little `python3 -c "requests…"` one-liners rather than `curl`,
which this column doesn't count.

"Looked at its output" means that, after writing files, it opened or printed one at least once
(share of attempts). As the next
sections show, *looking* isn't the same as *reading*: Spark looked the most and still waved
through wrong values.

### Look first, then write

Flint fetched and grepped the pages before writing any scraper (about three probes per
attempt, the most of the first batch). The distill almost never looked (0.1 probes per attempt) and wrote
scrapers from memory: blazingly fast and usually right, and wrong in exactly the ways memory is
wrong, like which of two links is which.

### More thinking ≠ more passing

Spark reasoned a lot (about 17,000 characters of thinking per attempt) and solved 14/18. The
distill thought little (about 4,400) and tied Qwen3.6. Flint and Gemma think a
lot *and* look at the data, which seems to be the winning combination. The Nemotrons make the
same point from both ends: the E4B thought the second most and came last, while Bolt 74B thought
less than half as much and solved 15.

### Bigger isn't automatically better

Bolt 74B has roughly twice the total parameters of anything else here, and more than twice the
active parameters of the other MoEs. It finished fifth of the first batch's eight: careful, tidy (the fewest tool calls and tool errors), and done in
by the same unlabelled character set that tripped the distill and Bonsai. Its 2-bit quantization
may cost it something we can't separate from the model itself. What the size *does* buy is
breadth (text, images and video, 128K), at a price: 6.4 minutes per attempt, the slowest of the first batch (Dory, at 7.2, is the only slower one
that finishes on time).

### Context is a capability

No model with 64K or more of context ran out of it. Some attempts ran out of *time* (one each for
Bolt, E4B, Ornith and Qwen3-VL, 11 for the Gemma 12B agentic, 14 for looping LFM3-VL), which is a
different story. Bonsai 2, at 16K,
got cut off mid-thought in two thirds of its attempts; nine of those still passed because the
files were already written. On a longer job, that's a job half done. Given 32K on the merged fork, it solved
16/18: the books chore went from 0/3 to 2/3 and the JavaScript page from 2/3 to 3/3, and only 2 attempts filled
the context, both after they had passed.

### The code chores barely separate them (first batch)

Bug fix, feature and CLI: 51 of 54 attempts solved across the first batch's six models (63 of 72 with the
two Nemotrons, the E4B accounting for most of the misses). The three web chores decided the
ranking: 45 of 54 (55 of 72). Real-world data is where agents trip.

## Five more, and only one of them an agent

The 2026-11-02 batch shows that a tool-calling template doesn't make a model an agent.

- **Dory** (4B) solved 14/18, tied with Spark and Bonsai, and passed 95% of the hidden tests. It thinks
  hard (31K characters a job; among models that solved at least half their jobs, only LFM3-1.7B thinks more) and
  takes its time (7.2 min a job). It fell into both of the books site's traps: five stars for every book in two attempts,
  "Noahâ\x80\x99s" in one, and in the third a curly apostrophe turned into a straight one, so the titles didn't match.
- **Qwen3-VL** solved one job, the login, with one script in 18 seconds. Elsewhere it got stuck
  re-sending the same edit: 181 of its 254 `edit` calls changed nothing.
- **The Gemma 12B agentic** worked hard and finished nothing. It made 31 tool calls a job, and 57 of
  them were missing a file path. At 13 tok/s on our fork (part of the model in
  RAM), 11 of its 18 jobs ran out of time. It came within one test of the bug fix (26/27), and in at least one job
  spent the rest of its time writing the same three lines over and over inside one `write`.
- **The Gemma 12B coder** never touched a file. It made 6 tool calls in 18 jobs and put its code in
  the reply instead, so nothing was saved and nothing was tested.
- **VibeThinker 3B** did the same, as its page predicts: 5 tool calls in 18 jobs, the rest
  answers in chat, and some of its calls were JSON with broken escaping that even the proxy can't repair.

## Same scores, less waiting: the MoE expert cache (2026-11-04)

The three MoE models ran the whole bench again with a GPU cache of recently used experts instead of fixed
layers ([how-it-fits §12](how-it-fits.md#12-let-a-cache-pick-the-experts-the-default-for-the-big-moes)),
same VRAM. Scores held (Qwen3.6 17/18, Gemma 26B 17/18 with the miss at 5 of 6 hidden tests, Flint 18/18)
and decode inside the sessions got faster: Qwen3.6 +59%, Gemma +26%, Flint +23%. The cache is now the
default in `scripts/llm` for qwen36, distill, flint, ornith, groot, lfm25 and gemma (default build `moe-cache-max-batch`);
the scoreboard above keeps the original runs.

## Three bugs we found in our own setup

Honesty corner. The first time Spark ran, it scored 0 on everything in 20 seconds flat: pi asks
for "medium" reasoning effort by default, and Spark's chat template only accepts "high" or "low",
so every request bounced with HTTP 400. We threw those results away, re-ran Spark with
`--thinking high` (what its model card uses), and taught the runner to stop instead of scoring
an attempt where the model never produced a single token.

And one of Bonsai's attempts wrote an empty CSV that still passed four of five tests, because
those tests looped over the rows and an empty loop passes. We tightened the tests and re-graded
all 108 saved attempts. Only that one score changed.

The third was in VibeThinker's tool-call proxy. Its first run crashed the proxy once, on a call
written OpenAI's way (`{"type": "function", "function": {...}}`), which pi saw as a dropped
connection; and the proxy sent no token counts, so every attempt showed 0 output tokens. We fixed
both, added the crashing form to the proxy's tests, and re-ran all 18 attempts. Same score, 0/18.
