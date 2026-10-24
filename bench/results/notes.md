# pibench — qualitative notes

Running notes taken while the benchmark ran (2026-10-23), from the transcripts
(`bench/trace.py RUN_DIR` for one run, `bench/trace.py --stats` for habits per model). Run paths
are under `bench/results/raw/` (git-ignored, kept locally). Observations, not scores: the scores
are in `summary.md`.

## What to look for in each model

- **Plan vs. probe:** does it inspect the site/code before writing, or write first from memory?
- **Self-checking:** does it run its code, and does the check actually test the output's content
  (not just "the file exists / the keys are there")?
- **Scope:** does it stay inside the task (no rewriting provided inputs, no stray files)?
- **Cost:** tokens, tool calls and time per solved task; how it spends them (thinking vs. trying).
- **Failure shape:** confident and wrong, stuck in a loop, gives up, times out?

## Flint (UD-Q3_K_XL) — 18/18 solved

- **Probes before it writes.** On the web tasks it fetches the pages with curl, greps the HTML for
  the classes it needs, and only then writes the scraper (≈3 curl calls per run on average, the
  most of any model so far). On code tasks it reads the files first, then thinks at length
  (one 7K-character reasoning block before the first edit of `bugfix-durations` r1).
- **Audits beyond the report.** In `bugfix-durations` it found the hidden bugs (leftover text,
  `0 → "0s"`) unprompted: all three runs 27/27. Its final message lists the extra bugs it found.
- **Checks content, not just existence.** After scraping it re-reads `travel.csv` with `csv.reader`,
  counts rows, checks encodings with `xxd`; on `login-goodreads` it asserts every value starts with
  `http://go…`. It also deletes its own scratch files (`explore.py`, `explore2.py`) before finishing.
- **Thorough to a fault on the web tasks.** `scrape-books` r3 took 12.7 min and 40 tool calls:
  it fetched the site's CSS to understand the star rating, compared two downloads of the same page
  with `diff`, wrote two probe scripts to `/tmp`. Correct, but 2× the time of its first run on
  the same task. Variance between runs of the same task is large (6.3 vs 12.7 min).
- **The most reasoning of all models so far:** ~11.6K thinking characters and ~5.7K output tokens
  per run, about 1.6× Qwen3.6.

## Qwen3.6-35B-A3B (UD-IQ4_XS) — 16/18 solved

- **Fast and direct.** Fewer tool calls (≈12/run) and tokens (≈3.6K/run) than Flint; on the code
  tasks it reads, edits, writes tests, runs pytest, done (`feature-inventory` r1: 8 calls, 3.6 min).
- **Writes from memory on the web tasks.** In `scrape-js-quotes` r1 it wrote the Playwright
  scraper without looking at the page, guessed the author selector (`span.small.text` instead of
  `small.author`), and every author came out as an empty string. Its "verification" checked that
  30 items with the keys `text`/`author`/`tags` existed, which they did, so it declared success
  after 74 s. **Failure shape: confident and shallow.**
- **Fixes what was reported, not what was hidden.** `bugfix-durations` r2: fixed both reported
  bugs, missed the leftover-text check (22/27); r1 got it (27/27). Flint found it every time.
- **Touches provided inputs.** `cli-wordfreq` r1 overwrote the provided `sample.txt` with its own
  text before testing (harmless here, since the hidden test reads whatever `sample.txt` is there,
  but in a real repo that is an unrequested change).

- Rounds 2–3 were clean apart from `bugfix-durations` r2: the two failures above are its only ones.
  Median 3.4 min per run vs Flint's 4.3.

## Qwen3.8-35B-A3B distill (empero-ai, IQ4_XS) — 16/18 solved

- **The fastest model so far:** median 2.5 min per run. Short reasoning between tool calls (most
  blocks under a few hundred characters), and almost no site inspection: 0.1 curl calls per run,
  against Flint's 2.8.
- **Writes from memory, and it often works.** `scrape-js-quotes` r1: checked Playwright was
  installed, wrote the scraper with the right selectors without looking at the page, ran it,
  printed a sample, done in 48 s and 4 tool calls (5/5).
- **Test-driven on code tasks.** It runs pytest more than the others (1.5 times per run), adds its
  own cases, and on `bugfix-durations` r1 used failing tests to find the hidden repeated-unit and
  leftover-text bugs (27/27 in all three runs).
- **Both failures are values it printed and did not read:**
  - `login-goodreads` r1: logged in correctly, then collected the "(about)" links
    (`/author/Albert-Einstein`) instead of the Goodreads links. It `cat`-ed `authors.json`, which
    showed the wrong URLs, and then checked only the number of keys. Its final message calls them
    "(about) / Goodreads page link".
  - `scrape-books` r1: the site does not declare a charset, so `requests` decoded the page as
    Latin-1 and one title came out as `Noahâs`. Everything else was right; that one title fails
    3 of the 5 tests, because they look books up by title. Flint checked its CSV bytes with `xxd`
    in the same task.
- Checked the *content* of its output in only half of its runs (Flint 0.83, Qwen3.6 0.72).

## Spark-X2.6-4B (Q8_0, `--thinking high`) — 14/18 solved

- **Setup note:** the first pass was invalid. pi requests reasoning effort "medium" by default and
  Spark's chat template rejects everything but high/low, so all 18 requests were HTTP 400 with no
  output. Discarded; re-run with `--thinking high`, the card's own evaluation setting.
- **The busiest model:** most tool calls (≈19/run), most output (≈9K tokens/run) and by far the
  most reasoning (≈18K thinking characters/run, 1.6× Flint) — yet a median of 3.3 min per run,
  because it generates at ~48 tok/s.
- **Checks often, but checks the wrong thing.** It inspected its own output in more runs than any
  other model, yet all four failures were wrong values that looked plausible:
  - `scrape-books` r1 (8.4 min, 56 tool calls): counted the `<i class="icon-star">` elements to get
    the rating. Every book has five; the real rating is in the class name (`star-rating Two`) and
    CSS colours the stars. Every book came out 5 stars and it did not find that odd.
  - `scrape-js-quotes` r2: a loop bug (`extract_quote(page)` instead of `extract_quote(quote)`)
    repeated each page's first quote 10 times. 30 entries, mostly Albert Einstein; finished in
    67 s without looking.
  - `cli-wordfreq` r1: implemented `--ignore` as a repeatable single-word flag although the spec
    (which it copied into its own docstring) says `--ignore WORD [WORD ...]`.
  - `feature-inventory` r2: restock report not sorted by quantity, as specified.
- **Strong on the code-reading task:** `bugfix-durations` 3/3 with every hidden bug found, like
  Flint. Its failures are in specs and data, not in debugging.

## Gemma 4 26B-A4B QAT (UD-Q4_K_XL, 64K, q8_0 KV) — 18/18 solved

- **The surprise of the run: a perfect score**, tied with Flint, although Qwen's published table puts
  it far behind on agentic coding (SWE-bench Verified 17.4 vs Qwen3.6's 73.4). These tasks are
  small, so this says "reliable at short agentic jobs", not "as strong as Qwen3.6 on SWE-bench".
- **Slowest per run** (median 4.8 min, ~30 tok/s) with long reasoning before acting (≈12K thinking
  characters per run, about Flint's level) but the **fewest tool calls** (≈12/run).
- **Notices and chases data problems.** In `scrape-books` r1 it spotted the garbled "Noah’s" in its
  own CSV, grepped the live page for it, checked the response headers with `curl -I` (the missing
  charset the distill tripped over), fixed the decoding and re-checked. On `scrape-js-quotes` it
  wrote a throw-away `check_structure.py` to inspect the rendered page first, then deleted it.
- **Clumsy with pi's tools, but recovers.** The only model with malformed tool calls: 8 in 18 runs,
  all `edit`/`write` calls missing the `path` argument (pi rejects them with a validation error).
  Each time it fell back to rewriting the whole file with `write`. It also creates
  `test_durations_new.py` instead of editing the given test file.

## Bonsai 2 27B (ternary PTQ1_0, 16K context) — 14/18 solved

- **The 16K context is the whole story.** 12 of its 18 runs ended because the conversation hit the
  context limit (`stopReason: length`), and one more was rejected outright when a tool result pushed
  the request to 22,482 tokens (HTTP 400, over 16,384). Only 5 runs finished on their own.
- **Nine of those truncated runs still passed**, because it had already written the deliverable when
  the context ran out. Typical: `feature-inventory` r1 thought for 22K characters in one block, wrote
  the class and its own tests, ran pytest, saw a failure, and was cut off before it could react. The
  hidden tests happened to pass. On a longer job this is an unfinished job.
- **Fails where pages are big:** `scrape-books` 0/3 (one context overflow, one garbled "Noah’s"
  plus a cut-off before it could notice, one crashed script and an empty CSV) and
  `scrape-js-quotes` r3 (cut off while comparing the rendered DOM with the page's embedded JSON).
- **Strong reasoning when it fits:** all code tasks 3/3, and its exploration of the sites was sound.
  It runs pytest the least of all models (0.3 per run), probably because there is no room left.
- Its base model is the strongest on paper here; on this rig the 16K window, not the ternary
  weights, is the limit. A larger context would need a smaller KV cache than the fork allows on
  Turing (q8_0 aborts; see `docs/models/bonsai2-27b.md`).

## Nemotron 3.5 Bolt 74B-A9B (NVIDIA "24GB" IQ2, 128K) — 15/18 solved

Run separately from the other six (2026-10-23 evening), same harness and settings. Median 6.4 min per
attempt (the slowest model, ~14 tok/s), 2.4 h for all 18; one timeout, no context overflows.

- **Probes first, with Python instead of curl.** On the web tasks it inspects pages with short
  `python3 -c "requests … BeautifulSoup …"` snippets (four of them before writing the login
  script), then writes the deliverable. (`trace.py --stats` counts only curl, so its "curl" number
  understates how much it looks.)
- **Tests edge cases nobody asked for.** `cli-wordfreq` r1: after the spec'd options it tried an
  empty file, `--top 0`, `O'Brien`, digits inside words and upper-case `--ignore` words.
  `feature-inventory` r1 checked names with embedded newlines in the CSV round trip. Both passed.
- **Found the hidden bugs** in `bugfix-durations` r1 (27/27) by iterating on its own failing tests;
  in r2 it fixed the two reported bugs and one more, but missed the leftover-text check (22/27:
  `"5x"`, `"1h30x"`, `"abc"` accepted), the same hidden bug Qwen3.6 missed once.
- **The charset trap got it too.** `scrape-books` r1: it printed `travel.csv`, which showed
  "Noahâs", re-read it with `csv.DictReader`, and declared it complete. Same miss as the distill
  and Bonsai, opposite of Flint and Gemma: the value was on screen and not read.
  In r2 it **chased the symptom instead of the cause**: the same mis-decoding turned "£" into
  "Â£", which broke its price parser; it patched the parser twice, decided "the price contains a
  non-breaking space", and ran out of the 20-minute budget mid-fix (the only timeout of the whole
  benchmark so far; the CSV on disk still had the garbled title). The root cause, one missing
  `response.encoding = "utf-8"`, was never considered. In r3 it did set `resp.encoding = "utf-8"` and solved it
  (14.8 min, 24 tool calls): 1 of 3 on the task every careful model got right.
- **Slow and thoughtful:** 5–10 min per attempt (about 2× the 35B MoEs; it generates at ~14 tok/s),
  with single reasoning blocks up to 8K characters before writing a file, yet the JS-quotes task
  took 2.8 min and 1K tokens: it knew the page and wrote the scraper straight away.

- **Vision probe: 91% overall.** OCR, understanding (chart, counting, grid) and CAD all perfect,
  including the dot count that tripped Qwen3-VL and the distill. Bounding boxes are its weak spot
  (21% in pixels as asked, 64% in its best format): it mixes conventions between images, and in
  pixels its boxes get x and the top edge right but run far too low (heights roughly 2× too tall),
  as if it measured height on another scale. Slowest at it too: 68–667 s per image (median 115 s),
  up to 6K reasoning tokens on the counting and CAD items, with the projector on the CPU.

## Nemotron 3.5 Bolt E4B-4GB (NVIDIA Q4_K_M, 64K, all on GPU) — 7/18 solved

Run after Bolt (2026-10-23/24), same harness. The fastest model (52 tok/s, 1,531 tok/s prompt,
4.4 GB VRAM) and by far the least reliable: 7 of 18, 78 % of hidden tests. NVIDIA pitches it at
voice/device assistants, not coding; this is what happens when you hand it a coding agent anyway.

- **Declares victory on empty output.** `login-goodreads` r1: `authors.json` was `{}`; it read the
  file and wrote "The task is complete". `scrape-books` r1 and r3: a CSV with only the header row;
  r2: no CSV at all (it once fetched `/work/https:/books.toscrape.com` as a local path). 0 of 3.
- **Skims the spec.** `cli-wordfreq` 0/3: kept the apostrophes around `'quoted'` although the spec
  uses exactly that example, never tested it, done in 1.2–1.5 min. `scrape-js-quotes` r1 kept the
  curly quotes the prompt said to strip, after reading a `quotes.json` where every line starts with
  them (r2 and r3 fixed it, in under a minute each).
- **Fumbles the tools more than anyone:** 5.5 tool errors per attempt (next worst Spark, 2.8):
  23 `edit` calls whose old text didn't match the file, 21 malformed tool calls, 11 crashes of its
  own scripts.
- **Wildly inconsistent.** The same task swings from solved to failed: `bugfix-durations` 27/27 in
  3.8 min (all hidden bugs), then 20/27 and 22/27; `login-goodreads` 1/3, 0/3, then 3/3 in 54 s.
  And once it didn't know when to stop: `feature-inventory` r2 was correct early but kept going for
  36K tokens until the 20-minute limit (still counted solved).
- **Thinks a lot for a small model** (~15K reasoning characters per attempt, close to Spark), but
  the reasoning doesn't turn into checking.
- Code check (`qwen-codecheck.py`): answered without a code block, so nothing to run.

## Cross-model observations

- **The pass/fail line is whether the self-check reads the values.** Every model runs its code;
  the ones that failed had printed the wrong result and moved on: empty authors (Qwen3.6), the
  "(about)" links (distill), five stars for every book and a quote repeated ten times (Spark), a
  garbled title (distill, Bonsai). The two perfect models reacted to what they saw: Flint checked
  CSV bytes with `xxd`, Gemma grepped the live page for the odd character and fixed the charset.
- **Probing beats recalling on the web tasks.** Flint (2.8 curl calls/run) and Gemma looked at the
  pages before writing scrapers; the distill (0.1) wrote from memory, which is fast and usually right,
  and wrong in the ways memory is wrong (which link is which).
- **The code tasks barely separate them.** `bugfix-durations`, `cli-wordfreq` and `feature-inventory`:
  51 of 54 runs solved across the first six models (Qwen3.6 missed one hidden bug, Spark two spec
  details); 63 of 72 with the Nemotrons (E4B 4/9, Bolt 8/9). The three web tasks decide the
  ranking: 45 of 54 (55 of 72).
- **More thinking did not buy more passes.** Spark thought the most (17K characters/run) and tied for
  last; the distill thought the least of the 35B models and tied Qwen3.6. Flint and Gemma think a
  lot *and* look at the data. The E4B thought the second most (15K) and came last; Bolt thought
  less than half as much (7K) and solved 15.
- **Size is not the whole story, but the floor is real.** The two 4B-class models (Spark, E4B)
  are the fastest and the least careful; the 2-bit 74B Bolt is the slowest and lands mid-table.
- **Context is a capability.** Every model with ≥64K context finished every run on its own; the
  16K model was cut off in two thirds of its runs.
- **Speed vs. reliability:** distill 2.5 min/run (89%), Spark 3.3 (78%), Qwen3.6 3.4 (89%), Flint
  4.3 (100%), Gemma 4.8 (100%), Bonsai 4.9 (78%), E4B 2.0 (39%), Bolt 6.4 (83%).
