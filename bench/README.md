# pibench — a toy agentic benchmark, run in a sandbox

Six small, realistic jobs for a coding agent. Each model drives plain **pi 1.3.0** (no
extensions) on its own, with no human in the loop, and is scored by **hidden tests** it never
sees. Results go into [`docs/scoreboard.md`](../docs/scoreboard.md).

## The tasks

| Task | What the agent must do | Hidden tests |
|---|---|---|
| `bugfix-durations` | Fix a duration parser/formatter from two bug reports; more bugs are hiding | 27: parse/format cases, error handling, a round trip |
| `feature-inventory` | Add error handling, a restock report and a CSV round trip to a small class | 8, incl. commas, quotes and non-ASCII names in CSV |
| `cli-wordfreq` | Write a word-frequency CLI from a spec (options, stdin, JSON output, exit codes) | 6, compared with a reference implementation |
| `scrape-books` | Scrape one category of books.toscrape.com, following each book's page | 5: titles, prices, ratings, stock counts, UPCs |
| `scrape-js-quotes` | Scrape 3 pages of a **JavaScript-rendered** site (needs a browser, or reading the page's JS) | 5: 30 quotes, authors, texts, tags, order |
| `login-goodreads` | Log in through a form with a CSRF token, then scrape links only visible when logged in | 3: 15 authors and their exact links |

The web tasks use [toscrape.com](https://toscrape.com), sandbox sites built for scraping
practice, so their content does not change between runs. Every task was checked both ways: a
reference solution (`tasks/*/solution/`, never shown to the model) passes all hidden tests, and
the untouched starter files fail them.

**A run counts as solved only if every hidden test passes.** Each run has 20 minutes. Every
model × task runs 3 times, because at temperature 1.0 one run is noise.

## The sandbox

```
host:       llama-server on the GPU, bound to 172.30.0.1:8080 (a private Docker bridge, not the LAN)
container:  pi + Python + Playwright/Chromium, one fresh container per run
            /work = a copy of the task's starter files; nothing else from the host
network:    internet yes; the LAN, the router, other Docker networks and every other host port: no
```

- pi runs **inside** the container, so its bash/edit/write tools can only touch the container
  and that run's `/work`. Non-root user, all capabilities dropped, `no-new-privileges`, 4 GB
  RAM, 4 CPUs, 512 processes.
- `firewall.sh` (root, run once per boot) adds the iptables rules that block private address
  ranges and every host port except 8080 for the `br-pibench` bridge.
- `run-bench.sh` refuses to start unless a probe from inside the sandbox **cannot** reach a
  listener on another host address but **can** reach the internet.
- Scoring happens in a second container with **no network**, on a copy of `/work`, with the
  model's own `conftest.py`/pytest config ignored.

## Running it

```bash
docker build -t pibench:1 bench/
docker network create --subnet 172.30.0.0/24 --gateway 172.30.0.1 \
  -o com.docker.network.bridge.name=br-pibench pibench
sudo sh bench/firewall.sh add                 # once per boot; `remove` undoes it
bash scripts/stop-server.sh                   # the benchmark needs the GPU to itself
bash bench/run-bench.sh                       # the default six: flint qwen36 distill spark gemma26b bonsai2, ~6–12 h; resumable
MODELS="flint qwen36" REPEATS=1 bash bench/run-bench.sh   # a subset
MOE_CACHE=auto MODELS=groot bash bench/run-bench.sh      # with the launcher's expert cache (the default is the static split)
python3 bench/summarize.py                    # -> bench/results/summary.md, results.csv
```

To repeat a batch under other settings without touching the main results, use `RAW`, `TEMP` and
`EXTRA_ARGS` (`bash bench/run-bench.sh --help`), as in [Reproducing our results](../docs/reproduce.md).

Each run keeps its full transcript in `bench/results/raw/<model>/<task>/r<N>/` (`pi.jsonl`,
the final `/work`, `junit.xml`, `score.json`); those are git-ignored, the summary is not.

## Limits

Six tasks are a smoke test, not a leaderboard: 3 runs per task give each model 18 attempts, so
a difference of one or two solved runs is noise. The tasks are small (minutes, not hours) and
favour careful, test-checking agents over raw knowledge. Every model gets the same plain pi,
with no extensions.

## The other benchmarks in `bench/`

pibench is the big one; the rest each have their own folder and README, and
[Reproducing our results](../docs/reproduce.md) lists the command behind every number.

| Folder | What it measures |
|---|---|
| [`vision/`](vision/README.md) | the vision probe: OCR, boxes, charts, counting, CAD drawings |
| [`visual-control/`](visual-control/) | look, decide, call a tool ([docs](../docs/visual-control.md)) |
| [`moe-cache/`](moe-cache/README.md) | the MoE expert cache: decode, prompts, agent loops, daytime sizes |
| [`smoke/`](smoke/README.md) | quick checks for the helper models (decisions, embeddings, boxes) |
| [`voice/`](voice/README.md) | LFM3-Audio: speech in, speech out, spoken tool calls, realtime |
| [`recheck/`](recheck/README.md) | every chat model launches: speed, a tool call, the code check |
| [`bonsai-ctx/`](bonsai-ctx/README.md) | how much context Bonsai 2 fits |
| [`gemma12-probe/`](gemma12-probe/README.md) | why the Gemma 12B models failed: the models or our setup |
| [`cactus/`](cactus/README.md) | Cactus's Mirage, offline, on the visual control scenes |
| `finetune/`, `imagegen/`, `steering/`, `retrieval/`, `grading/`, `relaxed/`, `beyond-vision/` | fine-tuning, pictures, steering vectors, search, judging transcripts, relaxed speculative acceptance; see [fine-tuning](../docs/fine-tuning.md), [making pictures](../docs/image-generation.md) and [reproduce.md](../docs/reproduce.md) |
