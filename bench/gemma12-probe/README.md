# 🔬 Gemma 12B probe: the models, or our setup?

Both Gemma 4 12B fine-tunes scored 0/18 on [pibench](../../docs/pibench.md) on 2026-11-02, while Gemma 4 26B scored
18/18 through the same harness. Their transcripts had two symptoms that could have been ours rather than theirs:

- the **coder**'s reply after every tool result arrived empty (1–2 tokens, or 264–391 tokens with no visible text);
- the **agentic** model's `write`/`edit` calls often had no `path`, and some argument names were mangled
  (`content=<|"|>import csv…`, `<channel|><|tool_call>call`, hex junk), with a repetition penalty of 1.1.

## How

1. `logproxy.py` sits between pi and llama-server and saves every request and response.
   `PROXY_LOG=DIR` in `bench/run-bench.sh` puts the server on `127.0.0.1:8081` and the proxy on the sandbox's
   port 8080. Captures: [`capture/`](capture/) (`NNNN-req.json` is what pi sent, `NNNN-resp.txt` the raw stream).
2. `replay.py` takes a captured request and asks the server twice per seed: through `/apply-template` +
   `/completion`, which shows what the model writes with no grammar and no parser, and through
   `/v1/chat/completions`, which is what pi gets.

## What we found

**Coder** ([`coder12-0001.txt`](results/coder12-0001.txt), [`coder12-raw-samples.txt`](results/coder12-raw-samples.txt);
pi's first request on `feature-inventory`):

- The prompt is standard Gemma 4: system turn with `<|think|>`, tools declared as `<|tool>declaration:read{…}`.
  The chat template is the 26B's plus one branch used only with thinking off.
- In 8 raw samples it emitted `<|tool_call>` **once**. Otherwise it narrated steps as text after stray `<channel|>`
  markers (`<channel|>read inventory.py<channel|>edit inventory.py`) and wrote the code out.
- The one call used JSON strings, `call:read{path: "inventory.py"}`, where Gemma 4 uses `<|"|>…<|"|>`.
- llama.cpp makes this look worse. Its Gemma 4 parser stops the content at an unmatched `<channel|>`, which is
  why pi saw empty turns. Its tool-call grammar (`common/parsers/gemma4.cpp`) only allows `<|"|>` strings, numbers,
  booleans, null, dicts and lists as values, so a model that starts a JSON `"` string is forced into something
  else. In the captured pibench turn the coder made several calls at once and then looped on
  `write{"\"content\"": -13562}`-style arguments for 20 minutes.

Verdict: the model wasn't trained to call tools (its card says as much), so there was nothing for the harness to
lose. Not our setup.

**Agentic** ([`agentic12-raw-samples.txt`](results/agentic12-raw-samples.txt); the same request):

- Its first call was clean native syntax in **16 of 16** raw samples (`read{path:<|"|>/work/inventory.py<|"|>}`),
  with the repetition penalty at 1.1 and at 1.0.
- A captured pibench job (`cli-wordfreq`, stopped after 16 requests): all 15 finished calls had well-formed
  arguments ([`capture/agentic12/`](capture/agentic12/)).

Verdict: the mangled calls are intermittent and the model writes good calls most of the time; nothing here points at
llama.cpp, pi or the repetition penalty. We stopped before catching a bad call in the act, and retired both models
(2026-11-02): see their pages.

## Rerun

The launchers were removed with the models; take `scripts/run-12b-coder.sh` and `run-12b-agentic.sh` from git
history (commit `2764962`), start one on port 8081, and run
`python3 bench/gemma12-probe/replay.py bench/gemma12-probe/capture/coder12/0001-req.json`.
