# Bonsai 2's chat template against Unsloth's (2026-11-11)

Unsloth re-released its Qwen3.8-27B GGUFs on 2026-11-11 with a chat template that, per their guide, adds
"developer role support" and "improved parsing" of tool calls. Bonsai 2 is the same model, so would its
template make Bonsai 2 better in pi?

**No: for what pi sends, the model gets exactly the same prompt.** llama-server itself rendered 18 real pi
requests (both sessions in [`../gemma12-probe/capture/`](../gemma12-probe/README.md): system prompt, tools,
tool calls and results, thinking text; pi sends the same messages for any model) with each template, at the
effort pi sent (`medium`) and at `low`, `xhigh` and `high`. Its `/apply-template` endpoint only builds the
prompt; nothing is generated.

| | Bonsai 2's template | Unsloth's |
|---|---|---|
| Prompts built | 54 of 72 | 72 of 72 |
| Identical bytes where both build | 54 of 54 | |
| Errors | 18, every one at effort `high` (HTTP 500) | none |

What Unsloth changed, read from the diff against Qwen's template:

- `reasoning_effort: high` becomes `xhigh` instead of an error. pi never sends `high` to Bonsai 2:
  `config/pi-provider.json` maps it ([the Bonsai 2 page](../../docs/models/bonsai2-27b.md#gotchas)).
- A `developer` message counts as `system`, and several leading system messages are merged. pi sends one
  `system` message.
- Tool calls: a call with no name, or with arguments as a JSON string, now raises a clear error. pi does
  send string arguments, but llama-server turns them into objects before the template sees them (Unsloth's
  template rendered every request).
- The "No user query found in messages" check is gone.

Bonsai 2's own template is Qwen's, minus the `{% generation %}` markers Qwen added later (they only mark
the assistant's text for fine-tuning).

A second run with `run.sh` gave the same sha256 for every prompt.

## Files

| File | What |
|---|---|
| [`fetch.sh`](fetch.sh) | Re-downloads Qwen's template and the first 32 MB of Unsloth's smallest GGUF (the header holds the template), both at pinned revisions, re-extracts Bonsai 2's from `models/`, and checks all three sha256s |
| [`gguf_kv.py`](gguf_kv.py) | Reads a GGUF header (works on a truncated file) and writes out `tokenizer.chat_template` |
| [`run.sh`](run.sh) | Starts Bonsai 2 on the CPU (`NGL=0`: rendering needs no GPU) with each template in turn and calls `render.py`, then `compare.py` |
| [`render.py`](render.py) | Posts each captured request to `/apply-template` at four efforts |
| [`compare.py`](compare.py) | Writes [`results/prompts.json`](results/prompts.json) (sha256 of each prompt, or its error) and [`results/summary.txt`](results/summary.txt) |
| [`templates/`](templates/) | The three templates. Unsloth's GGUF template is the same file as `chat_template.jinja` in `unsloth/Qwen3.8-27B` at `3ea932ce` |

Build: `1f6e660c4` (the default, fork branch `adventures-sync-2026-11-08`).
