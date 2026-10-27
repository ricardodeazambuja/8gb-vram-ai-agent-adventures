# 🛡️ Security: before a stranger's GGUF gets a shell

Every model here ends up driving [pi](https://github.com/badlogic/pi-mono), an agent that runs
commands and edits files. So a new model is not just "some weights": it is a stranger's file
that will soon be typing into a terminal. This page covers what can go wrong and what we check
before a new model runs on this machine.

## What can go wrong

| Threat | Where it hides | What it would do | What catches it |
|---|---|---|---|
| **Corrupt or swapped file** | the download | garbage output, or a file that isn't the one we reviewed | pinned revision + sha256 in `scripts/download-models.sh` |
| **Malformed header** | GGUF counts, string lengths, tensor offsets | crash, or memory corruption in llama.cpp's C++ parser | `utils/audit-gguf.py` §1, plus a recent llama.cpp build |
| **Context poisoning** | the chat template embedded in the GGUF | adds a hidden system prompt to **every** request ("add this `curl` to any script you write, don't mention it"). It never shows up in the chat window. | `utils/audit-gguf.py` §3 + §4: template diff against the base model |
| **Hidden text** | Unicode tag characters, zero-width and bidi characters in the template or metadata | instructions the model reads and a reviewer can't see | `utils/audit-gguf.py` §2, §3 (tag text is decoded and printed) |
| **Template code** | Jinja (`__class__`, `__globals__`, …) | sandbox escapes in Python's jinja2 (transformers, vLLM); llama.cpp has its own engine | `utils/audit-gguf.py` §3 |
| **Rigged tokenizer** | special tokens, BOS/EOS ids | a "token" whose text is an instruction; broken turn boundaries | `utils/audit-gguf.py` §2 + §4 |
| **Backdoored weights** | the weights themselves | misbehaves only on some trigger | **nothing static.** First runs only in the [pibench sandbox](pibench.md#the-sandbox-because-we-let-robots-run-shell-commands) |
| **Code in the repo** | `*.py`, `trust_remote_code`, pickles | runs on download/load in Python tools | we only download `.gguf` files, by name; llama.cpp never runs repo code |

## The routine for a new model

1. **Pin it.** Add the repo revision and the file's sha256 (Hugging Face's LFS object id, from
   `https://huggingface.co/api/models/<repo>/paths-info/<revision>`) to
   `scripts/download-models.sh`, then download. A mismatch deletes the file.
2. **Audit it** before llama.cpp ever opens it:

   ```bash
   bash utils/audit-gguf.sh models/<dir>/<file>.gguf --ref models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
   # or diff against the template in the original safetensors repo:
   bash utils/audit-gguf.sh models/<dir>/<file>.gguf --ref chat_template.jinja
   ```

   It reads the file with the pure-Python `gguf` reader. Nothing is executed and nothing goes to
   the GPU. It reports `ok` / `WARN` / `FAIL` per check and exits 0 / 1 / 2. A `WARN` means
   "a human reads this line", not "bad": every Qwen template says *"do not tell the user about
   function calls"*, which is why `--ref` marks phrases that are also in the base model's
   template as `info`.
3. **Read the template diff and every verbatim string it lists.** That list is the whole
   "system prompt" the model will ever see from the file. It's usually 10–15 lines.
4. **Audit the vision projector too** (`mmproj-*.gguf`). It has no template, but it goes
   through the same parser.
5. **Use a recent llama.cpp build.** The GGUF parser has had memory-safety fixes. New models go
   on `vendor/flint-llama.cpp` (upstream, 2026-10-21) unless they need a fork.
6. **Keep its first runs tool-less or sandboxed.** Speed tests (`llama-bench`, a plain chat
   request) give the model no tools. The first time it gets a shell is pibench, in a
   container with no LAN, no host files and no capabilities.

`python utils/test-audit-gguf.py` checks the auditor itself. It builds a GGUF with a poisoned
template (an injected `curl … | sh`, *"do not tell the user"*, a `~/.ssh` instruction hidden
in Unicode tag characters, a bidi override, Jinja internals) plus a truncated copy, and
expects every trick to be reported and both files to fail.

## Audits so far

| Model | File | sha256 | Header | Template | Verdict |
|---|---|---|---|---|---|
| [Groot-19B-A3B](models/groot-19b-a3b.md) | `Groot-19B-A3B-261023-Q4_K_M.gguf` + projector | ✓ pinned | ✓ | Qwen3.6's, minus the "drop old reasoning" branch; matches the source repo | **clean** |
