# 🔧 utils

Small standalone tools that aren't part of the run/serve flow but are handy for
poking at the model files.

## `inspect-gguf.{sh,py}` — what's inside a GGUF?

Reads a GGUF's header and reports its **architecture**, any **multimodal
metadata**, its **tensor inventory**, and a one-line **verdict** (text-only /
multimodal model / `mmproj` projector). This is how we established that the
Gemma 4 text GGUF carries no vision tensors and that `mmproj-BF16.gguf` is
vision-only — see [`docs/how-it-fits.md` §14](../docs/how-it-fits.md).

```bash
# the model (text-only):
bash utils/inspect-gguf.sh models/gemma4-26b-a4b-qat/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf

# the vision projector:
bash utils/inspect-gguf.sh models/gemma4-26b-a4b-qat/mmproj-BF16.gguf

# also dump every tensor name:
bash utils/inspect-gguf.sh <file.gguf> --tensors
```

Example verdicts:

```
verdict      : text-only (no vision/audio tensors — needs a separate --mmproj for images)
verdict      : projector (mmproj) — vision
```

**Notes**
- The `.sh` wrapper runs the `.py` inside the `llamacpp` conda env (`mamba run`)
  so `numpy` + `gguf` are available; it falls back to plain `python3` if mamba
  isn't on PATH. You can also call the `.py` directly in any env that has `gguf`.
- The `gguf` module is auto-resolved from our llama.cpp fork (`../llama.cpp/gguf-py`,
  or `$ADV_LLAMA/gguf-py`), otherwise an installed copy — no extra install needed
  once the fork is checked out.

## `check_docs.py` — do the docs still match the scripts?

```bash
python3 utils/check_docs.py          # report, exit 1 on any problem
python3 utils/check_docs.py --fix    # also regenerate docs/cheatsheet.md's model table from scripts/llm
```

Checks every `scripts/llm NAME`, `QMODEL=`, `download-models.sh`, `imagine -m` and `examples/` command in the
docs against the real scripts, every relative link and anchor, and that each model page's Launch row shows its
`scripts/llm` command. Standard library only, no models needed; CI runs it on every push
(`.github/workflows/docs.yml`).

## `audit-gguf.{sh,py}` — is this GGUF safe to load?

Security checks on a downloaded GGUF **before** llama.cpp opens it: header sanity (every tensor
in bounds, aligned, non-overlapping), hidden Unicode / URLs / shell commands / injection phrases
in every metadata string, the embedded **chat template** (hidden system prompts, Jinja
internals, text spelled in `\x` / `\u` / octal escapes, every sentence it emits verbatim), and with `--ref` a diff of template, tokenizer
and hyperparameters against the base model. Pure Python, nothing is executed. Why and how:
[`docs/security.md`](../docs/security.md).

```bash
bash utils/audit-gguf.sh models/groot-19b-a3b/Groot-19B-A3B-261023-Q4_K_M.gguf \
     --ref models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --dump-template /tmp/groot.jinja
```

Exit code 0 = clean, 1 = a `WARN` line for a human to read, 2 = `FAIL`, don't load it.

`python utils/test-audit-gguf.py` is its self-test: it builds a poisoned GGUF (injected
`curl … | sh`, text hidden in Unicode tag characters, a bidi override, Jinja internals, `wget`
spelled in hex, `ssh` in octal, tag characters written as `\U` escapes) and a
truncated one, and checks that the auditor reports every trick and fails both.
