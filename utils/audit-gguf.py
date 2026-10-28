#!/usr/bin/env python3
"""Security audit of a GGUF file, done before llama.cpp ever loads it.

A GGUF is not just weights. It also carries text that ends up in front of the
model, or in front of a parser, on every request:

  * the chat template (`tokenizer.chat_template*`), a Jinja program that builds
    every prompt. A poisoned one can slip a hidden system prompt into every
    conversation ("context poisoning"): exfiltrate files, run commands, plant
    backdoors in code. It never shows up in the chat window.
  * free-text metadata (names, descriptions, licences, URLs, default sampling)
  * the header itself: counts, lengths and offsets that a C++ parser trusts.

This script reads all of that with the pure-Python `gguf` reader (no model code
runs, nothing is executed) and reports:

  1. structure  every tensor inside the file, aligned, non-overlapping, sane dims
  2. strings    invisible / bidi / tag Unicode, URLs, shell commands, injection
                phrases, in every string field (vocab and merges excluded)
  3. template   the same checks on the chat template, Jinja sandbox-escape
                patterns, and every hard-coded sentence it would emit
  4. --ref      a diff of the chat template, tokenizer and hyperparameters
                against a reference (the base model's GGUF, or a .jinja file
                from the original safetensors repo)

What it cannot do: find a backdoor trained into the weights. That is what the
sandbox in bench/ is for. Run new models there before giving them a shell.

Usage:
    python utils/audit-gguf.py <file.gguf> [--ref <base.gguf | template.jinja>]
                               [--dump-template OUT.jinja] [--sha256 HEX]
    bash   utils/audit-gguf.sh <file.gguf> ...        # same, inside the env

Exit code: 0 clean, 1 something needs a human to read it, 2 broken file.
"""
import argparse
import difflib
import hashlib
import itertools
import os
import re
import sys
import unicodedata


def _import_gguf():
    # Prefer the newest vendored gguf-py (it knows the newest quant types), then
    # an installed one.
    here = os.path.dirname(os.path.abspath(__file__))
    for sub in ("flint-llama.cpp", "llama.cpp"):
        p = os.path.join(here, "..", "vendor", sub, "gguf-py")
        if os.path.isdir(p):
            sys.path.insert(0, p)
            break
    try:
        import gguf
        return gguf
    except ModuleNotFoundError:
        sys.exit("ERROR: the 'gguf' module is not available (pip install gguf, or check out llama.cpp in vendor/)")


gguf = _import_gguf()

# --- what we look for --------------------------------------------------------------------------

# Characters a human reviewer cannot see: zero-width, bidi overrides, BOM, Unicode "tag"
# characters (U+E0000-E007F, invisible ASCII clones used to hide prompt injections).
def hidden_chars(s):
    out = []
    for i, ch in enumerate(s):
        cp = ord(ch)
        if (0xE0000 <= cp <= 0xE007F or cp in (0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x180E)
                or 0x202A <= cp <= 0x202E or 0x2066 <= cp <= 0x2069
                or (unicodedata.category(ch) in ("Cc", "Cf") and ch not in "\n\r\t")):
            out.append((i, f"U+{cp:04X}", unicodedata.name(ch, "?")))
    return out


SUSPICIOUS = [
    ("url", r"\b(?:https?|ftp|file)://[^\s'\"]+"),
    ("shell", r"\b(?:curl|wget|nc|ncat|netcat|ssh|scp|bash\s+-c|sh\s+-c|chmod\s+\+x|rm\s+-rf|base64\s+-d|eval|exec)\b"),
    ("secrets", r"(?i)\b(?:password|passwd|api[_ -]?key|secret|private key|id_rsa|\.ssh|\.env|credentials?|aws_)\b"),
    ("injection", (r"(?i)\b(?:ignore (?:all |any )?(?:previous|prior|above)|disregard|do not (?:tell|mention|reveal)|"
                   r"never (?:tell|mention|reveal)|without telling|secretly|covertly|hidden instruction|"
                   r"you are now|new instructions|exfiltrat\w*|backdoor)")),
    ("ip", r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    ("blob", r"[A-Za-z0-9+/]{120,}={0,2}"),  # long base64-ish runs
]
# Jinja sandbox escapes (server-side template injection). Harmless for llama.cpp's own Jinja
# engine in principle, but a template that reaches for these has no business being trusted,
# and the same template is rendered by Python jinja2 in transformers / vLLM.
SSTI = r"__\w+__|\b(?:mro|subclasses|globals|builtins|import|popen|subprocess|os\.system|getattr|cycler|joiner|namespace\.__init__|lipsum|self\._)\b"


# Escapes in template string literals. llama.cpp's Jinja decodes \xHH (U+0000-U+00FF); Python's
# jinja2 (transformers, vLLM) also decodes \uHHHH and \UHHHHHHHH. A template can spell text in
# hex ("\x63\x75\x72\x6c" renders as curl) and slip past the patterns above, so we decode first.
ESCAPE = r"\\(?:x[0-9A-Fa-f]{2}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})"


def decode_escapes(s):
    def one(m):
        try:
            return chr(int(m.group(0)[2:], 16))  # noqa: FURB166 (strips "\\x", not "0x")
        except (ValueError, OverflowError):
            return m.group(0)
    return re.sub(ESCAPE, one, s)


def scan(text, skip=()):
    hits = []
    for name, pat in SUSPICIOUS:
        if name in skip:
            continue
        for m in re.finditer(pat, text):
            hits.append((name, m.group(0)[:80]))
    return hits


def template_literals(tpl, min_len=25):
    """Text the template emits verbatim: raw text between tags, plus quoted string
    literals inside tags. This is where a hidden system prompt would live."""
    out = []
    raw = re.split(r"\{%.*?%\}|\{\{.*?\}\}|\{#.*?#\}", tpl, flags=re.DOTALL)
    for chunk in raw:
        c = chunk.strip()
        if len(c) >= min_len:
            out.append(("raw", c))
    for tag in re.findall(r"\{%.*?%\}|\{\{.*?\}\}", tpl, flags=re.DOTALL):
        for q in re.findall(r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"", tag):
            s = q[0] or q[1]
            if len(s) >= min_len:
                out.append(("str", s))
    seen, uniq = set(), []
    for k, s in out:
        if s not in seen:
            seen.add(s)
            uniq.append((k, s))
    return uniq


# --- reading -----------------------------------------------------------------------------------

def value(field):
    try:
        return field.contents()
    except Exception as e:  # noqa: BLE001
        return f"<undecodable: {e}>"


def load(path):
    return gguf.GGUFReader(path, "r")


def templates(r):
    return {k: value(f) for k, f in r.fields.items() if k.startswith("tokenizer.chat_template")}


# --- checks ------------------------------------------------------------------------------------

class Report:
    def __init__(self):
        self.level = 0

    def ok(self, msg):
        print(f"  ok    {msg}")

    def warn(self, msg):
        print(f"  WARN  {msg}")
        self.level = max(self.level, 1)

    def fail(self, msg):
        print(f"  FAIL  {msg}")
        self.level = 2


def check_structure(r, path, rep):
    print("\n[1] structure")
    size = os.path.getsize(path)
    align = r.alignment
    rep.ok(f"header parsed: {len(r.fields)} metadata keys, {len(r.tensors)} tensors, alignment {align}")
    if r.data_offset > size:
        rep.fail(f"data section starts at {r.data_offset}, past end of file ({size})")
        return
    spans, bad = [], 0
    for t in r.tensors:
        dims = [int(d) for d in t.shape.tolist()]
        if not dims or len(dims) > 4 or any(d <= 0 for d in dims):
            rep.fail(f"tensor {t.name}: bad dims {dims}")
            bad += 1
        start, end = t.data_offset, t.data_offset + int(t.n_bytes)
        if (start - r.data_offset) % align:
            rep.fail(f"tensor {t.name}: misaligned offset {start}")
            bad += 1
        if end > size:
            rep.fail(f"tensor {t.name}: ends at {end}, past end of file ({size})")
            bad += 1
        spans.append((start, end, t.name))
    spans.sort()
    for (s0, e0, n0), (s1, e1, n1) in itertools.pairwise(spans):
        if s1 < e0:
            rep.fail(f"tensors overlap: {n0} [{s0},{e0}) and {n1} [{s1},{e1})")
            bad += 1
    if not bad:
        used = sum(e - s for s, e, _ in spans)
        tail = size - (spans[-1][1] if spans else r.data_offset)
        rep.ok(f"every tensor in bounds, aligned, no overlaps ({used / 1e9:.2f} GB of tensor data)")
        if tail > 4096:
            rep.warn(f"{tail:,} bytes after the last tensor (unused trailing data)")
        else:
            rep.ok(f"trailing bytes after last tensor: {tail}")


def check_strings(r, rep):
    print("\n[2] metadata strings (vocab and merges excluded)")
    skip_keys = ("tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.token_type",
                 "tokenizer.ggml.scores")
    n, flagged = 0, 0
    for k, f in r.fields.items():
        if k.startswith("tokenizer.chat_template") or k in skip_keys:
            continue
        v = value(f)
        vals = v if isinstance(v, list) else [v]
        for s in vals:
            if not isinstance(s, str):
                continue
            n += 1
            h = hidden_chars(s)
            if h:
                rep.warn(f"{k}: {len(h)} invisible/control chars, e.g. {h[:3]}")
                flagged += 1
            for kind, frag in scan(s, skip=("blob",) if k == "general.license.text" else ()):
                # A link to the model's own repo is normal metadata; anything else, a human decides.
                if kind == "url" and k.endswith(("url", "link")) and re.match(r"https://(huggingface\.co|github\.com)/", frag):
                    print(f"  info  {k}: {frag}")
                    continue
                rep.warn(f"{k}: {kind}: {frag!r}")
                flagged += 1
            if len(s) > 1024 and k not in ("general.license.text",):
                rep.warn(f"{k}: unusually long string ({len(s)} chars)")
                flagged += 1
    if not flagged:
        rep.ok(f"{n} strings, nothing suspicious")
    print("\n    descriptive metadata:")
    for k in sorted(r.fields):
        if k.startswith(("general.", "tokenizer.ggml.add", "tokenizer.ggml.pre", "tokenizer.ggml.model")) \
                and not k.startswith("general.license.text"):
            v = value(r.fields[k])
            s = repr(v)
            print(f"      {k} = {s[:150]}{'…' if len(s) > 150 else ''}")


def check_vocab(r, rep):
    # Special tokens are where a template-free injection could hide (a "token" whose text is
    # an instruction). Look only at control/user-defined tokens, not the normal vocab.
    if "tokenizer.ggml.tokens" not in r.fields or "tokenizer.ggml.token_type" not in r.fields:
        return
    toks = value(r.fields["tokenizer.ggml.tokens"])
    types = value(r.fields["tokenizer.ggml.token_type"])
    special = [(i, t) for i, (t, ty) in enumerate(zip(toks, types)) if ty in (3, 4)]  # CONTROL, USER_DEFINED
    long_special = [(i, t) for i, t in special if len(t) > 40 or scan(t, skip=("ip",)) or hidden_chars(t)]
    print(f"\n    vocab: {len(toks)} tokens, {len(special)} control/user-defined")
    for i, t in long_special[:20]:
        rep.warn(f"special token {i} looks like text, not a marker: {t[:100]!r}")
    if not long_special:
        rep.ok("special tokens are all short markers")


def check_template(r, rep, dump, ref_tpl=""):
    print("\n[3] chat template")
    tpls = templates(r)
    if not tpls:
        if "general.architecture" in r.fields and value(r.fields["general.architecture"]) == "clip":
            rep.ok("none (a vision/audio projector has no template)")
        else:
            rep.warn("no chat template embedded (llama.cpp falls back to a built-in one)")
        return {}
    for k, tpl in tpls.items():
        print(f"    {k}: {len(tpl)} chars, sha256 {hashlib.sha256(tpl.encode()).hexdigest()[:16]}")
        h = hidden_chars(tpl)
        if h:
            rep.fail(f"{k}: {len(h)} invisible/control characters: {h[:5]}")
            tags = "".join(chr(ord(c) - 0xE0000) for c in tpl if 0xE0000 <= ord(c) <= 0xE007F)
            if tags:
                rep.fail(f"{k}: text hidden in Unicode tag characters reads: {tags!r}")
        for kind, frag in scan(tpl, skip=("ip",)):
            if frag in ref_tpl:  # same words in the base model's template: stock text, not an addition
                print(f"  info  {k}: {kind}: {frag!r} (also in the reference template)")
            else:
                rep.warn(f"{k}: {kind}: {frag!r}")
        esc = sorted(set(re.findall(ESCAPE, tpl)))
        if esc:
            shown = ", ".join(f"{e} = {decode_escapes(e)!r}" for e in esc[:12])
            if all(e in ref_tpl for e in esc):
                print(f"  info  {k}: escapes {shown} (also in the reference template)")
            else:
                rep.warn(f"{k}: {len(esc)} distinct escape sequences (text written in hex): {shown}")
            invisible = [e for e in esc if hidden_chars(decode_escapes(e)) and e not in ref_tpl]
            if invisible:
                rep.warn(f"{k}: escapes that render as invisible/control characters: {invisible}")
            seen_raw = set(scan(tpl, skip=("ip",)))
            ref_dec = decode_escapes(ref_tpl)
            for kind, frag in scan(decode_escapes(tpl), skip=("ip",)):
                if (kind, frag) in seen_raw:
                    continue
                if frag in ref_dec:
                    print(f"  info  {k}: after decoding escapes: {kind}: {frag!r} (also in the reference)")
                else:  # only visible once decoded: written in hex on purpose
                    rep.fail(f"{k}: after decoding escapes: {kind}: {frag!r}")
        ssti = sorted(set(re.findall(SSTI, tpl)))
        if ssti:
            rep.warn(f"{k}: Jinja internals referenced: {ssti}")
        lits = template_literals(tpl)
        new = [(kind, s) for kind, s in lits if s not in ref_tpl]
        print(f"    text it emits verbatim: {len(lits)} pieces >= 25 chars"
              + (f", {len(new)} not in the reference (read these):" if ref_tpl else " (read these):"))
        for kind, s in (new if ref_tpl else lits):
            one = " ".join(s.split())
            print(f"      [{kind}] {one[:200]}{'…' if len(one) > 200 else ''}")
        if not h and not ssti:
            rep.ok(f"{k}: no hidden characters, no Jinja internals")
    if dump:
        with open(dump, "w") as fh:
            fh.write(tpls.get("tokenizer.chat_template", next(iter(tpls.values()))))
        print(f"    template written to {dump}")
    return tpls


def check_ref(r, tpls, ref, rep):
    print(f"\n[4] compared with {ref}")
    if ref.endswith(".gguf"):
        rr = load(ref)
        ref_tpl = templates(rr).get("tokenizer.chat_template", "")
    else:
        rr = None
        with open(ref, encoding="utf-8") as fh:
            ref_tpl = fh.read()
    tpl = tpls.get("tokenizer.chat_template", "")
    if tpl.strip() == ref_tpl.strip():
        rep.ok("chat template identical to the reference")
    else:
        diff = list(difflib.unified_diff(ref_tpl.splitlines(), tpl.splitlines(), "reference", "this file", lineterm="", n=1))
        rep.warn(f"chat template differs from the reference ({len(diff)} diff lines); read the diff:")
        for line in diff[:200]:
            print(f"      {line}")
        if len(diff) > 200:
            print(f"      … {len(diff) - 200} more lines")
    if rr is None:
        return
    a = value(r.fields["tokenizer.ggml.tokens"]) if "tokenizer.ggml.tokens" in r.fields else []
    b = value(rr.fields["tokenizer.ggml.tokens"]) if "tokenizer.ggml.tokens" in rr.fields else []
    changed = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if len(a) == len(b) and not changed:
        rep.ok(f"vocabulary identical ({len(a)} tokens)")
    else:
        rep.warn(f"vocabulary differs: {len(a)} vs {len(b)} tokens, {len(changed)} changed, e.g. "
                 + ", ".join(f"{i}: {b[i]!r}->{a[i]!r}" for i in changed[:5]))
    for k in ("tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id",
              "tokenizer.ggml.eot_token_id"):
        x = value(r.fields[k]) if k in r.fields else None
        y = value(rr.fields[k]) if k in rr.fields else None
        if x != y:
            rep.warn(f"{k}: {y} -> {x}")
    arch, rarch = value(r.fields["general.architecture"]), value(rr.fields["general.architecture"])
    print(f"    architecture: {rarch} -> {arch}")
    keys = sorted(k for k in set(r.fields) | set(rr.fields) if k.startswith((f"{arch}.", f"{rarch}.")))
    for k in keys:
        x = value(r.fields[k]) if k in r.fields else "—"
        y = value(rr.fields[k]) if k in rr.fields else "—"
        if x != y:
            print(f"      {k}: {str(y)[:60]} -> {str(x)[:60]}")


def main():
    ap = argparse.ArgumentParser(description="Security audit of a GGUF before loading it.")
    ap.add_argument("path")
    ap.add_argument("--ref", help="base model GGUF, or a .jinja chat template, to diff against")
    ap.add_argument("--dump-template", help="write the embedded chat template to this file")
    ap.add_argument("--sha256", help="expected sha256 of the file (hashes the whole file)")
    args = ap.parse_args()

    rep = Report()
    print(f"file: {args.path} ({os.path.getsize(args.path) / 1e9:.2f} GB)")
    if args.sha256:
        h = hashlib.sha256()
        with open(args.path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        (rep.ok if h.hexdigest() == args.sha256.lower() else rep.fail)(f"sha256 {h.hexdigest()}")
    try:
        r = load(args.path)
    except Exception as e:  # noqa: BLE001
        print(f"\n[1] structure\n  FAIL  the reader rejected the header: {e}")
        sys.exit(2)
    check_structure(r, args.path, rep)
    check_strings(r, rep)
    check_vocab(r, rep)
    ref_tpl = ""
    if args.ref:
        if args.ref.endswith(".gguf"):
            ref_tpl = templates(load(args.ref)).get("tokenizer.chat_template", "")
        else:
            with open(args.ref, encoding="utf-8") as fh:
                ref_tpl = fh.read()
    tpls = check_template(r, rep, args.dump_template, ref_tpl)
    if args.ref:
        check_ref(r, tpls, args.ref, rep)
    verdict = {0: "clean", 1: "needs a human read of the WARN lines above", 2: "FAIL, do not load this file"}[rep.level]
    print(f"\nverdict: {verdict}")
    sys.exit(rep.level)


if __name__ == "__main__":
    main()
