#!/usr/bin/env python3
"""check_docs: keep the docs in step with the scripts. Standard library only; no models or GPU needed.

    python3 utils/check_docs.py          # report problems, exit 1 if any
    python3 utils/check_docs.py --fix    # also regenerate docs/cheatsheet.md's model table

Checks, over README.md, docs/**/*.md, scripts/README.md, examples/README.md and bench/**/README.md:
  * the model table in docs/cheatsheet.md matches `scripts/llm list --md` (the single source of truth);
  * every `scripts/llm NAME`, `QMODEL=NAME`, `download-models.sh NAME…`, `imagine … -m NAME`,
    `examples/X.py` and `examples/chat.py -m NAME` names something that exists;
  * every relative link and #anchor resolves (GitHub's heading-anchor rules);
  * every model page of a `scripts/llm` model shows `scripts/llm NAME` in its TL;DR Launch row, and every
    page in docs/models/ belongs to a launcher (or is one of the known exceptions below).
Run it before committing doc or launcher changes; CI runs it on every push (.github/workflows/docs.yml).
"""
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = [REPO / "README.md", *sorted((REPO / "docs").rglob("*.md")), REPO / "scripts" / "README.md",
        REPO / "examples" / "README.md", REPO / "bench" / "README.md", *sorted((REPO / "bench").glob("*/README.md"))]
CHEAT = REPO / "docs" / "cheatsheet.md"
START, END = "<!-- llm-list:start (generated: python3 utils/check_docs.py --fix) -->", "<!-- llm-list:end -->"
LLM_SUBCOMMANDS = {"list", "ls", "names", "status", "stop", "help"}
PAGES_WITHOUT_LLM = {"README.md", "gemma4-12b-agentic.md", "gemma4-12b-coder.md",  # template; retired
                     "flux2-klein-4b.md", "iris-3b.md", "sdxl.md"}                   # scripts/imagine
IMAGINE_PAGES = {"flux2-klein-4b.md": "klein", "iris-3b.md": "iris", "sdxl.md": "sdxl"}
problems = []


def run(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=REPO).stdout


def fail(path, msg):
    problems.append(f"{path.relative_to(REPO)}: {msg}")


def known_names():
    llm = set(run("bash", "scripts/llm", "names").split())
    imagine = set(run(sys.executable, "-I", "scripts/imagine", "--names").split())
    rq = (REPO / "scripts" / "run-qwen.sh").read_text()
    start = rq.index('case "$QMODEL" in')
    block = rq[start:rq.index("\nesac", start)]
    qmodel = {n for labels in re.findall(r"^  ([a-z0-9|]+)\)", block, re.MULTILINE) for n in labels.split("|")}
    dl = set(re.findall(r'^  "([a-z0-9]+)\|', (REPO / "scripts" / "download-models.sh").read_text(), re.MULTILINE))
    return llm, imagine, qmodel, dl


def slug(heading):
    text = re.sub(r"\[([^]]*)\]\([^)]*\)", r"\1", heading).replace("`", "").strip().lower()
    keep = [c for c in text if c.isalnum() or c in "-_ ️" or unicodedata.category(c)[0] in "LN"]
    return "".join(keep).replace(" ", "-")


def anchors(path, cache={}):  # noqa: B006 - a deliberate per-run cache
    if path not in cache:
        found, seen, code = set(), {}, False
        for line in path.read_text().split("\n"):
            if line.startswith("```"):
                code = not code
            if code:
                continue
            m = re.match(r"#{1,6} (.*)", line)
            if m:
                s = slug(m.group(1))
                n = seen.get(s, 0)
                seen[s] = n + 1
                found.add(s if n == 0 else f"{s}-{n}")
            found.update(re.findall(r'<a id="([^"]+)"', line))
        cache[path] = found
    return cache[path]


def check_cheatsheet(fix):
    want = run("bash", "scripts/llm", "list", "--md").strip()
    text = CHEAT.read_text()
    if START not in text or END not in text:
        return fail(CHEAT, "the llm-list markers are missing")
    head, rest = text.split(START, 1)
    body, tail = rest.split(END, 1)
    if body.strip() != want:
        if fix:
            CHEAT.write_text(f"{head}{START}\n{want}\n{END}{tail}")
            print("fixed: docs/cheatsheet.md model table regenerated")
        else:
            fail(CHEAT, "the model table is out of date with scripts/llm (run: python3 utils/check_docs.py --fix)")
    return None


def check_commands(path, text, llm, imagine, qmodel, dl):
    for m in re.finditer(r"scripts/llm ([a-z0-9][a-z0-9-]*)", text):
        if m.group(1) not in llm | LLM_SUBCOMMANDS:
            fail(path, f"`scripts/llm {m.group(1)}`: no such model (scripts/llm names)")
    for m in re.finditer(r"\bQMODEL=([a-z0-9]+)", text):
        if m.group(1) not in qmodel:
            fail(path, f"`QMODEL={m.group(1)}`: not a run-qwen.sh model")
    for m in re.finditer(r"download-models\.sh((?: +[a-z0-9]+)+)", text):
        for name in m.group(1).split():
            if name not in dl:
                fail(path, f"`download-models.sh {name}`: no such download")
    for m in re.finditer(r"imagine\b[^`\n|]*?(?:-m|--model) ([a-z0-9]+)", text):
        if m.group(1) not in imagine:
            fail(path, f"`imagine -m {m.group(1)}`: no such image model")
    for m in re.finditer(r"examples/([a-z_]+\.py)", text):
        if not (REPO / "examples" / m.group(1)).exists():
            fail(path, f"examples/{m.group(1)} does not exist")
    for m in re.finditer(r"examples/chat\.py[^`\n|]*? -m ([a-z0-9]+)", text):
        if m.group(1) not in llm:
            fail(path, f"`examples/chat.py -m {m.group(1)}`: no such model")


TRACKED = None


def tracked(dest):
    """True if git tracks this file (or a file under this folder): what a GitHub reader actually gets."""
    global TRACKED
    if TRACKED is None:
        out = subprocess.run(["git", "ls-files", "-z"], capture_output=True, text=True, check=True, cwd=REPO).stdout
        TRACKED = {REPO / f for f in out.split("\0") if f}
        TRACKED |= {d for f in TRACKED for d in f.parents}
    return dest.resolve() in TRACKED


def check_links(path, text):
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    for m in re.finditer(r"\]\(([^)\s]+)\)", text):
        target = m.group(1)
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        file, _, frag = target.partition("#")
        dest = (path.parent / file) if file else path
        if not dest.exists():
            fail(path, f"broken link {target}")
        elif not tracked(dest):
            fail(path, f"link to a file git doesn't track (ignored or uncommitted): {target}")
        elif frag and dest.suffix == ".md" and frag not in anchors(dest.resolve()):
            fail(path, f"broken anchor {target}")


def check_model_pages(llm):
    rows = run("bash", "scripts/llm", "list", "--md")
    pages = {}  # page -> the scripts/llm names it documents (VibeThinker's page has two)
    for page, name in re.findall(r"\]\(models/([^)]+)\) \| \w+ \| `scripts/llm ([a-z0-9]+)`", rows):
        pages.setdefault(page, set()).add(name)
    if set().union(*pages.values()) != llm:
        problems.append(f"utils/check_docs.py: parsed {len(pages)} pages from `scripts/llm list --md` but they "
                        f"don't cover every model: {sorted(llm - set().union(*pages.values()))}")
    for page, names in pages.items():
        p = REPO / "docs" / "models" / page
        if not p.exists():
            fail(p, "scripts/llm points here, but the page doesn't exist")
            continue
        launch = next((ln for ln in p.read_text().split("\n") if re.match(r"> \| Launch \|", ln)), "")
        for n in names:
            if f"scripts/llm {n}" not in launch:
                fail(p, f"its TL;DR Launch row doesn't show `scripts/llm {n}`")
    for page, model in IMAGINE_PAGES.items():
        launch = next((ln for ln in (REPO / "docs/models" / page).read_text().split("\n")
                       if ln.startswith("> | Launch |")), "")
        if "scripts/imagine" not in launch or f"-m {model}" not in launch:
            fail(REPO / "docs/models" / page, f"its TL;DR Launch row doesn't show `scripts/imagine ... -m {model}`")
    for p in sorted((REPO / "docs" / "models").glob("*.md")):
        if p.name not in pages and p.name not in PAGES_WITHOUT_LLM:
            fail(p, "no scripts/llm entry launches this model (add one, or list it in check_docs.py's exceptions)")


def main():
    fix = "--fix" in sys.argv[1:]
    llm, imagine, qmodel, dl = known_names()
    check_cheatsheet(fix)
    for path in DOCS:
        text = path.read_text()
        check_commands(path, text, llm, imagine, qmodel, dl)
        check_links(path, text)
    check_model_pages(llm)
    for p in problems:
        print(p)
    print(f"{len(problems)} problem{'s' if len(problems) != 1 else ''} in {len(DOCS)} files")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
