"""compare.py A.json B.json OUTDIR: compare two render.py outputs prompt by prompt.
Writes OUTDIR/prompts.json (per request and effort: the sha256 of each rendered prompt, or its error) and
OUTDIR/summary.txt. Keys are made relative to the repo so the result doesn't depend on where it ran."""
import hashlib
import json
import sys
from collections import Counter

a_path, b_path, out = sys.argv[1:4]
a, b = json.load(open(a_path)), json.load(open(b_path))
rel = lambda k: "bench/gemma12-probe/capture/" + k.split("capture/", 1)[1]  # noqa: E731


def digest(v):
    if v is None or v.startswith("HTTP"):
        return "ERROR " + (v or "none").split("\n")[0][:60]
    return hashlib.sha256(v.encode()).hexdigest()


rows = {rel(k): {"bonsai2": digest(a[k]), "unsloth": digest(b[k])} for k in a}
json.dump(rows, open(f"{out}/prompts.json", "w"), indent=1, sort_keys=True)
effort = lambda k: k.split("#effort=")[1] if "#effort=" in k else "as sent (medium)"  # noqa: E731
lines = [f"{len(rows)} renders per template ({len({k.split('#')[0] for k in rows})} pi requests x 4 efforts)"]
for arm in ("bonsai2", "unsloth"):
    errs = Counter(effort(k) for k, r in rows.items() if r[arm].startswith("ERROR"))
    lines.append(f"{arm}: {len(rows) - sum(errs.values())} rendered, errors by effort: {dict(errs) or 'none'}")
both = [r for r in rows.values() if not r["bonsai2"].startswith("ERROR") and not r["unsloth"].startswith("ERROR")]
lines.append(f"identical where both rendered: {sum(r['bonsai2'] == r['unsloth'] for r in both)} / {len(both)}")
open(f"{out}/summary.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
