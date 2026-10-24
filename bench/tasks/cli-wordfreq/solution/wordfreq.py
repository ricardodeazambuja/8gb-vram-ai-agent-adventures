import argparse, json, re, sys
from collections import Counter

p = argparse.ArgumentParser()
p.add_argument("file"); p.add_argument("--top", type=int, default=10)
p.add_argument("--min-length", type=int, default=1); p.add_argument("--ignore", nargs="+", default=[])
a = p.parse_args()
try:
    text = sys.stdin.read() if a.file == "-" else open(a.file, encoding="utf-8").read()
except OSError as e:
    print(f"error: {e}", file=sys.stderr); sys.exit(2)
ign = {w.lower() for w in a.ignore}
c = Counter(w for w in re.findall(r"[a-z]+(?:'[a-z]+)*", text.lower()) if len(w) >= a.min_length and w not in ign)
print(json.dumps([{"word": w, "count": n} for w, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:a.top]]))
