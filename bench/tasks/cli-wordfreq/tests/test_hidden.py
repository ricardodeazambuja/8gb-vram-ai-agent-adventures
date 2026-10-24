import json
import os
import re
import subprocess
import sys
from collections import Counter

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "wordfreq.py")
TEXT = ("It's the end -- the END! Don't panic; 'quoted' words, x2y and rock'n'roll.\n"
        "the The tHe a an AN it's ITS cafe42cafe don't\n")


def ref(text, top=10, min_length=1, ignore=()):
    words = re.findall(r"[a-z]+(?:'[a-z]+)*", text.lower())
    ign = {w.lower() for w in ignore}
    c = Counter(w for w in words if len(w) >= min_length and w not in ign)
    return [{"word": w, "count": n} for w, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:top]]


def run(args, stdin=None):
    return subprocess.run([sys.executable, TOOL, *args], input=stdin, capture_output=True,
                          text=True, timeout=30)


@pytest.fixture
def textfile(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text(TEXT)
    return str(p)


def test_default(textfile):
    r = run([textfile])
    assert r.returncode == 0
    assert json.loads(r.stdout) == ref(TEXT)


def test_top_and_min_length(textfile):
    assert json.loads(run([textfile, "--top", "3", "--min-length", "3"]).stdout) == ref(TEXT, 3, 3)


def test_ignore(textfile):
    out = json.loads(run([textfile, "--ignore", "THE", "it's", "--top", "50"]).stdout)
    assert out == ref(TEXT, 50, 1, ["the", "it's"])


def test_stdin():
    r = run(["-", "--top", "50"], stdin=TEXT)
    assert json.loads(r.stdout) == ref(TEXT, 50)


def test_sample():
    with open(os.path.join(HERE, "sample.txt")) as f:
        s = f.read()
    assert json.loads(run([os.path.join(HERE, "sample.txt"), "--top", "5"]).stdout) == ref(s, 5)


def test_missing_file(tmp_path):
    assert os.path.exists(TOOL)
    r = run([str(tmp_path / "nope.txt")])
    assert r.returncode == 2
    assert r.stderr.strip()
    assert r.stdout.strip() == ""
