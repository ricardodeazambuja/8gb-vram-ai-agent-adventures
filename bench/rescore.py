#!/usr/bin/env python3
"""Re-run the hidden tests on every finished run's saved /work and update its score.json.

    python3 bench/rescore.py            # all runs
    python3 bench/rescore.py spark      # one model

Use after changing a task's hidden tests: the agent's work is kept per run, so nothing has to be
re-run with the model. Scoring is the same offline container run-bench.sh uses.
"""
import glob
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

BENCH = os.path.dirname(os.path.abspath(__file__))
SCORE = ("cp -r /work /tmp/s && cp -r /tests/. /tmp/s/ && cd /tmp/s && "
         "timeout 300 python3 -I -m pytest -q -p no:cacheprovider --noconftest -c /dev/null "
         "--rootdir /tmp/s --junitxml=/tmp/junit.xml test_hidden.py >/dev/null 2>&1; "
         "cat /tmp/junit.xml 2>/dev/null")

models = sys.argv[1:] or ["*"]
changed = 0
for m in models:
    for sp in sorted(glob.glob(os.path.join(BENCH, "results/raw", m, "*", "r*", "score.json"))):
        d = os.path.dirname(sp)
        s = json.load(open(sp))
        tests_dir = os.path.join(BENCH, "tasks", s["task"], "tests")
        xml = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", "--cap-drop", "ALL",
             "--security-opt", "no-new-privileges", "-v", f"{d}/work:/work:ro",
             "-v", f"{tests_dir}:/tests:ro", "pibench:1", "sh", "-c", SCORE],
            capture_output=True, text=True).stdout
        open(os.path.join(d, "junit.xml"), "w").write(xml)
        tests = failed = 0
        try:
            root = ET.fromstring(xml)
            for suite in ([root] if root.tag == "testsuite" else root.iter("testsuite")):
                tests += int(suite.get("tests", 0))
                failed += int(suite.get("failures", 0)) + int(suite.get("errors", 0))
        except ET.ParseError:
            pass
        new = {"tests": tests, "passed": max(tests - failed, 0), "solved": tests > 0 and failed == 0}
        if any(s[k] != v for k, v in new.items()):
            changed += 1
            print(f"{s['model']:9} {s['task']:18} r{s['rep']}: {s['passed']}/{s['tests']} -> {new['passed']}/{new['tests']}")
        s.update(new)
        json.dump(s, open(sp, "w"), indent=1)
print(f"{changed} run(s) changed")
