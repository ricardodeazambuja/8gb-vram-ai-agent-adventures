#!/usr/bin/env python3
"""Watchdog against systemd-oomd, for the whole bench queue.

    systemd-run --user --scope --unit=guard-mem python3 memguard.py [LIMIT=20] [N=3]

systemd-oomd (`oomctl`: 50% on user@UID.service, 20 s, swap 90%) kills the descendant cgroup with the most
reclaim, which can be a terminal rather than the cause. This stops our own bench scopes (`moe*.scope`, as
started by srv.sh, hits.sh and step3-models.sh) much earlier: when memory PSI avg10 (max of some and full)
is over LIMIT for N consecutive 1 s polls. Its own unit name must not start with `moe`. Logs every stop and
the 5-minute peak.
"""
import os
import subprocess
import sys
import time

LIMIT = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3
P = f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/memory.pressure"


def psi():
    v = {}
    with open(P) as f:
        for line in f:
            k, *fields = line.split()
            v[k] = float(dict(x.split("=") for x in fields)["avg10"])
    return max(v.values())


def scopes():
    out = subprocess.run(["systemctl", "--user", "list-units", "--type=scope", "--state=running", "--no-legend", "--plain"],
                         capture_output=True, text=True, check=False).stdout
    return [line.split()[0] for line in out.splitlines() if line.startswith("moe")]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main():
    log(f"memguard: stop moe*.scope when PSI avg10 > {LIMIT}% for {N} s ({P})")
    over, peak, last = 0, 0.0, 0.0
    while True:
        p = psi()
        peak = max(peak, p)
        over = over + 1 if p > LIMIT else 0
        if over >= N:
            s = scopes()
            log(f"PSI {p:.1f}% > {LIMIT}% for {N} s: stopping {s}")
            for u in s:
                subprocess.run(["systemctl", "--user", "kill", "-s", "KILL", u], check=False)
            over = 0
        if time.time() - last > 300:
            log(f"peak PSI avg10 last 5 min: {peak:.1f}%")
            peak, last = 0.0, time.time()
        time.sleep(1)


if __name__ == "__main__":
    main()
