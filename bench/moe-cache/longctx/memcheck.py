#!/usr/bin/env python3
"""Before loading MODEL: drop the page cache of every other .gguf under the models folder
(posix_fadvise DONTNEED, no root; pages a running process has mapped stay), then require
MemAvailable >= size(MODEL) + MARGIN GiB. Exit 1 if not. srv.sh calls it before every launch.

    python3 memcheck.py MODEL [MARGIN_GIB=4]
"""
import glob
import os
import sys


def main():
    m = os.path.realpath(sys.argv[1])
    margin = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    root = os.path.dirname(os.path.dirname(m))
    for f in glob.glob(os.path.join(root, "*", "*.gguf")):
        if os.path.realpath(f) != m:
            fd = os.open(f, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
    with open("/proc/meminfo") as f:
        avail = int(next(line for line in f if line.startswith("MemAvailable")).split()[1]) / 2**20
    size = os.path.getsize(m) / 2**30
    ok = avail >= size + margin
    print(f"    memcheck: {os.path.basename(m)} {size:.1f} GiB + {margin:g} margin, "
          f"MemAvailable {avail:.1f} GiB -> {'ok' if ok else 'REFUSED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
