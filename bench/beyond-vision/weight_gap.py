#!/usr/bin/env python3
"""How far apart are LFM3-1.7B and LFM3-VL-2B? Relative distance and cosine per tensor group."""
import collections
import re

import gguf
import numpy as np


def half(raw):
    return np.frombuffer(raw.tobytes(), "<f2").astype(np.float32)


def weights(t):
    raw = np.asarray(t.data).reshape(-1)
    if int(t.tensor_type) == 0:
        return np.frombuffer(raw.tobytes(), "<f4")
    blocks = np.frombuffer(raw.tobytes(), "u1").reshape(-1, 34)  # Q8_0: 2-byte half scale, 32 int8
    return (half(blocks[:, :2]).reshape(-1, 1) * blocks[:, 2:].view("i1").astype(np.float32)).reshape(-1)


a = gguf.GGUFReader("models/lfm3-1.7b/LFM3-1.7B-Q8_0.gguf")
b = gguf.GGUFReader("models/lfm3-vl-2b/LFM3-VL-2B-Q8_0.gguf")
B = {t.name: t for t in b.tensors}
groups = collections.defaultdict(list)
for t in a.tensors:
    x, y = weights(t), weights(B[t.name])
    name = re.sub(r"blk\.\d+\.", "", t.name)
    groups[name].append((np.linalg.norm(x - y) / np.linalg.norm(x), float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))))
for k, v in sorted(groups.items(), key=lambda kv: -np.mean([r[0] for r in kv[1]])):
    print(f"{k:38s} n={len(v):3d} rel_dist {np.mean([r[0] for r in v]):.3f} cos {np.mean([r[1] for r in v]):.3f}")
