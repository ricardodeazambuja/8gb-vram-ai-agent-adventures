#!/usr/bin/env python3
"""LFM3 text weights from a GGUF, under the Python model's names, for loading into VL's code.

check: VL's GGUF against VL's safetensors, to prove the mapping. export: write 1.7B's weights.
"""
import glob
import re
import sys

import gguf
import numpy as np
import torch
from safetensors import safe_open

HF = "models/lfm3-vl-2b-hf"
LAYER = {"attn_norm": "input_layernorm", "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
         "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj", "attn_gate": "self_attn.gate_proj",
         "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm"}


def half(raw):
    return np.frombuffer(raw.tobytes(), "<f2").astype(np.float32)


def values(t):
    raw = np.asarray(t.data).reshape(-1)
    if int(t.tensor_type) == 0:
        x = np.frombuffer(raw.tobytes(), "<f4")
    else:
        blocks = np.frombuffer(raw.tobytes(), "u1").reshape(-1, 34)
        x = (half(blocks[:, :2]).reshape(-1, 1) * blocks[:, 2:].view("i1").astype(np.float32)).reshape(-1)
    return torch.from_numpy(x.copy()).reshape(tuple(int(s) for s in reversed(t.shape)))


def hf_tensors():
    out = {}
    for f in glob.glob(f"{HF}/*.safetensors"):
        with safe_open(f, "pt") as s:
            for k in s.keys():  # noqa: SIM118
                if "language_model" in k or k.startswith("lm_head"):
                    out[k] = s.get_tensor(k)
    return out


def match(reader, hf):
    """Pair each GGUF tensor with the Python one holding the same numbers, using VL as the key."""
    pairs = {}
    by_shape = {}
    for k, v in hf.items():
        by_shape.setdefault(tuple(v.shape), []).append(k)
        if v.dim() == 3:  # shortconv: the GGUF drops the middle dimension
            by_shape.setdefault((v.shape[0], v.shape[2]), []).append(k)
    for t in reader.tensors:
        x = values(t)
        best, score = None, -1
        m = re.match(r"blk\.(\d+)\.", t.name)
        names = [k for k in by_shape.get(tuple(x.shape), []) + by_shape.get(tuple(x.shape[::-1]), [])
                 if not m or f"layers.{m.group(1)}." in k]
        for k in names:
            y = hf[k].float()
            y = y.reshape(x.shape) if y.numel() == x.numel() and y.dim() == 3 else y
            y = y if y.shape == x.shape else y.t()
            c = torch.nn.functional.cosine_similarity(x.reshape(-1), y.reshape(-1), dim=0).item()
            if c > score:
                best, score = k, c
        pairs[t.name] = (best, score)
    return pairs


mode = sys.argv[1]
hf = hf_tensors()
vl = gguf.GGUFReader("models/lfm3-vl-2b/LFM3-VL-2B-Q8_0.gguf")
pairs = match(vl, hf)
weak = {k: v for k, v in pairs.items() if v[1] < 0.99}
print(len(pairs), "tensors,", len(hf), "python tensors, weak matches:", {k: round(v[1], 3) for k, v in list(weak.items())[:10]},
      len(weak), "names used twice:", len(pairs) - len({v[0] for v in pairs.values()}))
if mode == "export":
    state = {}
    for t in gguf.GGUFReader("models/lfm3-1.7b/LFM3-1.7B-Q8_0.gguf").tensors:
        k = pairs[t.name][0]
        if k is None or pairs[t.name][1] < 0.99:
            continue
        x = values(t)
        state[k] = (x.reshape(hf[k].shape) if hf[k].dim() == 3 else x if x.shape == hf[k].shape else x.t()).half()
    missing = set(hf) - set(state)
    print("exported", len(state), "missing", sorted(missing)[:5], len(missing))
    torch.save(state, "models/lfm3-vl-2b-adapters/proj/lfm3-1.7b-text.pt")
