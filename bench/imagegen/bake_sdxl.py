#!/usr/bin/env python3
"""Bake a norm-only SDXL run into a full checkpoint that sd-cli can load.

Usage: bake_sdxl.py --run subject-norm
Finds each trained norm's place in the single-file checkpoint by matching the base model's exact
values, and refuses to write if any match isn't unique. Output: models/image-gen/style-runs/<run>/baked.safetensors.
"""
import argparse

import torch
from safetensors.torch import load_file, save_file
from train_style import CKPT, RUNS, pipe_load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    trained = torch.load(RUNS / a.run / "norms.pt", map_location="cpu", weights_only=True)
    base = pipe_load().unet.state_dict()
    ck = load_file(CKPT)
    index = {}
    for k, v in ck.items():
        if k.startswith("model.diffusion_model.") and v.ndim == 1:
            index.setdefault((v.shape[0], v.half().numpy().tobytes()), []).append(k)
    used, n = set(), 0
    groups = {}
    for name, v in trained.items():
        b = base[name].cpu().half()
        groups.setdefault(b.numpy().tobytes(), []).append(name)
    for raw, names in groups.items():
        hits = index.get((len(raw) // 2, raw), [])
        if len(hits) != len(names):
            raise SystemExit(f"{names[:2]}: {len(names)} trained tensors but {len(hits)} checkpoint matches")
        if len(names) > 1:
            raise SystemExit(f"ambiguous values shared by {names[:2]}")
        ck[hits[0]] = trained[names[0]].to(ck[hits[0]].dtype)
        used.add(hits[0])
        n += 1
    out = RUNS / a.run / "baked.safetensors"
    save_file(ck, out)
    print(f"baked {n} of {len(trained)} tensors into {out}")


if __name__ == "__main__":
    main()
