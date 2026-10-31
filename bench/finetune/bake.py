#!/usr/bin/env python3
"""Bake a norm-only or IA3 adapter into a copy of the model's weights, so llama.cpp can run it.

Plain tensor maths on model.safetensors: nothing from the model repo runs. Norm-only replaces the
gains; IA3 scales a layer's outputs (rows of its weight) or, for feed-forward layers, its inputs
(columns). The result converts to GGUF like the original.

Usage: bake.py --model HF_DIR --adapter ADAPTER_DIR --out OUT_DIR   (no --adapter: plain copy)
"""
import argparse
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

KEEP = (".json", ".jinja", ".txt", ".model")


def norm_updates(adapter, weights):
    state = torch.load(os.path.join(adapter, "norms.pt"), weights_only=True)
    out = {}
    for name, value in state.items():
        assert weights[name].shape == value.shape, name
        out[name] = value
    return out


def ia3_updates(adapter, weights):
    path = os.path.join(adapter, "adapter_model.safetensors")
    prefix, suffix = "base_model.model.", ".ia3_l"
    out = {}
    with safe_open(path, "pt") as f:
        for key in f.keys():  # noqa: SIM118 (safe_open has no __iter__)
            assert key.startswith(prefix) and key.endswith(suffix), key
            name = key[len(prefix):-len(suffix)]
            assert name + ".bias" not in weights, f"{name} has a bias, which IA3 would scale too"
            name += ".weight"
            scale = f.get_tensor(key).float()
            base = weights[name]
            rows, cols = scale.shape
            assert (rows == base.shape[0] and cols == 1) or (rows == 1 and cols == base.shape[1]), name
            out[name] = base.float() * scale
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    weights = load_file(os.path.join(a.model, "model.safetensors"))
    updates = {}
    if a.adapter:
        is_norm = os.path.exists(os.path.join(a.adapter, "norms.pt"))
        updates = (norm_updates if is_norm else ia3_updates)(a.adapter, weights)
    for name, value in updates.items():
        weights[name] = value.to(weights[name].dtype)
    os.makedirs(a.out, exist_ok=True)
    save_file(weights, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
    for f in os.listdir(a.model):
        if f.endswith(KEEP):
            shutil.copy(os.path.join(a.model, f), a.out)
    print(f"baked {len(updates)} tensors into {a.out}")


if __name__ == "__main__":
    main()
