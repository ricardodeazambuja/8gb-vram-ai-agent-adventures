#!/usr/bin/env python3
"""Make a stand-in for "your own photos": one invented person, drawn by FLUX.2 Klein in many places.

Usage: make_subject.py
One portrait, then reference edits of it. Pictures go to models/image-gen/train/subject/ (git-ignored).
The last four are held out of training, to compare against.
"""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SD = ROOT.parent / "stable-diffusion.cpp/build/bin/sd-cli"
M = ROOT / "models/image-gen"
OUT = M / "train/subject"
KLEIN = ["--diffusion-model", M / "flux2-klein/flux-2-klein-4b-Q4_0.gguf",
         "--vae", M / "flux2-klein/split_files/vae/flux2-vae.safetensors",
         "--llm", M / "flux2-klein/Qwen3-4B-Q4_K_M.gguf",
         "--cfg-scale", "1.0", "--steps", "4", "--backend", "te=cpu", "--vae-tiling", "-H", "1024", "-W", "1024"]
PERSON = ("portrait photograph of an invented man in his forties with a short red beard, round tortoiseshell "
          "glasses, a gap in his left eyebrow and freckles, neutral background, natural light")
SITUATIONS = [
    "laughing outdoors in a park, close-up", "reading a book in a cafe", "riding a bicycle on a city street",
    "cooking in a home kitchen", "standing on a snowy mountain in a red jacket", "at a desk with a laptop in an office",
    "side profile on a beach at sunset", "wearing a suit at a wedding", "playing an acoustic guitar on a sofa",
    "hiking in a pine forest with a backpack", "walking in the rain with an umbrella",
    "looking out of a train window", "gardening with a straw hat", "in a workshop holding a hammer",
    "seen from the waist up in a museum", "sitting on a stone bench in an old town square",
    "drinking coffee on a balcony, morning light", "in a supermarket holding apples", "fishing on a lake from a small boat",
    "in a library between the shelves", "at a street market at night", "looking down, three-quarter view, in a greenhouse",
    "wearing a grey hoodie on a rooftop", "in a blue shirt at a kitchen table, candid",
]


def run(out, prompt, seed, ref=None):
    cmd = [SD, *KLEIN, "-p", prompt, "-s", seed, "-o", out, *(["-r", ref] if ref else [])]
    subprocess.run(list(map(str, cmd)), check=True, capture_output=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seed_pic = OUT / "ref.png"
    if not seed_pic.exists():
        run(seed_pic, PERSON, 11)
    rows = []
    for i, s in enumerate(SITUATIONS):
        out = OUT / f"{i:02d}.png"
        prompt = f"photograph of the same man with the same face, beard and glasses, {s}"
        if not out.exists():
            run(out, prompt, 100 + i, seed_pic)
        rows.append({"file": out.name, "caption": s, "held_out": i >= len(SITUATIONS) - 4})
        print(i, s, flush=True)
    (OUT / "captions.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
