"""Download Cactus's engine and model weights from Hugging Face only, pinned and sha256-checked.

    python -I bench/cactus/fetch.py models/cactus

The revisions and hashes are the ones in cactus-compute/cactus@ee2ffc6 (cactus/agent/fetch.py); the
wheel's hash comes from the pinned release manifest. Nothing here contacts cactuscompute.com. The engine
library is unzipped from the wheel into OUT/engine/libcactus.so; it only ever runs inside sandbox.sh.
"""
import hashlib
import json
import os
import shutil
import sys
import zipfile

from huggingface_hub import hf_hub_download

ENGINE = ("Cactus-Compute/cactus", "1a028f5a3972b67dd8d8752f882113d475e7ec4c")
MANIFEST_SHA256 = "b6a39fb21bc77780c029c45d4cc206b60f883146fa2dbc28bfcc9995fe72b871"
WHEEL = "python/cactus_runtime-4.0.0-py3-none-manylinux2014_x86_64.whl"
MODELS = {
    "mirage.cact": ("Cactus-Compute/mirage", "76c1e9e53381aca8fc3843c636216c96ff220122",
                    "4249a0f7515272ff76b9414bec17d7e7daf84b15b66c7fb4f17fce1e66f9f237"),
    "whistle.cact": ("Cactus-Compute/whistle", "741118687e1f008f06d1351a402a731274789790",
                     "b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb"),
    "needle3.cact": ("Cactus-Compute/needle3", "d6c7d6a2e9655505e0f8e23339356dd1705900eb",
                     "c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get(out, repo, rev, name, want):
    path = hf_hub_download(repo_id=repo, filename=name, revision=rev)
    got = sha256(path)
    if got != want:
        sys.exit(f"MISMATCH {repo}/{name}: {got}")
    dst = os.path.join(out, os.path.basename(name))
    shutil.copyfile(path, dst)
    print(f"ok {repo}@{rev[:7]} {name} {os.path.getsize(dst) / 1e6:.1f} MB {got}")
    return dst


def main(out):
    os.makedirs(os.path.join(out, "engine"), exist_ok=True)
    with open(get(out, *ENGINE, "releases/4.0.0/manifest.json", MANIFEST_SHA256)) as f:
        files = json.load(f)["files"]
    wheel = get(out, *ENGINE, "releases/4.0.0/" + WHEEL, files[WHEEL]["sha256"])
    with zipfile.ZipFile(wheel) as z, open(os.path.join(out, "engine", "libcactus.so"), "wb") as lib:
        lib.write(z.read("cactus/libcactus.so"))
    for name, (repo, rev, want) in MODELS.items():
        get(out, repo, rev, name, want)


if __name__ == "__main__":
    main(sys.argv[1])
