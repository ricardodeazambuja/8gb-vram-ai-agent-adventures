# 🔧 Troubleshooting: things that bit us, so they don't bite you

Every entry here happened on this laptop (RTX 2070 Max-Q 8 GB, 32 GB RAM), usually at the worst
possible moment. Format: what you see → what's really going on → what to do.

**Contents:** [Downloads & model files](#downloads--model-files) ·
[Memory](#memory) · [Model-specific](#model-specific) · [Thinking & templates](#thinking--templates) ·
[pi & processes](#pi--processes) · [Images](#images)

---

## Downloads & model files

### The model loads fine, then spews garbage or `<unused32><unused32>…`

**What's going on:** most likely a **corrupt download**. A GGUF can have the right size and a
valid header and still contain wrong tensor data. It happened here with an `aria2c -x16` download
from Hugging Face's CDN that was interrupted and resumed: it loaded, ran at full speed, and printed
nothing but `<unused32>`. It looked exactly like a kernel bug. It wasn't.

**Fix:** check the file's sha256 against the one Hugging Face lists for it (the LFS object id)
**before** blaming llama.cpp, the quant or the fork:

```bash
sha256sum models/<dir>/<file>.gguf
```

`scripts/download-models.sh` does this for you: every file is pinned to a repo revision and
a sha256, and a mismatch deletes the file. Never reuse an interrupted partial download.

### A new model gives garbage on a quick raw-prompt test

**What's going on:** thinking/instruct models are trained on their chat format. Feed them a raw,
unformatted prompt through `/completion` and you push them out of distribution: repetition, CJK
noise, nonsense. It's expected behaviour, not a broken build.

**Fix:** test the way pi talks to it, through the chat template:
`/v1/chat/completions` with the server started with `--jinja` (all scripts here do). If it's still
garbage, try a second, less aggressive quant (a Q2_K can be genuinely broken; see
[the graveyard](graveyard.md)). Only then suspect the build.

### Hugging Face downloads stall at 0 B/s, no error

**What's going on:** on this network, IPv6 to Hugging Face was broken: `curl -6
https://huggingface.co` timed out while `curl -4` answered in ~0.2 s. Downloads would start, then
hang forever with a connection stuck in `SYN-SENT` to an IPv6 address.

**Check:** `ss -tnp | grep <pid of the download>`, look for IPv6 `SYN-SENT`.

**Fix (no root needed):** force IPv4 for that one process. Put this in a folder as
`sitecustomize.py`:

```python
import socket
_orig = socket.getaddrinfo
def _v4(host, port, family=0, *a, **k):
    return _orig(host, port, socket.AF_INET, *a, **k)
socket.getaddrinfo = _v4
```

then run the download with that folder on `PYTHONPATH`, and disable the Rust "Xet" transfer path,
which ignores the Python patch:

```bash
PYTHONPATH=/path/to/that/folder HF_HUB_DISABLE_XET=1 bash scripts/download-models.sh flint
```

### `ERROR: no 'hf' CLI in env 'llamacpp'`

**What's going on:** the download script insists on the `hf` CLI from inside a specific conda
environment (`ENV_NAME`, default `llamacpp`), so results don't depend on whatever else is on your
`PATH`.

**Fix:** install it there, or point the script at an env that has it:

```bash
mamba install -n llamacpp -c conda-forge huggingface_hub
ENV_NAME=myenv bash scripts/download-models.sh spark
```

## Memory

### The server loads cleanly, then dies with `Killed` a few seconds later

**What's going on:** that's the Linux **OOM killer**, and it's about **system RAM**, not VRAM. The
MoE models keep 15–20 GB of experts in RAM; add a big context and a busy desktop and the kernel
shoots the biggest process. Here it happened ~15 s after pi connected, twice, with a 128K f16
context.

**Check:**

```bash
journalctl -k | grep -i oom
```

**Fix:** a smaller context (`CTX=65536`), a quantized KV cache (`KVQUANT=q8_0`), a smaller quant,
fewer browser tabs, or more swap. Re-tuning `NCMOE` won't help: the GPU was fine.

### It loads, then crashes on the first request with `CUDA error: out of memory`

**What's going on:** some memory is only allocated when real work starts: compute buffers and the
cuBLAS workspace. Two cases here: Qwen3.8-27B at 36 GPU layers loaded at 7.65 GB and died on the
first request; the Gemma 12B agentic model at `-ngl 99` OOMed with llama.cpp's default
`n_parallel=4` + `n_batch=2048` (a ~0.5 GB compute buffer).

**Fix:** leave real headroom (the defaults here aim for ~0.3–0.5 GB free next to a desktop), use
`--parallel 1` and a smaller batch (`NP=1 BATCH=512 UBATCH=512`), or one layer fewer on the GPU.
And always test with an actual request, not just a successful load.

### 128K "should fit" by the math, but runs out of memory

**What's going on:** with a quantized KV cache, flash attention keeps an f16 copy of one layer's
K and V in the compute buffer, which grows with context. Spark-X2.6-4B's q8_0 cache "costs"
~19 KiB/token on paper and ~24 KiB/token in practice; 128K missed by ~165 MiB.

**Fix:** treat your calculated KV size as a floor. Spark's default is the measured fit, 108K
(`CTX=110592`). More in [how it fits](how-it-fits.md).

## Model-specific

### Bonsai 2: server aborts on the first request, `no device code compatible with CUDA arch 750`

**What's going on:** PrismML's llama.cpp fork has no Turing (sm_75) code for its flash-attention
kernel with a **q8_0** KV cache.

**Fix:** use an **f16** KV cache. It's the default for `QMODEL=bonsai2` (and the reason Bonsai
stops at 16K context on 8 GB).

### Dory: slow, and VRAM goes *down* as you add loops

**What's going on:** the Dory build counts every pass through the looped block as extra layers,
so `-ngl 99` doesn't mean "everything" any more. At 4 loops it left 2.9 GB of weights and most of
the KV cache on the CPU: 3.8 tok/s instead of 19.

**Fix:** `-ngl 999`. The repo's Dory default already uses it.

### Spark: HTTP 400, `Unexpected reasoning effort medium. Supported types are high (default) and low.`

**What's going on:** pi sends its thinking level as the request's `reasoning_effort`, llama-server
hands it to the chat template, and Spark's template accepts only `high` or `low`. pi's default is
`medium`, so every request is rejected before a single token is generated.

**Fix:** `--thinking high` (what Spark's card evaluates with) or `--thinking low`.
`scripts/start_spark.sh` passes `--thinking high` for you unless you pick a level yourself.

### Bonsai 2: HTTP 500 when pi asks for "high" thinking

**What's going on:** same idea, different template: Bonsai's accepts a different set of effort
values than pi's levels.

**Fix:** already handled: `config/pi-provider.json` maps pi's levels onto ones Bonsai accepts,
and `start_bonsai2.sh` defaults to `medium` (higher levels think for minutes at 27 tok/s). Register
the provider with `bash scripts/configure-pi.sh`.

### Qwen3-VL: a big photo takes minutes, or kills the server

**What's going on:** a 4K photo becomes ~4,096 image tokens. With the vision projector on the CPU
that took **270 s**; on the GPU, uncapped, it ran the card out of memory and aborted the server.

**Fix:** projector on the GPU and a cap on image tokens: `MMPROJ_GPU=1 IMAGE_MAX_TOKENS=1024`
(both the `qwen3vl` defaults). The same photo then encodes in **5.4 s**. 2048 also fits, with less
headroom.

## Thinking & templates

### I can't turn thinking off from pi

**What's going on:** with llama.cpp, whether a model thinks is decided **server-side**: by the
chat template and llama-server's `--reasoning` flag. pi only forwards a `reasoning_effort` value,
and what happens to it depends on the template: Gemma's ignores it, Spark's and Bonsai's react to
it (see above), Flint's template has no thinking switch at all.

**Fix:** to really disable thinking, start the server with `--reasoning off` (or
`--chat-template-kwargs '{"enable_thinking":false}'`, for templates that support it), e.g. through
`EXTRA_ARGS`.

### The answer comes back empty

**What's going on:** a small `max_tokens`. The model spent the whole budget thinking
(`reasoning_content` is full, `content` is empty). Not a bug.

**Fix:** give it room, thousands of tokens for the thinking models.

## pi & processes

### `pi -p "…"` hangs forever

**What's going on:** in non-interactive mode pi waits on stdin if it's left open, which happens
under scripts and harnesses.

**Fix:** `pi -p "your prompt" --no-session </dev/null`.

### The launcher refuses: `port 8080 is already serving 'X'`

**What's going on:** another model is still running. The `start_*.sh` launchers refuse to reuse a
server that's serving a different model, so pi never talks to the wrong one by accident.

**Fix:** `bash scripts/stop-server.sh`, then start again. Only one model fits on the card at a
time anyway.

### A `llama-server` is still around after you killed things

**What's going on:** the servers are launched through `mamba run`, which wraps the real process,
so killing the PID you got back (`kill $!`) can miss the actual `llama-server`. After a timeout or
an interrupted script you can end up with an orphan holding all the VRAM, and the next model fails
to load.

**Fix:** `bash scripts/stop-server.sh` (it finds the process by its port). If something survives,
`kill -9` it, and check that `nvidia-smi` shows the memory free before loading the next model.

### `pkill -f llama-server` kills your own shell (exit 144)

**What's going on:** `pkill -f` matches full command lines, including the shell that's running
the `pkill`, because its command line contains the pattern.

**Fix:** stop servers by port (`scripts/stop-server.sh`) or by PID. If you must pattern-match, use
the bracket trick: `pgrep -f '[l]lama-server'`.

## Images

### The model "can't see" an image you gave it

**What's going on:** two different ways to get here:

1. **pi doesn't send it.** pi only attaches images for models whose entry says
   `"input": ["text", "image"]`. An unknown model id falls back to text-only, and the model
   politely replies that it can't view images. Register the repo's model entries with
   `bash scripts/configure-pi.sh`.
2. **The model never looked.** In an agent session the image only reaches the vision projector if
   the model actually **reads the file** with pi's `read` tool. Gemma 4 once "judged" a PNG it had
   just rendered by reasoning about the SVG source, without ever opening it. If seeing the image
   matters, say so: "open the PNG with the read tool and look at it".

Also check the server was started with its projector (`--image` on the run scripts) and that the
model can take images at all: Flint, Spark and Dory are text only.

### Bounding boxes come back in the wrong place

**What's going on:** most models ignore "give me pixel coordinates" and answer in the format they
were trained on. In the [vision probe](vision-probe.md), all the models except Bonsai 2 answered
on a 0–1000 grid instead of in pixels. The Qwen models used `[x_min, y_min, x_max, y_max]` (often
under the key `bbox_2d`). Gemma 4 used Gemma's native `box_2d` = `[y_min, x_min, y_max, x_max]`
in two of three images, and `[x_min, y_min, x_max, y_max]` when it kept the requested `bbox` key.
Converted, their boxes were good.

**Fix:** convert from the model's native format instead of fighting it:

```python
# Gemma 4: box_2d = [y1, x1, y2, x2] in 0..1000
x1, y1, x2, y2 = b[1] * W / 1000, b[0] * H / 1000, b[3] * W / 1000, b[2] * H / 1000
# Qwen: bbox_2d = [x1, y1, x2, y2] in 0..1000
x1, y1, x2, y2 = b[0] * W / 1000, b[1] * H / 1000, b[2] * W / 1000, b[3] * H / 1000
```
