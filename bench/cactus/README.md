# 🌵 Cactus's Mirage, offline, on the visual control scenes

[Cactus Compute](https://huggingface.co/Cactus-Compute) makes tiny on-device models: Needle 3 (tool calls, 35 MB),
Whistle (speech to text, 17 MB) and Mirage (vision, **21.6 MB**). They don't use GGUF or llama.cpp; they need
Cactus's own engine. We tested Mirage first, for the security-camera idea, with **no data sent anywhere**.

## Keeping it offline

What the package talks to, read from its source (cactus-compute/cactus@ee2ffc6):

| What | Where | Default | Here |
|---|---|---|---|
| Anonymous usage counts (function name, version, OS, a random install id; no prompts, images or outputs) | a Supabase endpoint | **on** | `CACTUS_TELEMETRY=0`, `DO_NOT_TRACK=1` |
| A "download" ping: re-fetches the model's `config.json` on every load | Hugging Face | on | skipped: the weights are passed as local files |
| Engine and weights, first use only, pinned and sha256-checked | Hugging Face | on | fetched once by [`fetch.py`](fetch.py), same pins and hashes |
| Hosted fine-tuning, synthetic data through OpenRouter | cactuscompute.com, openrouter.ai | only when asked | not used |

The engine itself (`libcactus.so`, 0.9 MB wheel) imports no networking functions at all (no `socket`,
`connect`, `send` or `getaddrinfo`; only libc, libm, libpthread and libdl) and holds no URLs. On top of
that, everything runs in [`sandbox.sh`](sandbox.sh): Docker with `--network none`, read-only, not root, CPU
only. So nothing tried to leave, and nothing could have.

Licences: the Python package and all three models are Apache 2.0. The engine's source
([cactus-compute/engine](https://github.com/cactus-compute/engine)) has its own licence: free for
individuals (personal, research, non-commercial), companies under $2M in funding and revenue, schools and
non-profits; others need a commercial licence. Mirage's network is defined only in that engine; Needle 3 and
Whistle also have JAX code in the Apache package.

## The test

The [visual control probe](../../docs/visual-control.md)'s glance scenes, the same 150 (30 per kind, same
seeds, same scorers) that LFM3-VL-2B was scored on. Mirage can't call tools, so it reports what it sees
and [`mirage_glance.py`](mirage_glance.py) applies each scene's rule in Python:

- **tanks**: a yes/no form field per tank ("tank_A_below_30_percent_full").
- **click**: the text Mirage reads, with its box; click the centre of the one that matches.
- **pick**: the objects it finds (name, colour, box); the matching one, or the left/right/top/bottom-most.
- **signal**: a form field for the light's colour, plus a person from the form or the object list.
- **gauge**: a form field "value the red needle points at", 0, 10, … 100.

That split, the model looks and code decides, is easier than the probe's job, where the model applies the
rule itself. Forms only take choices, counts, text and (undocumented) boxes; an integer field is a
*count*, so "gauge reading 0–100" comes back 0 ([`mirage_forms.py`](mirage_forms.py),
[log](results/forms.log)).

## Results

RTX 2070 Max-Q laptop, CPU only (12 threads), measured 2026-11-06. [Log](results/glance-30.log),
[every answer](results/mirage-glance-30.json).

| | 🛢️ | 🖱️ | 🤏 | 🚦 | ⏲️ | **All** | Per decision |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Mirage** (21.6 MB, CPU), code applies the rules | 6 | **30** | 17 | 17 | 10 | **80/150** | 2.2 s |
| [LFM3-VL-2B](../../docs/models/lfm3-vl-2b.md) (GPU), thinking on / off, forced call | 25 / 24 | 30 / 30 | 29 / 26 | 30 / 27 | 30 / 10 | *144 / 117* | 2.7 / 0.4 s |

- **Reading text is excellent.** It found the right button 30 of 30 times, and declined when the button
  wasn't there. On the gauge it read every dial number and "PRESSURE".
- **Where things are, yes; what colour, no.** "The leftmost object" 14/15, but "the green circle" 3/15:
  it names every shape on a table the same ("triangle", "triangle", …) and gives no colour.
- **It doesn't read values.** The gauge answer is "100" whatever the needle says (a "40" three times), and the
  tank answers barely change between pictures. Both scores are what a constant answer gets (10/30 and 6/30).
- **No stick figures.** It never saw the drawn person on the crosswalk (0/15; it called it a "banner",
  "flag" or "text"). The light's colour was right 15/15 without a person and 8/15 with one.
- Fast and tiny: 0.2 s to load, 1.2–1.9 s a picture on the CPU (signal is two calls, 4.5 s).

These are flat drawings. Mirage is trained for photos (its own descriptions of our two image-gen ornaments
were good: "yellow star ornament with red ribbon, pine branch, string lights"), so stick figures and dials
are a hard test for it. The question for the camera, real photos of people, is still open.

## Rerun

```bash
python -I bench/cactus/fetch.py models/cactus                      # once, from Hugging Face only
git clone https://github.com/cactus-compute/cactus /tmp/cx && git -C /tmp/cx checkout ee2ffc67fe0f9537d4928e158440c7d228a63100
mkdir -p models/cactus/py && cp -r /tmp/cx/cactus models/cactus/py/
bash bench/cactus/sandbox.sh -I /bench/cactus/mirage_glance.py 30   # no network
```

The first run (`results/glance-30-v1-bbox-scaled-twice.log`) had a harness bug: Mirage's boxes are already
0–1000 and we scaled them again, so click and pick missed. Fixed before the run above.
