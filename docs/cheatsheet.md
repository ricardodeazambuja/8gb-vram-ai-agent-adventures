# 🗺️ Cheatsheet: I want to…, so I run…

Every model here, by job, with the one command that starts it. The commands assume the
[one-time setup](getting-started.md) is done and the model is downloaded (`scripts/llm list` says which are,
and how to get the rest). One model on the GPU at a time: `scripts/llm stop` frees it.

## By task

| I want to… | Run | Notes |
|---|---|---|
| **have an agent write or fix code** | `scripts/llm groot` | the everyday pick: fastest of the big ones, 18/18 on [pibench](pibench.md) |
| …the most careful one | `scripts/llm flint` | 18/18, looks before it leaps |
| …one that also looks at pictures | `scripts/llm gemma --image` | 18/18 and 100% on the [vision probe](vision-probe.md); paste images into pi |
| …in 6 GB, leaving room on the GPU | `scripts/llm bonsai2` | 16/18, 32K context |
| …as fast as possible, for small chores | `scripts/llm lfm3` | 106 tok/s, 12/18 |
| **ask one question, no agent** | `python3 examples/chat.py -m groot "…"` | any chat model by name; streams the answer |
| …and get a one-shot answer from pi | `scripts/llm groot -p "explain @README.md"` | pi reads the files you `@`-mention |
| **look at a picture and decide** | `python3 examples/look.py photo.jpg --task "…" --tools a,b` | [LFM3-VL-2B](models/lfm3-vl-2b.md): one tool call (or none), 2.3 s; it prints the call, you act on it |
| **find things in a picture** | `python3 examples/locate.py photo.jpg "the red mug"` | [LocateAnything-3B](models/locateanything-3b.md): boxes in pixels; add `--all` when it may be absent, `--draw out.png` to see them |
| **read text in a picture, describe it** | `scripts/llm qwen3vl`, then send the image to its API | [Qwen3-VL-8B](models/qwen3-vl-8b.md); or `scripts/llm gemma --image` and paste the image into pi |
| **search my docs or code by meaning** | `python3 examples/search.py "question" --dir notes/` | [EmbeddingGemma 2](models/embeddinggemma-2.md); `--ext .py --code` for code |
| **a fast yes/no or pick-one decision** | `python3 examples/decide.py yesno "text" --question "…"` | [d1-2B](models/d1-2b.md), ~30 ms; `choose … --options a,b,c` to pick one |
| …is this shell command safe to run? | `python3 examples/decide.py guard "rm -rf ~/"` | `-m clefflash` for the bigger, better judge |
| **turn speech into text** | `python3 examples/listen.py clip.wav` | [LFM3-Audio-2B](models/lfm3-audio-2b.md), 0.3 s; `--tools a,b` lets a spoken request pick a tool |
| **turn text into speech** | `python3 examples/speak.py "Hello"` | writes a WAV; stop any server first |
| **make a picture** | `scripts/imagine "a red fox in the snow"` | [FLUX.2 Klein](models/flux2-klein-4b.md), ~25 s; prints the file's path |
| …with readable text in it | `scripts/imagine "a sign that says OPEN" -m iris` | [Iris-3B Turbo](models/iris-3b.md), ~33 s |
| …in a style I trained | `scripts/imagine "…" -m sdxl --lora FILE.safetensors` | [SDXL](models/sdxl.md), ~47 s |
| **change a picture** | `scripts/imagine "the same street at night" -i in.png` | `--mask m.png` redraws only the white part; `-r ref.png` edits from a reference (Klein) |
| **serve a model to my own program** | `scripts/llm <name> serve` | OpenAI API on `http://127.0.0.1:8080/v1`; [examples/](../examples/README.md) show the requests |
| **see what's running, stop it** | `scripts/llm status`, `scripts/llm stop` | `stop` only stops what `scripts/llm` started; `--force` for anything on the port |

## Every model

Generated from [`scripts/llm`](../scripts/llm) (`scripts/llm list --md`); `python3 utils/check_docs.py --fix`
refreshes it. Chat models open pi; vision and helper models start the server and print its URL. Add `serve` to
skip pi, `pi` to chat with a vision model, `--image` to give a chat model its vision projector, `--static` to run
an MoE model without the expert cache.

<!-- llm-list:start (generated: python3 utils/check_docs.py --fix) -->
| Model | Job | Command | What it is |
|---|---|---|---|
| [Groot-19B-A3B](models/groot-19b-a3b.md) | agent | `scripts/llm groot` | the everyday coding agent, fastest of the big ones (pibench 18/18) |
| [Flint 40B MoE](models/flint.md) | agent | `scripts/llm flint` | the careful coder (18/18) |
| [Gemma 4 26B-A4B](models/gemma4-26b-a4b.md) | agent | `scripts/llm gemma` | agent that also sees (18/18, vision 100%) |
| [Qwen3.6-35B-A3B](models/qwen3.6-35b-a3b.md) | agent | `scripts/llm qwen36` | our yardstick, agent + vision (16/18, vision 100%) |
| [Bonsai 2 27B ternary](models/bonsai2-27b.md) | agent | `scripts/llm bonsai2` | big brain in 6 GB, 32K context (16/18) |
| [Qwen3.8 35B-A3B distill](models/qwen3.8-35b-a3b-distill.md) | agent | `scripts/llm distill` | quick jobs (16/18) |
| [Ornith-1.5-35B-A3B](models/ornith-1.5-35b-a3b.md) | agent | `scripts/llm ornith` | Qwen3.6 that trained on its own tasks (14/18) |
| [ThinkingCap Gemma 4 26B](models/thinkingcap-gemma4-26b-a4b.md) | agent | `scripts/llm thinkingcap` | Gemma told to think less (17/18) |
| [Nemotron 3.5 Bolt 74B](models/nemotron-bolt-74b.md) | agent | `scripts/llm bolt` | the biggest, slow and thoughtful (15/18) |
| [LFM2.5-24B-A2B](models/lfm2.5-24b-a2b.md) | agent | `scripts/llm lfm25` | not a Qwen, trained in agent harnesses (10/18) |
| [Spark-X2.6-4B](models/spark-x2.6-4b.md) | agent | `scripts/llm spark` | small and fast (14/18) |
| [Dory1.0-4B looped](models/dory1.0-4b.md) | agent | `scripts/llm dory` | Spark's score at 4B (14/18) |
| [LFM3-1.7B](models/lfm3-1.7b.md) | agent | `scripts/llm lfm3` | tiny, 106 tok/s, good at chores (12/18) |
| [Nemotron 3.5 Bolt E4B](models/nemotron-bolt-e4b.md) | agent | `scripts/llm e4b` | voice-assistant model (7/18) |
| [TwIL-8B](models/twil-8b.md) | agent | `scripts/llm twil` (server only) | math and code reasoner, not measured yet (serve only until pibench) |
| [VibeThinker 3B](models/vibethinker.md) | agent | `scripts/llm vibe3b` | one-shot code and math (pi through a tool-call shim) |
| [VibeThinker 1.5B](models/vibethinker.md) | agent | `scripts/llm vibe15` | one-shot math and code (pi through a tool-call shim) |
| [Qwen3-VL-8B](models/qwen3-vl-8b.md) | vision | `scripts/llm qwen3vl` | reads an image in 5 s |
| [LFM3-VL-2B](models/lfm3-vl-2b.md) | vision | `scripts/llm lfm3vl` | one picture, one decision (144/150); examples/look.py |
| [LFM2.5-VL-450M](models/lfm2.5-vl-450m.md) | vision | `scripts/llm lfm25vl` | tiny vision model for fine-tuning |
| [LocateAnything-3B](models/locateanything-3b.md) | vision | `scripts/llm locate` | boxes for what you describe; examples/locate.py |
| [EmbeddingGemma 2](models/embeddinggemma-2.md) | helper | `scripts/llm eg2` | search over code and docs; examples/search.py |
| [d1-2B](models/d1-2b.md) | helper | `scripts/llm d1` | yes/no and pick-one decisions in 27 ms; examples/decide.py |
| [Clef-Flash](models/clef-flash.md) | helper | `scripts/llm clefflash` | the same, bigger and better (7 GB); examples/decide.py |
| [LFM3-Audio-2B](models/lfm3-audio-2b.md) | helper | `scripts/llm lfm3audio` | hears and speaks; examples/listen.py |
<!-- llm-list:end -->

Pictures: `scripts/imagine --list` (Klein, Iris, SDXL). Every option: `scripts/llm -h`, `scripts/imagine -h`,
and each example's `-h`.

## When something goes wrong

| It says… | Do this |
|---|---|
| `port 8080 already serves X` | another model is running: `scripts/llm stop` |
| `the expert cache doesn't fit next to what is on the GPU now` | close the browser, or `scripts/llm <name> --static` |
| `is not downloaded` | the command it prints: `bash scripts/download-models.sh <name>` |
| `a llama-server … is using the GPU` (imagine) | `scripts/llm stop`, then try again |
| anything else | [troubleshooting](troubleshooting.md) |
