# 🗣️ Voice tests

First tests of LFM3-Audio-2B: speech out, speech in, spoken tool calls and a live conversation over
`/v1/realtime`. The write-up and the plans are in
[Voice agents](../../docs/up-next.md#-voice-agents-agents-that-talk-to-people).

## Run it

Takes about a minute, all on the GPU in 3.3 GB. Stop any other llama-server first: it uses port 8080.

```bash
bash scripts/download-models.sh lfm3audio
(cd ../llama.cpp && cmake --build build --target llama-tts)   # the fork, as in docs/reproduce.md
pip install numpy websockets
bash bench/voice/run.sh
```

Results go to `bench/voice/rerun/`, which git ignores. `LLAMA_BIN` picks another build and
`VOICE_OUT` another output folder.

| Script | What it does |
|---|---|
| `run.sh` | Starts llama-server in its own scope, runs everything, stops the server |
| `audio_probe.py tts-asr` | Speaks 6 sentences with `llama-tts`, transcribes them, scores the word error rate |
| `audio_probe.py tools` | Speaks 4 requests with 3 tools offered, and checks the tool call or answer |
| `realtime_probe.py` | Streams 3 clips from `audio/` at real-time pace, times the first audio back, saves each reply |
| `audio_probe.py replies` | Transcribes those saved replies, to check what was said |

`audio/` holds the 2026-10-28 clips. The live test always reads those; `tts-asr` and `tools` make
new clips in the output folder, so their audio differs from run to run.

## Results

| | [2026-10-28](results/2026-10-28-8ca4060/) (`8ca4060`) | [2026-11-01](results/2026-11-01-fork-3cb2992/) (fork, `3cb2992`) |
|---|---:|---:|
| VRAM | 3,271 MiB | 3,348 MiB |
| Word error rate | 10.1%, all from numbers written as digits | 10.1%, the same |
| Tool calls that fit | 3/3 | 3/3 |
| No tool fits | miss | miss, `set_timer(minutes=0)` |
| First audio back | 0.13–0.17 s | 0.14–0.16 s |

The October raw output doesn't show which tool the miss called, because the script printed only the
empty answer. It prints the call now.

All of this is the model hearing its own generated voice, so it says little about real speech.
