# 🗣️ LFM3-Audio-2B — speech in, speech out, tool calls

> **TL;DR**
>
> | | |
> |---|---|
> | On disk | 1.85 GB (`Q8_0`) + 0.57 GB audio projector |
> | Where it lives | entirely on the GPU (3.3 GB VRAM at 16K context) |
> | Speech to text | 6/6 sentences right, 0.2 s a clip |
> | Spoken tool calls | 3/3 over the chat API; none over `/v1/realtime` yet |
> | Live reply | first audio in 0.10–0.18 s |
> | [pibench](../pibench.md) | not run |
> | [Security audit](../security.md) | clean |
> | Launch | `scripts/llm lfm3audio` · [`examples/listen.py`](../../examples/listen.py) · [`examples/speak.py`](../../examples/speak.py) |

## What is it?

[LiquidAI's speech model](https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF) (LFM Open License v1.0,
commercial use under $10M revenue). It hears and speaks 18 languages and can call tools from a voice
request, which makes it the speech-native route in our [voice agent plans](../up-next.md#-voice-agents-agents-that-talk-to-people).

## Install & run

Build our llama.cpp once: [getting-started.md](../getting-started.md). Then:

```bash
bash scripts/download-models.sh lfm3audio
(cd ../llama.cpp && cmake --build build --target llama-tts)     # only for text to speech
scripts/llm lfm3audio                                           # server on port 8080
```

It's a server to call, not a pi model, so `scripts/llm` only serves it. The projector always loads.
[`examples/listen.py`](../../examples/listen.py) and [`examples/speak.py`](../../examples/speak.py) are the short
versions of Listen and Speak below. (The building block: `QMODEL=lfm3audio bash scripts/run-qwen.sh`.)

- **Listen:** send a WAV to `/v1/chat/completions` as `input_audio`, with the system message
  "Perform ASR in English."
- **Spoken tool calls:** the same request, with `tools` and a short assistant system message.
- **Talk:** stream audio to `ws://127.0.0.1:8080/v1/realtime` and read the audio that comes back.
- **Speak:** `llama-tts -m MODEL -mm PROJECTOR -ngl 99 --tts-voice female -p "Hello" -o out.wav`

[`bench/voice/`](../../bench/voice/README.md) has working code for each one, and `bash bench/voice/run.sh`
runs them all in about a minute.

## Our numbers

**First test of LFM3-Audio-2B (2026-10-28, Q8_0 + F16 projector, `vendor/llama.cpp-lfm3`
@`8ca4060`, all on the GPU in 3.3 GB).** Audited clean first ([security.md](../security.md)). Test
audio was made by the model itself (`llama-tts`), so the text is known and scoring is automatic.
Reruns on our fork gave the same scores.

| Test | Result |
|---|---|
| Text to speech (`llama-tts`, female and male voices) | 3.4–5.4 s of speech in 2.5–2.9 s, faster than real time |
| Speech to text (llama-server chat API, "Perform ASR in English.") | 6/6 sentences right word for word; the raw WER is 10% only because it writes numbers as digits ("7.30", "8GB"). 0.2 s a clip |
| Spoken request → tool call (3 tools offered) | 3/3 right: `get_weather(city="Paris")`, `set_timer(minutes=10)`, `send_message(to="Maria", text="I'll be late")`, 0.2–0.3 s each |
| Spoken question with tools offered but none fitting | **miss**: it called a tool with empty or useless arguments instead of answering (`get_weather` with no city, on the rerun `set_timer(minutes=0)`); without tools it answers "312 divided by 4 equals 78." Too eager with tools |
| **Live conversation over `/v1/realtime`** (audio streamed at real-time pace, 100 ms chunks) | **first audio back 0.13–0.17 s after the end of the question**; a full spoken reply (3–9 s of speech) generated in 0.7–1.9 s. Its replies, transcribed back, matched what it meant, with two slips ("WhatsApp" for "website", "311" for "312") |

So the speech-native route works on this laptop today, with latency well under a human pause.

## Gotchas

- **It's too keen on tools.** With tools offered and none fitting, it calls one anyway, with empty or
  useless arguments.
- **No tools over `/v1/realtime` yet** in our setup, so in a live conversation it politely refuses.
- All tests use its own clean generated voice, so real speech is untested.
- A default server build doesn't include `llama-tts`: build it as above.
