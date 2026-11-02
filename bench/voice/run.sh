#!/usr/bin/env bash
# First LFM3-Audio-2B tests: speech out and in, spoken tool calls, live /v1/realtime.
#   bash bench/voice/run.sh              # writes to bench/voice/rerun/
# See README.md. LLAMA_BIN picks the build (default ../llama.cpp/build/bin); it needs llama-server
# and llama-tts. realtime_probe.py needs numpy and websockets.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
export LLAMA_BIN="${LLAMA_BIN:-$REPO/../llama.cpp/build/bin}"
export VOICE_OUT="${VOICE_OUT:-$HERE/rerun}"
mkdir -p "$VOICE_OUT"
M="$REPO/models/lfm3-audio-2b"
systemd-run --user --scope --unit=llama-audio "$LLAMA_BIN/llama-server" \
  -m "$M/LFM3-Audio-2B-Q8_0.gguf" --mmproj "$M/mmproj-LFM3-Audio-2B-F16.gguf" \
  -ngl 99 -c 16384 --jinja --host 127.0.0.1 --port 8080 > "$VOICE_OUT/server.log" 2>&1 &
trap 'systemctl --user stop llama-audio.scope' EXIT
for _ in $(seq 60); do curl -sf localhost:8080/health >/dev/null && break; sleep 2; done
echo "VRAM with server: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" > "$VOICE_OUT/audio-tts-asr.txt"
python3 "$HERE/audio_probe.py" tts-asr >> "$VOICE_OUT/audio-tts-asr.txt"
python3 "$HERE/audio_probe.py" tools > "$VOICE_OUT/audio-tools.txt"
timeout 600 python3 "$HERE/realtime_probe.py" > "$VOICE_OUT/audio-realtime.txt"
python3 "$HERE/audio_probe.py" replies >> "$VOICE_OUT/audio-realtime.txt"
