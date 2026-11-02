#!/usr/bin/env bash
# First LFM3-Audio-2B tests: speech out and in, spoken tool calls, live /v1/realtime.
#   bash bench/voice/run.sh              # writes to bench/voice/rerun/
# See README.md. LLAMA_BIN picks the build (default ../llama.cpp/build/bin); it needs llama-server
# and llama-tts. The server starts through scripts/run-qwen.sh. realtime_probe.py needs numpy and websockets.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
export LLAMA_BIN="${LLAMA_BIN:-$REPO/../llama.cpp/build/bin}"
export VOICE_OUT="${VOICE_OUT:-$HERE/rerun}"
mkdir -p "$VOICE_OUT"
CUDA_BIN="$LLAMA_BIN/llama-server" QMODEL=lfm3audio \
  systemd-run --user --scope --unit=llama-audio bash "$REPO/scripts/run-qwen.sh" > "$VOICE_OUT/server.log" 2>&1 &
trap 'systemctl --user stop llama-audio.scope' EXIT
for _ in $(seq 60); do curl -sf localhost:8080/health >/dev/null && break; sleep 2; done
echo "VRAM with server: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" > "$VOICE_OUT/audio-tts-asr.txt"
python3 "$HERE/audio_probe.py" tts-asr >> "$VOICE_OUT/audio-tts-asr.txt"
python3 "$HERE/audio_probe.py" tools > "$VOICE_OUT/audio-tools.txt"
timeout 600 python3 "$HERE/realtime_probe.py" > "$VOICE_OUT/audio-realtime.txt"
python3 "$HERE/audio_probe.py" replies >> "$VOICE_OUT/audio-realtime.txt"
