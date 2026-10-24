#!/bin/sh
# Install the per-run pi provider config (mounted read-only at /bench/models.json), then run the
# given command. pi writes settings and sessions under ~/.pi, so it needs a writable copy.
set -e
if [ -f /bench/models.json ]; then
  mkdir -p "$HOME/.pi/agent"
  cp /bench/models.json "$HOME/.pi/agent/models.json"
fi
exec "$@"
