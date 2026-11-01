#!/usr/bin/env bash
# One GPU job in its own scope, so a crash or an oomd kill takes only the job.  gpu.sh NAME TIMEOUT_S CMD...
n=$1; t=$2; shift 2
cd "$(dirname "$0")/.."; mkdir -p loop/logs
exec systemd-run --user --scope --quiet --unit="ov-$n-$(date +%s)" -p MemoryMax=20G -p MemorySwapMax=4G \
  timeout "$t" "$@" >"loop/logs/$n.log" 2>&1
