#!/usr/bin/env bash
# does the warm-up fire? llama-completion, Qwen, ~10K-token prompt, 300 tokens, -lv 4, without and with warm-up
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADV="$(cd "$Q/../../.." && pwd)"
W=${WARM_LLAMA:-$ADV/../llama.cpp-warm}
M=$ADV/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
for cfg in nowarm warm; do
  if [ $cfg = nowarm ]; then export LLAMA_MOE_CACHE_NO_WARM=1; else unset LLAMA_MOE_CACHE_NO_WARM; fi
  cd "$W" || exit 1
  systemd-run --user --scope -q --unit=moew-$$-$cfg mamba run --no-capture-output -n llamacpp-cuda build-warm/bin/llama-completion -m $M -ngl 99 -c 32768 -ctk q8_0 -ctv q8_0 -fa on -t 6 -b 512 -ub 512 -cmoe --moe-cache-mib 2900 --moe-cache-cpu-share 0.95 -f $Q/logs/prompt-qwen-10k.txt -n 300 --ignore-eos --temp 0 -no-cnv -lv 4 < /dev/null > $Q/logs/warmcheck-$cfg.log 2>&1
  echo "### $cfg (exit $?)"
  grep -a 'MoE cache size\|llama_moe_cache\|prompt eval time\|       eval time' $Q/logs/warmcheck-$cfg.log | sed -E 's/^[0-9.]+ [IWED] //' | cut -c1-200
done
