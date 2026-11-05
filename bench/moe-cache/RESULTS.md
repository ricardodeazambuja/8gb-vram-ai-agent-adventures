# MoE cache / prompt-cache measurements, 2026-11-04 (RTX 2070 Max-Q, i7-8750H, 32 GB DDR4)


(Copied into this repo on 2026-11-04: the `srv-*.log` and `cache_run.log` files are in `logs/`, and the scripts take paths relative to this folder; see README.md.)
Console outputs below are verbatim (they were not saved to log files at the time).
Builds: llama.cpp `/home/ricardodeazambuja/backup/GitStuff/llama.cpp`, build dir `build/` (CUDA, sm75, Release, env llamacpp-cuda).
- Sections 1-4 (bandwidth, llama-bench sweep, top-k sweep, merged-cache share-0 run): branch `adventures` @ 7ff495fc4.
- Sections 5-9: branch `moe-offload` @ 3a72c57be (= 7ff495fc4 + PR #30966's 8 commits).
Models: `~/GitStuff/8gb-vram-ai-agent-adventures/models/{qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf, gemma4-26b-a4b-qat/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf, flint-40b-a2.5b/flint-UD-Q3_K_XL.gguf}`.
Every llama process ran inside `systemd-run --user --scope -q`.

## 1. PCIe / RAM bandwidth (bw/bw.cu; `nvcc -O3 -arch=sm_75 -o bw bw.cu -lpthread && ./bw`)
```
H2D    1.41 MiB: pinned  6.42 GB/s  pageable  6.29 GB/s
H2D    3.19 MiB: pinned  6.45 GB/s  pageable  6.21 GB/s
H2D    8.00 MiB: pinned  6.49 GB/s  pageable  6.37 GB/s
H2D   64.00 MiB: pinned  6.47 GB/s  pageable  6.31 GB/s
RAM read  1 threads:  20.3 GB/s
RAM read  2 threads:  29.1 GB/s
RAM read  4 threads:  35.4 GB/s
RAM read  6 threads:  36.7 GB/s
RAM read 12 threads:  35.0 GB/s
concurrent: RAM read 6 thr  29.8 GB/s + pinned H2D  6.43 GB/s
nvidia-smi under load: pcie.link.gen.current 3, width.current 8
```

## 2. Per-expert bytes (expert_bytes.py MODEL...; gpu_exp_bytes.py MODEL NCMOE)
```
Qwen3.6-35B-A3B-UD-IQ4_XS.gguf: arch=qwen35moe layers=40 moe_layers=40 n_exp=256 top_k=8
  file tensors 16.50 GiB, experts 14.12 GiB, non-expert 2.38 GiB
  per-expert bytes (all 3 mats) min 1.391 MiB max 1.680 MiB mean 1.412 MiB
gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf: arch=gemma4 layers=30 moe_layers=30 n_exp=128 top_k=8
  file tensors 13.26 GiB, experts 11.96 GiB, non-expert 1.29 GiB
  per-expert bytes (all 3 mats) min 3.190 MiB max 3.190 MiB mean 3.190 MiB
flint-UD-Q3_K_XL.gguf: arch=flint layers=49 moe_layers=47 n_exp=256 top_k=6
  file tensors 17.22 GiB, experts 15.40 GiB, non-expert 1.82 GiB
  per-expert bytes (all 3 mats) min 1.172 MiB max 1.625 MiB mean 1.311 MiB
Qwen3.6 ncmoe=32: GPU expert layers [32..39] = 3070 MiB (21.2% of experts)
flint   ncmoe=42: GPU expert layers [42..48] = 2168 MiB (13.7% of experts)
gemma   ncmoe=22: GPU expert layers [22..29] = 3267 MiB (26.7% of experts)
```

## 3. Static split sweep (Qwen3.6)
`llama-bench -m QWEN -ngl 99 -fa 1 -t 6 -p 0 -n 64 -r 3 -ncmoe 40,36,32,28 -o csv`
```
n_cpu_moe avg_ts stddev_ts
40 22.21 1.17 | 36 24.62 1.38 | 32 26.98 1.25 | 28 29.82 1.02
fit (estimate.py): ms/token = 6.74 + 0.951 * n_cpu_layers
```
Top-k sweep: `llama-completion -m QWEN -ngl 99 -fa on -t 6 -ncmoe 40 --override-kv qwen35moe.expert_used_count=int:K -p "Write a long story about a lighthouse keeper." -n 160 -no-cnv --temp 0 --ignore-eos`
```
k=2: 19.80 ms/token | k=4: 26.97 ms/token | k=8: 45.53 ms/token
-> ~0.1 ms per CPU-computed expert, ~0.1 ms fixed per CPU layer; upload of one expert 1.48 MB / 6.45 GB/s = 0.23 ms
```

## 4. llama-completion A/B, Qwen3.6, 8K ctx (decode t/s, 400 tokens greedy)
`llama-completion -m QWEN -ngl 99 -fa on -t 6 -c 8192 ARGS -p "Write a Python module that implements an LRU cache with TTL expiry, thread safety, and statistics. Include docstrings and unit tests." -n 400 -no-cnv --temp 0 --ignore-eos`
On adventures 7ff495fc4 (no CPU share yet):
```
-ncmoe 32 : 26.17 | -cmoe --moe-cache-mib 2900 : 30.05 | -ncmoe 32 : 23.72
cache log (cache_run.log): ubatch <= 8: hits = 98997, misses = 29306, hit rate = 77.16%, uploaded = 41565.05 MiB
```
On moe-offload 3a72c57be:
```
-ncmoe 32 : 24.71
-cmoe --moe-cache-mib 2900 --moe-cache-cpu-share 0    : 30.21
-cmoe --moe-cache-mib 2900 --moe-cache-cpu-share 0.5  : 37.00
-cmoe --moe-cache-mib 2900 --moe-cache-cpu-share 0.8  : 41.56
-cmoe --moe-cache-mib 2900 --moe-cache-cpu-share 0.95 : 42.37
-ncmoe 32 : 24.47
```

## 5. llama-server A/B, real launcher-like flags (srvbench.sh LABEL ARGS; 2 runs each; prefill = ~6K-token prompt, decode = 400 tokens greedy)
Common flags: `-ngl 99 -c CTX -ctk q8_0 -ctv q8_0 -fa on -t 6 -b BATCH -ub BATCH --parallel 1 --jinja --port 8090 --host 127.0.0.1`
Qwen3.6, CTX=131072 BATCH=512:
```
ncmoe32            vram 7571 MiB  prefill 6236 tok 239.9 / 239.2 t/s  decode 24.81 / 25.01   ARGS: -ncmoe 32
cache2900-s0.95    vram 7485 MiB  prefill 6236 tok 225.1 / 245.3 t/s  decode 39.88 / 39.01   ARGS: -cmoe --moe-cache-mib 2900 --moe-cache-cpu-share 0.95
cache2900-s0.8     vram 7485 MiB  prefill 6236 tok 224.9 / 245.4 t/s  decode 38.55 / 39.95   ARGS: -cmoe --moe-cache-mib 2900 --moe-cache-cpu-share 0.8
ncmoe32-b          vram 7671 MiB  prefill 6236 tok 237.7 / 238.1 t/s  decode 23.91 / 23.98   ARGS: -ncmoe 32
```
Flint, CTX=131072 BATCH=2048:
```
flint-ncmoe42      vram 7405 MiB  prefill 5708 tok 533.5 / 531.9 t/s  decode 27.29 / 26.53   ARGS: -ncmoe 42
flint-cache-s0.95  vram 7411 MiB  prefill 5708 tok 505.8 / 526.4 t/s  decode 31.82 / 32.33   ARGS: -cmoe --moe-cache-mib 2168 --moe-cache-cpu-share 0.95
flint-ncmoe42-b    vram 7401 MiB  prefill 5708 tok 529.7 / 470.4 t/s  decode 24.22 / 17.38   ARGS: -ncmoe 42
```
Gemma 4 26B, CTX=32768 BATCH=512 (note: q8_0 KV here; run-server.sh defaults to f16 KV and --load-mode none):
```
gemma-ncmoe22      vram 6501 MiB  prefill 7197 tok 255.4 / 274.2 t/s  decode 21.05 / 21.02   ARGS: -ncmoe 22
gemma-cache-s0.95  vram 6509 MiB  prefill 7197 tok 243.1 / 262.1 t/s  decode 29.19 / 30.81   ARGS: -cmoe --moe-cache-mib 3267 --moe-cache-cpu-share 0.95
gemma-ncmoe22-b    vram 6501 MiB  prefill 7197 tok 255.1 / 255.2 t/s  decode 21.55 / 21.10   ARGS: -ncmoe 22
```
Gemma hit rate (from srv-dbg-gemma.log, short agent session, 3267 MiB = 26.7%): ubatch<=8 hit rate 61.44%.

## 6. Tools vs no tools (toolsdiff.py via srvbench.sh, Qwen3.6 128K, chat API, 700 tokens, temp 0.6 seed 7)
```
ncmoe32:        chat+tools 24.37 | no tools 23.38 | chat+tools 22.16
cache s0.95:    chat+tools 33.53 | no tools 35.72 | chat+tools 35.44
```

## 7. Long generation (longgen.py, 2500 tokens greedy, t/s per 500-token window)
```
cache s0.95: 34.0 32.7 33.0 37.0 | overall 33.87   (CPU package 83 C after)
ncmoe32:     21.0 20.6 19.0 17.9 | overall 19.56   (CPU package 100 C after)
```

## 8. Prompt-cache reuse per agent turn (agentloop.py URL TURNS; env TASK, NEWUSER_AT, PI_CAPTURE, MAXTOK)
Qwen3.6, cache s0.95, default task, 8 turns (srv-agentloop.log):
```
turn prompt_tok cached reproc gen think_chars tool
 0 443 0 443 68 164 bash | 1 535 510 25 98 73 read | 2 844 632 212 62 96 bash | 3 930 905 25 65 91 bash
 4 1019 994 25 52 56 bash | 5 1095 1070 25 79 117 bash | 6 1198 1173 25 70 104 bash | 7 1292 1267 25 56 139 bash
```
Qwen3.6, write task, 3000-token write call (srv-agentloop2.log): next turn reprocessed 19 tokens, f_keep = 1.000 (the run then hit HTTP 500 because max_tokens=3000 truncated the second write's JSON - harness issue).
Qwen3.6, new user message after turn 2 (srv-agentloop3.log): turn 3 reprocessed 47, others 25.
Gemma, short prompt, stock template (srv-agentloop-gemma.log): reprocessed 91 100 213 365 96 (inflated by the short-context SWA restore quirk, see below).
Gemma, pi system prompt + tools (PI_CAPTURE=.../bench/gemma12-probe/capture/agentic12/0001-req.json MAXTOK=6000, write task, 5 turns):
```
stock template (srv-pi-gemma-stock.log)          patched template (srv-pi-gemma-nonl.log)
 0 1416    0 1416  724 508 write                   0 1416    0 1416 724 508 write
 1 2154 1411  743 1129  67 write                   1 2153 2139   14 883  66 write
 2 3296 2149 1147   86 235 bash                    2 3049 3035   14  55 101 bash
 3 3406 3359   47  237 813 bash                    3 3128 3103   25 147 451 bash
 4 3667 3626   41   95 226 bash                    4 3299 3274   25 170 521 bash
```

## 9. Gemma divergence diagnosis
divergence.py (stock template, srv-div-gemma.log):
```
first difference at char 1090 (221 chars into the generation)
GENERATED : '...overview of the project structure.<channel|><|tool_call>call:bash{...'
RERENDERED: '...overview of the project structure.\n<channel|><|tool_call>call:bash{...'
```
divergence.py (patched, srv-div-gemma2.log): first difference only at the end of the generation (pure extension).
toklcp.py (patched, srv-toklcp-gemma.log): generated token ids == re-tokenized ids for 3 seeds.
srv-dbg-gemma.log (-lv 4): `checking checkpoint with [0, 196] against 0` -> `restored context checkpoint (... n_past = 196)` although f_keep = 1.000:
in tools/server/server-context.cpp, when context < n_swa, pos_min_thold clamps to 0 and the SWA pos_min is 0, so `pos_min >= pos_min_thold` forces a needless restore. Only for prompts < ~1K tokens (Gemma n_swa = 1024).
