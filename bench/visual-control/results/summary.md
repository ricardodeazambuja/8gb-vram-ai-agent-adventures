# Visual control probe: results

## glance: one picture, one decision

Scenes: *probe* = the probe's own scenes; *new* = never-seen drawings (`--new`); *shifted* = degraded pictures (`--shift`).

| Model | Scenes | Thinking | Tool call | tanks | click | pick | signal | gauge | All | Median time | Invalid calls |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| agent-lora | probe | off | auto | 19/30 | 24/30 | 29/30 | 19/30 | 10/30 | **101/150** | 0.4 s | 0 |
| gemma26b | probe | on | auto | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | **30/30** | 16.8 s | 0 |
| graft | probe | off | auto | 8/30 | 6/30 | 0/30 | 0/30 | 0/30 | **14/150** | 5.9 s | 2 |
| groot | probe | on | auto | 6/6 | 4/6 | 5/6 | 6/6 | 6/6 | **27/30** | 16.0 s | 0 |
| lfm25-base-gguf | new | off | auto | 8/30 | 6/30 | 0/30 | 0/30 | 0/30 | **14/150** | 0.3 s | 0 |
| lfm25-base-gguf | shifted | off | auto | 8/30 | 6/30 | 0/30 | 0/30 | 3/30 | **17/150** | 0.3 s | 0 |
| lfm25-base-gguf | probe | off | auto | 8/30 | 6/30 | 0/30 | 0/30 | 0/30 | **14/150** | 0.4 s | 0 |
| lfm25-both-norm-gguf | new | off | auto | 21/30 | 20/30 | 17/30 | 24/30 | 19/30 | **101/150** | 0.1 s | 0 |
| lfm25-both-norm-gguf | shifted | off | auto | 15/30 | 23/30 | 22/30 | 29/30 | 29/30 | **118/150** | 0.1 s | 0 |
| lfm25-both-norm-gguf | probe | off | auto | 29/30 | 28/30 | 21/30 | 30/30 | 30/30 | **138/150** | 0.1 s | 0 |
| lfm25-ia3-gguf | new | off | auto | 21/30 | 24/30 | 18/30 | 30/30 | 18/30 | **111/150** | 0.1 s | 0 |
| lfm25-ia3-gguf | shifted | off | auto | 26/30 | 24/30 | 27/30 | 30/30 | 27/30 | **134/150** | 0.1 s | 0 |
| lfm25-ia3-gguf | probe | off | auto | 27/30 | 28/30 | 27/30 | 30/30 | 27/30 | **139/150** | 0.1 s | 0 |
| lfm25-norm-gguf | new | off | auto | 18/30 | 20/30 | 22/30 | 7/30 | 10/30 | **77/150** | 0.1 s | 0 |
| lfm25-norm-gguf | shifted | off | auto | 21/30 | 26/30 | 25/30 | 30/30 | 19/30 | **121/150** | 0.1 s | 0 |
| lfm25-norm-gguf | probe | off | auto | 27/30 | 25/30 | 21/30 | 30/30 | 30/30 | **133/150** | 0.1 s | 0 |
| lfm3vl-base-gguf | new | off | auto | 25/30 | 30/30 | 29/30 | 14/30 | 12/30 | **110/150** | 0.5 s | 0 |
| lfm3vl-base-gguf | shifted | off | auto | 27/30 | 30/30 | 30/30 | 14/30 | 8/30 | **109/150** | 0.3 s | 0 |
| lfm3vl-base-gguf | probe | off | auto | 26/30 | 28/30 | 28/30 | 17/30 | 10/30 | **109/150** | 0.4 s | 0 |
| lfm3vl-both-ia3-gguf | new | off | auto | 30/30 | 30/30 | 30/30 | 30/30 | 29/30 | **149/150** | 0.3 s | 0 |
| lfm3vl-both-ia3-gguf | probe | off | auto | 30/30 | 30/30 | 30/30 | 30/30 | 30/30 | **150/150** | 0.3 s | 0 |
| lfm3vl-both-norm-gguf | new | off | auto | 28/30 | 30/30 | 30/30 | 30/30 | 30/30 | **148/150** | 0.3 s | 0 |
| lfm3vl-both-norm-gguf | probe | off | auto | 30/30 | 30/30 | 30/30 | 30/30 | 30/30 | **150/150** | 0.3 s | 0 |
| lfm3vl-ia3-gguf | new | off | auto | 28/30 | 30/30 | 30/30 | 30/30 | 30/30 | **148/150** | 0.3 s | 0 |
| lfm3vl-ia3-gguf | shifted | off | auto | 29/30 | 30/30 | 30/30 | 30/30 | 30/30 | **149/150** | 0.3 s | 0 |
| lfm3vl-ia3-gguf | probe | off | auto | 30/30 | 30/30 | 30/30 | 30/30 | 30/30 | **150/150** | 0.3 s | 0 |
| lfm3vl-lora-gguf | probe | off | auto | 30/30 | 30/30 | 29/30 | 30/30 | 30/30 | **149/150** | 0.4 s | 0 |
| lfm3vl-n30 | probe | off | auto-noaction | 25/30 | 30/30 | 25/30 | 29/30 | 10/30 | **119/150** | 0.4 s | 0 |
| lfm3vl-n30 | probe | off | required | 24/30 | 30/30 | 26/30 | 27/30 | 10/30 | **117/150** | 0.4 s | 0 |
| lfm3vl-n30 | probe | off | auto | 25/30 | 28/30 | 27/30 | 19/30 | 11/30 | **110/150** | 0.4 s | 0 |
| lfm3vl-n30 | probe | on | required | 25/30 | 30/30 | 29/30 | 30/30 | 30/30 | **144/150** | 2.7 s | 0 |
| lfm3vl-norm-gguf | new | off | auto | 29/30 | 30/30 | 30/30 | 30/30 | 30/30 | **149/150** | 0.3 s | 0 |
| lfm3vl-norm-gguf | shifted | off | auto | 30/30 | 30/30 | 30/30 | 30/30 | 30/30 | **150/150** | 0.3 s | 0 |
| lfm3vl-norm-gguf | probe | off | auto | 30/30 | 30/30 | 30/30 | 30/30 | 30/30 | **150/150** | 0.3 s | 0 |
| lfm3vl | probe | off | required | 12/12 | 12/12 | 10/12 | 12/12 | 4/12 | **50/60** | 0.4 s | 0 |
| lfm3vl | probe | off | auto | 12/12 | 11/12 | 11/12 | 7/12 | 4/12 | **45/60** | 0.4 s | 0 |
| lfm3vl | probe | on | required | 12/12 | 12/12 | 12/12 | 11/12 | 12/12 | **59/60** | 2.6 s | 0 |
| lfm3vl | probe | on | auto | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | **60/60** | 2.9 s | 0 |
| qwen3vl-n30 | probe | off | required | 18/30 | 30/30 | 29/30 | 30/30 | 17/30 | **124/150** | 0.9 s | 0 |
| qwen3vl | probe | off | required | 8/12 | 12/12 | 11/12 | 12/12 | 6/12 | **49/60** | 1.0 s | 0 |
| qwen3vl | probe | off | auto | 8/12 | 12/12 | 11/12 | 12/12 | 8/12 | **51/60** | 1.0 s | 0 |

## grid: steer a robot to the goal, one move per step

| Model | Thinking | Tool call | Memory | Open grid | With walls | Steps / best (solved) | Steps without a move | Blocked moves | Longest run of one move | Time per step |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| lfm3vl-both-ia3-gguf | off | auto | stateless | 8/8 | 4/8 | 1.00 | 0/179 | 80 | 24 | 0.3 s |
| lfm3vl-both-lora-gguf | off | auto | stateless | 8/8 | 1/8 | 1.00 | 0/226 | 148 | 24 | 0.3 s |
| lfm3vl-both-norm-gguf | off | auto | stateless | 8/8 | 5/8 | 1.00 | 0/162 | 45 | 20 | 0.3 s |
| lfm3vl | off | required | history | 0/8 | 0/8 | — | 218/354 | 36 | 2 | 4.1 s |
| lfm3vl | off | auto | history | 5/8 | 0/8 | 1.00 | 148/308 | 45 | 9 | 2.5 s |
| lfm3vl | off | required | stateless | 6/8 | 0/8 | 1.00 | 4/280 | 167 | 28 | 0.3 s |
| lfm3vl | off | auto | stateless | 3/8 | 0/8 | 1.29 | 157/318 | 50 | 10 | 0.7 s |
| lfm3vl | on | auto | history | 6/8 | 2/8 | 1.60 | 58/274 | 27 | 4 | 38.7 s |
| lfm3vl | on | required | stateless | 8/8 | 3/8 | 1.57 | 90/243 | 8 | 5 | 43.2 s |
| lfm3vl | on | auto | stateless | 8/8 | 3/8 | 1.50 | 77/229 | 6 | 5 | 44.5 s |
| qwen3vl | off | auto | history | 4/8 | 2/8 | 1.12 | 68/274 | 71 | 14 | 1.6 s |
| qwen3vl | off | required | stateless | 8/8 | 0/8 | 1.62 | 0/278 | 73 | 3 | 0.7 s |
| qwen3vl | off | auto | stateless | 8/8 | 1/8 | 1.80 | 0/283 | 67 | 2 | 0.7 s |
