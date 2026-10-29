# Visual control probe: results

## glance: one picture, one decision

| Model | Thinking | Tool call | tanks | click | pick | signal | gauge | All | Median time | Invalid calls |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lfm3vl-n30 | off | auto-noaction | 25/30 | 30/30 | 25/30 | 29/30 | 10/30 | **119/150** | 0.4 s | 0 |
| lfm3vl-n30 | off | required | 24/30 | 30/30 | 26/30 | 27/30 | 10/30 | **117/150** | 0.4 s | 0 |
| lfm3vl-n30 | off | auto | 25/30 | 28/30 | 27/30 | 19/30 | 11/30 | **110/150** | 0.4 s | 0 |
| lfm3vl-n30 | on | required | 25/30 | 30/30 | 29/30 | 30/30 | 30/30 | **144/150** | 2.7 s | 0 |
| lfm3vl | off | required | 12/12 | 12/12 | 10/12 | 12/12 | 4/12 | **50/60** | 0.4 s | 0 |
| lfm3vl | off | auto | 12/12 | 11/12 | 11/12 | 7/12 | 4/12 | **45/60** | 0.4 s | 0 |
| lfm3vl | on | required | 12/12 | 12/12 | 12/12 | 11/12 | 12/12 | **59/60** | 2.6 s | 0 |
| lfm3vl | on | auto | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | **60/60** | 2.9 s | 0 |
| qwen3vl-n30 | off | required | 18/30 | 30/30 | 29/30 | 30/30 | 17/30 | **124/150** | 0.9 s | 0 |
| qwen3vl | off | required | 8/12 | 12/12 | 11/12 | 12/12 | 6/12 | **49/60** | 1.0 s | 0 |
| qwen3vl | off | auto | 8/12 | 12/12 | 11/12 | 12/12 | 8/12 | **51/60** | 1.0 s | 0 |

## grid: steer a robot to the goal, one move per step

| Model | Thinking | Tool call | Memory | Open grid | With walls | Steps / best (solved) | Steps without a move | Blocked moves | Longest run of one move | Time per step |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| lfm3vl | off | required | history | 0/8 | 0/8 | — | 218/354 | 36 | 2 | 4.1 s |
| lfm3vl | off | auto | history | 5/8 | 0/8 | 1.00 | 148/308 | 45 | 9 | 2.5 s |
| lfm3vl | off | required | stateless | 6/8 | 0/8 | 1.00 | 4/280 | 167 | 28 | 0.3 s |
| lfm3vl | off | auto | stateless | 3/8 | 0/8 | 1.29 | 157/318 | 50 | 10 | 0.7 s |
| lfm3vl | on | required | stateless | 8/8 | 3/8 | 1.57 | 90/243 | 8 | 5 | 43.2 s |
| lfm3vl | on | auto | stateless | 8/8 | 3/8 | 1.50 | 77/229 | 6 | 5 | 44.5 s |
| qwen3vl | off | auto | history | 4/8 | 2/8 | 1.12 | 68/274 | 71 | 14 | 1.6 s |
| qwen3vl | off | required | stateless | 8/8 | 0/8 | 1.62 | 0/278 | 73 | 3 | 0.7 s |
| qwen3vl | off | auto | stateless | 8/8 | 1/8 | 1.80 | 0/283 | 67 | 2 | 0.7 s |
