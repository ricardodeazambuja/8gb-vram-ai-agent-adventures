# Visual control probe: results

## glance: one picture, one decision

| Model | Thinking | Tool call | tanks | click | pick | signal | gauge | All | Median time | Invalid calls |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lfm3vl | off | auto | 12/12 | 12/12 | 11/12 | 6/12 | 4/12 | **45/60** | 0.4 s | 0 |
| lfm3vl | on | auto | 12/12 | 8/12 | 8/12 | 12/12 | 12/12 | **52/60** | 3.1 s | 0 |

## grid: steer a robot to the goal, one move per step

| Model | Thinking | Tool call | Memory | Open grid | With walls | Steps / best (solved) | Steps without a move | Blocked moves | Longest run of one move | Time per step |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| lfm3vl | off | auto | history | 5/8 | 0/8 | 1.00 | 148/308 | 45 | 9 | 2.5 s |
| lfm3vl | off | auto | stateless | 3/8 | 0/8 | 1.29 | 157/318 | 50 | 10 | 0.7 s |
| lfm3vl | on | auto | stateless | 8/8 | 3/8 | 1.50 | 77/229 | 6 | 5 | 44.5 s |
