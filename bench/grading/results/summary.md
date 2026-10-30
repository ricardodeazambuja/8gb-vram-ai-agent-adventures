# Judge probe: results

Each finished pibench attempt; truth = the hidden tests. Baseline "solved unless it timed out": 204/252 = 81%, and it catches none of the 47 failures that didn't time out.

| Judge | Chars | Attempts | Accuracy (p>0.5) | AUC | Quiet failures caught | Solved runs flagged | Median time |
|---|---:|---:|---:|---:|---:|---:|---:|
| clefflash | 6000 | 252 | 86% | 0.75 | 15/47 | 4/189 | 1.55 s |
| d1 | 20000 | 252 | 80% | 0.77 | 15/47 | 18/189 | 0.31 s |
| d1 | 6000 | 252 | 80% | 0.78 | 15/47 | 17/189 | 0.30 s |
