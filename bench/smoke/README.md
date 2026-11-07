# 💨 Quick checks for the helper models

One command starts each model with the backend script under `scripts/llm` (`scripts/run-qwen.sh`), tests it, and
stops it.

```bash
bash scripts/download-models.sh clefflash d1 eg2 locate
bash bench/smoke/run.sh                # or name some: bash bench/smoke/run.sh d1 eg2
```

Stop any other llama-server first: it uses port 8080. Output goes to `bench/smoke/rerun/`, which git
ignores. Saved runs are in [`results/`](results/).

| Model | Test | 2026-10-28 | 2026-11-01, fork `3cb2992` |
|---|---|---:|---:|
| [Clef-Flash](../../docs/models/clef-flash.md) | 12 labelled decisions + 4 "angry?" | 12/12, 4/4, 183 ms | 12/12, 4/4, 187 ms |
| [d1-2B](../../docs/models/d1-2b.md) | the same | 12/12, 4/4, 27 ms | 12/12, 4/4, 31 ms |
| [EmbeddingGemma 2](../../docs/models/embeddinggemma-2.md) | 6 toy searches, then throughput | 5/6, 280/s | 5/6, 270/s |
| [LocateAnything-3B](../../docs/models/locateanything-3b.md) | 15 shapes, then detect-all | 15/15, IoU 0.94 | 15/15, IoU 0.94, detect-all 15/15 |

The first LocateAnything run's raw output wasn't kept, so only the November output is saved.
