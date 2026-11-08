# 💨 Quick checks for the helper models

One command starts each model with the backend script under `scripts/llm` (`scripts/run-qwen.sh`), tests it, and
stops it.

```bash
bash scripts/download-models.sh clefflash d1 eg2 locate
bash bench/smoke/run.sh                # or name some: bash bench/smoke/run.sh d1 eg2
```

Stop any other llama-server first: it uses port 8080. Output goes to `bench/smoke/rerun/`, which git
ignores. Saved runs are in [`results/`](results/).

| Model | Test | 2026-10-28 | 2026-11-01, fork `3cb2992` | 2026-11-08, `84137489b` → `3f2b2d60a` | sync `1f6e660c4` |
|---|---|---:|---:|---:|---:|
| [Clef-Flash](../../docs/models/clef-flash.md) | 12 labelled decisions + 4 "angry?" | 12/12, 4/4, 183 ms | 12/12, 4/4, 187 ms | 12/12, 4/4, 199 → 195 ms | 12/12, 4/4, 196 ms |
| [d1-2B](../../docs/models/d1-2b.md) | the same | 12/12, 4/4, 27 ms | 12/12, 4/4, 31 ms | 12/12, 4/4, 34 → 35 ms | 12/12, 4/4, 30 ms |
| [EmbeddingGemma 2](../../docs/models/embeddinggemma-2.md) | 6 toy searches, then throughput | 5/6, 280/s | 5/6, 270/s | 5/6, 210 → 216/s | 5/6, 258/s |
| [LocateAnything-3B](../../docs/models/locateanything-3b.md) | 15 shapes, then detect-all | 15/15, IoU 0.94 | 15/15, IoU 0.94, detect-all 15/15 | the same on both | the same |

The first LocateAnything run's raw output wasn't kept, so only the November output is saved. The 2026-11-08 column
is the build before the merged expert cache and the one with it ([bench/buildcheck](../buildcheck/README.md) checked
every other model); a llama.cpp compile ran in the background, which is likely why EmbeddingGemma's throughput is
lower on both. The sync column is the full upstream sync, run with nothing else on the CPU.
