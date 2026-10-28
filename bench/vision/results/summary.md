| Model | OCR | Boxes (pixels, as asked) | Boxes (any format) | Understanding | CAD | Overall (any format) | Median time | Median tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3vl | 100% | 7% | 100% (norm1000_xyxy) | 61% | 96% | **89%** | 6 s | 152 |
| gemma26b | 100% | 0% | 100% (norm1000_xyxy, norm1000_yxyx) | 100% | 100% | **100%** | 71 s | 1,117 |
| qwen36 | 100% | 7% | 100% (norm1000_xyxy) | 100% | 100% | **100%** | 44 s | 461 |
| distill | 100% | 13% | 87% (norm1000_xyxy) | 67% | 100% | **88%** | 36 s | 458 |
| bonsai2 | 100% | 67% | 100% (norm1000_xyxy, pixel) | 100% | 93% | **98%** | 38 s | 553 |
| bolt | 100% | 21% | 64% (norm1000_xyxy, pixel) | 100% | 100% | **91%** | 115 s | 1,171 |
| groot | 97% | 21% | 21% (pixel) | 78% | 100% | **74%** | 49 s | 1,016 |
| ornith | 100% | 13% | 87% (norm1000_xyxy) | 100% | 100% | **97%** | 34 s | 436 |
