| Model | OCR | Boxes (pixels, as asked) | Boxes (any format) | Understanding | CAD | Overall (any format) | Median time | Median tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3vl | 100% | 7% | 100% (norm1000_xyxy) | 61% | 96% | **89%** | 5 s | 125 |
| gemma26b | 100% | 0% | 100% (norm1000_xyxy, norm1000_yxyx) | 100% | 100% | **100%** | 70 s | 1,080 |
| qwen36 | 100% | 7% | 100% (norm1000_xyxy) | 100% | 100% | **100%** | 43 s | 445 |
| distill | 100% | 13% | 87% (norm1000_xyxy) | 67% | 100% | **88%** | 35 s | 458 |
| bonsai2 | 100% | 67% | 100% (norm1000_xyxy, pixel) | 100% | 93% | **98%** | 38 s | 542 |
| bolt | 100% | 21% | 64% (norm1000_xyxy, pixel) | 100% | 100% | **91%** | 110 s | 1,137 |
| groot | 97% | 21% | 21% (pixel) | 78% | 100% | **74%** | 44 s | 975 |
| ornith | 100% | 13% | 87% (norm1000_xyxy) | 100% | 100% | **97%** | 33 s | 421 |
| lfm3vl | 100% | 7% | 100% (norm1000_xyxy) | 69% | 93% | **91%** | 8 s | 804 |
