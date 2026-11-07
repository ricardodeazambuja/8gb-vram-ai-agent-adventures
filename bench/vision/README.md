# 👁️ Vision probe — what the image-capable models can actually see

Results, 2026-10-23 to 2026-10-28 (one answer per image): [`results/summary.md`](results/summary.md), written up in
[docs/vision-probe.md](../../docs/vision-probe.md).

Twelve synthetic images with exact ground truth, three per category, asked straight through
llama-server's chat API (no agent, no tools, nothing executed, so no sandbox is needed). Each answer
must be JSON and is scored field by field.

| Category | Images | What is scored |
|---|---|---|
| **OCR** | 3 invoices; the third is a "scan" (rotated 1.5°, blurred, greyscale) | invoice number, date, total, and each line's description, quantity and amount |
| **Bounding boxes** | 3 scenes with 4–6 coloured circles, squares, triangles and stars | a shape counts if its box overlaps the true one with IoU ≥ 0.5 **and** its shape and colour are right |
| **Understanding** | a bar chart, a dot-counting scene, a 3×3 grid of shapes | chart reading (value ±2), counting, spatial relations ("directly left of…") |
| **CAD** | 3 dimensioned drawings of plates with a row of holes, plus a title block | width, height, hole count, diameter, pitch, edge offsets, part number, material |

Boxes are scored in **pixels, as the prompt asks**, and again in the best of three other common
conventions (0–1000 `xyxy`, 0–1000 `yxyx` as Gemma uses, 0–1). The models' native key names
(`box_2d` for Gemma, `bbox_2d` for Qwen) are accepted as well as the requested `bbox`, and a shape
named inside a longer label ("blue circle") counts. The gap between the two columns
separates "cannot localise" from "localises but ignores the requested format". Overlays (green =
truth, magenta = the model's boxes in its best convention) are written to `results/overlays/`.

## Rerun

```bash
python3 bench/vision/gen_images.py               # regenerate the images (fixed seeds, identical every time)
bash scripts/stop-server.sh                      # needs the GPU to itself
python3 bench/vision/run_vision.py               # every model in its MODELS list; resumable
python3 bench/vision/run_vision.py --score-only  # re-score saved answers -> results/summary.md
```

Every model runs with its own projector (`--image`) and the backend script's defaults (`run_vision.py` starts
`run-qwen.sh` or `run-server.sh`, not `scripts/llm`), so the MoEs ran with the static `NCMOE` split. Twelve items, one answer
each: a probe of capabilities, not a benchmark; a single wrong field moves a category by several
points.
