# mojo-supervision

<!-- badges -->
<!-- /badges -->

Roboflow [supervision](https://github.com/roboflow/supervision) ported to Mojo. Every op is checked
against Python supervision on the same inputs: 3308 recorded rows pass.

```mojo
from supervision.detection.utils.iou_and_nms import box_iou, box_non_max_suppression


def main() raises:
    var xyxy: List[Float64] = [0, 0, 10, 10, 1, 1, 11, 11, 50, 50, 60, 60]
    var scores: List[Float64] = [0.9, 0.8, 0.7]
    var cats: List[Int] = [0, 0, 0]
    print(box_iou([0.0, 0, 10, 10], [1.0, 1, 11, 11]))   # 0.680672268907563
    var keep = box_non_max_suppression(xyxy, scores, cats, 0.5)
    for k in keep:
        print(k)                                           # True False True
```

Runnable copy: `examples/quickstart.mojo`.

## Modules

| Module | Contents |
|---|---|
| `detection` | `Detections` (boxes, masks, confidence, class and tracker ids, data columns), filter, merge, area; IoU, NMS, NMM; box, mask and polygon converters |
| `detection/tools` | polygon zones, line counter, inference slicer, smoother, CSV and JSON sinks |
| `metrics` | mAP (50 and 50:95), precision, recall, F1, confusion matrix |
| `dataset` | COCO, YOLO and Pascal VOC load, save, split and merge |
| `draw`, `geometry`, `annotators`, `key_points` | drawing primitives, annotators, skeletons |
| `classification`, `validators` | classification results, input checks |

Image operations come from [mojo-cv](https://github.com/amarbaro/mojo-cv), JSON from
[mojo-json](https://github.com/amarbaro/mojo-json).

## How it is checked

- Numeric rows agree with Python supervision within 1e-6.
- Dataset round trips are byte-equal to Python's output.
- Annotator and draw rows record the max pixel difference against Python supervision on OpenCV.
  Each row states its own bar.

```sh
tests/build_all.sh     # builds the row drivers into build/
tests/run.sh           # runs every row in tests/cases/
```

## Limits

- Needs Mojo 1.2 (`uv sync`), with mojo-cv, mojo-json and mojo-gzip checked out next to this repo.
  For other locations, set `MOJO_CV`, `MOJO_JSON` and `MOJO_GZIP` to their `src` directories.
- Not on a package channel yet.
- Pixel output is not bit-exact with OpenCV. The per-row bar says how far it may differ.

## License

Apache-2.0 (`LICENSE`). Ported from Roboflow supervision (MIT). Its notice is in
`NOTICE-supervision-LICENSE.md`.
