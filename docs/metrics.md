# metrics (lane S2)

Port of supervision `metrics/` for box targets, on plain arrays (`Dets`), with `adapt.to_dets` from S0 `Detections`.

| file | Python source | entry |
|---|---|---|
| `coco.mojo` | `COCOEvaluator`, `MeanAveragePrecision` | `mean_average_precision(preds, targets, image_ids, max_dets)` -> `MapResult`; `evaluate(gt, dt, ids, max_det, sigmas)` is the shared core (OKS overlap when `sigmas` is given) |
| `keypoints.mojo` | `KeyPointMeanAveragePrecision` | `keypoint_mean_average_precision(preds, targets, sigmas, agnostic)` on `KpDets`; all, medium and large buckets, 20 detections; OKS in `detection/utils/iou_and_nms.mojo` `keypoint_oks_batch` |
| `prf.mojo` | `_ConfusionMatrixMetric`, `Precision`, `Recall`, `F1Score` | `prf(preds, targets, kind, averaging)` |
| `mar.mojo` | `MeanAverageRecall` | `mean_average_recall(preds, targets)` |
| `confmat.mojo` | `ConfusionMatrix.from_detections` | `confusion_matrix(preds, targets, n_classes, conf, iou)` |
| `iou.mojo`, `sortutil.mojo` | `box_iou_batch`, greedy matcher, numpy stable argsort | helpers |

Float rules that matter: boxes and confidences are float32 (round on load); `box_iou_batch` computes in float64 and returns float32; the P/R/F1 and mAR IoU thresholds are float32; COCO mAP thresholds are float64 `linspace` values reproduced bit for bit.

Not ported: mask and oriented-box targets, small/medium/large buckets of P/R/F1/mAR, plots, pandas, `class_mapping`. Paths without oracle rows (UNVERIFIED): `micro` averaging, `class_agnostic`, `iscrowd`/`ignore` fields, `image_indices`, custom max-det limits, mAP per-class and per-size values.

Gate: `tests/build_metrics.sh && tests/run.sh tests/cases/metrics_*.txt` (354 rows; `metrics_kp.txt` has 34: 17-point COCO sigmas and custom ones, visibility, crowds, areas, boxes of unlabelled targets, class-agnostic, empties). The keypoint input is `KpDets` plain arrays; `key_points/core.mojo` needed no new field.
Trap: `input()` cuts lines near 1 KB in Mojo 1.2; read `/dev/stdin` whole.
