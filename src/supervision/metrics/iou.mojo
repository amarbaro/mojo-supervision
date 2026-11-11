"""Float32 box IoU and the greedy matcher shared by P/R/F1, mAR and the confusion matrix."""

from supervision.metrics.dets import Dets
from supervision.metrics.sortutil import argsort_stable

comptime N_THR = 10


def thresholds_f32() -> List[Float32]:
    """`np.linspace(0.5, 0.95, 10, dtype=np.float32)`: float64 values cast down."""
    var step = (0.95 - 0.5) / 9.0
    var out = List[Float32]()
    for i in range(N_THR):
        out.append(Float32(Float64(i) * step + 0.5))
    out[N_THR - 1] = Float32(0.95)
    return out^


def box_iou_f32(true: Dets, det: Dets) -> List[Float32]:
    """`box_iou_batch`: float64 arithmetic, float32 result, rows = true boxes."""
    var nt = true.n()
    var nd = det.n()
    var out = List[Float32](length=nt * nd, fill=0.0)
    for i in range(nt):
        var tx1 = true.xyxy[4 * i]
        var ty1 = true.xyxy[4 * i + 1]
        var tx2 = true.xyxy[4 * i + 2]
        var ty2 = true.xyxy[4 * i + 3]
        var area_t = (tx2 - tx1) * (ty2 - ty1)
        for j in range(nd):
            var dx1 = det.xyxy[4 * j]
            var dy1 = det.xyxy[4 * j + 1]
            var dx2 = det.xyxy[4 * j + 2]
            var dy2 = det.xyxy[4 * j + 3]
            var iw = max(min(tx2, dx2) - max(tx1, dx1), 0.0)
            var ih = max(min(ty2, dy2) - max(ty1, dy1), 0.0)
            var inter = iw * ih
            var norm = area_t + (dx2 - dx1) * (dy2 - dy1) - inter
            if norm > 0.0:
                out[i * nd + j] = Float32(inter / norm)
    return out^


def match_batch(
    pred_cls: List[Int],
    true_cls: List[Int],
    iou: List[Float32],
    thr: List[Float32],
) -> List[Bool]:
    """`_match_detection_batch_with_target_indices` without a size mask: P x Th flags.

    Per threshold, candidate pairs (same class, iou >= threshold) are taken highest
    iou first (stable, row-major ties) and each target and prediction is used once.
    """
    var nt = len(true_cls)
    var np_ = len(pred_cls)
    var out = List[Bool](length=np_ * len(thr), fill=False)
    for ti in range(len(thr)):
        var ti_idx = List[Int]()
        var pi_idx = List[Int]()
        var keys = List[Float64]()
        for t in range(nt):
            for p in range(np_):
                if true_cls[t] == pred_cls[p] and iou[t * np_ + p] >= thr[ti]:
                    ti_idx.append(t)
                    pi_idx.append(p)
                    keys.append(-Float64(iou[t * np_ + p]))
        var order = argsort_stable(keys)
        var used_t = List[Bool](length=nt, fill=False)
        var used_p = List[Bool](length=np_, fill=False)
        for o in order:
            if not used_t[ti_idx[o]] and not used_p[pi_idx[o]]:
                used_t[ti_idx[o]] = True
                used_p[pi_idx[o]] = True
                out[pi_idx[o] * len(thr) + ti] = True
    return out^
