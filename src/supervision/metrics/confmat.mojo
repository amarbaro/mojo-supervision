"""Detection confusion matrix, port of `ConfusionMatrix.from_detections` for boxes."""

from supervision.metrics.dets import Dets
from supervision.metrics.iou import box_iou_f32
from supervision.metrics.sortutil import argsort_stable


def _check_classes(cls: List[Int], num_classes: Int, role: String) raises:
    for c in cls:
        if c < 0 or c >= num_classes:
            raise Error(role + " class ids must be in [0, " + String(num_classes - 1) + "]")


def _evaluate_batch(
    pred: Dets, target: Dets, nc: Int, conf: Float32, iou_thr: Float32
) raises -> List[Int]:
    var w = nc + 1
    var m = List[Int](length=w * w, fill=0)
    var keep = Dets()
    if pred.n() > 0 and len(pred.conf) == 0:
        raise Error("predictions require `confidence`")
    for j in range(pred.n()):
        if Float32(pred.conf[j]) >= conf:
            for q in range(4):
                keep.xyxy.append(pred.xyxy[4 * j + q])
            keep.class_id.append(pred.class_id[j])
    _check_classes(target.class_id, nc, "Target")
    _check_classes(keep.class_id, nc, "Prediction")
    if keep.n() == 0:
        for c in target.class_id:
            m[c * w + nc] += 1
        return m^
    if target.n() == 0:
        for c in keep.class_id:
            m[nc * w + c] += 1
        return m^
    var iou = box_iou_f32(target, keep)
    var nt = target.n()
    var nd = keep.n()
    # candidate pairs above the threshold: same class first, then highest iou (stable)
    var gt_i = List[Int]()
    var dt_i = List[Int]()
    var keys = List[Float64]()
    for g in range(nt):
        for d in range(nd):
            if iou[g * nd + d] > iou_thr:
                gt_i.append(g)
                dt_i.append(d)
                var same = target.class_id[g] == keep.class_id[d]
                keys.append(-Float64(iou[g * nd + d]) + (0.0 if same else 4.0))
    var order = argsort_stable(keys)
    var used_g = List[Bool](length=nt, fill=False)
    var used_d = List[Bool](length=nd, fill=False)
    for o in order:
        if not used_g[gt_i[o]] and not used_d[dt_i[o]]:
            used_g[gt_i[o]] = True
            used_d[dt_i[o]] = True
            m[target.class_id[gt_i[o]] * w + keep.class_id[dt_i[o]]] += 1
    for g in range(nt):
        if not used_g[g]:
            m[target.class_id[g] * w + nc] += 1
    for d in range(nd):
        if not used_d[d]:
            m[nc * w + keep.class_id[d]] += 1
    return m^


def confusion_matrix(
    preds: List[Dets],
    targets: List[Dets],
    num_classes: Int,
    conf_threshold: Float64,
    iou_threshold: Float64,
) raises -> List[Int]:
    """Row-major (num_classes+1)^2 matrix; row = target class, column = prediction class."""
    if len(preds) != len(targets):
        raise Error("Number of predictions and targets must be equal")
    var w = num_classes + 1
    var total = List[Int](length=w * w, fill=0)
    for i in range(len(preds)):
        var m = _evaluate_batch(
            preds[i], targets[i], num_classes, Float32(conf_threshold), Float32(iou_threshold)
        )
        for k in range(w * w):
            total[k] += m[k]
    return total^
