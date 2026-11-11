"""Precision, recall and F1 from per-class (TP, FP, FN), port of `_ConfusionMatrixMetric`."""

from supervision.metrics.dets import Dets
from supervision.metrics.iou import N_THR, box_iou_f32, match_batch, thresholds_f32
from supervision.metrics.sortutil import sorted_unique

comptime PRECISION = 0
comptime RECALL = 1
comptime F1 = 2
comptime MACRO = 0
comptime MICRO = 1
comptime WEIGHTED = 2


struct PrfResult(Copyable, Movable):
    var scores: List[Float64]  # per IoU threshold
    var per_class: List[Float64]  # class x threshold
    var classes: List[Int]

    def __init__(out self):
        self.scores = List[Float64](length=N_THR, fill=0.0)
        self.per_class = List[Float64]()
        self.classes = List[Int]()


def _score(kind: Int, tp: Float64, fp: Float64, fneg: Float64) -> Float64:
    var num = tp
    var den = tp + fp
    if kind == RECALL:
        den = tp + fneg
    elif kind == F1:
        num = 2.0 * tp
        den = 2.0 * tp + fp + fneg
    return num / den if den != 0.0 else 0.0


def _need_class(d: Dets, what: String) raises:
    if d.n() > 0 and len(d.class_id) == 0:
        raise Error(what + " requires `class_id`")


def prf(
    preds: List[Dets], targets: List[Dets], kind: Int, averaging: Int
) raises -> PrfResult:
    """`Precision`/`Recall`/`F1Score(...).update(preds, targets).compute()` (all sizes).

    Prediction order inside a class does not change TP/FP/FN, so the confidence sort of
    the Python code is not repeated here.
    """
    if len(preds) != len(targets):
        raise Error("predictions and targets must have the same length")
    var thr = thresholds_f32()
    var pcls = List[Int]()
    var pmatch = List[Bool]()  # P x Th
    var tcls = List[Int]()
    for i in range(len(preds)):
        ref p = preds[i]
        ref t = targets[i]
        if t.n() == 0 and p.n() > 0:
            if len(p.class_id) == 0 or len(p.conf) == 0:
                raise Error("metric requires `class_id` and `confidence` on predictions")
            for c in p.class_id:
                pcls.append(c)
                for _ in range(N_THR):
                    pmatch.append(False)
        elif t.n() > 0:
            if len(t.class_id) == 0 or (p.n() > 0 and len(p.class_id) == 0):
                raise Error("metric requires `class_id` on both predictions and targets")
            for c in t.class_id:
                tcls.append(c)
            if p.n() == 0:
                continue
            if len(p.conf) == 0:
                raise Error("metric requires `confidence` on predictions")
            var m = match_batch(p.class_id, t.class_id, box_iou_f32(t, p), thr)
            for c in p.class_id:
                pcls.append(c)
            pmatch.extend(m^)
    var res = PrfResult()
    if len(pcls) == 0 and len(tcls) == 0:
        return res^
    var all_cls = pcls.copy()
    all_cls.extend(tcls.copy())
    var classes = sorted_unique(all_cls)
    var nc = len(classes)
    var counts = List[Int](length=nc, fill=0)
    var tp = List[Float64](length=nc * N_THR, fill=0.0)
    var fp = List[Float64](length=nc * N_THR, fill=0.0)
    var fneg = List[Float64](length=nc * N_THR, fill=0.0)
    for c in tcls:
        for k in range(nc):
            if classes[k] == c:
                counts[k] += 1
    for k in range(nc):
        var n_pred = 0
        for c in pcls:
            if c == classes[k]:
                n_pred += 1
        for t in range(N_THR):
            var hit = 0
            for j in range(len(pcls)):
                if pcls[j] == classes[k] and pmatch[j * N_THR + t]:
                    hit += 1
            var i = k * N_THR + t
            if n_pred == 0:
                fneg[i] = Float64(counts[k])
            elif counts[k] == 0:
                fp[i] = Float64(n_pred)
            else:
                tp[i] = Float64(hit)
                fp[i] = Float64(n_pred - hit)
                fneg[i] = Float64(counts[k] - hit)
    res.per_class = List[Float64](length=nc * N_THR, fill=0.0)
    for i in range(nc * N_THR):
        res.per_class[i] = _score(kind, tp[i], fp[i], fneg[i])
    var total = 0.0
    for k in range(nc):
        total += Float64(counts[k])
    for t in range(N_THR):
        var s = 0.0
        if averaging == MACRO:
            for k in range(nc):
                s += res.per_class[k * N_THR + t]
            s /= Float64(nc)
        elif averaging == MICRO:
            var a = 0.0
            var b = 0.0
            var c = 0.0
            for k in range(nc):
                a += tp[k * N_THR + t]
                b += fp[k * N_THR + t]
                c += fneg[k * N_THR + t]
            s = _score(kind, a, b, c)
        elif total > 0.0:
            for k in range(nc):
                s += res.per_class[k * N_THR + t] * Float64(counts[k])
            s /= total
        res.scores[t] = s
    res.classes = classes^
    return res^
