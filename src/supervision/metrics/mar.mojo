"""Mean average recall at 1, 10 and 100 detections per class, port of `MeanAverageRecall`."""

from supervision.metrics.dets import Dets
from supervision.metrics.iou import N_THR, box_iou_f32, thresholds_f32
from supervision.metrics.sortutil import argsort_desc_stable, sorted_unique

comptime N_K = 3


def _limit(k: Int) -> Int:
    return 1 if k == 0 else (10 if k == 1 else 100)


def _drop_ignored(t: Dets) -> Dets:
    """Targets with `iscrowd` or `ignore` set are ignore regions, not targets."""
    var out = Dets()
    for j in range(t.n()):
        var skip = (len(t.iscrowd) > 0 and t.iscrowd[j] != 0) or (
            len(t.ignore) > 0 and t.ignore[j] != 0
        )
        if skip:
            continue
        for q in range(4):
            out.xyxy.append(t.xyxy[4 * j + q])
        if len(t.class_id) > 0:
            out.class_id.append(t.class_id[j])
    return out^


def _match_top(
    cls: List[Int],
    ranks: List[Int],
    true_cls: List[Int],
    iou: List[Float32],
    order: List[Int],
    thr: List[Float32],
) -> List[Bool]:
    """`_match_top_predictions`: P x Th x K flags; `iou` is T x P in input order."""
    var np_ = len(cls)
    var nt = len(true_cls)
    var out = List[Bool](length=np_ * N_THR * N_K, fill=False)
    for k in range(N_K):
        for ti in range(N_THR):
            var taken = List[Bool](length=nt, fill=False)
            for p in range(np_):
                if ranks[p] >= _limit(k):
                    continue
                var best = -1
                var best_iou = Float32(-1.0)
                for t in range(nt):
                    if true_cls[t] != cls[p] or taken[t]:
                        continue
                    var v = iou[t * np_ + order[p]]
                    if v < thr[ti]:
                        continue
                    if v >= best_iou:  # ties go to the last eligible target
                        best_iou = v
                        best = t
                if best >= 0:
                    taken[best] = True
                    out[(p * N_THR + ti) * N_K + k] = True
    return out^


def mean_average_recall(preds: List[Dets], targets: List[Dets]) raises -> List[Float64]:
    """`MeanAverageRecall(...).update(preds, targets).compute().recall_scores`."""
    if len(preds) != len(targets):
        raise Error("predictions and targets must have the same length")
    var thr = thresholds_f32()
    var pcls = List[Int]()  # sorted by confidence inside each image
    var prank = List[Int]()
    var pmatch = List[Bool]()  # P x Th x K
    var tcls = List[Int]()
    for i in range(len(preds)):
        ref p = preds[i]
        var t = _drop_ignored(targets[i])
        if t.n() == 0:
            continue
        if len(t.class_id) == 0 or (p.n() > 0 and len(p.class_id) == 0):
            raise Error("MeanAverageRecall requires `class_id` on both predictions and targets")
        for c in t.class_id:
            tcls.append(c)
        if p.n() == 0:
            continue
        if len(p.conf) == 0:
            raise Error("MeanAverageRecall requires `confidence` on predictions")
        var iou = box_iou_f32(t, p)
        var order = argsort_desc_stable(p.conf)
        var cls = List[Int]()
        for o in order:
            cls.append(p.class_id[o])
        var ranks = List[Int]()
        for q in range(len(cls)):
            var r = 0
            for s in range(q):
                if cls[s] == cls[q]:
                    r += 1
            ranks.append(r)
        pmatch.extend(_match_top(cls, ranks, t.class_id, iou, order, thr))
        pcls.extend(cls^)
        prank.extend(ranks^)
    var out = List[Float64](length=N_K, fill=0.0)
    var classes = sorted_unique(tcls)
    if len(classes) == 0:
        return out^
    for k in range(N_K):
        var total = 0.0
        for c in classes:
            var n_true = 0
            for x in tcls:
                if x == c:
                    n_true += 1
            var rec_sum = 0.0
            for ti in range(N_THR):
                var tp = 0
                for j in range(len(pcls)):
                    if pcls[j] == c and prank[j] < _limit(k) and pmatch[(j * N_THR + ti) * N_K + k]:
                        tp += 1
                var den = Float64(n_true)  # tp + fn = n_true whatever the prediction count
                rec_sum += Float64(tp) / den if den != 0.0 else 0.0
            total += rec_sum / Float64(N_THR)
        out[k] = total / Float64(len(classes))
    return out^
