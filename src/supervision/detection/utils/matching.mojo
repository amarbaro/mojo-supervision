"""Greedy IoU matching (port of supervision/detection/utils/matching.py)."""

from supervision.detection.core import Detections
from supervision.detection.utils.iou_and_nms import argsort, box_iou_batch, validate_iou_threshold


def greedy_match(iou: List[Float32], n_cols: Int, rows: List[Int], cols: List[Int]) -> List[Tuple[Int, Int]]:
    """Pairs (row, col) among the candidates, best IoU first (stable), each row and col used once."""
    var neg = List[Float64]()
    for k in range(len(rows)):
        neg.append(-Float64(iou[rows[k] * n_cols + cols[k]]))
    var order = argsort(neg)
    var used_r = List[Int]()
    var used_c = List[Int]()
    var out = List[Tuple[Int, Int]]()
    for o in order:
        var r = rows[o]
        var c = cols[o]
        var seen = False
        for x in used_r:
            if x == r:
                seen = True
        for x in used_c:
            if x == c:
                seen = True
        if not seen:
            used_r.append(r)
            used_c.append(c)
            out.append((r, c))
    return out^


struct BatchMatch(Copyable, Movable):
    """correct[p * levels + l], matched_targets[p * levels + l] (-1 when unmatched)."""

    var correct: List[Bool]
    var matched_targets: List[Int]

    def __init__(out self):
        self.correct = List[Bool]()
        self.matched_targets = List[Int]()


def match_detection_batch(
    prediction_classes: List[Int],
    target_classes: List[Int],
    iou: List[Float32],
    iou_thresholds: List[Float64],
    target_scored: List[Bool],
    use_scored: Bool,
) -> BatchMatch:
    """Per IoU level, greedily match predictions to same-class targets (scored targets first when asked).
    `iou` is (n_targets x n_predictions) row-major."""
    var np = len(prediction_classes)
    var nt = len(target_classes)
    var nl = len(iou_thresholds)
    var out = BatchMatch()
    for _ in range(np * nl):
        out.correct.append(False)
        out.matched_targets.append(-1)
    for lv in range(nl):
        var thr = Float32(iou_thresholds[lv])
        var rounds = 2 if use_scored else 1
        for rd in range(rounds):
            var rs = List[Int]()
            var cs = List[Int]()
            for t in range(nt):
                for p in range(np):
                    if out.matched_targets[p * nl + lv] >= 0:
                        continue
                    if not (iou[t * np + p] >= thr and target_classes[t] == prediction_classes[p]):
                        continue
                    if use_scored and target_scored[t] != (rd == 0):
                        continue
                    rs.append(t)
                    cs.append(p)
            for pr in greedy_match(iou, np, rs, cs):
                out.correct[pr[1] * nl + lv] = True
                out.matched_targets[pr[1] * nl + lv] = pr[0]
    return out^


struct DetectionMatches(Copyable, Movable):
    var pairs: List[Tuple[Int, Int]]
    var unmatched_a: List[Int]
    var unmatched_b: List[Int]

    def __init__(out self):
        self.pairs = List[Tuple[Int, Int]]()
        self.unmatched_a = List[Int]()
        self.unmatched_b = List[Int]()


def match_detections(a: Detections, b: Detections, iou_threshold: Float64 = 0.5, class_agnostic: Bool = False) raises -> DetectionMatches:
    validate_iou_threshold(iou_threshold)
    if not class_agnostic and (not a.has_class_id or not b.has_class_id):
        raise Error("Both detections must provide class_id when class_agnostic is False.")
    var out = DetectionMatches()
    var na = len(a)
    var nb = len(b)
    if na > 0 and nb > 0:
        var iou = box_iou_batch(a.xyxy, b.xyxy)
        var thr = Float32(iou_threshold)
        var rs = List[Int]()
        var cs = List[Int]()
        for i in range(na):
            for j in range(nb):
                if iou[i * nb + j] >= thr and (class_agnostic or a.class_id[i] == b.class_id[j]):
                    rs.append(i)
                    cs.append(j)
        out.pairs = greedy_match(iou, nb, rs, cs)
    var ma = List[Bool]()
    var mb = List[Bool]()
    for _ in range(na):
        ma.append(False)
    for _ in range(nb):
        mb.append(False)
    for pr in out.pairs:
        ma[pr[0]] = True
        mb[pr[1]] = True
    for i in range(na):
        if not ma[i]:
            out.unmatched_a.append(i)
    for j in range(nb):
        if not mb[j]:
            out.unmatched_b.append(j)
    return out^
