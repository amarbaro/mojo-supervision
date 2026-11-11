"""Keypoint mean average precision (metrics/keypoint_mean_average_precision.py).

The COCO evaluation of coco.mojo with the object keypoint similarity as the overlap. Like the box metric it takes plain
arrays, one `KpDets` per image. Only the all, medium and large size buckets exist (as pycocotools), at 20 detections.
"""
from std.math import isfinite
from supervision.metrics.coco import Anns, MapResult, evaluate

comptime KEYPOINT_MAX_DETECTIONS = 20


def coco_keypoint_sigmas() -> List[Float64]:
    var v: List[Float64] = [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89]
    for i in range(len(v)):
        v[i] = v[i] / 10.0
    return v^


struct KpDets(Copyable, Movable):
    """Skeletons of one image. An empty optional list means the field is absent."""

    var n: Int
    var k: Int
    var xy: List[Float64]  # n * k * 2
    var visible: List[Int]  # n * k (targets)
    var class_id: List[Int]
    var conf: List[Float64]  # detection confidence (predictions)
    var area: List[Float64]  # targets
    var iscrowd: List[Int]  # targets
    var xyxy: List[Float64]  # `data["xyxy"]`, 4 per skeleton

    def __init__(out self, n: Int, k: Int, var xy: List[Float64]):
        self.n = n
        self.k = k
        self.xy = xy^
        self.visible = List[Int]()
        self.class_id = List[Int]()
        self.conf = List[Float64]()
        self.area = List[Float64]()
        self.iscrowd = List[Int]()
        self.xyxy = List[Float64]()


def _span_area(d: KpDets, i: Int, only_visible: Bool, vis: List[Bool]) -> Float64:
    """Width times height of the box spanning the (visible) keypoints of skeleton i."""
    var lx = Float64.MAX_FINITE
    var ly = Float64.MAX_FINITE
    var hx = -Float64.MAX_FINITE
    var hy = -Float64.MAX_FINITE
    for j in range(d.k):
        if only_visible and not vis[j]:
            continue
        lx = min(lx, d.xy[(i * d.k + j) * 2])
        hx = max(hx, d.xy[(i * d.k + j) * 2])
        ly = min(ly, d.xy[(i * d.k + j) * 2 + 1])
        hy = max(hy, d.xy[(i * d.k + j) * 2 + 1])
    return (hx - lx) * (hy - ly)


def _category(d: KpDets, i: Int, agnostic: Bool, agnostic_id: Int) -> Int:
    if agnostic:
        return agnostic_id
    return d.class_id[i] if len(d.class_id) > 0 else 0


def keypoint_mean_average_precision(
    preds: List[KpDets], targets: List[KpDets], sigmas: List[Float64] = coco_keypoint_sigmas(), agnostic: Bool = False,
) raises -> MapResult:
    """`KeyPointMeanAveragePrecision(sigmas, class_agnostic).update(preds, targets).compute()`."""
    if len(preds) != len(targets):
        raise Error("The number of predictions (" + String(len(preds)) + ") and targets (" + String(len(targets)) + ") during the update must be the same.")
    var k = 0
    for d in preds:
        if d.n > 0:
            k = d.k
    for d in targets:
        if d.n > 0:
            k = d.k
    if k > 0 and len(sigmas) != k:
        raise Error("`sigmas` must have shape (" + String(k) + ",) to match the number of keypoints; got (" + String(len(sigmas)) + ",).")
    var agnostic_id = 0
    for d in preds:
        if d.n > 0 and len(d.class_id) > 0:
            agnostic_id = -1
    for d in targets:
        if d.n > 0 and len(d.class_id) > 0:
            agnostic_id = -1
    var ids = List[Int]()
    var gt = Anns()
    var dt = Anns()
    for img in range(len(targets)):
        ids.append(img)
        ref t = targets[img]
        for i in range(t.n):
            var vis = List[Bool]()
            var any_vis = False
            for j in range(k):
                var x = t.xy[(i * k + j) * 2]
                var y = t.xy[(i * k + j) * 2 + 1]
                vis.append((len(t.visible) == 0 or t.visible[i * k + j] != 0) and isfinite(x) and isfinite(y))
                any_vis = any_vis or vis[j]
            var ignored = not any_vis
            if ignored and len(t.xyxy) == 0:
                continue
            var area: Float64
            if len(t.area) > 0:
                area = t.area[i]
            elif ignored:
                area = (t.xyxy[4 * i + 2] - t.xyxy[4 * i]) * (t.xyxy[4 * i + 3] - t.xyxy[4 * i + 1])
            else:
                area = _span_area(t, i, True, vis)
            var crowd = len(t.iscrowd) > 0 and t.iscrowd[i] != 0
            gt.img.append(img)
            gt.cat.append(_category(t, i, agnostic, agnostic_id))
            gt.area.append(area)
            gt.crowd.append(1 if crowd else 0)
            gt.ignore.append(1 if (crowd or ignored) else 0)
            gt.score.append(0.0)
            gt.id.append(len(gt.id) + 1)
            for j in range(k):
                gt.kp.append(t.xy[(i * k + j) * 2])
                gt.kp.append(t.xy[(i * k + j) * 2 + 1])
                gt.kp.append(1.0 if vis[j] else 0.0)
            for c in range(4):
                gt.box.append(t.xyxy[4 * i + c] if ignored else 0.0)
        ref p = preds[img]
        for i in range(p.n):
            var all_vis = List[Bool](length=k, fill=True)
            var area: Float64
            if len(p.xyxy) > 0:
                area = (p.xyxy[4 * i + 2] - p.xyxy[4 * i]) * (p.xyxy[4 * i + 3] - p.xyxy[4 * i + 1])
            else:
                area = _span_area(p, i, False, all_vis)
            dt.img.append(img)
            dt.cat.append(_category(p, i, agnostic, agnostic_id))
            dt.area.append(area)
            dt.crowd.append(0)
            dt.ignore.append(0)
            dt.score.append(p.conf[i] if len(p.conf) > 0 else 0.0)
            dt.id.append(len(dt.id) + 1)
            for j in range(k):
                dt.kp.append(p.xy[(i * k + j) * 2])
                dt.kp.append(p.xy[(i * k + j) * 2 + 1])
                dt.kp.append(1.0)
    return evaluate(gt, dt, ids, KEYPOINT_MAX_DETECTIONS, sigmas)
