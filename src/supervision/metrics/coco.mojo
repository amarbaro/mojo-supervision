"""COCO-style mean average precision on boxes.

Port of `COCOEvaluator` and `MeanAveragePrecision` from supervision
(metrics/mean_average_precision.py). Inputs are plain arrays, one `Dets` per image;
the S0 `Detections` adapter lives with the caller.
"""

from supervision.detection.utils.iou_and_nms import keypoint_oks_batch
from supervision.metrics.dets import Dets
from supervision.metrics.sortutil import (
    argsort_desc_stable,
    argsort_stable,
    sorted_unique,
)

comptime EPS = 2.220446049250313e-16  # np.spacing(1)
comptime SMALL_AREA = 1024.0  # 32**2
comptime MEDIUM_AREA = 9216.0  # 96**2
comptime MAX_AREA = 1e10  # 1e5**2
comptime N_THR = 10
comptime N_REC = 101
comptime N_AREA = 4


struct Anns(Copyable, Movable):
    """COCO annotations in flat arrays (boxes as x, y, w, h)."""

    var img: List[Int]
    var cat: List[Int]
    var bbox: List[Float64]
    var area: List[Float64]
    var crowd: List[Int]
    var ignore: List[Int]
    var score: List[Float64]
    var id: List[Int]
    var kp: List[Float64]  # keypoint metrics: (x, y, visible) per keypoint, K * 3 per annotation
    var box: List[Float64]  # keypoint targets without visible keypoints: xyxy, 4 per annotation (zeros otherwise)

    def __init__(out self):
        self.img = List[Int]()
        self.cat = List[Int]()
        self.bbox = List[Float64]()
        self.area = List[Float64]()
        self.crowd = List[Int]()
        self.ignore = List[Int]()
        self.score = List[Float64]()
        self.id = List[Int]()
        self.kp = List[Float64]()
        self.box = List[Float64]()


struct ApSet(Copyable, Movable):
    """AP per IoU threshold (`scores`, T) and per class (`per_class`, C x T)."""

    var scores: List[Float64]
    var per_class: List[Float64]

    def __init__(out self):
        self.scores = List[Float64]()
        self.per_class = List[Float64]()


struct MapResult(Copyable, Movable):
    var all: ApSet
    var small: ApSet
    var medium: ApSet
    var large: ApSet
    var iou_thresholds: List[Float64]
    var matched_classes: List[Int]

    def __init__(out self):
        self.all = ApSet()
        self.small = ApSet()
        self.medium = ApSet()
        self.large = ApSet()
        self.iou_thresholds = List[Float64]()
        self.matched_classes = List[Int]()


struct ImgEval(Copyable, Movable):
    """Per image and category result of `_evaluate_image` for one area range."""

    var valid: Bool
    var dt_score: List[Float64]
    var dt_match: List[Int]  # N_THR x nd
    var dt_ignore: List[Int]  # N_THR x nd
    var gt_ignore: List[Int]

    def __init__(out self):
        self.valid = False
        self.dt_score = List[Float64]()
        self.dt_match = List[Int]()
        self.dt_ignore = List[Int]()
        self.gt_ignore = List[Int]()


def iou_thresholds() -> List[Float64]:
    """`np.linspace(0.5, 0.95, 10)` bit for bit."""
    var step = (0.95 - 0.5) / 9.0
    var out = List[Float64]()
    for i in range(N_THR):
        out.append(Float64(i) * step + 0.5)
    out[N_THR - 1] = 0.95
    return out^


def recall_thresholds() -> List[Float64]:
    """`np.linspace(0, 1, 101)` bit for bit."""
    var step = 1.0 / 100.0
    var out = List[Float64]()
    for i in range(N_REC):
        out.append(Float64(i) * step + 0.0)
    out[N_REC - 1] = 1.0
    return out^


def area_range(a: Int) -> Tuple[Float64, Float64]:
    if a == 0:
        return (0.0, MAX_AREA)
    if a == 1:
        return (0.0, SMALL_AREA)
    if a == 2:
        return (SMALL_AREA, MEDIUM_AREA)
    return (MEDIUM_AREA, MAX_AREA)


def _prepare(
    dets: List[Dets],
    image_ids: List[Int],
    is_target: Bool,
    agnostic: Bool,
    any_class: Bool,
) -> Anns:
    """`_prepare_targets` / `_prepare_predictions`: ids start at 1, 0 means no match."""
    var out = Anns()
    for i in range(len(dets)):
        ref d = dets[i]
        for j in range(d.n()):
            var x1 = d.xyxy[4 * j]
            var y1 = d.xyxy[4 * j + 1]
            var w = d.xyxy[4 * j + 2] - x1
            var h = d.xyxy[4 * j + 3] - y1
            var cat = 0
            if agnostic:
                cat = -1 if any_class else 0
            elif len(d.class_id) > 0:
                cat = d.class_id[j]
            out.img.append(image_ids[i])
            out.cat.append(cat)
            out.bbox.append(x1)
            out.bbox.append(y1)
            out.bbox.append(w)
            out.bbox.append(h)
            out.area.append(d.area[j] if len(d.area) > 0 else w * h)
            if is_target:
                var crowd = d.iscrowd[j] if len(d.iscrowd) > 0 else 0
                var ign = d.ignore[j] if len(d.ignore) > 0 else 0
                out.crowd.append(crowd)
                # `_prepare_targets_and_predictions`: crowd implies ignore.
                out.ignore.append(1 if (crowd != 0 or ign != 0) else 0)
                out.score.append(0.0)
            else:
                out.crowd.append(0)
                out.ignore.append(0)
                out.score.append(d.conf[j] if len(d.conf) > 0 else 0.0)
            out.id.append(len(out.id) + 1)
    return out^


def _jaccard(
    gt: Anns, dt: Anns, gsel: List[Int], dsel: List[Int]
) -> List[Float64]:
    """`box_iou_batch_with_jaccard`: len(dsel) x len(gsel), crowd union = det area."""
    var ng = len(gsel)
    var out = List[Float64](length=len(dsel) * ng, fill=0.0)
    for di in range(len(dsel)):
        var d = dsel[di]
        var dx = dt.bbox[4 * d]
        var dy = dt.bbox[4 * d + 1]
        var dx2 = dx + dt.bbox[4 * d + 2]
        var dy2 = dy + dt.bbox[4 * d + 3]
        var area_det = max(dx2 - dx, 0.0) * max(dy2 - dy, 0.0)
        for gi in range(ng):
            var g = gsel[gi]
            var gx = gt.bbox[4 * g]
            var gy = gt.bbox[4 * g + 1]
            var gx2 = gx + gt.bbox[4 * g + 2]
            var gy2 = gy + gt.bbox[4 * g + 3]
            var area_gt = max(gx2 - gx, 0.0) * max(gy2 - gy, 0.0)
            var iw = max(min(dx2, gx2) - max(dx, gx), 0.0)
            var ih = max(min(dy2, gy2) - max(dy, gy), 0.0)
            var inter = iw * ih
            var norm: Float64
            if gt.crowd[g] != 0:
                norm = area_det + EPS
            else:
                norm = area_det + area_gt - inter + EPS
            out[di * ng + gi] = inter / norm
    return out^


def _oks(gt: Anns, dt: Anns, gsel: List[Int], dsel: List[Int], sigmas: List[Float64]) -> List[Float64]:
    """OKS between the selected targets and detections, len(dsel) x len(gsel) like `_jaccard`."""
    var k = len(sigmas)
    var kt = List[Float64]()
    var vis = List[Bool]()
    var area = List[Float64]()
    var box = List[Float64]()
    for g in gsel:
        for j in range(k):
            kt.append(gt.kp[g * k * 3 + 3 * j])
            kt.append(gt.kp[g * k * 3 + 3 * j + 1])
            vis.append(gt.kp[g * k * 3 + 3 * j + 2] > 0)
        area.append(gt.area[g])
        box.extend(gt.box[4 * g : 4 * g + 4])
    var kd = List[Float64]()
    for d in dsel:
        for j in range(k):
            kd.append(dt.kp[d * k * 3 + 3 * j])
            kd.append(dt.kp[d * k * 3 + 3 * j + 1])
    var o = keypoint_oks_batch(kt, kd, k, area, sigmas, vis, box)  # targets x detections
    var out = List[Float64](length=len(dsel) * len(gsel), fill=0.0)
    for gi in range(len(gsel)):
        for di in range(len(dsel)):
            out[di * len(gsel) + gi] = o[gi * len(dsel) + di]
    return out^


def _evaluate_image(
    gt: Anns,
    dt: Anns,
    gsel: List[Int],
    dsel: List[Int],
    iou: List[Float64],
    amin: Float64,
    amax: Float64,
    thr: List[Float64],
) -> ImgEval:
    """`_evaluate_image` for one image, category and area range.

    `dsel` is already sorted by descending score and cut to the largest max-det;
    `iou` is len(dsel) x len(gsel) in the original `gsel` order.
    """
    var res = ImgEval()
    var ng = len(gsel)
    var nd = len(dsel)
    if ng == 0 and nd == 0:
        return res^
    res.valid = True
    # Ground truths ignored by flag or area go last (stable).
    var gign = List[Float64]()
    for g in gsel:
        var a = gt.area[g]
        if gt.ignore[g] != 0 or not (amin <= a and a <= amax):
            gign.append(1.0)
        else:
            gign.append(0.0)
    var gorder = argsort_stable(gign)
    var gign_sorted = List[Int]()
    var crowd = List[Bool]()
    var gid = List[Int]()
    for o in gorder:
        gign_sorted.append(Int(gign[o]))
        crowd.append(gt.crowd[gsel[o]] != 0)
        gid.append(gt.id[gsel[o]])
    res.gt_ignore = gign_sorted.copy()
    var dt_match = List[Int](length=N_THR * nd, fill=0)
    var dt_ign = List[Int](length=N_THR * nd, fill=0)
    if nd > 0 and ng > 0:
        var gt_match = List[Int](length=N_THR * ng, fill=0)
        var avail = List[Bool](length=N_THR * ng, fill=True)
        for di in range(nd):
            for t in range(N_THR):
                var th = min(thr[t], 1.0 - 1e-10)
                var has_kept = False
                for gi in range(ng):
                    if avail[t * ng + gi] and iou[di * ng + gorder[gi]] >= th:
                        if gign_sorted[gi] == 0:
                            has_kept = True
                var best = -1
                var best_iou = -1.0
                for gi in range(ng):
                    if not avail[t * ng + gi]:
                        continue
                    var v = iou[di * ng + gorder[gi]]
                    if v < th:
                        continue
                    if has_kept and gign_sorted[gi] != 0:
                        continue
                    # ties go to the last index (argmax over the reversed row)
                    if v >= best_iou:
                        best_iou = v
                        best = gi
                if best < 0:
                    continue
                dt_ign[t * nd + di] = gign_sorted[best]
                dt_match[t * nd + di] = gid[best]
                gt_match[t * ng + best] = dt.id[dsel[di]]
                avail[t * ng + best] = crowd[best]
    for di in range(nd):
        var a = dt.area[dsel[di]]
        if a < amin or a > amax:
            for t in range(N_THR):
                if dt_match[t * nd + di] == 0:
                    dt_ign[t * nd + di] = 1
    for d in dsel:
        res.dt_score.append(dt.score[d])
    res.dt_match = dt_match^
    res.dt_ignore = dt_ign^
    return res^


def _any_class(dets: List[Dets]) -> Bool:
    for d in dets:
        if d.n() > 0 and len(d.class_id) > 0:
            return True
    return False


def _mean_ap(prec: List[Float32], a: Int, n_cat: Int) -> ApSet:
    """`compute_average_precision`: means that skip the -1 sentinel of absent classes."""
    var out = ApSet()
    out.per_class = List[Float64](length=n_cat * N_THR, fill=-1.0)
    for t in range(N_THR):
        var total = 0.0
        var count = 0
        for c in range(n_cat):
            var csum = 0.0
            var ccount = 0
            var base = ((a * n_cat + c) * N_THR + t) * N_REC
            for r in range(N_REC):
                var v = prec[base + r]
                if v != -1.0:
                    csum += Float64(v)
                    ccount += 1
            total += csum
            count += ccount
            if ccount > 0:
                out.per_class[c * N_THR + t] = csum / Float64(ccount)
        out.scores.append(total / Float64(count) if count > 0 else -1.0)
    return out^


def mean_average_precision(
    preds: List[Dets],
    targets: List[Dets],
    image_ids: List[Int],
    max_dets: List[Int],
    agnostic: Bool = False,
) raises -> MapResult:
    """`MeanAveragePrecision(...).update(preds, targets).compute()` for boxes.

    `image_ids[i]` is the id of image i (`image_indices`, or 0..n-1). `max_dets` holds the
    three limits; AP uses the largest, so only that one is accumulated.
    """
    if len(preds) != len(targets):
        raise Error("predictions and targets must have the same length")
    var any_class = _any_class(preds) or _any_class(targets)
    var gt = _prepare(targets, image_ids, True, agnostic, any_class)
    var dt = _prepare(preds, image_ids, False, agnostic, any_class)
    return evaluate(gt, dt, image_ids, max_dets[len(max_dets) - 1])


def evaluate(gt: Anns, dt: Anns, image_ids: List[Int], max_det: Int, sigmas: List[Float64] = List[Float64]()) raises -> MapResult:
    """The shared COCO evaluation (matching, accumulation, 101-point interpolation). With `sigmas` the overlap is
    the object keypoint similarity (`gt.kp`, `dt.kp`), otherwise the box IoU."""
    var cats = sorted_unique(gt.cat)
    var imgs = sorted_unique(image_ids)
    var n_cat = len(cats)
    var n_img = len(imgs)
    var cat_idx = Dict[Int, Int]()
    for i in range(n_cat):
        cat_idx[cats[i]] = i
    var img_idx = Dict[Int, Int]()
    for i in range(n_img):
        img_idx[imgs[i]] = i
    var gts = List[List[Int]]()
    var dts = List[List[Int]]()
    for _ in range(n_cat * n_img):
        gts.append(List[Int]())
        dts.append(List[Int]())
    for i in range(len(gt.id)):
        if gt.cat[i] in cat_idx:
            gts[cat_idx[gt.cat[i]] * n_img + img_idx[gt.img[i]]].append(i)
    for i in range(len(dt.id)):
        if dt.cat[i] in cat_idx:
            dts[cat_idx[dt.cat[i]] * n_img + img_idx[dt.img[i]]].append(i)

    var thr = iou_thresholds()
    var rec_thr = recall_thresholds()
    var evals = List[ImgEval]()
    for _ in range(n_cat * N_AREA * n_img):
        evals.append(ImgEval())
    for c in range(n_cat):
        for m in range(n_img):
            ref gsel = gts[c * n_img + m]
            var scores = List[Float64]()
            for d in dts[c * n_img + m]:
                scores.append(dt.score[d])
            var dsel = List[Int]()
            var order = argsort_desc_stable(scores)
            for o in order:
                if len(dsel) < max_det:
                    dsel.append(dts[c * n_img + m][o])
            var iou = _jaccard(gt, dt, gsel, dsel) if len(sigmas) == 0 else _oks(gt, dt, gsel, dsel, sigmas)
            for a in range(N_AREA):
                var rng = area_range(a)
                evals[(c * N_AREA + a) * n_img + m] = _evaluate_image(
                    gt, dt, gsel, dsel, iou, rng[0], rng[1], thr
                )

    var prec = List[Float32](length=N_AREA * n_cat * N_THR * N_REC, fill=-1.0)
    for c in range(n_cat):
        for a in range(N_AREA):
            var ev_of = List[Int]()
            var pos_of = List[Int]()
            var scores = List[Float64]()
            var npig = 0
            for m in range(n_img):
                ref e = evals[(c * N_AREA + a) * n_img + m]
                if not e.valid:
                    continue
                for p in range(min(len(e.dt_score), max_det)):
                    scores.append(e.dt_score[p])
                    ev_of.append(m)
                    pos_of.append(p)
                for g in e.gt_ignore:
                    if g == 0:
                        npig += 1
            if npig == 0:
                continue
            var order = argsort_desc_stable(scores)
            var n = len(order)
            for t in range(N_THR):
                var rc = List[Float64](capacity=n)
                var pr = List[Float64](capacity=n)
                var tp = 0.0
                var fp = 0.0
                for o in order:
                    ref e = evals[(c * N_AREA + a) * n_img + ev_of[o]]
                    var k = t * len(e.dt_score) + pos_of[o]
                    if e.dt_ignore[k] == 0:
                        if e.dt_match[k] != 0:
                            tp += 1.0
                        else:
                            fp += 1.0
                    rc.append(tp / Float64(npig))
                    pr.append(tp / (fp + tp + EPS))
                for i in range(n - 1, 0, -1):
                    if pr[i] > pr[i - 1]:
                        pr[i - 1] = pr[i]
                var base = ((a * n_cat + c) * N_THR + t) * N_REC
                var j = 0
                for ri in range(N_REC):
                    while j < n and rc[j] < rec_thr[ri]:
                        j += 1
                    prec[base + ri] = Float32(pr[j]) if j < n else 0.0

    var res = MapResult()
    res.all = _mean_ap(prec, 0, n_cat)
    res.small = _mean_ap(prec, 1, n_cat)
    res.medium = _mean_ap(prec, 2, n_cat)
    res.large = _mean_ap(prec, 3, n_cat)
    res.iou_thresholds = thr^
    res.matched_classes = cats^
    return res^


def mean_valid(scores: List[Float64]) -> Float64:
    """`_mean_valid_score`: mean of the scores above -1, or -1."""
    var total = 0.0
    var count = 0
    for s in scores:
        if s > -1.0:
            total += s
            count += 1
    return total / Float64(count) if count > 0 else -1.0
