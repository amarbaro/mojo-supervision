"""IoU, NMS, soft NMS and NMM (port of supervision/detection/utils/iou_and_nms.py).

Layout: boxes are flat `List[Float64]` (4 per box), masks are flat `List[UInt8]` (n masks of `pixels`
each, row-major H x W), oriented boxes are flat (8 per box). Categories are `List[Int]`; pass all
zeros for class-agnostic work. Sorting is stable: where Python's `argsort` breaks a tie by quicksort
order (only seen above 16 elements) the order of tied scores can differ, so row generators must not
emit equal scores.
"""

from std.math import exp, isfinite


comptime OVERLAP_IOU = 0
comptime OVERLAP_IOS = 1


def overlap_metric_from_value(value: String) raises -> Int:
    var v = value.upper()
    if v == "IOU":
        return OVERLAP_IOU
    if v == "IOS":
        return OVERLAP_IOS
    raise Error("Invalid value: " + v + ". Must be one of ['IOU', 'IOS']")


def validate_iou_threshold(iou_threshold: Float64) raises:
    if not (0.0 <= iou_threshold and iou_threshold <= 1.0):
        raise Error("Value of `iou_threshold` must be in the closed range from 0 to 1, " + String(iou_threshold) + " given.")


def validate_sigma(sigma: Float64) raises:
    if not sigma > 0.0:
        raise Error("Value of `sigma` must be greater than 0, " + String(sigma) + " given.")


def argsort(values: List[Float64]) -> List[Int]:
    """Stable ascending argsort (bottom-up merge sort)."""
    var n = len(values)
    var idx = List[Int]()
    for i in range(n):
        idx.append(i)
    var tmp = idx.copy()
    var width = 1
    while width < n:
        var lo = 0
        while lo < n:
            var mid = min(lo + width, n)
            var hi = min(lo + 2 * width, n)
            var a = lo
            var b = mid
            var k = lo
            while a < mid and b < hi:
                if values[idx[b]] < values[idx[a]]:
                    tmp[k] = idx[b]
                    b += 1
                else:
                    tmp[k] = idx[a]
                    a += 1
                k += 1
            while a < mid:
                tmp[k] = idx[a]
                a += 1
                k += 1
            while b < hi:
                tmp[k] = idx[b]
                b += 1
                k += 1
            lo += 2 * width
        var swap = idx^
        idx = tmp^
        tmp = swap^
        width *= 2
    return idx^


def _descending(values: List[Float64]) -> List[Int]:
    """np.flip(values.argsort())."""
    var asc = argsort(values)
    var out = List[Int]()
    for i in range(len(asc)):
        out.append(asc[len(asc) - 1 - i])
    return out^


def _ratio(inter: Float64, norm: Float64) -> Float64:
    return inter / norm if norm > 0.0 else 0.0


def box_iou(a: List[Float64], b: List[Float64], metric: Int = OVERLAP_IOU) raises -> Float64:
    """IoU (or IoS) of two boxes given as four numbers each."""
    var iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    var ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    var inter = iw * ih
    var area_a = (a[2] - a[0]) * (a[3] - a[1])
    var area_b = (b[2] - b[0]) * (b[3] - b[1])
    var norm: Float64
    if metric == OVERLAP_IOU:
        norm = area_a + area_b - inter
    elif metric == OVERLAP_IOS:
        norm = min(area_a, area_b)
    else:
        raise Error("overlap_metric " + String(metric) + " is not supported, only 'IOU' and 'IOS' are supported")
    if norm <= 0.0:
        return 0.0
    return inter / norm


def box_iou_batch(boxes_true: List[Float64], boxes_det: List[Float64], metric: Int = OVERLAP_IOU) raises -> List[Float32]:
    """Row-major (n_true x n_det) matrix, float32 like the oracle."""
    var nt = len(boxes_true) // 4
    var nd = len(boxes_det) // 4
    var out = List[Float32]()
    for i in range(nt):
        var a = List[Float64]()
        for k in range(4):
            a.append(boxes_true[4 * i + k])
        for j in range(nd):
            var b = List[Float64]()
            for k in range(4):
                b.append(boxes_det[4 * j + k])
            out.append(Float32(box_iou(a, b, metric)))
    return out^


def box_iou_batch_with_jaccard(boxes_true: List[Float64], boxes_det: List[Float64], is_crowd: List[Bool]) raises -> List[Float64]:
    """pycocotools iou on xywh boxes; result is (n_det x n_true) row-major."""
    var nt = len(boxes_true) // 4
    var nd = len(boxes_det) // 4
    if len(is_crowd) != nt:
        raise Error("`is_crowd` length (" + String(len(is_crowd)) + ") must match `boxes_true` length (" + String(nt) + ").")
    var eps = 2.220446049250313e-16
    var out = List[Float64]()
    for d in range(nd):
        var dx = boxes_det[4 * d]
        var dy = boxes_det[4 * d + 1]
        var dx2 = dx + boxes_det[4 * d + 2]
        var dy2 = dy + boxes_det[4 * d + 3]
        var area_d = max(dx2 - dx, 0.0) * max(dy2 - dy, 0.0)
        for g in range(nt):
            var gx = boxes_true[4 * g]
            var gy = boxes_true[4 * g + 1]
            var gx2 = gx + boxes_true[4 * g + 2]
            var gy2 = gy + boxes_true[4 * g + 3]
            var inter = max(min(dx2, gx2) - max(dx, gx), 0.0) * max(min(dy2, gy2) - max(dy, gy), 0.0)
            var area_g = max(gx2 - gx, 0.0) * max(gy2 - gy, 0.0)
            var norm = area_d + eps if is_crowd[g] else area_d + area_g - inter + eps
            out.append(inter / norm)
    return out^


def mask_iou_batch(masks_true: List[UInt8], masks_det: List[UInt8], pixels: Int, metric: Int = OVERLAP_IOU) raises -> List[Float64]:
    """Overlap of every true mask with every detection mask (n_true x n_det, row-major)."""
    if pixels == 0:
        return List[Float64]()
    var nt = len(masks_true) // pixels
    var nd = len(masks_det) // pixels
    var area_t = List[Int]()
    for i in range(nt):
        var c = 0
        for p in range(pixels):
            c += Int(masks_true[i * pixels + p] != 0)
        area_t.append(c)
    var area_d = List[Int]()
    for j in range(nd):
        var c = 0
        for p in range(pixels):
            c += Int(masks_det[j * pixels + p] != 0)
        area_d.append(c)
    var out = List[Float64]()
    for i in range(nt):
        for j in range(nd):
            var inter = 0
            for p in range(pixels):
                inter += Int(masks_true[i * pixels + p] != 0 and masks_det[j * pixels + p] != 0)
            var norm: Int
            if metric == OVERLAP_IOU:
                norm = area_t[i] + area_d[j] - inter
            elif metric == OVERLAP_IOS:
                norm = min(area_t[i], area_d[j])
            else:
                raise Error("overlap_metric " + String(metric) + " is not supported, only 'IOU' and 'IOS' are supported")
            out.append(Float64(inter) / Float64(norm) if norm != 0 else 0.0)
    return out^


# ---- oriented boxes ---------------------------------------------------------------------------------------


def _clip_convex(subject: List[Float64], clip: List[Float64]) -> List[Float64]:
    """Sutherland-Hodgman: subject polygon (x, y pairs) clipped by the convex clip polygon."""
    var out = subject.copy()
    var nc = len(clip) // 2
    # orientation of the clip polygon: +1 counter clockwise in math axes
    var area2 = 0.0
    for k in range(nc):
        var j = (k + 1) % nc
        area2 += clip[2 * k] * clip[2 * j + 1] - clip[2 * j] * clip[2 * k + 1]
    var sgn = 1.0 if area2 >= 0.0 else -1.0
    for k in range(nc):
        if len(out) == 0:
            break
        var j = (k + 1) % nc
        var ex = clip[2 * j] - clip[2 * k]
        var ey = clip[2 * j + 1] - clip[2 * k + 1]
        var inp = out^
        out = List[Float64]()
        var m = len(inp) // 2
        for t in range(m):
            var u = (t + m - 1) % m
            var cx = inp[2 * t]
            var cy = inp[2 * t + 1]
            var px = inp[2 * u]
            var py = inp[2 * u + 1]
            var sc = sgn * (ex * (cy - clip[2 * k + 1]) - ey * (cx - clip[2 * k]))
            var sp = sgn * (ex * (py - clip[2 * k + 1]) - ey * (px - clip[2 * k]))
            if sc >= 0.0:
                if sp < 0.0:
                    var f = sp / (sp - sc)
                    out.append(px + (cx - px) * f)
                    out.append(py + (cy - py) * f)
                out.append(cx)
                out.append(cy)
            elif sp >= 0.0:
                var f = sp / (sp - sc)
                out.append(px + (cx - px) * f)
                out.append(py + (cy - py) * f)
    return out^


def _poly_area(p: List[Float64]) -> Float64:
    var n = len(p) // 2
    if n < 3:
        return 0.0
    var s = 0.0
    for k in range(n):
        var j = (k + 1) % n
        s += (p[2 * k] - p[0]) * (p[2 * j + 1] - p[1]) - (p[2 * j] - p[0]) * (p[2 * k + 1] - p[1])
    return 0.5 * abs(s)


def intersect_convex_convex(a: List[Float64], b: List[Float64]) -> Float64:
    """Area of the intersection of two convex polygons (x, y pairs)."""
    return _poly_area(_clip_convex(a, b))


def _obb_envelope(c: List[Float64], i: Int) -> List[Float64]:
    var x0 = c[8 * i]
    var x1 = x0
    var y0 = c[8 * i + 1]
    var y1 = y0
    for k in range(1, 4):
        var x = c[8 * i + 2 * k]
        var y = c[8 * i + 2 * k + 1]
        x0 = min(x0, x)
        x1 = max(x1, x)
        y0 = min(y0, y)
        y1 = max(y1, y)
    return [x0, y0, x1, y1]


def _poly(c: List[Float64], i: Int) -> List[Float64]:
    var p = List[Float64]()
    for k in range(8):
        p.append(c[8 * i + k])
    return p^


def oriented_box_iou_batch(boxes_true: List[Float64], boxes_det: List[Float64], metric: Int = OVERLAP_IOU) raises -> List[Float64]:
    """IoU of oriented boxes (four corners each, flat); polygons are clipped in Float64.
    cv2 clips in float32, so rows should carry an absolute tolerance of 1e-5."""
    if metric != OVERLAP_IOU and metric != OVERLAP_IOS:
        raise Error("overlap_metric " + String(metric) + " is not supported, only 'IOU' and 'IOS' are supported")
    var n = len(boxes_true) // 8
    var m = len(boxes_det) // 8
    var out = List[Float64]()
    for _ in range(n * m):
        out.append(0.0)
    if n == 0 or m == 0:
        return out^
    var area_t = obb_areas(boxes_true)
    var area_d = obb_areas(boxes_det)
    for i in range(n):
        var et = _obb_envelope(boxes_true, i)
        for j in range(m):
            var ed = _obb_envelope(boxes_det, j)
            if not (min(et[2], ed[2]) > max(et[0], ed[0]) and min(et[3], ed[3]) > max(et[1], ed[1])):
                continue
            var inter = intersect_convex_convex(_poly(boxes_true, i), _poly(boxes_det, j))
            if inter <= 0.0:
                continue
            var den = area_t[i] + area_d[j] - inter if metric == OVERLAP_IOU else min(area_t[i], area_d[j])
            if den > 0.0:
                out[i * m + j] = min(max(inter / den, 0.0), 1.0)
    return out^


def obb_areas(c: List[Float64]) -> List[Float64]:
    var out = List[Float64]()
    for i in range(len(c) // 8):
        out.append(_poly_area(_poly(c, i)))
    return out^


# ---- NMS --------------------------------------------------------------------------------------------------


def _nms_keep_from_matrix_f32(ious: List[Float32], cats: List[Int], n: Int, thr: Float64) -> List[Bool]:
    var t = Float32(thr)
    var keep = List[Bool]()
    for _ in range(n):
        keep.append(True)
    for index in range(n):
        if not keep[index]:
            continue
        for j in range(n):
            var v = ious[index * n + j] - (Float32(1.0) if index == j else Float32(0.0))
            if v > t and cats[index] == cats[j]:
                keep[j] = False
    return keep^


def _nms_keep_from_matrix_f64(ious: List[Float64], cats: List[Int], n: Int, thr: Float64) -> List[Bool]:
    var keep = List[Bool]()
    for _ in range(n):
        keep.append(True)
    for row in range(n):
        if keep[row]:
            for j in range(row + 1, n):
                if ious[row * n + j] > thr and cats[row] == cats[j]:
                    keep[j] = False
    return keep^


def _gather_boxes(xyxy: List[Float64], order: List[Int], width: Int) -> List[Float64]:
    var out = List[Float64]()
    for i in range(len(order)):
        for k in range(width):
            out.append(xyxy[width * order[i] + k])
    return out^


def _gather_masks(masks: List[UInt8], order: List[Int], pixels: Int) -> List[UInt8]:
    var out = List[UInt8]()
    for i in range(len(order)):
        for p in range(pixels):
            out.append(masks[order[i] * pixels + p])
    return out^


def _gather_int(v: List[Int], order: List[Int]) -> List[Int]:
    var out = List[Int]()
    for i in range(len(order)):
        out.append(v[order[i]])
    return out^


def _scatter_keep(keep_sorted: List[Bool], order: List[Int]) -> List[Bool]:
    var out = List[Bool]()
    for _ in range(len(order)):
        out.append(False)
    for k in range(len(order)):
        out[order[k]] = keep_sorted[k]
    return out^


def box_non_max_suppression(xyxy: List[Float64], scores: List[Float64], cats: List[Int], iou_threshold: Float64 = 0.5, metric: Int = OVERLAP_IOU) raises -> List[Bool]:
    """Boolean keep flag per input box."""
    validate_iou_threshold(iou_threshold)
    var order = _descending(scores)
    var b = _gather_boxes(xyxy, order, 4)
    var ious = box_iou_batch(b, b, metric)
    var keep = _nms_keep_from_matrix_f32(ious, _gather_int(cats, order), len(order), iou_threshold)
    return _scatter_keep(keep, order)


def oriented_box_non_max_suppression(corners: List[Float64], scores: List[Float64], cats: List[Int], iou_threshold: Float64 = 0.5, metric: Int = OVERLAP_IOU) raises -> List[Bool]:
    validate_iou_threshold(iou_threshold)
    var order = _descending(scores)
    var b = _gather_boxes(corners, order, 8)
    var ious = oriented_box_iou_batch(b, b, metric)
    var keep = _nms_keep_from_matrix_f64(ious, _gather_int(cats, order), len(order), iou_threshold)
    return _scatter_keep(keep, order)


def mask_non_max_suppression(masks: List[UInt8], pixels: Int, scores: List[Float64], cats: List[Int], iou_threshold: Float64 = 0.5, metric: Int = OVERLAP_IOU) raises -> List[Bool]:
    validate_iou_threshold(iou_threshold)
    var order = _descending(scores)
    var m = _gather_masks(masks, order, pixels)
    var ious = mask_iou_batch(m, m, pixels, metric)
    var keep = _nms_keep_from_matrix_f64(ious, _gather_int(cats, order), len(order), iou_threshold)
    return _scatter_keep(keep, order)


def _soft_decay(ious: List[Float64], cats: List[Int], scores: List[Float64], n: Int, sigma: Float64) -> List[Float64]:
    var out = List[Float64]()
    for j in range(n):
        var s = 0.0
        for i in range(j):
            if cats[i] == cats[j]:
                s += ious[i * n + j] * ious[i * n + j]
        out.append(scores[j] * exp(-s / sigma))
    return out^


def _scatter_f64(v: List[Float64], order: List[Int]) -> List[Float64]:
    var out = List[Float64]()
    for _ in range(len(order)):
        out.append(0.0)
    for k in range(len(order)):
        out[order[k]] = v[k]
    return out^


def box_soft_non_max_suppression(xyxy: List[Float64], scores: List[Float64], cats: List[Int], sigma: Float64 = 0.5, metric: Int = OVERLAP_IOU) raises -> List[Float64]:
    """Decayed score per input box."""
    validate_sigma(sigma)
    var order = _descending(scores)
    var b = _gather_boxes(xyxy, order, 4)
    var f32 = box_iou_batch(b, b, metric)
    var ious = List[Float64]()
    for i in range(len(f32)):
        ious.append(Float64(f32[i]))
    var sorted_scores = List[Float64]()
    for k in range(len(order)):
        sorted_scores.append(scores[order[k]])
    return _scatter_f64(_soft_decay(ious, _gather_int(cats, order), sorted_scores, len(order), sigma), order)


def mask_soft_non_max_suppression(masks: List[UInt8], pixels: Int, scores: List[Float64], cats: List[Int], sigma: Float64 = 0.5, metric: Int = OVERLAP_IOU) raises -> List[Float64]:
    validate_sigma(sigma)
    var order = _descending(scores)
    var m = _gather_masks(masks, order, pixels)
    var ious = mask_iou_batch(m, m, pixels, metric)
    var sorted_scores = List[Float64]()
    for k in range(len(order)):
        sorted_scores.append(scores[order[k]])
    return _scatter_f64(_soft_decay(ious, _gather_int(cats, order), sorted_scores, len(order), sigma), order)


# ---- NMM --------------------------------------------------------------------------------------------------


def _unique_sorted(cats: List[Int]) -> List[Int]:
    var u = List[Int]()
    for c in cats:
        var found = False
        for x in u:
            if x == c:
                found = True
        if not found:
            u.append(c)
    # insertion sort
    for i in range(1, len(u)):
        var v = u[i]
        var j = i - 1
        while j >= 0 and u[j] > v:
            u[j + 1] = u[j]
            j -= 1
        u[j + 1] = v
    return u^


def _overlap_col(oriented: Bool, data: List[Float64], order: List[Int], idx: Int, metric: Int) raises -> List[Float64]:
    """Overlap of every box in `order` with box `idx`."""
    var r = List[Float64]()
    if oriented:
        return oriented_box_iou_batch(_gather_boxes(data, order, 8), _gather_boxes(data, [idx], 8), metric)
    var f = box_iou_batch(_gather_boxes(data, order, 4), _gather_boxes(data, [idx], 4), metric)
    for k in range(len(f)):
        r.append(Float64(f[k]))
    return r^


def _group_overlapping(oriented: Bool, data: List[Float64], metric: Int, scores: List[Float64], threshold: Float64) raises -> List[List[Int]]:
    """Greedy NMM grouping: take the best remaining score, absorb everything at or above the threshold."""
    var groups = List[List[Int]]()
    var order = argsort(scores)
    while len(order) > 0:
        var idx = order.pop()
        if len(order) == 0:
            groups.append([idx])
            break
        var ious = _overlap_col(oriented, data, order, idx, metric)
        var group: List[Int] = [idx]
        var rest = List[Int]()
        var above = List[Int]()
        for k in range(len(order)):
            if ious[k] >= threshold:
                above.append(order[k])
            else:
                rest.append(order[k])
        for k in range(len(above)):
            group.append(above[len(above) - 1 - k])
        groups.append(group^)
        order = rest^
    return groups^


def _subset(v: List[Float64], idx: List[Int]) -> List[Float64]:
    var out = List[Float64]()
    for i in idx:
        out.append(v[i])
    return out^


def box_non_max_merge(xyxy: List[Float64], scores: List[Float64], cats: List[Int], iou_threshold: Float64 = 0.5, metric: Int = OVERLAP_IOU) raises -> List[List[Int]]:
    """Groups of indices to merge, one list per output box, winner first."""
    validate_iou_threshold(iou_threshold)
    var out = List[List[Int]]()
    for c in _unique_sorted(cats):
        var cur = List[Int]()
        for i in range(len(cats)):
            if cats[i] == c:
                cur.append(i)
        var sub_boxes = _gather_boxes(xyxy, cur, 4)
        var sub_scores = _subset(scores, cur)

        for g in _group_overlapping(False, sub_boxes, metric, sub_scores, iou_threshold):
            var glob = List[Int]()
            for i in g:
                glob.append(cur[i])
            out.append(glob^)
    return out^


def oriented_box_non_max_merge(corners: List[Float64], scores: List[Float64], cats: List[Int], iou_threshold: Float64 = 0.5, metric: Int = OVERLAP_IOU) raises -> List[List[Int]]:
    validate_iou_threshold(iou_threshold)
    var out = List[List[Int]]()
    for c in _unique_sorted(cats):
        var cur = List[Int]()
        for i in range(len(cats)):
            if cats[i] == c:
                cur.append(i)
        var sub = _gather_boxes(corners, cur, 8)
        var sub_scores = _subset(scores, cur)

        for g in _group_overlapping(True, sub, metric, sub_scores, iou_threshold):
            var glob = List[Int]()
            for i in g:
                glob.append(cur[i])
            out.append(glob^)
    return out^


def mask_non_max_merge(masks: List[UInt8], pixels: Int, scores: List[Float64], cats: List[Int], iou_threshold: Float64 = 0.5, metric: Int = OVERLAP_IOU) raises -> List[List[Int]]:
    """Mask NMM: the candidate grows to the union of every mask it absorbs."""
    validate_iou_threshold(iou_threshold)
    var out = List[List[Int]]()
    for c in _unique_sorted(cats):
        var cur = List[Int]()
        for i in range(len(cats)):
            if cats[i] == c:
                cur.append(i)
        var sub = _gather_masks(masks, cur, pixels)
        var sub_scores = _subset(scores, cur)
        var groups = List[List[Int]]()
        var order = argsort(sub_scores)
        while len(order) > 0:
            var idx = order.pop()
            if len(order) == 0:
                groups.append([idx])
                break
            var cand = _gather_masks(sub, [idx], pixels)
            var group: List[Int] = [idx]
            while len(order) > 0:
                var ious = mask_iou_batch(_gather_masks(sub, order, pixels), cand, pixels, metric)
                var above = List[Int]()
                var rest = List[Int]()
                for k in range(len(order)):
                    if ious[k] >= iou_threshold:
                        above.append(order[k])
                    else:
                        rest.append(order[k])
                if len(above) == 0:
                    break
                for a in above:
                    for p in range(pixels):
                        if sub[a * pixels + p] != 0:
                            cand[p] = 1
                for k in range(len(above)):
                    group.append(above[len(above) - 1 - k])
                order = rest^
            groups.append(group^)
        for g in groups:
            var glob = List[Int]()
            for i in g:
                glob.append(cur[i])
            out.append(glob^)
    return out^


def keypoint_oks_batch(
    kt: List[Float64], kd: List[Float64], k: Int, area: List[Float64], sigmas: List[Float64],
    visible: List[Bool], boxes: List[Float64],
) -> List[Float64]:
    """`_keypoint_oks_batch`: object keypoint similarity, len(area) targets x (len(kd) / (2 k)) detections.
    `kt` / `kd` hold x, y per keypoint; `visible` is N x K; `boxes` (N x 4 xyxy, or empty) stand in for a target with no
    visible keypoint: its keypoints count at their distance to the box grown by its width and height."""
    var nt = len(area)
    var nd = len(kd) // (2 * k) if k > 0 else 0
    var out = List[Float64](length=nt * nd, fill=0.0)
    for t in range(nt):
        var num_visible = 0
        for j in range(k):
            num_visible += Int(visible[t * k + j])
        var use_box = len(boxes) > 0 and num_visible == 0
        var x = boxes[4 * t] if len(boxes) > 0 else 0.0
        var y = boxes[4 * t + 1] if len(boxes) > 0 else 0.0
        var w = (boxes[4 * t + 2] - x) if len(boxes) > 0 else 0.0
        var h = (boxes[4 * t + 3] - y) if len(boxes) > 0 else 0.0
        var counted_n = k if use_box else num_visible
        if counted_n == 0:
            continue
        for d in range(nd):
            var sim = 0.0
            for j in range(k):
                if not (use_box or visible[t * k + j]):
                    continue
                var xd = kd[(d * k + j) * 2]
                var yd = kd[(d * k + j) * 2 + 1]
                if not (isfinite(xd) and isfinite(yd)):
                    continue
                var dx: Float64
                var dy: Float64
                if use_box:
                    dx = max(0.0, (x - w) - xd) + max(0.0, xd - (x + w * 2))
                    dy = max(0.0, (y - h) - yd) + max(0.0, yd - (y + h * 2))
                else:
                    dx = xd - kt[(t * k + j) * 2]
                    dy = yd - kt[(t * k + j) * 2 + 1]
                if not (isfinite(dx) and isfinite(dy)):
                    continue
                var var_j = (sigmas[j] * 2) ** 2
                sim += exp(-((dx**2 + dy**2) / var_j / (area[t] + 2.220446049250313e-16) / 2))
            out[t * nd + d] = sim / Float64(counted_n)
    return out^
