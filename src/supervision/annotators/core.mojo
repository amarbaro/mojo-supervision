# Port of supervision/annotators/core.py for the annotators that need no text: Box, Mask, Polygon, Circle, Dot, Halo, Blur,
# Pixelate, Trace. Free functions on a BGR `Mat[uint8]` scene, edited in place; colors come from a ColorPalette (a single
# Color is a one-entry palette, by_idx wraps) and `lookup` is LOOKUP_INDEX / LOOKUP_CLASS / LOOKUP_TRACK.
# LabelAnnotator is label_annotate (border_radius, smart_position and max_line_length are not ported). Not ported yet: the other
# text annotators, the other ~12 annotators.
from supervision.geometry.core import Position
from std.math import sqrt
from mojocv.mat import Mat
from mojocv.imgproc import draw_rect, find_contours, resize, cvt_color, cvt_code, _circle_fill
from supervision.detection.core import Detections
from supervision.detection.utils.boxes import clip_boxes
from supervision.draw.color import Color, ColorPalette
from supervision._cv import draw
from supervision._cv.font.layout import TextEngine
from supervision._cv.ops import add_weighted, blur, mean
from supervision.annotators.utils import resolve_color, Trace, PENDING_TRACK_ID, LOOKUP_CLASS

comptime Img = Mat[DType.uint8]


def _bgr(c: Color) -> List[Float64]:
    return [Float64(c.b), Float64(c.g), Float64(c.r)]


def box_annotate(mut scene: Img, det: Detections, palette: ColorPalette, thickness: Int = 2, lookup: Int = LOOKUP_CLASS) raises:
    for i in range(len(det)):
        var c = resolve_color(palette, det, i, lookup)
        draw_rect(scene, Int(det.xyxy[4 * i]), Int(det.xyxy[4 * i + 1]), Int(det.xyxy[4 * i + 2]), Int(det.xyxy[4 * i + 3]), _bgr(c), thickness)


def circle_annotate(mut scene: Img, det: Detections, palette: ColorPalette, thickness: Int = 2, lookup: Int = LOOKUP_CLASS) raises:
    for i in range(len(det)):
        var c = resolve_color(palette, det, i, lookup)
        var x1 = Int(det.xyxy[4 * i])
        var y1 = Int(det.xyxy[4 * i + 1])
        var x2 = Int(det.xyxy[4 * i + 2])
        var y2 = Int(det.xyxy[4 * i + 3])
        var cx = (x1 + x2) // 2
        var cy = (y1 + y2) // 2
        var distance = sqrt(Float64((x1 - cx) * (x1 - cx) + (y1 - cy) * (y1 - cy)))
        draw.circle(scene, cx, cy, Int(distance), _bgr(c), thickness)


def dot_annotate(mut scene: Img, det: Detections, palette: ColorPalette, radius: Int = 4, position: Int = 0, lookup: Int = LOOKUP_CLASS) raises:
    var xy = det.get_anchors_coordinates(Position(position))
    for i in range(len(det)):
        var c = resolve_color(palette, det, i, lookup)
        _circle_fill(scene, Int(xy[2 * i]), Int(xy[2 * i + 1]), radius, _bgr(c))


def _paint_masks_by_area(mut canvas: Img, det: Detections, palette: ColorPalette, lookup: Int, mut union: List[Bool]) raises:
    """Largest masks first so smaller ones end up on top; `union` receives the footprint of all masks."""
    var area = det.area()
    var order = List[Int]()
    for i in range(len(det)):
        order.append(i)
    for a in range(1, len(order)):  # stable insertion sort, descending area
        var cur = order[a]
        var b = a - 1
        while b >= 0 and area[order[b]] < area[cur]:
            order[b + 1] = order[b]
            b -= 1
        order[b + 1] = cur
    var px = det.mask_h * det.mask_w
    for i in order:
        var col = _bgr(resolve_color(palette, det, i, lookup))
        for p in range(px):
            if det.mask[i * px + p] != 0:
                for k in range(canvas.ch):
                    canvas.data[p * canvas.ch + k] = UInt8(col[k])
                union[p] = True


def mask_annotate(mut scene: Img, det: Detections, palette: ColorPalette, opacity: Float64 = 0.5, lookup: Int = LOOKUP_CLASS) raises:
    if not det.has_mask:
        return
    var colored = Img(copy=scene)
    var union = List[Bool](length=scene.rows * scene.cols, fill=False)
    _paint_masks_by_area(colored, det, palette, lookup, union)
    scene = add_weighted(colored, opacity, scene, 1.0 - opacity, 0.0)


def polygon_annotate(mut scene: Img, det: Detections, palette: ColorPalette, thickness: Int = 2, lookup: Int = LOOKUP_CLASS) raises:
    if not det.has_mask:
        return
    var px = det.mask_h * det.mask_w
    for i in range(len(det)):
        var c = resolve_color(palette, det, i, lookup)
        var m = Img(det.mask_h, det.mask_w, 1)
        for p in range(px):
            m.data[p] = 1 if det.mask[i * px + p] != 0 else 0
        var res = find_contours(m, 0, True)
        var pts = Mat[DType.int32](copy=res[0])
        var lens = Mat[DType.int32](copy=res[1])
        var at = 0
        for k in range(lens.rows):
            var n = Int(lens.data[k])
            if n >= 3:
                var xs = List[Int]()
                var ys = List[Int]()
                for j in range(n):
                    xs.append(Int(pts.data[2 * (at + j)]))
                    ys.append(Int(pts.data[2 * (at + j) + 1]))
                draw.polylines(scene, xs, ys, True, _bgr(c), thickness)
            at += n


def blur_annotate(mut scene: Img, det: Detections, kernel_size: Int = 15) raises:
    if kernel_size < 1:
        raise Error("kernel_size must be >= 1, got " + String(kernel_size))
    var b = clip_boxes(det.xyxy, scene.cols, scene.rows)
    for i in range(len(det)):
        var x1 = Int(b[4 * i])
        var y1 = Int(b[4 * i + 1])
        var x2 = Int(b[4 * i + 2])
        var y2 = Int(b[4 * i + 3])
        if x2 <= x1 or y2 <= y1:
            continue
        var roi = _crop(scene, x1, y1, x2, y2)
        _paste(scene, blur(roi, kernel_size), x1, y1)


def pixelate_annotate(mut scene: Img, det: Detections, pixel_size: Int = 20) raises:
    if pixel_size < 1:
        raise Error("pixel_size must be >= 1, got " + String(pixel_size))
    var b = clip_boxes(det.xyxy, scene.cols, scene.rows)
    for i in range(len(det)):
        var x1 = Int(b[4 * i])
        var y1 = Int(b[4 * i + 1])
        var x2 = Int(b[4 * i + 2])
        var y2 = Int(b[4 * i + 3])
        if x2 <= x1 or y2 <= y1:
            continue
        var roi = _crop(scene, x1, y1, x2, y2)
        if min(y2 - y1, x2 - x1) < pixel_size:
            for c in range(scene.ch):
                var v = UInt8(Int(mean(roi, c)))  # numpy assigns a float to uint8 by truncation
                for p in range(roi.rows * roi.cols):
                    roi.data[p * roi.ch + c] = v
            _paste(scene, roi, x1, y1)
            continue
        var dw = max(1, Int(Float64(roi.cols) / Float64(pixel_size) + 0.5))
        var dh = max(1, Int(Float64(roi.rows) / Float64(pixel_size) + 0.5))
        var small = resize(roi, dw, dh, 1)
        _paste(scene, resize(small, roi.cols, roi.rows, 0), x1, y1)


def halo_annotate(mut scene: Img, det: Detections, palette: ColorPalette, opacity: Float64 = 0.8, kernel_size: Int = 40, lookup: Int = LOOKUP_CLASS) raises:
    if not det.has_mask:
        return
    var colored = Img(scene.rows, scene.cols, scene.ch)
    var union = List[Bool](length=scene.rows * scene.cols, fill=False)
    _paint_masks_by_area(colored, det, palette, lookup, union)
    colored = blur(colored, kernel_size)
    for p in range(scene.rows * scene.cols):
        if union[p]:
            for k in range(scene.ch):
                colored.data[p * scene.ch + k] = 0
    var gray = cvt_color(colored, cvt_code("bgr2gray"))
    var gmax = 0
    for p in range(len(gray.data)):
        gmax = max(gmax, Int(gray.data[p]))
    if gmax == 0:
        return
    for p in range(scene.rows * scene.cols):
        var alpha = opacity * Float64(gray.data[p]) / Float64(gmax)
        for k in range(scene.ch):
            var v = Float64(scene.data[p * scene.ch + k]) * (1.0 - alpha) + Float64(colored.data[p * scene.ch + k]) * alpha
            scene.data[p * scene.ch + k] = UInt8(Int(min(255.0, max(0.0, v))))


def trace_annotate(mut scene: Img, det: Detections, mut trace: Trace, palette: ColorPalette, thickness: Int = 2, lookup: Int = LOOKUP_CLASS) raises:
    if len(det) == 0:
        trace.put(det)
        return
    if not det.has_tracker_id:
        raise Error("The `tracker_id` field is missing in the provided detections.")
    var keep = List[Int]()
    for i in range(len(det)):
        if det.tracker_id[i] != PENDING_TRACK_ID:
            keep.append(i)
    var filtered = det.select(keep)
    trace.put(filtered)
    for i in range(len(filtered)):
        var c = resolve_color(palette, filtered, i, lookup)
        var xs = List[Float32]()
        var ys = List[Float32]()
        trace.get(filtered.tracker_id[i], xs, ys)
        if len(xs) > 1:
            var ix = List[Int]()
            var iy = List[Int]()
            for k in range(len(xs)):
                ix.append(Int(xs[k]))
                iy.append(Int(ys[k]))
            draw.polylines(scene, ix, iy, False, _bgr(c), thickness)


def _crop(a: Img, x1: Int, y1: Int, x2: Int, y2: Int) -> Img:
    var o = Img(y2 - y1, x2 - x1, a.ch)
    for y in range(y2 - y1):
        for x in range(x2 - x1):
            for c in range(a.ch):
                o.data[(y * o.cols + x) * a.ch + c] = a.data[((y1 + y) * a.cols + x1 + x) * a.ch + c]
    return o^


def _paste(mut a: Img, o: Img, x1: Int, y1: Int):
    for y in range(o.rows):
        for x in range(o.cols):
            for c in range(a.ch):
                a.data[((y1 + y) * a.cols + x1 + x) * a.ch + c] = o.data[(y * o.cols + x) * a.ch + c]


def _label_box(p: Position, cx: Int, cy: Int, w: Int, h: Int) raises -> List[Int]:
    """resolve_text_background_xyxy: the label background for text anchored at `p`."""
    if p == Position.TOP_LEFT:
        return [cx, cy - h, cx + w, cy]
    if p == Position.TOP_RIGHT:
        return [cx - w, cy - h, cx, cy]
    if p == Position.TOP_CENTER:
        return [cx - w // 2, cy - h, cx + w // 2, cy]
    if p == Position.CENTER or p == Position.CENTER_OF_MASS:
        return [cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2]
    if p == Position.BOTTOM_LEFT:
        return [cx, cy, cx + w, cy + h]
    if p == Position.BOTTOM_RIGHT:
        return [cx - w, cy, cx, cy + h]
    if p == Position.BOTTOM_CENTER:
        return [cx - w // 2, cy, cx + w // 2, cy + h]
    if p == Position.CENTER_LEFT:
        return [cx - w, cy - h // 2, cx, cy + h // 2]
    if p == Position.CENTER_RIGHT:
        return [cx, cy - h // 2, cx + w, cy + h // 2]
    raise Error("Unsupported position")


def _lines(text: String) -> List[String]:
    """wrap_text(text, None): splitlines() or [""]."""
    var out = List[String]()
    for ln in text.replace("\r\n", "\n").split("\n"):
        out.append(String(ln))
    if len(out) > 0 and out[len(out) - 1].byte_length() == 0:
        _ = out.pop()  # splitlines() has no trailing empty line
    if len(out) == 0:
        out.append(String(""))
    return out^


def label_annotate(mut scene: Img, det: Detections, palette: ColorPalette, labels: List[String], mut engine: TextEngine, text_color: Color, text_scale: Float64 = 0.5, text_thickness: Int = 1, text_padding: Int = 10, position: Position = Position.TOP_LEFT, offset_x: Int = 0, offset_y: Int = 0, lookup: Int = LOOKUP_CLASS) raises:
    """sv.LabelAnnotator.annotate with a given label per detection: a filled box at the anchor, then each line with cv2 putText."""
    if len(labels) != len(det):
        raise Error("The number of labels provided (" + String(len(labels)) + ") does not match the number of detections (" + String(len(det)) + "). Each detection should have a corresponding label.")
    var xy = det.get_anchors_coordinates(position)
    for i in range(len(det)):
        var lines = _lines(labels[i])
        var heights = List[Int]()
        var max_w = 0
        var total = 0
        for line in lines:
            var sz = engine.text_size(line, text_scale, text_thickness)
            max_w = max(max_w, sz[0])
            heights.append(sz[1] if line.byte_length() > 0 else engine.text_size("Tg", text_scale, text_thickness)[1])
            total += heights[len(heights) - 1]
        total += (len(lines) - 1) * text_padding
        var b = _label_box(position, Int(xy[2 * i]) + offset_x, Int(xy[2 * i + 1]) + offset_y, max_w + 2 * text_padding, total + 2 * text_padding)
        draw_rect(scene, b[0], b[1], b[2], b[3], _bgr(resolve_color(palette, det, i, lookup)), -1)
        var y = b[1] + text_padding
        for k in range(len(lines)):
            if lines[k].byte_length() > 0:
                engine.put_text(scene, lines[k], b[0] + text_padding, y + heights[k], text_scale, _bgr(text_color), text_thickness)
            y += heights[k] + text_padding
