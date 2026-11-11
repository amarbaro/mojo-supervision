"""Box / mask / RLE converters (port of supervision/detection/utils/converters.py).

Boxes are flat `List[Float64]` (4 per box), masks flat `List[UInt8]` (n x h x w, row major).
Functions that need cv2 drawing (polygon_to_mask, mask_to_polygons) wait for the mojo-cv ops and live in
detection/utils/polygons.mojo.
"""

comptime MIN_POLYGON_POINT_COUNT = 3


def xywh_to_xyxy(xywh: List[Float64]) -> List[Float64]:
    var out = xywh.copy()
    for i in range(len(out) // 4):
        out[4 * i + 2] = xywh[4 * i] + xywh[4 * i + 2]
        out[4 * i + 3] = xywh[4 * i + 1] + xywh[4 * i + 3]
    return out^


def xyxy_to_xywh(xyxy: List[Float64]) -> List[Float64]:
    var out = xyxy.copy()
    for i in range(len(out) // 4):
        out[4 * i + 2] = xyxy[4 * i + 2] - xyxy[4 * i]
        out[4 * i + 3] = xyxy[4 * i + 3] - xyxy[4 * i + 1]
    return out^


def xcycwh_to_xyxy(xcycwh: List[Float64]) -> List[Float64]:
    var out = List[Float64]()
    for i in range(len(xcycwh) // 4):
        var hw = xcycwh[4 * i + 2] / 2
        var hh = xcycwh[4 * i + 3] / 2
        out.append(xcycwh[4 * i] - hw)
        out.append(xcycwh[4 * i + 1] - hh)
        out.append(xcycwh[4 * i] + hw)
        out.append(xcycwh[4 * i + 1] + hh)
    return out^


def xyxy_to_xcycarh(xyxy: List[Float64]) -> List[Float64]:
    """(center x, center y, aspect ratio w/h (0 when h is 0), height)."""
    var out = List[Float64]()
    for i in range(len(xyxy) // 4):
        var w = xyxy[4 * i + 2] - xyxy[4 * i]
        var h = xyxy[4 * i + 3] - xyxy[4 * i + 1]
        out.append(xyxy[4 * i] + w / 2)
        out.append(xyxy[4 * i + 1] + h / 2)
        out.append(w / h if h != 0.0 else 0.0)
        out.append(h)
    return out^


def xyxy_to_polygons(xyxy: List[Float64]) -> List[Float64]:
    """Four corners per box, flat (8 numbers): top-left, top-right, bottom-right, bottom-left."""
    var out = List[Float64]()
    for i in range(len(xyxy) // 4):
        var x1 = xyxy[4 * i]
        var y1 = xyxy[4 * i + 1]
        var x2 = xyxy[4 * i + 2]
        var y2 = xyxy[4 * i + 3]
        out.extend([x1, y1, x2, y1, x2, y2, x1, y2])
    return out^


def polygon_to_xyxy(polygon: List[Float64]) -> List[Float64]:
    """Bounding box of (x, y) pairs; four zeros for an empty polygon."""
    if len(polygon) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    var x0 = polygon[0]
    var y0 = polygon[1]
    var x1 = x0
    var y1 = y0
    for i in range(1, len(polygon) // 2):
        x0 = min(x0, polygon[2 * i])
        x1 = max(x1, polygon[2 * i])
        y0 = min(y0, polygon[2 * i + 1])
        y1 = max(y1, polygon[2 * i + 1])
    return [x0, y0, x1, y1]


def mask_to_xyxy(masks: List[UInt8], n: Int, height: Int, width: Int, exclusive: Bool = False) -> List[Int]:
    """Tight box of every mask; four zeros for an empty mask. `exclusive` adds one to x2 and y2."""
    var out = List[Int]()
    var px = height * width
    for i in range(n):
        var x0 = width
        var y0 = height
        var x1 = -1
        var y1 = -1
        for y in range(height):
            for x in range(width):
                if masks[i * px + y * width + x] != 0:
                    x0 = min(x0, x)
                    x1 = max(x1, x)
                    y0 = min(y0, y)
                    y1 = max(y1, y)
        if x1 < 0:
            out.extend([0, 0, 0, 0])
        else:
            var add = 1 if exclusive else 0
            out.extend([x0, y0, x1 + add, y1 + add])
    return out^


def xyxy_to_mask(boxes: List[Float64], width: Int, height: Int, exclusive: Bool = False) -> List[UInt8]:
    """One mask per box: pixels inside the box (inclusive corners, or half open when `exclusive`)."""
    var out = List[UInt8]()
    for i in range(len(boxes) // 4):
        var x_min = max(0, Int(boxes[4 * i]))
        var y_min = max(0, Int(boxes[4 * i + 1]))
        var xm = boxes[4 * i + 2]
        var ym = boxes[4 * i + 3]
        var x_lo = x_min
        var x_hi = -1
        var y_lo = y_min
        var y_hi = -1
        if exclusive:
            x_hi = min(width, Int(xm)) - 1
            y_hi = min(height, Int(ym)) - 1
        elif xm >= 0 and ym >= 0:
            x_hi = min(width - 1, Int(xm))
            y_hi = min(height - 1, Int(ym))
        for y in range(height):
            for x in range(width):
                out.append(UInt8(1) if (x >= x_lo and x <= x_hi and y >= y_lo and y <= y_hi) else UInt8(0))
    return out^


# ---- COCO RLE ---------------------------------------------------------------------------------------------


def mask_to_rle(mask: List[UInt8], height: Int, width: Int) raises -> List[Int]:
    """Uncompressed COCO run lengths (column major, starting with a run of zeros)."""
    if height * width == 0:
        raise Error("Input mask cannot be empty")
    var counts = List[Int]()
    var cur = False
    var run = 0
    for x in range(width):
        for y in range(height):
            var v = mask[y * width + x] != 0
            if v == cur:
                run += 1
            else:
                counts.append(run)
                cur = v
                run = 1
    counts.append(run)
    return counts^


def rle_to_mask(counts: List[Int], width: Int, height: Int) raises -> List[UInt8]:
    var total = 0
    for c in counts:
        total += c
    if total != width * height:
        raise Error("the sum of the number of pixels in the RLE must be the same as the number of pixels in the expected mask")
    var out = List[UInt8]()
    for _ in range(width * height):
        out.append(0)
    var pos = 0
    var val: UInt8 = 0
    for c in counts:
        for _ in range(c):
            var x = pos // height
            var y = pos % height
            out[y * width + x] = val
            pos += 1
        val = 1 - val
    return out^


def base48_decode(s: String) raises -> List[Int]:
    var values = List[Int]()
    var b = s.as_bytes()
    var i = 0
    while i < len(b):
        var x = 0
        var k = 0
        var more = True
        while more:
            if i >= len(b):
                raise Error("Malformed compressed RLE string: unexpected end at position " + String(i))
            var c = Int(b[i]) - 48
            x |= (c & 31) << (5 * k)
            more = (c & 32) != 0
            i += 1
            k += 1
            if not more and (c & 16) != 0:
                x |= (-1) << (5 * k)
        values.append(x)
    return values^


def base48_encode(values: List[Int]) -> String:
    var out = List[UInt8]()
    for v in values:
        var x = v
        var more = True
        while more:
            var c = x & 31
            x >>= 5
            more = x != -1 if (c & 16) != 0 else x != 0
            if more:
                c |= 32
            out.append(UInt8(c + 48))
    return String(unsafe_from_utf8=out)


def delta_decode(values: List[Int]) -> List[Int]:
    var c = values.copy()
    for i in range(3, len(c)):
        c[i] = values[i] + c[i - 2]
    return c^


def delta_encode(counts: List[Int]) -> List[Int]:
    var d = counts.copy()
    for i in range(3, len(d)):
        d[i] = counts[i] - counts[i - 2]
    return d^


def mask_to_rle_string(mask: List[UInt8], height: Int, width: Int) raises -> String:
    """mask_to_rle(compressed=True): the pycocotools string form."""
    return base48_encode(delta_encode(mask_to_rle(mask, height, width)))


def rle_string_to_mask(rle: String, width: Int, height: Int) raises -> List[UInt8]:
    return rle_to_mask(delta_decode(base48_decode(rle)), width, height)
