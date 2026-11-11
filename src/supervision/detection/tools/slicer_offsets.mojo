# Port of supervision InferenceSlicer._generate_offset / _validate_overlap (pure Int math).


def _axis_starts(image_size: Int, slice_size: Int, stride: Int) -> List[Int]:
    var starts = List[Int]()
    if image_size <= slice_size:
        starts.append(0)
        return starts^
    if stride == slice_size:
        var s = 0
        while s < image_size:
            starts.append(s)
            s += stride
        return starts^
    var last = image_size - slice_size
    var s = 0
    while s < last:
        starts.append(s)
        s += stride
    if len(starts) == 0 or starts[len(starts) - 1] != last:
        starts.append(last)
    return starts^


def validate_overlap(slice_w: Int, slice_h: Int, overlap_w: Int, overlap_h: Int) raises:
    if overlap_w < 0 or overlap_h < 0:
        raise Error("Overlap values must be greater than or equal to 0.")
    if overlap_w >= slice_w or overlap_h >= slice_h:
        raise Error("`overlap_wh` must be smaller than `slice_wh` in both dimensions to keep a positive stride.")


def generate_offsets(
    image_w: Int, image_h: Int, slice_w: Int, slice_h: Int, overlap_w: Int, overlap_h: Int
) raises -> List[List[Int]]:
    """Rows of [x_min, y_min, x_max, y_max], row-major over y then x (numpy meshgrid order)."""
    if slice_w <= 0 or slice_h <= 0:
        raise Error("`slice_wh` must be positive.")
    validate_overlap(slice_w, slice_h, overlap_w, overlap_h)
    var xs = _axis_starts(image_w, slice_w, slice_w - overlap_w)
    var ys = _axis_starts(image_h, slice_h, slice_h - overlap_h)
    var out = List[List[Int]]()
    for y in ys:
        for x in xs:
            out.append([x, y, min(x + slice_w, image_w), min(y + slice_h, image_h)])
    return out^
