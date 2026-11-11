"""Shape validators (port of supervision/validators/__init__.py). Arrays are flat lists plus explicit sizes,
so each check takes the sizes NumPy would read from `.shape`. Detections.validate() uses the same messages."""

from std.math import isfinite


def validate_xyxy(xyxy: List[Float64]) raises:
    if len(xyxy) % 4 != 0:
        raise Error("xyxy must be a 2D np.ndarray with shape (_, 4), but got shape (" + String(len(xyxy)) + ",)")
    for v in xyxy:
        if not isfinite(v):
            raise Error("xyxy must contain only finite numeric values (no NaN or inf).")


def validate_mask(mask_len: Int, h: Int, w: Int, n: Int) raises:
    if h * w * n != mask_len:
        raise Error("mask must be a 3D np.ndarray with shape (" + String(n) + ", H, W), but got shape (" + String(mask_len // max(h * w, 1)) + ", " + String(h) + ", " + String(w) + ")")


def validate_class_id(length: Int, n: Int) raises:
    if length != n:
        raise Error("class_id must be a 1D np.ndarray with shape (" + String(n) + ",), but got shape (" + String(length) + ",)")


def validate_confidence(length: Int, n: Int) raises:
    if length != n:
        raise Error("confidence must be a 1D np.ndarray with shape (" + String(n) + ",), but got shape (" + String(length) + ",)")


def validate_tracker_id(length: Int, n: Int) raises:
    if length != n:
        raise Error("tracker_id must be a 1D np.ndarray with shape (" + String(n) + ",), but got shape (" + String(length) + ",)")


def validate_keypoint_confidence(rows: Int, cols: Int, n: Int, m: Int) raises:
    if rows != n:
        raise Error("keypoint_confidence first dimension must be " + String(n) + ", but got shape (" + String(rows) + ", " + String(cols) + ")")
    if n > 0 and cols != m:
        raise Error("keypoint_confidence second dimension must be " + String(m) + ", but got shape (" + String(rows) + ", " + String(cols) + ")")


def validate_visible(rows: Int, cols: Int, n: Int, m: Int) raises:
    if rows != n:
        raise Error("visible first dimension must be " + String(n) + ", but got shape (" + String(rows) + ", " + String(cols) + ")")
    if n > 0 and cols != m:
        raise Error("visible second dimension must be " + String(m) + ", but got shape (" + String(rows) + ", " + String(cols) + ")")


def validate_xy(depth: Int) raises:
    """`depth` is the last dimension of the (n, m, d) array."""
    if depth != 2 and depth != 3:
        raise Error("xy must be a 3D np.ndarray with shape (n, m, 2) or (n, m, 3)")


def validate_resolution(width: Int, height: Int) raises -> Tuple[Int, Int]:
    if width <= 0 or height <= 0:
        raise Error("Both dimensions in resolution must be positive. Got (" + String(width) + ", " + String(height) + ").")
    return (width, height)
