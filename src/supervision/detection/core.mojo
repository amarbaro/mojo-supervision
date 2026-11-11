"""S0: the Detections struct (port of supervision/detection/core.py).

Storage is flat and typed, not NumPy:

    xyxy        List[Float64]   4 per detection
    mask        List[UInt8]     n * mask_h * mask_w, row major (dense masks only: no CompactMask)
    confidence  List[Float64]   present when has_confidence
    class_id    List[Int]       present when has_class_id
    tracker_id  List[Int]       present when has_tracker_id
    data        List[DataCol]   named columns: numeric (width numbers per detection) or string
    metadata    parallel key / value String lists (collection level)

Python's "None" is the matching `has_*` flag. `validate()` is Python's `__post_init__`: build the
struct, set the fields, call `validate()` (every method here that builds a result validates it).
"""

from std.math import isfinite, nan
from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.geometry.core import Position
from supervision.detection.utils.masks import calculate_masks_centroids
from supervision.detection.utils.boxes import (
    box_midpoint,
    obb_polygon_area,
    oriented_box_anchors,
    xyxyxyxy_to_xyxy,
)
from supervision.detection.utils.iou_and_nms import (
    OVERLAP_IOU,
    box_iou_batch,
    box_non_max_merge,
    box_non_max_suppression,
    box_soft_non_max_suppression,
    mask_iou_batch,
    mask_non_max_merge,
    mask_non_max_suppression,
    mask_soft_non_max_suppression,
    oriented_box_iou_batch,
    oriented_box_non_max_merge,
    oriented_box_non_max_suppression,
)


struct DataCol(Copyable, Movable):
    """One entry of Detections.data: `width` numbers per detection, or one string per detection."""

    var name: String
    var is_str: Bool
    var width: Int
    var f: List[Float64]
    var s: List[String]

    def __init__(out self, name: String, f: List[Float64], width: Int = 1):
        self.name = name
        self.is_str = False
        self.width = width
        self.f = f.copy()
        self.s = List[String]()

    def __init__(out self, name: String, s: List[String]):
        self.name = name
        self.is_str = True
        self.width = 1
        self.f = List[Float64]()
        self.s = s.copy()

    def rows(self) -> Int:
        return len(self.s) if self.is_str else (len(self.f) // self.width if self.width > 0 else 0)

    def pick(self, index: List[Int]) -> DataCol:
        if self.is_str:
            var s = List[String]()
            for i in index:
                s.append(self.s[i])
            return DataCol(self.name, s)
        var f = List[Float64]()
        for i in index:
            for k in range(self.width):
                f.append(self.f[i * self.width + k])
        return DataCol(self.name, f, self.width)

    def equals(self, other: DataCol) -> Bool:
        if self.is_str != other.is_str or self.width != other.width:
            return False
        if self.is_str:
            return self.s == other.s
        if len(self.f) != len(other.f):
            return False
        for i in range(len(self.f)):
            if self.f[i] != other.f[i]:
                return False
        return True


def _round32(x: Float64) -> Float64:
    return Float64(Float32(x))


struct Detections(Copyable, Movable, Sized):
    var xyxy: List[Float64]
    var has_mask: Bool
    var mask: List[UInt8]
    var mask_h: Int
    var mask_w: Int
    var has_confidence: Bool
    var confidence: List[Float64]
    var has_class_id: Bool
    var class_id: List[Int]
    var has_tracker_id: Bool
    var tracker_id: List[Int]
    var data: List[DataCol]
    var meta_keys: List[String]
    var meta_vals: List[String]

    def __init__(out self, var xyxy: List[Float64]):
        self.xyxy = xyxy^
        self.has_mask = False
        self.mask = List[UInt8]()
        self.mask_h = 0
        self.mask_w = 0
        self.has_confidence = False
        self.confidence = List[Float64]()
        self.has_class_id = False
        self.class_id = List[Int]()
        self.has_tracker_id = False
        self.tracker_id = List[Int]()
        self.data = List[DataCol]()
        self.meta_keys = List[String]()
        self.meta_vals = List[String]()

    @staticmethod
    def empty() -> Detections:
        var d = Detections(List[Float64]())
        d.has_confidence = True
        d.has_class_id = True
        return d^

    def __len__(self) -> Int:
        return len(self.xyxy) // 4

    def is_empty(self) -> Bool:
        return len(self.xyxy) == 0

    def set_mask(mut self, var mask: List[UInt8], h: Int, w: Int):
        self.mask = mask^
        self.mask_h = h
        self.mask_w = w
        self.has_mask = True

    def set_confidence(mut self, var v: List[Float64]):
        self.confidence = v^
        self.has_confidence = True

    def set_class_id(mut self, var v: List[Int]):
        self.class_id = v^
        self.has_class_id = True

    def set_tracker_id(mut self, var v: List[Int]):
        self.tracker_id = v^
        self.has_tracker_id = True

    def set_metadata(mut self, key: String, value: String):
        for i in range(len(self.meta_keys)):
            if self.meta_keys[i] == key:
                self.meta_vals[i] = value
                return
        self.meta_keys.append(key)
        self.meta_vals.append(value)

    # ---- validation (supervision/validators) -------------------------------------------------------------

    def validate(self) raises:
        if len(self.xyxy) % 4 != 0:
            raise Error("xyxy must be a 2D np.ndarray with shape (_, 4), but got shape (" + String(len(self.xyxy)) + ",)")
        for v in self.xyxy:
            if not isfinite(v):
                raise Error("xyxy must contain only finite numeric values (no NaN or inf).")
        var n = len(self)
        if self.has_mask:
            if self.mask_h * self.mask_w * n != len(self.mask):
                raise Error("mask must be a 3D np.ndarray with shape (" + String(n) + ", H, W), but got shape (" + String(len(self.mask) // max(self.mask_h * self.mask_w, 1)) + ", " + String(self.mask_h) + ", " + String(self.mask_w) + ")")
        if self.has_class_id and len(self.class_id) != n:
            raise Error("class_id must be a 1D np.ndarray with shape (" + String(n) + ",), but got shape (" + String(len(self.class_id)) + ",)")
        if self.has_confidence and len(self.confidence) != n:
            raise Error("confidence must be a 1D np.ndarray with shape (" + String(n) + ",), but got shape (" + String(len(self.confidence)) + ",)")
        if self.has_tracker_id and len(self.tracker_id) != n:
            raise Error("tracker_id must be a 1D np.ndarray with shape (" + String(n) + ",), but got shape (" + String(len(self.tracker_id)) + ",)")
        for c in self.data:
            if c.rows() != n:
                if c.is_str:
                    raise Error("Length of list for key '" + c.name + "' must be " + String(n))
                if c.width == 1:
                    raise Error("Shape of np.ndarray for key '" + c.name + "' must be (" + String(n) + ",)")
                raise Error("First dimension of np.ndarray for key '" + c.name + "' must have size " + String(n))

    # ---- data columns -------------------------------------------------------------------------------------

    def has_data(self, key: String) -> Bool:
        for c in self.data:
            if c.name == key:
                return True
        return False

    def data_index(self, key: String) raises -> Int:
        for i in range(len(self.data)):
            if self.data[i].name == key:
                return i
        raise Error("no data key '" + key + "'")

    def get_data(self, key: String) raises -> DataCol:
        return self.data[self.data_index(key)].copy()

    def set_data(mut self, var col: DataCol) raises:
        if col.rows() != len(self):
            raise Error("Length of list for key '" + col.name + "' must be " + String(len(self)))
        for i in range(len(self.data)):
            if self.data[i].name == col.name:
                self.data[i] = col^
                return
        self.data.append(col^)

    # ---- indexing -----------------------------------------------------------------------------------------

    def select(self, index: List[Int]) raises -> Detections:
        """Rows `index` (negative counts from the end, repeats allowed) as a new Detections."""
        var n = len(self)
        var idx = List[Int]()
        if n == 0:
            for i in range(0):
                idx.append(i)
        else:
            for i in index:
                var j = i + n if i < 0 else i
                if j < 0 or j >= n:
                    raise Error("index " + String(i) + " is out of bounds for axis 0 with size " + String(n))
                idx.append(j)
        var out = Detections(List[Float64]())
        for j in idx:
            for k in range(4):
                out.xyxy.append(self.xyxy[4 * j + k])
        if self.has_mask:
            var px = self.mask_h * self.mask_w
            var m = List[UInt8]()
            for j in idx:
                for p in range(px):
                    m.append(self.mask[j * px + p])
            out.set_mask(m^, self.mask_h, self.mask_w)
        if self.has_confidence:
            var v = List[Float64]()
            for j in idx:
                v.append(self.confidence[j])
            out.set_confidence(v^)
        if self.has_class_id:
            var v = List[Int]()
            for j in idx:
                v.append(self.class_id[j])
            out.set_class_id(v^)
        if self.has_tracker_id:
            var v = List[Int]()
            for j in idx:
                v.append(self.tracker_id[j])
            out.set_tracker_id(v^)
        for c in self.data:
            out.data.append(c.pick(idx))
        out.meta_keys = self.meta_keys.copy()
        out.meta_vals = self.meta_vals.copy()
        out.validate()
        return out^

    def select_where(self, keep: List[Bool]) raises -> Detections:
        """Boolean-mask indexing: `detections[keep]`."""
        if len(keep) != len(self):
            raise Error("boolean index did not match: size " + String(len(keep)) + " vs " + String(len(self)))
        var idx = List[Int]()
        for i in range(len(keep)):
            if keep[i]:
                idx.append(i)
        return self.select(idx)

    def slice(self, start: Int, stop: Int) raises -> Detections:
        """`detections[start:stop]` with Python clamping."""
        var n = len(self)
        var a = start + n if start < 0 else start
        var b = stop + n if stop < 0 else stop
        a = max(0, min(a, n))
        b = max(0, min(b, n))
        var idx = List[Int]()
        for i in range(a, b):
            idx.append(i)
        return self.select(idx)

    # ---- equality -----------------------------------------------------------------------------------------

    def equals(self, other: Detections) -> Bool:
        if len(self.xyxy) != len(other.xyxy):
            return False
        for i in range(len(self.xyxy)):
            if self.xyxy[i] != other.xyxy[i]:
                return False
        if self.has_mask != other.has_mask:
            return False
        if self.has_mask and (self.mask_h != other.mask_h or self.mask_w != other.mask_w or self.mask != other.mask):
            return False
        if self.has_class_id != other.has_class_id or self.class_id != other.class_id:
            return False
        if self.has_tracker_id != other.has_tracker_id or self.tracker_id != other.tracker_id:
            return False
        if self.has_confidence != other.has_confidence or len(self.confidence) != len(other.confidence):
            return False
        for i in range(len(self.confidence)):
            if self.confidence[i] != other.confidence[i]:
                return False
        if len(self.data) != len(other.data):
            return False
        for c in self.data:
            var found = False
            for o in other.data:
                if o.name == c.name:
                    found = c.equals(o)
            if not found:
                return False
        if len(self.meta_keys) != len(other.meta_keys):
            return False
        for i in range(len(self.meta_keys)):
            var found = False
            for j in range(len(other.meta_keys)):
                if other.meta_keys[j] == self.meta_keys[i]:
                    found = other.meta_vals[j] == self.meta_vals[i]
            if not found:
                return False
        return True

    # ---- areas --------------------------------------------------------------------------------------------

    def box_area(self) -> List[Float64]:
        var out = List[Float64]()
        for i in range(len(self)):
            out.append((self.xyxy[4 * i + 2] - self.xyxy[4 * i]) * (self.xyxy[4 * i + 3] - self.xyxy[4 * i + 1]))
        return out^

    def box_aspect_ratio(self) -> List[Float64]:
        """width / height, NaN where the height is 0."""
        var out = List[Float64]()
        for i in range(len(self)):
            var w = self.xyxy[4 * i + 2] - self.xyxy[4 * i]
            var h = self.xyxy[4 * i + 3] - self.xyxy[4 * i + 1]
            out.append(w / h if h != 0.0 else nan[DType.float64]())
        return out^

    def mask_pixel_counts(self) -> List[Float64]:
        var out = List[Float64]()
        var px = self.mask_h * self.mask_w
        for i in range(len(self)):
            var c = 0
            for p in range(px):
                c += Int(self.mask[i * px + p] != 0)
            out.append(Float64(c))
        return out^

    def area(self) raises -> List[Float64]:
        """Mask pixel count, else oriented-box polygon area, else box area."""
        if self.has_mask:
            return self.mask_pixel_counts()
        if self.has_data(ORIENTED_BOX_COORDINATES):
            return obb_polygon_area(self.get_data(ORIENTED_BOX_COORDINATES).f)
        return self.box_area()

    # ---- anchors ------------------------------------------------------------------------------------------

    def mask_centroids(self) -> List[Float64]:
        """(x, y) per mask, truncated to integers like `astype(int)`."""
        var out = List[Float64]()
        for v in calculate_masks_centroids(self.mask, len(self), self.mask_h, self.mask_w):
            out.append(Float64(v))
        return out^

    def get_anchors_coordinates(self, anchor: Position) raises -> List[Float64]:
        """(x, y) per detection, flat."""
        if self.has_data(ORIENTED_BOX_COORDINATES) and anchor != Position.CENTER_OF_MASS:
            return oriented_box_anchors(self.get_data(ORIENTED_BOX_COORDINATES).f, anchor)
        if anchor == Position.CENTER_OF_MASS:
            if not self.has_mask:
                raise Error("Cannot use `Position.CENTER_OF_MASS` without a detection mask.")
            return self.mask_centroids()
        var out = List[Float64]()
        for i in range(len(self)):
            var x1 = self.xyxy[4 * i]
            var y1 = self.xyxy[4 * i + 1]
            var x2 = self.xyxy[4 * i + 2]
            var y2 = self.xyxy[4 * i + 3]
            var x = x1
            var y = y1
            if anchor == Position.CENTER:
                x = box_midpoint(x1, x2)
                y = box_midpoint(y1, y2)
            elif anchor == Position.CENTER_LEFT:
                y = box_midpoint(y1, y2)
            elif anchor == Position.CENTER_RIGHT:
                x = x2
                y = box_midpoint(y1, y2)
            elif anchor == Position.BOTTOM_CENTER:
                x = box_midpoint(x1, x2)
                y = y2
            elif anchor == Position.BOTTOM_LEFT:
                y = y2
            elif anchor == Position.BOTTOM_RIGHT:
                x = x2
                y = y2
            elif anchor == Position.TOP_CENTER:
                x = box_midpoint(x1, x2)
            elif anchor == Position.TOP_LEFT:
                pass
            elif anchor == Position.TOP_RIGHT:
                x = x2
            else:
                raise Error(anchor.name() + " is not supported.")
            out.append(x)
            out.append(y)
        return out^

    # ---- merge --------------------------------------------------------------------------------------------

    @staticmethod
    def merge(detections_list: List[Detections]) raises -> Detections:
        """Concatenate the non-empty entries; a field must be present in all or none."""
        var parts = List[Int]()
        for i in range(len(detections_list)):
            if not detections_list[i].is_empty():
                parts.append(i)
        if len(parts) == 0:
            return Detections.empty()
        for i in parts:
            detections_list[i].validate()
        var first = parts[0]
        var out = Detections(List[Float64]())
        var mask_count = 0
        var conf_count = 0
        var cls_count = 0
        var trk_count = 0
        for i in parts:
            ref d = detections_list[i]
            out.xyxy.extend(d.xyxy.copy())
            mask_count += Int(d.has_mask)
            conf_count += Int(d.has_confidence)
            cls_count += Int(d.has_class_id)
            trk_count += Int(d.has_tracker_id)
        var np = len(parts)
        if mask_count != 0 and mask_count != np:
            raise Error("All or none of the 'mask' fields must be None")
        if conf_count != 0 and conf_count != np:
            raise Error("All or none of the 'confidence' fields must be None")
        if cls_count != 0 and cls_count != np:
            raise Error("All or none of the 'class_id' fields must be None")
        if trk_count != 0 and trk_count != np:
            raise Error("All or none of the 'tracker_id' fields must be None")
        if mask_count == np:
            var m = List[UInt8]()
            var h = detections_list[first].mask_h
            var w = detections_list[first].mask_w
            for i in parts:
                ref d = detections_list[i]
                if d.mask_h != h or d.mask_w != w:
                    raise Error("cannot concatenate masks of different shapes")
                m.extend(d.mask.copy())
            out.set_mask(m^, h, w)
        if conf_count == np:
            var v = List[Float64]()
            for i in parts:
                v.extend(detections_list[i].confidence.copy())
            out.set_confidence(v^)
        if cls_count == np:
            var v = List[Int]()
            for i in parts:
                v.extend(detections_list[i].class_id.copy())
            out.set_class_id(v^)
        if trk_count == np:
            var v = List[Int]()
            for i in parts:
                v.extend(detections_list[i].tracker_id.copy())
            out.set_tracker_id(v^)
        # data: identical key sets, same kind per key
        ref f0 = detections_list[first]
        for i in parts:
            ref d = detections_list[i]
            var same = len(d.data) == len(f0.data)
            if same:
                for c in d.data:
                    if not f0.has_data(c.name):
                        same = False
            if not same:
                raise Error("All data dictionaries must have the same keys to merge.")
        for c0 in f0.data:
            var f = List[Float64]()
            var s = List[String]()
            for i in parts:
                var c = detections_list[i].get_data(c0.name)
                if c.is_str != c0.is_str or c.width != c0.width:
                    raise Error("Inconsistent data types for key '" + c0.name + "'. Only np.ndarray and list types are allowed.")
                f.extend(c.f.copy())
                s.extend(c.s.copy())
            if c0.is_str:
                out.data.append(DataCol(c0.name, s))
            else:
                out.data.append(DataCol(c0.name, f, c0.width))
        # metadata: same keys, equal values
        for i in parts:
            ref d = detections_list[i]
            if len(d.meta_keys) != len(f0.meta_keys):
                raise Error("All metadata dictionaries must have the same keys to merge.")
            for j in range(len(d.meta_keys)):
                var found = False
                for k in range(len(f0.meta_keys)):
                    if f0.meta_keys[k] == d.meta_keys[j]:
                        found = True
                        if f0.meta_vals[k] != d.meta_vals[j]:
                            raise Error("Conflicting metadata for key: '" + d.meta_keys[j] + "'.")
                if not found:
                    raise Error("All metadata dictionaries must have the same keys to merge.")
        out.meta_keys = f0.meta_keys.copy()
        out.meta_vals = f0.meta_vals.copy()
        out.validate()
        return out^

    # ---- NMS / soft NMS / NMM ----------------------------------------------------------------------------

    def _cats(self, class_agnostic: Bool, operation: String) raises -> List[Int]:
        var cats = List[Int]()
        if class_agnostic:
            for _ in range(len(self)):
                cats.append(0)
            return cats^
        if not self.has_class_id:
            raise Error("Detections class_id must be given for " + operation + " to be executed. If you intended to perform class agnostic " + operation + " set class_agnostic=True.")
        return self.class_id.copy()

    def with_nms(self, threshold: Float64 = 0.5, class_agnostic: Bool = False, overlap_metric: Int = OVERLAP_IOU) raises -> Detections:
        if len(self) == 0:
            return self.copy()
        if not self.has_confidence:
            raise Error("Detections confidence must be given for NMS to be executed.")
        var cats = self._cats(class_agnostic, "NMS")
        var keep: List[Bool]
        if self.has_mask:
            keep = mask_non_max_suppression(self.mask, self.mask_h * self.mask_w, self.confidence, cats, threshold, overlap_metric)
        elif self.has_data(ORIENTED_BOX_COORDINATES):
            keep = oriented_box_non_max_suppression(self.get_data(ORIENTED_BOX_COORDINATES).f, self.confidence, cats, threshold, overlap_metric)
        else:
            keep = box_non_max_suppression(self.xyxy, self.confidence, cats, threshold, overlap_metric)
        return self.select_where(keep)

    def with_soft_nms(self, sigma: Float64 = 0.5, class_agnostic: Bool = False, score_threshold: Float64 = -1.0, use_score_threshold: Bool = False) raises -> Detections:
        if len(self) == 0:
            return self.copy()
        if not self.has_confidence:
            raise Error("Detections confidence must be given for Soft-NMS to be executed.")
        var cats = self._cats(class_agnostic, "Soft-NMS")
        var decayed: List[Float64]
        if self.has_mask:
            decayed = mask_soft_non_max_suppression(self.mask, self.mask_h * self.mask_w, self.confidence, cats, sigma)
        else:
            decayed = box_soft_non_max_suppression(self.xyxy, self.confidence, cats, sigma)
        var result = self.copy()
        result.confidence = decayed.copy()
        if not use_score_threshold:
            return result^
        var keep = List[Bool]()
        for v in decayed:
            keep.append(v > score_threshold)
        return result.select_where(keep)

    def with_nmm(self, threshold: Float64 = 0.5, class_agnostic: Bool = False, overlap_metric: Int = OVERLAP_IOU) raises -> Detections:
        if len(self) == 0:
            return self.copy()
        if not self.has_confidence:
            raise Error("Detections confidence must be given for NMM to be executed.")
        var cats = self._cats(class_agnostic, "NMM")
        var groups: List[List[Int]]
        if self.has_mask:
            groups = mask_non_max_merge(self.mask, self.mask_h * self.mask_w, self.confidence, cats, threshold, overlap_metric)
        elif self.has_data(ORIENTED_BOX_COORDINATES):
            groups = oriented_box_non_max_merge(self.get_data(ORIENTED_BOX_COORDINATES).f, self.confidence, cats, threshold, overlap_metric)
        else:
            groups = box_non_max_merge(self.xyxy, self.confidence, cats, threshold, overlap_metric)
        var merged = List[Detections]()
        for g in groups:
            var members = List[Detections]()
            for i in g:
                members.append(self.select([i]))
            merged.append(merge_detection_group(members))
        return Detections.merge(merged)


def _merge_obb_corners(corners_list: List[List[Float64]]) raises -> List[Float64]:
    """Oriented box that encloses every box, rotated to the winner's first edge (corners_list[0])."""
    from std.math import atan2, cos, sin

    var ox = corners_list[0][0]
    var oy = corners_list[0][1]
    var px = List[Float64]()
    var py = List[Float64]()
    for c in corners_list:
        for k in range(4):
            px.append(c[2 * k] - ox)
            py.append(c[2 * k + 1] - oy)
    var angle = atan2(py[1] - py[0], px[1] - px[0])
    var cs = cos(angle)
    var sn = sin(angle)
    var x_min = Float64.MAX_FINITE
    var x_max = -Float64.MAX_FINITE
    var y_min = Float64.MAX_FINITE
    var y_max = -Float64.MAX_FINITE
    for i in range(len(px)):
        var lx = px[i] * cs + py[i] * sn
        var ly = -px[i] * sn + py[i] * cs
        x_min = min(x_min, lx)
        x_max = max(x_max, lx)
        y_min = min(y_min, ly)
        y_max = max(y_max, ly)
    var local: List[Float64] = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
    var out = List[Float64]()
    for k in range(4):
        out.append(local[2 * k] * cs - local[2 * k + 1] * sn + ox)
        out.append(local[2 * k] * sn + local[2 * k + 1] * cs + oy)
    return out^


def merge_detection_group(detections: List[Detections]) raises -> Detections:
    """Merge a group of one-row Detections into one (NMM): area-weighted confidence, union box and mask,
    the highest-confidence member supplies class, tracker and data."""
    if len(detections) == 1:
        return detections[0].copy()
    var confs = List[Float64]()
    for d in detections:
        confs.append(d.confidence[0] if d.has_confidence else 0.0)
    var win = 0
    for i in range(1, len(confs)):
        if confs[i] > confs[win]:
            win = i
    var areas = List[Float64]()
    var total = 0.0
    for d in detections:
        var a = (d.xyxy[2] - d.xyxy[0]) * (d.xyxy[3] - d.xyxy[1])
        areas.append(a)
        total += a
    var out = Detections(List[Float64]())
    ref winner = detections[win]
    if winner.has_confidence:
        if total > 0.0:
            var dot = 0.0
            for i in range(len(areas)):
                dot += areas[i] * confs[i]
            out.set_confidence([_round32(dot / total)])
        else:
            out.set_confidence(winner.confidence.copy())
    if winner.has_data(ORIENTED_BOX_COORDINATES):
        var corners = List[List[Float64]]()
        for d in detections:
            var c = d.get_data(ORIENTED_BOX_COORDINATES)
            if c.width != 8 or len(c.f) != 8:
                raise Error("corners must have shape (N, 4, 2); got (" + String(c.rows()) + ", " + String(c.width) + ")")
            corners.append(c.f.copy())
        var merged = _merge_obb_corners(corners)
        out.xyxy = xyxyxyxy_to_xyxy(merged)
        for c in winner.data:
            if c.name == ORIENTED_BOX_COORDINATES:
                out.data.append(DataCol(c.name, merged, 8))
            else:
                out.data.append(c.copy())
    else:
        var x1 = detections[0].xyxy[0]
        var y1 = detections[0].xyxy[1]
        var x2 = detections[0].xyxy[2]
        var y2 = detections[0].xyxy[3]
        for d in detections:
            x1 = min(x1, d.xyxy[0])
            y1 = min(y1, d.xyxy[1])
            x2 = max(x2, d.xyxy[2])
            y2 = max(y2, d.xyxy[3])
        out.xyxy = [_round32(x1), _round32(y1), _round32(x2), _round32(y2)]
        for c in winner.data:
            out.data.append(c.copy())
    var any_mask = False
    for d in detections:
        any_mask = any_mask or d.has_mask
    if any_mask:
        var h = 0
        var w = 0
        for d in detections:
            if d.has_mask:
                h = d.mask_h
                w = d.mask_w
        var u = List[UInt8]()
        for _ in range(h * w):
            u.append(0)
        for d in detections:
            if d.has_mask:
                for p in range(h * w):
                    if d.mask[p] != 0:
                        u[p] = 1
        out.set_mask(u^, h, w)
    if winner.has_class_id:
        out.set_class_id(winner.class_id.copy())
    if winner.has_tracker_id:
        out.set_tracker_id(winner.tracker_id.copy())
    out.meta_keys = winner.meta_keys.copy()
    out.meta_vals = winner.meta_vals.copy()
    for d in detections:
        if len(d.meta_keys) != len(out.meta_keys):
            raise Error("All metadata dictionaries must have the same keys to merge.")
        for j in range(len(d.meta_keys)):
            var ok = False
            for k in range(len(out.meta_keys)):
                if out.meta_keys[k] == d.meta_keys[j]:
                    if out.meta_vals[k] != d.meta_vals[j]:
                        raise Error("Conflicting metadata for key: '" + d.meta_keys[j] + "'.")
                    ok = True
            if not ok:
                raise Error("All metadata dictionaries must have the same keys to merge.")
    out.validate()
    return out^
