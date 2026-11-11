"""Detections (S0) to the plain arrays the metrics run on."""

from supervision.config import AREA_DATA_FIELD, ISCROWD_DATA_FIELD
from supervision.detection.core import Detections
from supervision.metrics.dets import Dets


def _data_ints(d: Detections, key: String) raises -> List[Int]:
    var out = List[Int]()
    if d.has_data(key):
        for v in d.get_data(key).f:
            out.append(Int(v))
    return out^


def to_dets(d: Detections) raises -> Dets:
    """Copy boxes, class ids, confidences and the `area`, `iscrowd`, `ignore` data fields.

    `area` goes through float32, as `np.asarray(data["area"], dtype=np.float32)` does.
    """
    d.validate()
    var out = Dets()
    out.xyxy = d.xyxy.copy()
    if d.has_class_id:
        out.class_id = d.class_id.copy()
    if d.has_confidence:
        out.conf = d.confidence.copy()
    if d.has_data(AREA_DATA_FIELD):
        for v in d.get_data(AREA_DATA_FIELD).f:
            out.area.append(Float64(Float32(v)))
    out.iscrowd = _data_ints(d, ISCROWD_DATA_FIELD)
    out.ignore = _data_ints(d, "ignore")
    return out^


def to_dets_list(ds: List[Detections]) raises -> List[Dets]:
    var out = List[Dets]()
    for d in ds:
        out.append(to_dets(d))
    return out^
