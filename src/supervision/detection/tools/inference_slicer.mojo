# Port of the image-free half of supervision InferenceSlicer (MIT, see NOTICE-supervision-LICENSE.md):
# move each slice's detections to image coordinates, merge them, apply the overlap filter.
# Offsets: slicer_offsets.mojo. NOT ported: cropping the image (mojo-cv), calling the model, thread_workers, batch_size,
# compact masks, raster datasets, the out-of-bounds warning; masks raise until S1's move_masks lands.
from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.core import Detections, DataCol
from supervision.detection.utils.boxes import move_boxes, move_oriented_boxes
from supervision.detection.utils.iou_and_nms import OVERLAP_IOU

comptime FILTER_NONE = 0
comptime FILTER_NMS = 1
comptime FILTER_NMM = 2


def move_detections(detections: Detections, dx: Int, dy: Int) raises -> Detections:
    var d = detections.copy()
    if d.has_mask:
        raise Error("move_detections: masks are not supported yet (S1 move_masks)")
    d.xyxy = move_boxes(d.xyxy, Float64(dx), Float64(dy))
    if d.has_data(ORIENTED_BOX_COORDINATES):
        d.set_data(DataCol(ORIENTED_BOX_COORDINATES,
                           move_oriented_boxes(d.get_data(ORIENTED_BOX_COORDINATES).f, Float64(dx), Float64(dy)), 8))
    return d^


def merge_and_filter(
    per_slice: List[Detections], offsets: List[List[Int]], overlap_filter: Int, iou_threshold: Float64, overlap_metric: Int = OVERLAP_IOU
) raises -> Detections:
    """per_slice[i] holds the callback's detections in slice-local coordinates for offsets[i]."""
    if len(per_slice) != len(offsets):
        raise Error("Callback returned " + String(len(per_slice)) + " Detections for " + String(len(offsets)) + " slices.")
    var moved = List[Detections]()
    for i in range(len(per_slice)):
        moved.append(move_detections(per_slice[i], offsets[i][0], offsets[i][1]))
    var merged = Detections.merge(moved)
    if overlap_filter == FILTER_NMS:
        return merged.with_nms(iou_threshold, False, overlap_metric)
    if overlap_filter == FILTER_NMM:
        return merged.with_nmm(iou_threshold, False, overlap_metric)
    return merged^
