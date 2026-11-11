from supervision.detection.utils.iou_and_nms import box_iou, box_non_max_suppression


def main() raises:
    # two overlapping boxes and one apart, xyxy flat
    var xyxy: List[Float64] = [0, 0, 10, 10, 1, 1, 11, 11, 50, 50, 60, 60]
    var scores: List[Float64] = [0.9, 0.8, 0.7]
    var cats: List[Int] = [0, 0, 0]
    print(box_iou([0.0, 0, 10, 10], [1.0, 1, 11, 11]))
    var keep = box_non_max_suppression(xyxy, scores, cats, 0.5)
    for k in keep:
        print(k)
