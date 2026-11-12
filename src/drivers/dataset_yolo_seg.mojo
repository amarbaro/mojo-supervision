"""Row binary for tests/cases/dataset_yolo_seg.txt (build/dataset_yolo_seg): the same driver, segmentation fixtures."""
from supervision.dataset.rowds import run


def main() raises:
    run("yolo_seg")
