"""Row binary for tests/cases/dataset_coco_seg.txt (build/dataset_coco_seg): the same driver, segmentation fixtures."""
from supervision.dataset.rowds import run


def main() raises:
    run("coco_seg")
