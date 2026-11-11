"""Row binary for tests/cases/dataset_voc_seg.txt (build/dataset_voc_seg): the same driver, segmentation fixtures."""
from supervision.dataset.rowds import run


def main() raises:
    run("voc_seg")
