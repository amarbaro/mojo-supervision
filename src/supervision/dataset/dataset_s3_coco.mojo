"""Row binary for tests/cases/dataset_s3_coco.txt (build/dataset_s3_coco)."""
from supervision.dataset.rowds import run


def main() raises:
    run("s3coco")
