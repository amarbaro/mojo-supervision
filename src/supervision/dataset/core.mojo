"""DetectionDataset (dataset/core.py): classes, image paths, per-image Detections.

Images stay on disk (the lazy form of the Python class); the dict-of-arrays form is not ported.
"""
from std.collections import Dict
from std.time import perf_counter_ns
from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.dataset.fsutil import copy_file, join, mkdirs, name_of
from supervision.dataset.pyrandom import PyRandom
from supervision.dataset.utils import (
    build_class_index_mapping, check_no_basename_collisions, find_duplicates, merge_class_lists,
    py_repr, train_test_split_indices,
)
from supervision.detection.core import DataCol, Detections


def map_detections_class_id(mapping: Dict[Int, Int], det: Detections) raises -> Detections:
    """utils.map_detections_class_id: source class ids to target ids; every id must be mapped."""
    if not det.has_class_id:
        raise Error("Detections must have class_id attribute.")
    for c in det.class_id:
        if c not in mapping:
            raise Error("Detections class_id must be a subset of source_to_target_mapping keys.")
    var out = det.copy()
    for i in range(len(out.class_id)):
        out.class_id[i] = mapping[det.class_id[i]]
    return out^


struct DetectionDataset(Copyable, Movable, Sized):
    var classes: List[String]
    var image_paths: List[String]
    var annotations: List[Detections]  # parallel to image_paths

    def __init__(out self, classes: List[String], images: List[String], annotations: List[Detections]) raises:
        if len(images) != len(annotations):
            raise Error("The keys of the images and annotations dictionaries must match.")
        self.classes = classes.copy()
        self.image_paths = List[String]()
        self.annotations = List[Detections]()
        var seen = Dict[String, Int]()
        for i in range(len(images)):
            if images[i] in seen:
                self.annotations[seen[images[i]]] = annotations[i].copy()  # dict: the later one wins
                continue
            seen[images[i]] = len(self.image_paths)
            self.image_paths.append(images[i])
            self.annotations.append(annotations[i].copy())
        for k in range(len(self.annotations)):
            ref a = self.annotations[k]
            if not a.has_class_id:
                continue
            for cid in a.class_id:
                if cid < 0 or cid >= len(self.classes):
                    var rng = String("empty") if len(self.classes) == 0 else "[0, " + String(len(self.classes) - 1) + "]"
                    raise Error(
                        "Detection annotation for image " + py_repr(self.image_paths[k]) + " contains class_id "
                        + String(cid) + ", outside the valid range " + rng + " for " + String(len(self.classes))
                        + " classes."
                    )
            var names = List[String]()
            for cid in a.class_id:
                names.append(self.classes[cid])
            a.set_data(DataCol(CLASS_NAME_DATA_FIELD, names))

    def __len__(self) -> Int:
        return len(self.image_paths)

    def split(self, split_ratio: Float64 = 0.8, random_state: Int = -1, shuffle: Bool = True) raises -> Tuple[DetectionDataset, DetectionDataset]:
        """Train and test parts; `random_state < 0` means unseeded (a clock seed)."""
        var seed = UInt64(random_state) if random_state >= 0 else UInt64(perf_counter_ns())
        var t = train_test_split_indices(len(self), split_ratio, seed, shuffle)
        var tp = List[String]()
        var ta = List[Detections]()
        var vp = List[String]()
        var va = List[Detections]()
        for i in t[0]:
            tp.append(self.image_paths[i])
            ta.append(self.annotations[i].copy())
        for i in t[1]:
            vp.append(self.image_paths[i])
            va.append(self.annotations[i].copy())
        return (DetectionDataset(self.classes, tp, ta), DetectionDataset(self.classes, vp, va))

    @staticmethod
    def merge(datasets: List[DetectionDataset]) raises -> DetectionDataset:
        var all_paths = List[String]()
        for d in datasets:
            for p in d.image_paths:
                all_paths.append(p)
        var dups = find_duplicates(all_paths)
        if len(dups) > 0:
            var s = String("[")
            for i in range(len(dups)):
                s += (", " if i > 0 else "") + py_repr(dups[i])
            raise Error("Image paths " + s + "] are not unique across datasets.")
        var class_lists = List[List[String]]()
        for d in datasets:
            class_lists.append(d.classes.copy())
        var classes = merge_class_lists(class_lists)
        var paths = List[String]()
        var anns = List[Detections]()
        for d in datasets:
            var idx = build_class_index_mapping(d.classes, classes)
            var mapping = Dict[Int, Int]()
            for i in range(len(idx)):
                mapping[i] = idx[i]
            for i in range(len(d.image_paths)):
                paths.append(d.image_paths[i])
                anns.append(map_detections_class_id(mapping, d.annotations[i]))
        return DetectionDataset(classes, paths, anns)


def save_dataset_images(ds: DetectionDataset, images_directory_path: String) raises:
    """utils.save_dataset_images for on-disk images: copy each to <dir>/<basename>."""
    var names = List[String]()
    for p in ds.image_paths:
        names.append(name_of(p))
    check_no_basename_collisions(ds.image_paths, names, "image")
    mkdirs(images_directory_path)
    for i in range(len(ds.image_paths)):
        copy_file(ds.image_paths[i], join(images_directory_path, names[i]))
