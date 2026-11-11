"""Classifications (port of supervision/classification/core.py; model adapters from_clip/from_timm/from_ultralytics
take framework tensors and are not ported)."""

from supervision.detection.utils.iou_and_nms import argsort


struct Classifications(Copyable, Movable, Sized):
    var class_id: List[Int]
    var has_confidence: Bool
    var confidence: List[Float64]

    def __init__(out self, var class_id: List[Int]):
        self.class_id = class_id^
        self.has_confidence = False
        self.confidence = List[Float64]()

    def set_confidence(mut self, var v: List[Float64]):
        self.confidence = v^
        self.has_confidence = True

    def __len__(self) -> Int:
        return len(self.class_id)

    def validate(self) raises:
        if self.has_confidence and len(self.confidence) != len(self.class_id):
            raise Error("confidence must be 1d np.ndarray with (n, ) shape")

    def equals(self, other: Classifications) -> Bool:
        return self.class_id == other.class_id and self.has_confidence == other.has_confidence and self.confidence == other.confidence

    def get_top_k(self, k: Int) raises -> Tuple[List[Int], List[Float64]]:
        """Top-k (class_id, confidence), highest first; ties keep numpy's reversed ascending order."""
        if not self.has_confidence:
            raise Error("top_k could not be calculated, confidence is None")
        if k < 0:
            raise Error("k must be non-negative")
        var asc = argsort(self.confidence)
        var ids = List[Int]()
        var conf = List[Float64]()
        for t in range(min(k, len(asc))):
            var i = asc[len(asc) - 1 - t]
            ids.append(self.class_id[i])
            conf.append(self.confidence[i])
        return (ids^, conf^)
