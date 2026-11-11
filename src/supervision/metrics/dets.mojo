"""Plain-array detections of one image, the input of every metric."""


struct Dets(Copyable, Movable):
    """Detections of one image. An empty optional list means the field is absent."""

    var xyxy: List[Float64]  # 4 per detection
    var class_id: List[Int]
    var conf: List[Float64]
    var area: List[Float64]  # the `area` data field, already rounded to float32
    var iscrowd: List[Int]
    var ignore: List[Int]

    def __init__(out self):
        self.xyxy = List[Float64]()
        self.class_id = List[Int]()
        self.conf = List[Float64]()
        self.area = List[Float64]()
        self.iscrowd = List[Int]()
        self.ignore = List[Int]()

    def n(self) -> Int:
        return len(self.xyxy) // 4
