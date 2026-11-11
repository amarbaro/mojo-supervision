# Minimal KeyPoints (xy only): enough for VertexAnnotator. The rest of key_points/core.py (confidence, class_id, data,
# visible, slicing, from_* converters, skeletons) is not ported yet.
struct KeyPoints(Copyable, Movable):
    var xy: List[Float64]  # flat (n, m, 2)
    var n: Int
    var m: Int

    def __init__(out self, var xy: List[Float64], n: Int, m: Int):
        self.xy = xy^
        self.n = n
        self.m = m

    def __init__(out self, *, copy: Self):
        self.xy = copy.xy.copy()
        self.n = copy.n
        self.m = copy.m

    def __len__(self) -> Int:
        return self.n
