# Port of VertexAnnotator (supervision/key_points/annotators.py).
from std.math import isfinite
from mojocv.imgproc import _circle_fill
from mojocv.mat import Mat
from supervision.draw.color import Color
from supervision.key_points.core import KeyPoints


def vertex_annotate(mut scene: Mat[DType.uint8], kp: KeyPoints, color: Color, radius: Int = 4):
    var col: List[Float64] = [Float64(color.b), Float64(color.g), Float64(color.r)]
    for d in range(kp.n):
        for p in range(kp.m):
            var x = kp.xy[(d * kp.m + p) * 2]
            var y = kp.xy[(d * kp.m + p) * 2 + 1]
            if (abs(x) <= 1e-8 and abs(y) <= 1e-8) or not isfinite(x) or not isfinite(y):
                continue
            _circle_fill(scene, Int(x), Int(y), radius, col)
