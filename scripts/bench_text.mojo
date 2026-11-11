# putText speed on a 640x480 label scene: 12 labels per frame, 300 frames, one resident process, warm glyph cache.
# Compare with `python -I scripts/bench_text.py` (cv2). Build: mojo build -O3 -I src -I $MOJO_CV scripts/bench_text.mojo
from std.time import perf_counter_ns
from mojocv.mat import Mat
from supervision._cv.font.layout import TextEngine

def labels() -> List[String]:
    return ["car 0.92", "bus 0.75", "person 0.61", "traffic light", "bicycle 0.88", "dog", "truck 0.55", "cat", "#14 stop sign", "bench", "bird 0.31", "handbag 0.47"]


def main() raises:
    var eng = TextEngine()
    var labels = labels()
    var img = Mat[DType.uint8](480, 640, 3)
    var color: List[Float64] = [255.0, 255.0, 255.0]
    var sink = 0
    var t0 = perf_counter_ns()
    for f in range(300):
        for i in range(12):
            eng.put_text(img, labels[i], 10 + 50 * (i % 6), 40 + 38 * i, 0.5 + 0.1 * Float64(i % 4), color, 1 + (i % 2))
        sink += Int(img.data[1000 + f])
    var dt = Float64(perf_counter_ns() - t0) / 1e6
    print("mojo putText:", dt / 300.0, "ms/frame (12 labels)", sink)
