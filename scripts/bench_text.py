"""cv2 side of scripts/bench_text.mojo. Run: .work/oracle-venv/bin/python -I scripts/bench_text.py"""
import time

import cv2
import numpy as np

LABELS = ["car 0.92", "bus 0.75", "person 0.61", "traffic light", "bicycle 0.88", "dog", "truck 0.55", "cat", "#14 stop sign", "bench", "bird 0.31", "handbag 0.47"]
img = np.zeros((480, 640, 3), np.uint8)
t0 = time.perf_counter()
for f in range(300):
    for i in range(12):
        cv2.putText(img, LABELS[i], (10 + 50 * (i % 6), 40 + 38 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5 + 0.1 * (i % 4), (255, 255, 255), 1 + (i % 2), cv2.LINE_AA)
print("cv2 putText:", (time.perf_counter() - t0) * 1000 / 300, "ms/frame (12 labels)")
