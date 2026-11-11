# Detection tools (lane S4)

Ports of `supervision/detection/tools/*` and `LineZone` onto the S0 `Detections`. Rows: `tests/cases/tools_*.txt`
(468 rows, written by `tools/gen_tools.py` seed 4 from Python supervision itself). Build: `scripts/build_tools.sh`;
run: `tests/run.sh tests/cases/tools_*.txt`.

| Python | Mojo | Rows |
|---|---|---|
| `InferenceSlicer._generate_offset`, `_validate_overlap` | `tools/slicer_offsets.mojo` | `tools_slicer` 158 |
| `InferenceSlicer.__call__` after the callback (move, merge, NMS / NMM / none) | `tools/inference_slicer.mojo` | `tools_slicer_merge` 60 |
| `LineZone` (`trigger`, counts, per-class counts) | `tools/line_zone.mojo` | `tools_line` 70 |
| `PolygonZone.trigger` | `tools/polygon_zone.mojo` | `tools_zone` 60 |
| `DetectionsSmoother` | `tools/smoother.mojo` | `tools_smoother` 60 |
| `CSVSink`, `JSONSink` (byte-exact files) | `tools/csv_sink.mojo`, `json_sink.mojo`, `sink_rows.mojo` | `tools_sinks` 60 |

## Not ported (stated, not silent)
- `PolygonZoneAnnotator`, `LineZoneAnnotator*`: S5 (draw).
- `PolygonZone` builds no mask yet: the constructor takes it. Rows feed the oracle's own `zone.mask`, so they test `trigger`,
  not `polygon_to_mask`. Wire `polygon_to_mask` in when S1/S5 land `fillPoly`.
- `coord_transform` argument of `LineZone.trigger` / `PolygonZone.trigger` (needs `CoordinatesTransformation`).
- `InferenceSlicer`: image cropping, calling the model, `thread_workers`, `batch_size`, compact masks, raster datasets, the
  out-of-bounds warning. Detections with a dense mask raise (waits on S1 `move_masks`).
- Sinks: S1 stores numeric `data` columns as Float64, so an int column prints `1.0`; pass ints through `custom_data`.
  Confidences print as float64 repr (Python float32 would print the shortest float32).

## Traps found
- `input()` in Mojo 1.2 returns the first line, then EOF. Drivers read fd 0 with `external_call["read", Int](Int32(0), ...)`;
  any other signature aborts at LLVM lowering because the stdlib already declares `read`.
- Rows whose confidences are rounded to 3 digits make argsort tie order decide NMS / NMM; numpy's quicksort and S1's sort
  differ on ties (4 of 60 rows). Unrounded values in the rows; ties are an S1 question.
