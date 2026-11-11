# supervision/_cv: the cv2 layer

supervision calls cv2 through `supervision._cv2`; this package is the Mojo side. Built on mojo-cv, verified pixel-for-pixel
against cv2 5.0.0.93 by `tests/cases/pixels_cv.txt` (driver `tests/s5/pixels_cv.mojo`, binary `build/pixels_cv`).

| file | ops |
|---|---|
| `draw.mojo` | circle (thin, thick, filled), ellipse and arcs, polylines, fillPoly; LINE_8 only (no LINE_AA, no drawMarker) |
| `ops.mojo` | addWeighted, convertScaleAbs, applyColorMap(JET) on uint8 |
| `png.mojo` | PNG read/write, 8-bit gray/RGB/RGBA/palette, no interlace, no 16-bit, no JPEG; zlib via mojo-gzip |
| `sintable.mojo`, `jet.mojo` | data: cv2's float32 degree sine table (from drawing.cpp), JET LUT (tools/gen_jet.py) |

## Text: `font/` (putText, getTextSize)

cv2 5.0.0 draws `FONT_HERSHEY_SIMPLEX` with an embedded Rubik face, size `round(100*scale/3.7)`, weight 400 or 600 by
thickness. The pip wheel has no HarfBuzz: `drawing_text.cpp` takes its `#else` path, one glyph per codepoint with no shaping or
kerning, rasterized by OpenCV's own `stb_truetype.cpp` fork. `font/` ports exactly that path, pixel for pixel.

| file | role |
|---|---|
| `font/ttf.mojo` | cmap, hmtx, glyf (composites), fvar/avar/gvar with stbtt's integer arithmetic; `fmul` blocks FMA contraction |
| `font/raster.mojo` | stbtt rasterizer v2 (exact-area coverage) and `glyph_bitmap` |
| `font/layout.mojo` | `TextEngine.put_text` / `text_size`: legacy wrappers + `putText_` layout, blend, glyph cache |
| `font/sans.mojo`, `font/rubik_data.mojo`, `font/Rubik-sans.ttf.gz`, `font/NOTICE` | the face (OFL-1.1, hex-embedded, credited) |

Bar: rows `font_ttf` (885), `pixels_text` (242), `cv_put_text_aa`, `cv_text_size` and `pixel_label` (6) are all max abs diff 0.
`tools/stbtt_ref.sh` builds OpenCV's C source and shows the glyph rows equal it. Not ported: RTL/BiDi, HarfBuzz shaping, wrapping
and alignment flags, bottom-left origin, the italic fallback face (a codepoint missing from "sans" draws as '?').

## Known gaps

- `ThickLine` clips lines that leave the image only for shift 0 (as cv2); polylines with points far outside are untested.
- `fillPoly` takes no offset and uses shift 0.
