# cv2's built-in "sans" face: Rubik Regular (variable, wght 300 to 900), embedded gzipped as hex in rubik_data.mojo.
from gzip import decompress
from supervision._cv.font.rubik_data import RUBIK_SANS_GZ_HEX
from supervision._cv.font.ttf import Font


def _nib(c: UInt8) -> UInt8:
    return c - 48 if c <= 57 else c - 87


def load_sans() raises -> Font:
    var hx = RUBIK_SANS_GZ_HEX.as_bytes()
    var gz = List[UInt8](capacity=len(hx) // 2)
    for i in range(len(hx) // 2):
        gz.append((_nib(hx[2 * i]) << 4) | _nib(hx[2 * i + 1]))
    return Font(decompress(gz))
