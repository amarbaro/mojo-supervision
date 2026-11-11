"""Stable index sorts, the numpy `argsort(kind="stable")` the Python metrics rely on."""


def argsort_stable(keys: List[Float64]) -> List[Int]:
    """Indices that order `keys` ascending; equal keys keep their input order."""
    var n = len(keys)
    var src = List[Int](capacity=n)
    for i in range(n):
        src.append(i)
    var dst = List[Int](length=n, fill=0)
    var width = 1
    while width < n:
        var lo = 0
        while lo < n:
            var mid = min(lo + width, n)
            var hi = min(lo + 2 * width, n)
            var i = lo
            var j = mid
            var k = lo
            while i < mid and j < hi:
                if keys[src[j]] < keys[src[i]]:
                    dst[k] = src[j]
                    j += 1
                else:
                    dst[k] = src[i]
                    i += 1
                k += 1
            while i < mid:
                dst[k] = src[i]
                i += 1
                k += 1
            while j < hi:
                dst[k] = src[j]
                j += 1
                k += 1
            lo += 2 * width
        swap(src, dst)
        width *= 2
    return src^


def argsort_desc_stable(scores: List[Float64]) -> List[Int]:
    """`np.argsort(-scores, kind="stable")`: highest first, ties in input order."""
    var neg = List[Float64](capacity=len(scores))
    for s in scores:
        neg.append(-s)
    return argsort_stable(neg)


def sorted_unique(xs: List[Int]) -> List[Int]:
    """`np.unique(xs)` for ints."""
    var idx = List[Float64](capacity=len(xs))
    for x in xs:
        idx.append(Float64(x))
    var order = argsort_stable(idx)
    var out = List[Int]()
    for o in order:
        if len(out) == 0 or out[len(out) - 1] != xs[o]:
            out.append(xs[o])
    return out^
