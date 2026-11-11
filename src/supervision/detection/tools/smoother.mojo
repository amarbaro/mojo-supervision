# Port of supervision/detection/tools/smoother.py DetectionsSmoother (MIT, see NOTICE-supervision-LICENSE.md).
from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.core import Detections, DataCol
from supervision.detection.utils.boxes import xyxyxyxy_to_xyxy


def _align_oriented_corners(corners: List[Float64], reference: List[Float64]) -> List[Float64]:
    """Of the 8 cyclic/reversed orderings of the 4 corners, the one closest to `reference` (first minimum wins)."""
    var best = 0
    var best_d = Float64.MAX_FINITE
    for variant in range(2):
        for shift in range(4):
            var d = 0.0
            for j in range(4):
                var src = (j - shift) % 4  # np.roll: out[(j + shift) % 4] = in[j]
                if variant == 1:
                    src = 3 - src
                var dx = corners[2 * src] - reference[2 * j]
                var dy = corners[2 * src + 1] - reference[2 * j + 1]
                d += dx * dx
                d += dy * dy
            if d < best_d:
                best_d = d
                best = variant * 4 + shift
    var out = List[Float64]()
    for j in range(4):
        var src = (j - best % 4) % 4
        if best >= 4:
            src = 3 - src
        out.append(corners[2 * src])
        out.append(corners[2 * src + 1])
    return out^


struct DetectionsSmoother(Movable):
    var length: Int
    var ids: List[Int]  # track ids in first-seen order (Python dict order)
    var tracks: List[List[Optional[Detections]]]

    def __init__(out self, length: Int = 5) raises:
        if length < 0:
            raise Error("maxlen must be non-negative")
        self.length = length
        self.ids = List[Int]()
        self.tracks = List[List[Optional[Detections]]]()

    def reset(mut self):
        self.ids.clear()
        self.tracks.clear()

    def _find(self, tid: Int) -> Int:
        for i in range(len(self.ids)):
            if self.ids[i] == tid:
                return i
        return -1

    def _push(mut self, k: Int, var v: Optional[Detections]):
        self.tracks[k].append(v^)
        if len(self.tracks[k]) > self.length:
            _ = self.tracks[k].pop(0)

    def _age_tracks(mut self, active: List[Int]):
        for k in range(len(self.ids)):
            var is_active = False
            for a in active:
                if a == self.ids[k]:
                    is_active = True
            if not is_active:
                self._push(k, Optional[Detections]())
        var ids = List[Int]()
        var tracks = List[List[Optional[Detections]]]()
        for k in range(len(self.ids)):
            var any_valid = False
            for e in self.tracks[k]:
                if e:
                    any_valid = True
            if any_valid:
                ids.append(self.ids[k])
                tracks.append(self.tracks[k].copy())
        self.ids = ids^
        self.tracks = tracks^

    def update_with_detections(mut self, detections: Detections) raises -> Detections:
        if not detections.has_tracker_id:
            # Python warns "Smoothing skipped" and ages every track.
            self._age_tracks(List[Int]())
            return detections.copy()
        for i in range(len(detections)):
            var tid = detections.tracker_id[i]
            var k = self._find(tid)
            if k < 0:
                self.ids.append(tid)
                self.tracks.append(List[Optional[Detections]]())
                k = len(self.ids) - 1
            self._push(k, Optional[Detections](detections.select([i])))
        var active = detections.tracker_id.copy()
        self._age_tracks(active)
        return self.get_smoothed_detections(active)

    def get_track(self, track_id: Int) raises -> Optional[Detections]:
        var k = self._find(track_id)
        if k < 0:
            return None
        var valid = List[Detections]()
        for e in self.tracks[k]:
            if e:
                valid.append(e.value().copy())
        var nv = len(valid)
        if nv == 0:
            return None
        var ret = valid[nv - 1].copy()
        for c in range(4):
            var s = 0.0
            for d in valid:
                s += d.xyxy[c]
            ret.xyxy[c] = s / Float64(nv)
        var cs = 0.0
        var cn = 0
        for d in valid:
            if d.has_confidence:
                cs += d.confidence[0]
                cn += 1
        if cn > 0:
            ret.confidence = [cs / Float64(cn)]
            ret.has_confidence = True
        else:
            ret.confidence = List[Float64]()
            ret.has_confidence = False
        var all_obb = True
        for d in valid:
            if not d.has_data(ORIENTED_BOX_COORDINATES):
                all_obb = False
        if all_obb:
            var reference = valid[0].get_data(ORIENTED_BOX_COORDINATES).f.copy()
            var acc = List[Float64](length=8, fill=0.0)
            for d in valid:
                var al = _align_oriented_corners(d.get_data(ORIENTED_BOX_COORDINATES).f, reference)
                for j in range(8):
                    acc[j] += al[j]
            for j in range(8):
                acc[j] /= Float64(nv)
            ret.set_data(DataCol(ORIENTED_BOX_COORDINATES, acc, 8))
            ret.xyxy = xyxyxyxy_to_xyxy(acc)
        else:
            var kept = List[DataCol]()
            for c in ret.data:
                if c.name != ORIENTED_BOX_COORDINATES:
                    kept.append(c.copy())
            ret.data = kept^
        return Optional[Detections](ret^)

    def get_smoothed_detections(self, track_ids: List[Int]) raises -> Detections:
        var tracked = List[Detections]()
        for tid in self.ids:
            var wanted = False
            for t in track_ids:
                if t == tid:
                    wanted = True
            if not wanted:
                continue
            var tr = self.get_track(tid)
            if tr:
                tracked.append(tr.take())
        var any_missing = False
        for d in tracked:
            if not d.has_confidence:
                any_missing = True
        if any_missing:
            for i in range(len(tracked)):
                tracked[i].has_confidence = False
                tracked[i].confidence = List[Float64]()
        var out = Detections.merge(tracked)
        if len(out) == 0:
            out.has_tracker_id = True
            out.tracker_id = List[Int]()
        return out^
