"""Row runner for tests/cases/metrics_*.txt: one JSON result per stdin row (tests/README.md)."""

from supervision.detection.core import Detections
from supervision.metrics.adapt import to_dets_list
from supervision.metrics.coco import mean_average_precision, mean_valid
from supervision.metrics.confmat import confusion_matrix
from supervision.metrics.dets import Dets
from supervision.metrics.keypoints import KpDets, coco_keypoint_sigmas, keypoint_mean_average_precision
from supervision.metrics.mar import mean_average_recall
from supervision.metrics.prf import F1, MACRO, MICRO, PRECISION, RECALL, WEIGHTED, prf


struct Cursor:
    var b: List[UInt8]
    var i: Int

    def __init__(out self, s: String):
        self.b = List[UInt8]()
        for c in s.as_bytes():
            self.b.append(c)
        self.i = 0

    def ws(mut self):
        while self.i < len(self.b) and (self.b[self.i] == 32 or self.b[self.i] == 10 or self.b[self.i] == 13 or self.b[self.i] == 9):
            self.i += 1

    def peek(mut self) -> UInt8:
        self.ws()
        return self.b[self.i] if self.i < len(self.b) else 0

    def eat(mut self, c: UInt8) raises:
        if self.peek() != c:
            raise Error("json: expected '" + chr(Int(c)) + "' at " + String(self.i))
        self.i += 1

    def maybe(mut self, c: UInt8) -> Bool:
        if self.peek() == c:
            self.i += 1
            return True
        return False

    def string(mut self) raises -> String:
        self.eat(34)
        var s = String()
        while self.i < len(self.b) and self.b[self.i] != 34:
            s += chr(Int(self.b[self.i]))
            self.i += 1
        self.eat(34)
        return s^

    def number(mut self) raises -> Float64:
        self.ws()
        var s = String()
        while self.i < len(self.b):
            var c = self.b[self.i]
            if (c >= 48 and c <= 57) or c == 45 or c == 43 or c == 46 or c == 101 or c == 69:
                s += chr(Int(c))
                self.i += 1
            else:
                break
        return Float64(s)

    def numbers(mut self) raises -> List[Float64]:
        var out = List[Float64]()
        self.eat(91)
        if self.maybe(93):
            return out^
        while True:
            out.append(self.number())
            if not self.maybe(44):
                break
        self.eat(93)
        return out^

    def flat(mut self) raises -> List[Float64]:
        """Every number of a (nested) array, in order."""
        var out = List[Float64]()
        if self.maybe(91):
            if not self.maybe(93):
                while True:
                    out.extend(self.flat())
                    if not self.maybe(44):
                        break
                self.eat(93)
        else:
            out.append(self.number())
        return out^

    def skip(mut self) raises:
        var c = self.peek()
        if c == 34:
            _ = self.string()
        elif c == 91 or c == 123:
            var depth = 0
            while self.i < len(self.b):
                var x = self.b[self.i]
                if x == 34:
                    _ = self.string()
                    continue
                if x == 91 or x == 123:
                    depth += 1
                elif x == 93 or x == 125:
                    depth -= 1
                self.i += 1
                if depth == 0:
                    break
        else:
            while self.i < len(self.b) and self.b[self.i] != 44 and self.b[self.i] != 125 and self.b[self.i] != 93:
                self.i += 1


def f32(x: Float64) -> Float64:
    return Float64(Float32(x))


def parse_image(mut c: Cursor) raises -> Detections:
    """One image: {"xyxy": [[x1,y1,x2,y2],...], "class_id": [...], "confidence": [...]}."""
    var xyxy = List[Float64]()
    var cls = List[Int]()
    var conf = List[Float64]()
    var has_cls = False
    var has_conf = False
    c.eat(123)
    while True:
        var key = c.string()
        c.eat(58)
        if key == "xyxy":
            c.eat(91)
            if not c.maybe(93):
                while True:
                    for v in c.numbers():
                        xyxy.append(f32(v))
                    if not c.maybe(44):
                        break
                c.eat(93)
        elif key == "class_id":
            has_cls = True
            for v in c.numbers():
                cls.append(Int(v))
        elif key == "confidence":
            has_conf = True
            for v in c.numbers():
                conf.append(f32(v))
        else:
            c.skip()
        if not c.maybe(44):
            break
    c.eat(125)
    var d = Detections(xyxy^)
    if has_cls:
        d.set_class_id(cls^)
    if has_conf:
        d.set_confidence(conf^)
    return d^


def parse_kp_image(mut c: Cursor) raises -> KpDets:
    """One image of skeletons: {"n", "k", "xy", "visible", "class_id", "confidence", "area", "iscrowd", "xyxy"}."""
    var n = 0
    var k = 0
    var xy = List[Float64]()
    var visible = List[Int]()
    var iscrowd = List[Int]()
    var cls = List[Int]()
    var conf = List[Float64]()
    var area = List[Float64]()
    var xyxy = List[Float64]()
    c.eat(123)
    while True:
        var key = c.string()
        c.eat(58)
        if key == "n":
            n = Int(c.number())
        elif key == "k":
            k = Int(c.number())
        elif key == "xy":
            for v in c.flat():
                xy.append(f32(v))
        elif key == "visible":
            for v in c.flat():
                visible.append(Int(v))
        elif key == "iscrowd":
            for v in c.flat():
                iscrowd.append(Int(v))
        elif key == "class_id":
            for v in c.flat():
                cls.append(Int(v))
        elif key == "confidence":
            for v in c.flat():
                conf.append(f32(v))
        elif key == "area":
            area = c.flat()
        elif key == "xyxy":
            xyxy = c.flat()
        else:
            c.skip()
        if not c.maybe(44):
            break
    c.eat(125)
    var d = KpDets(n, k, xy^)
    d.visible = visible^
    d.iscrowd = iscrowd^
    d.class_id = cls^
    d.conf = conf^
    d.area = area^
    d.xyxy = xyxy^
    return d^


def parse_kp_images(mut c: Cursor) raises -> List[KpDets]:
    var out = List[KpDets]()
    c.eat(91)
    if c.maybe(93):
        return out^
    while True:
        out.append(parse_kp_image(c))
        if not c.maybe(44):
            break
    c.eat(93)
    return out^


def parse_images(mut c: Cursor) raises -> List[Detections]:
    var out = List[Detections]()
    c.eat(91)
    if c.maybe(93):
        return out^
    while True:
        out.append(parse_image(c))
        if not c.maybe(44):
            break
    c.eat(93)
    return out^


struct Args(Movable):
    var pred: List[Detections]
    var target: List[Detections]
    var averaging: String
    var classes: Int
    var conf: Float64
    var iou: Float64
    var kp_pred: List[KpDets]
    var kp_target: List[KpDets]
    var sigmas: List[Float64]
    var agnostic: Bool

    def __init__(out self):
        self.kp_pred = List[KpDets]()
        self.kp_target = List[KpDets]()
        self.sigmas = coco_keypoint_sigmas()
        self.agnostic = False
        self.pred = List[Detections]()
        self.target = List[Detections]()
        self.averaging = "weighted"
        self.classes = 0
        self.conf = 0.3
        self.iou = 0.5


def parse_args(text: String) raises -> Args:
    var c = Cursor(text)
    var a = Args()
    c.eat(123)
    while True:
        var key = c.string()
        c.eat(58)
        if key == "pred":
            a.pred = parse_images(c)
        elif key == "target":
            a.target = parse_images(c)
        elif key == "kp_pred":
            a.kp_pred = parse_kp_images(c)
        elif key == "kp_target":
            a.kp_target = parse_kp_images(c)
        elif key == "sigmas":
            a.sigmas = c.numbers()
        elif key == "agnostic":
            a.agnostic = c.number() != 0
        elif key == "averaging":
            a.averaging = c.string()
        elif key == "conf":
            a.conf = c.number()
        elif key == "iou":
            a.iou = c.number()
        elif key == "classes":
            c.eat(91)
            if not c.maybe(93):
                while True:
                    _ = c.string()
                    a.classes += 1
                    if not c.maybe(44):
                        break
                c.eat(93)
        else:
            c.skip()
        if not c.maybe(44):
            break
    c.eat(125)
    return a^


def fmt(x: Float64) -> String:
    return String(x)


def fmt_list(xs: List[Float64]) -> String:
    var s = String("[")
    for i in range(len(xs)):
        s += ("," if i > 0 else "") + fmt(xs[i])
    return s + "]"


def run_row(line: String) raises -> String:
    var parts = line.split("\t")
    if len(parts) < 2:
        raise Error("row needs <op> TAB <args> TAB <expected>")
    var op = String(parts[0])
    var a = parse_args(String(parts[1]))
    var preds = to_dets_list(a.pred)
    var targets = to_dets_list(a.target)
    if op == "map50" or op == "map50_95":
        var ids = List[Int]()
        for i in range(len(preds)):
            ids.append(i)
        var r = mean_average_precision(preds, targets, ids, [1, 10, 100])
        return fmt(r.all.scores[0]) if op == "map50" else fmt(mean_valid(r.all.scores))
    if op == "kp_map":
        var r = keypoint_mean_average_precision(a.kp_pred, a.kp_target, a.sigmas, a.agnostic)
        var out = List[Float64]()
        for ap in [r.all.scores.copy(), r.medium.scores.copy(), r.large.scores.copy()]:
            out.append(mean_valid(ap))
            out.append(ap[0])
            out.append(ap[5])
        return fmt_list(out)
    if op == "precision" or op == "recall" or op == "f1":
        var kind = PRECISION if op == "precision" else (RECALL if op == "recall" else F1)
        var avg = MACRO if a.averaging == "macro" else (MICRO if a.averaging == "micro" else WEIGHTED)
        return fmt_list(prf(preds, targets, kind, avg).scores)
    if op == "mar":
        return fmt_list(mean_average_recall(preds, targets))
    if op == "confusion":
        var m = confusion_matrix(preds, targets, a.classes, a.conf, a.iou)
        var w = a.classes + 1
        var s = String("[")
        for i in range(w):
            s += ("," if i > 0 else "") + "["
            for j in range(w):
                s += ("," if j > 0 else "") + String(m[i * w + j])
            s += "]"
        return s + "]"
    raise Error("unknown op " + op)


def main() raises:
    # `input()` cuts lines at about 1 KB; read the whole stream instead.
    var text: String
    with open("/dev/stdin", "r") as f:
        text = f.read()
    for line in text.split("\n"):
        if line.byte_length() > 0:
            print(run_row(String(line)))
