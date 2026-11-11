# Row driver for tests/cases/pixels_text.txt: cv2.putText / cv2.getTextSize (FONT_HERSHEY_SIMPLEX) vs src/supervision/_cv/font.
# text_put prints {"out": png path} for tools/pixdiff.py, text_size prints {"value": [width, height, baseline]}.
from json import Json, parse
from supervision._cv.font.layout import TextEngine
from supervision._cv.png import imread, imwrite


def main() raises:
    var eng = TextEngine()
    var n = 0
    for ln in open("/dev/stdin", "r").read().split("\n"):
        var line = String(ln)
        if line.byte_length() == 0:
            continue
        var cols = line.split("\t")
        var op = String(cols[0])
        var args = parse(String(cols[1]))
        var p = args.get("params")
        var res: String
        try:
            if op == "text_size":
                var s = eng.text_size(p.get("text").str(), p.get("scale").num(), p.get("thickness").int())
                res = '{"value": [' + String(s[0]) + ", " + String(s[1]) + ", " + String(s[2]) + "]}"
            elif op == "text_put":
                var img = imread(String(args.text("input")))
                var color = List[Float64]()
                for i in range(p.get("color").len()):
                    color.append(p.get("color").at(i).num())
                eng.put_text(img, p.get("text").str(), p.get("org").at(0).int(), p.get("org").at(1).int(), p.get("scale").num(), color, p.get("thickness").int())
                var path = "build/tests-out/px_text_" + String(n) + ".png"
                imwrite(path, img, 6)
                res = '{"out": "' + path + '"}'
            else:
                raise Error("unknown op " + op)
        except e:
            res = '{"error": "' + String(e) + '"}'
        n += 1
        print(res)
