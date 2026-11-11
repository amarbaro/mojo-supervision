"""The XML subset Pascal VOC needs: parse into a flat tree, escape text like minidom.

Parsed: elements, character data, CDATA, the five named entities and numeric references.
Attributes are skipped; comments, processing instructions and the XML declaration are skipped.
A DOCTYPE or ENTITY raises, as defusedxml does.
"""


struct XmlDoc(Movable):
    var tag: List[String]
    var text: List[String]
    var has_text: List[Bool]  # ElementTree .text is None when an element holds no character data before its first child
    var children: List[List[Int]]

    def __init__(out self):
        self.tag = List[String]()
        self.text = List[String]()
        self.has_text = List[Bool]()
        self.children = List[List[Int]]()

    def new_node(mut self, tag: String) -> Int:
        self.tag.append(tag)
        self.text.append(String())
        self.has_text.append(False)
        self.children.append(List[Int]())
        return len(self.tag) - 1

    def find(self, node: Int, tag: String) -> Int:
        """First direct child named `tag`, or -1."""
        for c in self.children[node]:
            if self.tag[c] == tag:
                return c
        return -1

    def findall(self, node: Int, tag: String) -> List[Int]:
        var out = List[Int]()
        for c in self.children[node]:
            if self.tag[c] == tag:
                out.append(c)
        return out^

    def descendants(self, node: Int) -> List[Int]:
        """`.//*`: every descendant in document order."""
        var out = List[Int]()
        for c in self.children[node]:
            out.append(c)
            out.extend(self.descendants(c))
        return out^


def xml_escape(s: String) -> String:
    """minidom's _write_data: & < > and the double quote."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;").replace(">", "&gt;")


def _num(s: String, base: Int) raises -> Int:
    var v = 0
    for c in s.as_bytes():
        var d = Int(c) - 48 if c <= 57 else (Int(c) - 87 if c >= 97 else Int(c) - 55)
        if d < 0 or d >= base:
            raise Error("xml: bad character reference")
        v = v * base + d
    return v


def _decode(raw: String) raises -> String:
    if raw.find("&") < 0:
        return raw
    var out = String()
    var b = raw.as_bytes()
    var i = 0
    var n = len(b)
    while i < n:
        if b[i] != 38:
            var j = i
            while j < n and b[j] != 38:
                j += 1
            out += String(raw[byte = i : j])
            i = j
            continue
        var semi = i + 1
        while semi < n and b[semi] != 59:
            semi += 1
        if semi >= n:
            raise Error("xml: unterminated entity")
        var ent = String(raw[byte = i + 1 : semi])
        if ent == "amp":
            out += "&"
        elif ent == "lt":
            out += "<"
        elif ent == "gt":
            out += ">"
        elif ent == "quot":
            out += '"'
        elif ent == "apos":
            out += "'"
        elif ent.startswith("#x") or ent.startswith("#X"):
            out += chr(_num(String(ent[byte = 2 :]), 16))
        elif ent.startswith("#"):
            out += chr(_num(String(ent[byte = 1 :]), 10))
        else:
            raise Error("xml: undefined entity &" + ent + ";")
        i = semi + 1
    return out^


def parse_xml(text: String) raises -> XmlDoc:
    """Returns the tree; the root element is node 0."""
    var doc = XmlDoc()
    var b = text.as_bytes()
    var n = len(b)
    var i = 0
    var stack = List[Int]()
    var seen_root = False
    # text accumulates into the element on top of the stack until its first child
    var open_text_ok = List[Bool]()
    while i < n:
        if b[i] != 60:
            var j = i
            while j < n and b[j] != 60:
                j += 1
            if len(stack) > 0:
                var top = stack[len(stack) - 1]
                if open_text_ok[len(open_text_ok) - 1]:
                    doc.text[top] += _decode(String(text[byte = i : j]))
                    doc.has_text[top] = True
            i = j
            continue
        if text.find("<!--", i) == i:
            var e = text.find("-->", i)
            if e < 0:
                raise Error("xml: unterminated comment")
            i = e + 3
        elif text.find("<![CDATA[", i) == i:
            var e = text.find("]]>", i)
            if e < 0:
                raise Error("xml: unterminated CDATA")
            if len(stack) > 0 and open_text_ok[len(open_text_ok) - 1]:
                var top = stack[len(stack) - 1]
                doc.text[top] += String(text[byte = i + 9 : e])
                doc.has_text[top] = True
            i = e + 3
        elif text.find("<?", i) == i:
            var e = text.find("?>", i)
            if e < 0:
                raise Error("xml: unterminated processing instruction")
            i = e + 2
        elif text.find("<!", i) == i:
            raise Error("xml: DTD and entity declarations are forbidden")
        elif i + 1 < n and b[i + 1] == 47:
            var e = text.find(">", i)
            if e < 0 or len(stack) == 0:
                raise Error("xml: unexpected closing tag")
            var name = String(String(text[byte = i + 2 : e]).strip())
            if name != doc.tag[stack[len(stack) - 1]]:
                raise Error("xml: mismatched tag </" + name + ">")
            _ = stack.pop()
            _ = open_text_ok.pop()
            i = e + 1
        else:
            # start tag: name, skip attributes (quote aware), maybe self closing
            var j = i + 1
            while j < n and b[j] != 62 and b[j] != 32 and b[j] != 47 and b[j] != 9 and b[j] != 10 and b[j] != 13:
                j += 1
            var name = String(text[byte = i + 1 : j])
            var q = 0
            while j < n and (b[j] != 62 or q != 0):
                if q != 0:
                    if Int(b[j]) == q:
                        q = 0
                elif b[j] == 34 or b[j] == 39:
                    q = Int(b[j])
                j += 1
            if j >= n:
                raise Error("xml: unterminated tag <" + name + ">")
            var self_closing = b[j - 1] == 47
            if len(stack) == 0 and seen_root:
                raise Error("xml: junk after document element")
            var node = doc.new_node(name)
            if len(stack) > 0:
                var parent = stack[len(stack) - 1]
                doc.children[parent].append(node)
                open_text_ok[len(open_text_ok) - 1] = False
            else:
                seen_root = True
            if not self_closing:
                stack.append(node)
                open_text_ok.append(True)
            i = j + 1
    if not seen_root:
        raise Error("xml: no element found")
    if len(stack) != 0:
        raise Error("xml: unclosed tag <" + doc.tag[stack[len(stack) - 1]] + ">")
    return doc^
