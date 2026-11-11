"""Python's random.Random(seed): MT19937 seeded by init_by_array, plus shuffle.

Dataset split must pick the same images as Python supervision, so the generator
and `shuffle` follow CPython's `random` module exactly (getrandbits, _randbelow).
"""


struct PyRandom(Movable):
    var mt: List[UInt32]
    var idx: Int

    def __init__(out self, seed: UInt64):
        self.mt = List[UInt32](length=624, fill=0)
        self.idx = 624
        self._init_genrand(19650218)
        # CPython seeds with the 32-bit chunks of abs(seed), least significant first.
        var key = List[UInt32]()
        var s = seed
        if s == 0:
            key.append(0)
        while s > 0:
            key.append(UInt32(s & 0xFFFFFFFF))
            s >>= 32
        self._init_by_array(key)

    def _init_genrand(mut self, s: UInt32):
        self.mt[0] = s
        for i in range(1, 624):
            var prev = self.mt[i - 1]
            self.mt[i] = UInt32(1812433253) * (prev ^ (prev >> 30)) + UInt32(i)

    def _init_by_array(mut self, key: List[UInt32]):
        var n = len(key)
        var i = 1
        var j = 0
        var k = 624 if 624 > n else n
        while k > 0:
            var prev = self.mt[i - 1]
            self.mt[i] = (
                (self.mt[i] ^ ((prev ^ (prev >> 30)) * UInt32(1664525)))
                + key[j]
                + UInt32(j)
            )
            i += 1
            j += 1
            if i >= 624:
                self.mt[0] = self.mt[623]
                i = 1
            if j >= n:
                j = 0
            k -= 1
        k = 623
        while k > 0:
            var prev = self.mt[i - 1]
            self.mt[i] = (
                (self.mt[i] ^ ((prev ^ (prev >> 30)) * UInt32(1566083941)))
                - UInt32(i)
            )
            i += 1
            if i >= 624:
                self.mt[0] = self.mt[623]
                i = 1
            k -= 1
        self.mt[0] = 0x80000000
        self.idx = 624

    def _twist(mut self):
        for kk in range(624):
            var y = (self.mt[kk] & 0x80000000) | (self.mt[(kk + 1) % 624] & 0x7FFFFFFF)
            var v = self.mt[(kk + 397) % 624] ^ (y >> 1)
            if (y & 1) != 0:
                v ^= 0x9908B0DF
            self.mt[kk] = v
        self.idx = 0

    def genrand_u32(mut self) -> UInt32:
        if self.idx >= 624:
            self._twist()
        var y = self.mt[self.idx]
        self.idx += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y

    def getrandbits(mut self, k: Int) -> UInt32:
        """k in 1..32, like CPython's getrandbits for k <= 32."""
        return self.genrand_u32() >> UInt32(32 - k)

    def randbelow(mut self, n: Int) -> Int:
        if n <= 0:
            return 0
        var k = 0
        var t = n
        while t > 0:
            k += 1
            t >>= 1
        var r = Int(self.getrandbits(k))
        while r >= n:
            r = Int(self.getrandbits(k))
        return r

    def shuffle_indices(mut self, n: Int) -> List[Int]:
        """The permutation `random.shuffle` applies to `list(range(n))`."""
        var out = List[Int](capacity=n)
        for i in range(n):
            out.append(i)
        var i = n - 1
        while i >= 1:
            var j = self.randbelow(i + 1)
            var tmp = out[i]
            out[i] = out[j]
            out[j] = tmp
            i -= 1
        return out^
