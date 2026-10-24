"""Descriptor-program compressibility study (feasibility of a hardware loop generator).

Encoding (lossless, verified by decode()): a stream of descriptor vectors is covered greedily by
  RAW  : one entry
  REP  : p base entries + 1 delta entry, expanding to n ops with  op[t] = op[t-p] + delta  for t >= p
A hardware loop generator would need exactly this property (per-field constant increments per period).
This measures POTENTIAL; it is not an RTL implementation.
"""
from .schedule import Program


def dma_vec(o):
    return (o.kind, o.dep_cp, o.rows, o.cols, o.ext_addr, o.ext_stride, o.base, o.stride)


def cmp_vec(o):
    return (o.tm, o.tn, o.tk, int(o.first), o.dep_dma, o.a_base, o.b_base, o.o_base)


def encode(vecs, maxp=8):
    out, n, i = [], len(vecs), 0
    while i < n:
        best = None                                   # (saved, p, covered, delta)
        for p in range(1, maxp + 1):
            if i + 2 * p > n:
                break
            delta = tuple(b - a for a, b in zip(vecs[i], vecs[i + p]))
            t = i + p
            while t < n and all(vecs[t][k] - vecs[t - p][k] == delta[k] for k in range(len(delta))):
                t += 1
            covered = t - i
            if covered >= 2 * p:
                saved = covered - (p + 1)
                if saved > 0 and (best is None or saved > best[0]):
                    best = (saved, p, covered, delta)
        if best:
            _, p, covered, delta = best
            out.append(("rep", vecs[i:i + p], delta, covered))
            i += covered
        else:
            out.append(("raw", vecs[i]))
            i += 1
    return out


def decode(enc):
    ops = []
    for e in enc:
        if e[0] == "raw":
            ops.append(e[1])
        else:
            _, base, delta, covered = e
            p = len(base)
            seg = list(base)
            while len(seg) < covered:
                prev = seg[-p]
                seg.append(tuple(a + d for a, d in zip(prev, delta)))
            ops.extend(seg)
    return ops


def cost_entries(enc):
    return sum(1 if e[0] == "raw" else len(e[1]) + 1 for e in enc)


def analyse(prog: Program):
    d, c = [dma_vec(o) for o in prog.dma], [cmp_vec(o) for o in prog.cmp]
    ed, ec = encode(d), encode(c)
    assert decode(ed) == d and decode(ec) == c, "lossless check failed"
    return dict(raw=len(d) + len(c), compressed=cost_entries(ed) + cost_entries(ec),
                raw_dma=len(d), comp_dma=cost_entries(ed), raw_cmp=len(c), comp_cmp=cost_entries(ec))
