"""Bank-mapping analysis: skewed (implemented in RTL) vs naive row-major (ablation).

A cycle with m simultaneous requests to one bank costs m-1 extra cycles (1R/1W port banks).
Skewed mapping: element (row r, col c) -> bank (r + c) mod NB, word address r*stride + c//NB.
"""
import random
from math import gcd
from .config import cdiv


def extra_cycles(banks):
    cnt = {}
    for b in banks:
        cnt[b] = cnt.get(b, 0) + 1
    return max(cnt.values()) - 1 if cnt else 0


def skew_bank(r, c, nb):
    return (r + c) % nb


def naive_bank(r, c, stride_elems, nb):
    return (r * stride_elems + c) % nb


def compute_read_conflicts(tm, tn, tk, nb, mode):
    """Extra cycles over one tile's compute reads (A lane vector + B lane vector per cycle)."""
    kg = cdiv(tk, nb)
    total = 0
    for g in range(kg):
        ks = [g * nb + l for l in range(nb) if g * nb + l < tk]
        for i in range(min(tm, nb)):                       # pattern is periodic in i mod nb
            if mode == "skew":
                ea = extra_cycles([skew_bank(i, k, nb) for k in ks])
            else:
                ea = extra_cycles([naive_bank(i, k, tk, nb) for k in ks])
            total += ea * (cdiv(tm - i, nb))
        for j in range(min(tn, nb)):
            if mode == "skew":
                eb = extra_cycles([skew_bank(j, k, nb) for k in ks])
            else:                                          # B stored [k][j], row stride tn
                eb = extra_cycles([naive_bank(k, j, tn, nb) for k in ks])
            total += eb * (cdiv(tn - j, nb))
    return total * 1  # per (i or j) vector, counted once; scaling by pair count done by caller


def dma_write_conflicts(rows, cols, nb, mode, which):
    """Extra cycles of DMA write beats (4 consecutive elements per beat) for one tile load."""
    total = 0
    kg = cdiv(rows if which == "B" else cols, nb)
    for r in range(rows):
        for c0 in range(0, cols, 4):
            cs = range(c0, min(c0 + 4, cols))
            if which == "A":        # element (i=r, k=c)
                if mode == "skew":
                    bs = [skew_bank(r, c, nb) for c in cs]
                else:
                    bs = [naive_bank(r, c, cols, nb) for c in cs]
            else:                   # B: ext row k=r, cols j=c ; buffer element (row j, col k)
                if mode == "skew":
                    bs = [skew_bank(c, r, nb) for c in cs]
                else:               # naive keeps [k][j] row-major with stride cols
                    bs = [naive_bank(r, c, cols, nb) for c in cs]
            total += extra_cycles(bs)
    return total


def synthetic_stride_study(nb=8, strides=range(1, 65), trials=200, seed=1):
    """NB parallel accesses with a fixed element stride: naive linear bank = (i*s) mod nb versus
    a skewed mapping bank = (i*s + i*s//nb) mod nb (the classic +1 skew per wrap)."""
    rows = []
    for s in strides:
        naive = extra_cycles([(i * s) % nb for i in range(nb)])
        skew = extra_cycles([((i * s) + (i * s) // nb) % nb for i in range(nb)])
        rows.append((s, naive, skew))
    rng = random.Random(seed)
    rnd = [extra_cycles([rng.randrange(nb) for _ in range(nb)]) for _ in range(trials)]
    return rows, sum(rnd) / len(rnd)
