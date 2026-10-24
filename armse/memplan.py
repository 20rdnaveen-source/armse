"""Tensor-lifetime memory planner (MODEL ONLY): peak live memory vs naive allocation vs planned arena.

A tensor is live over the half-open time interval [birth, death).  Op i reads its inputs at time i and
writes its outputs at time i.  Non-in-place op: input death = i+1, output birth = i (they overlap).
In-place-capable op: output birth = i+1 (may share the input's bytes).
plan() = greedy first-fit by decreasing size (a heuristic, not optimal); verify() proves non-overlap."""
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Tensor:
    name: str
    size: int
    birth: int
    death: int           # exclusive


def overlaps(a: Tensor, b: Tensor) -> bool:
    return a.birth < b.death and b.birth < a.death


def naive_total(ts: List[Tensor]) -> int:
    return sum(t.size for t in ts)


def peak_live(ts: List[Tensor]) -> int:
    if not ts:
        return 0
    times = sorted({t.birth for t in ts})
    return max(sum(t.size for t in ts if t.birth <= x < t.death) for x in times)


def plan(ts: List[Tensor], align: int = 1):
    """Returns (offsets, arena_size)."""
    up = lambda v: -(-v // align) * align
    placed, offsets = [], {}
    for t in sorted(ts, key=lambda t: (-t.size, t.birth, t.name)):
        busy = sorted((offsets[p.name], offsets[p.name] + up(p.size)) for p in placed if overlaps(p, t))
        off = 0
        for lo, hi in busy:
            if off + up(t.size) <= lo:
                break
            off = max(off, hi)
        offsets[t.name] = off
        placed.append(t)
    arena = max((offsets[t.name] + up(t.size) for t in ts), default=0)
    return offsets, arena


def verify(ts: List[Tensor], offsets: Dict[str, int], align: int = 1) -> bool:
    up = lambda v: -(-v // align) * align
    for i, a in enumerate(ts):
        if offsets[a.name] % align:
            return False
        for b in ts[i + 1:]:
            if overlaps(a, b):
                if not (offsets[a.name] + up(a.size) <= offsets[b.name] or offsets[b.name] + up(b.size) <= offsets[a.name]):
                    return False
    return True


# ---------------------------------------------------------------- graph builders (activation tensors only, INT8)
def chain(sizes: List[int], inplace: bool = False) -> List[Tensor]:
    """sizes[i] = bytes of activation i; op i maps activation i -> i+1."""
    n = len(sizes) - 1
    ts = []
    for i, s in enumerate(sizes):
        birth = 0 if i == 0 else ((i - 1) if not inplace else i)   # output of op i-1: born at i-1, or at i if in place
        death = (i + 1) if i < n else n + 1
        ts.append(Tensor(f"a{i}", s, birth, death))
    return ts


def mlp_activations(batch: int, dims: List[int], inplace=False) -> List[Tensor]:
    return chain([batch * d for d in dims], inplace)


def transformer_block(seq: int, d: int, ffn: int = 4, heads_scores: bool = True, inplace=False) -> List[Tensor]:
    """One decoder-style block, INT8 activations, ops 0..9 (QKV, scores, softmax, ctx, outproj, add, ffn1, act, ffn2, add).
    Lifetimes are ILLUSTRATIVE hand-set intervals for a standard block, not extracted from a real compiler graph."""
    sc = seq * seq
    f = lambda name, size, b, dth: Tensor(name, size, b, dth)
    ip = 1 if inplace else 0                                  # in-place capable ops: softmax(2), act(7), adds(5, 9)
    return [
        f("x", seq * d, 0, 6),                                # residual input, last read by add (op 5)
        f("q", seq * d, 0, 2), f("k", seq * d, 0, 2), f("v", seq * d, 0, 4),
        f("scores", sc, 1, 3 if not ip else 3),
        f("probs", sc, 2 + ip, 4),
        f("ctx", seq * d, 3, 5),
        f("attn_out", seq * d, 4, 6),
        f("y", seq * d, 5 + ip, 10),                          # residual 1, kept for the second add (op 9)
        f("h", seq * d * ffn, 6, 8 + 0),
        f("h_act", seq * d * ffn, 7 + ip, 9),
        f("z", seq * d, 8, 10),
        f("out", seq * d, 9 + ip, 11),
    ]
