"""Fused GEMM-chain extension (MODEL ONLY - no RTL support).

Chain:  C1 = requant(A @ B1)  (INT8, arithmetic shift + saturate),   C2 = C1 @ B2  (INT32)
  attention:  A=Q (seq x d), B1=K^T (d x seq), B2=V (seq x d)   [softmax between the GEMMs is NOT modelled]
  MLP pair :  A=activations, B1/B2 = consecutive weight matrices [activation function not modelled]

Unfused : two independent programs; C1 (int8) is written to and re-read from external memory.
Fused   : the M dimension is split into row blocks of `tm` rows. For each block
            phase 1 computes the block's C1 rows (tm x N1) and requantises them into an on-chip
            intermediate region carved out of the A buffer (tm*ceil(N1/lanes) words per bank),
            phase 2 uses that region as its resident A operand and streams B2 tiles.
          Intermediate traffic disappears, but B1 and B2 are re-streamed once per row block.
Conservative modelling choices (all stated in the report):
  * every phase is a separate program with a barrier before it (no overlap across phases/blocks)
  * requantisation costs ceil(tm_tile*tn_tile/lanes) extra compute-engine cycles per output tile
  * phase-2 resident-operand addressing would need an independent stride field in the descriptor
"""
from dataclasses import dataclass
import numpy as np
from .config import HwConfig, Workload, NoFitError, cdiv
from .schedule import build_program
from .sim import simulate, reference
from .scheduler import ranked, policy_D_fb


@dataclass(frozen=True)
class Chain:
    name: str
    M: int
    K1: int
    N1: int
    N2: int
    shift: int = 8
    family: str = "chain"


def requant(c32: np.ndarray, shift: int) -> np.ndarray:
    return np.clip(c32 >> shift, -128, 127).astype(np.int8)


def chain_reference(A, B1, B2, shift=8):
    return reference(requant(reference(A, B1), shift), B2)


def chain_data(ch: Chain, seed=0):
    rng = np.random.default_rng(seed)
    g = lambda r, c: rng.integers(-128, 128, size=(r, c), dtype=np.int64).astype(np.int8)
    return g(ch.M, ch.K1), g(ch.K1, ch.N1), g(ch.N1, ch.N2)


# --------------------------------------------------------------------------- unfused
def run_unfused(ch: Chain, hw: HwConfig, data=None, k=8):
    w1 = Workload(ch.name + "_g1", ch.M, ch.N1, ch.K1, "chain", out_bytes=1)
    w2 = Workload(ch.name + "_g2", ch.M, ch.N2, ch.N1, "chain")
    p1 = build_program(w1, hw, policy_D_fb(w1, hw, k))
    p2 = build_program(w2, hw, policy_D_fb(w2, hw, k))
    if data is None:
        r1, r2 = simulate(p1), simulate(p2)
        C2 = None
    else:
        A, B1, B2 = data
        r1 = simulate(p1, A, B1, functional=True)
        r2 = simulate(p2, requant(r1.C, ch.shift), B2, functional=True)
        C2 = r2.C
    return dict(cycles=r1.cycles + r2.cycles, bytes_rd=r1.metrics["bytes_rd"] + r2.metrics["bytes_rd"],
                bytes_wr=r1.metrics["bytes_wr"] + r2.metrics["bytes_wr"], C2=C2,
                mem_stall=r1.metrics["mem_stall"] + r2.metrics["mem_stall"],
                compute_busy=r1.metrics["compute_busy"] + r2.metrics["compute_busy"],
                sched1=w1 and policy_D_fb(w1, hw, k).key(), sched2=policy_D_fb(w2, hw, k).key())


# --------------------------------------------------------------------------- fused
def _best(wl, hw, flags, k=4, filt=None):
    cands = [s for _, s in ranked(wl, hw) if filt is None or filt(s)][:k]
    if not cands:
        raise NoFitError(f"no legal schedule for {wl.name}")
    return min(cands, key=lambda s: (simulate(build_program(wl, hw, s, **flags)).cycles, s.key()))


def inter_words(ch: Chain, hw: HwConfig, tm: int) -> int:
    return tm * cdiv(ch.N1, hw.lanes)


def _plan_block(ch, hw, tm, bs):
    iw = inter_words(ch, hw, tm)
    if hw.a_depth - iw < 1:
        raise NoFitError("intermediate block does not fit in the A buffer")
    hw1 = hw.replace(a_depth=hw.a_depth - iw)                        # phase 1 keeps what is left of A
    w1 = Workload(ch.name + "_f1", bs, ch.N1, ch.K1, "chain", out_bytes=1)
    s1 = _best(w1, hw1, dict(no_wb=True))
    p1 = build_program(w1, hw1, s1, no_wb=True)
    hw2 = hw.replace(a_depth=10 ** 6)                                 # A operand is the resident region
    w2 = Workload(ch.name + "_f2", bs, ch.N2, ch.N1, "chain")
    s2 = _best(w2, hw2, dict(a_resident=True),
               filt=lambda s: s.tk % hw.lanes == 0 or s.tk == ch.N1)  # resident chunks must be bank-aligned
    p2 = build_program(w2, hw2, s2, a_resident=True)
    return s1, p1, s2, p2


def run_fused(ch: Chain, hw: HwConfig, tm: int, data=None):
    if tm < 1 or tm > min(ch.M, hw.max_dim):
        raise NoFitError("bad block height")
    blocks = [tm] * (ch.M // tm) + ([ch.M % tm] if ch.M % tm else [])
    plans, timing = {}, {}
    cyc = brd = bwr = stall = busy = 0
    C2 = np.zeros((ch.M, ch.N2), np.int32) if data is not None else None
    for i, bs in enumerate(blocks):
        if bs not in plans:
            plans[bs] = _plan_block(ch, hw, tm, bs)
            _, p1, _, p2 = plans[bs]
            r1, r2 = simulate(p1), simulate(p2)
            timing[bs] = (r1.cycles + r2.cycles, r1.metrics["bytes_rd"] + r2.metrics["bytes_rd"],
                          r2.metrics["bytes_wr"], r1.metrics["mem_stall"] + r2.metrics["mem_stall"],
                          r1.metrics["compute_busy"] + r2.metrics["compute_busy"])
        c, r, w, st, bz = timing[bs]
        cyc, brd, bwr, stall, busy = cyc + c, brd + r, bwr + w, stall + st, busy + bz
        if data is not None:
            A, B1, B2 = data
            _, p1, _, p2 = plans[bs]
            r1 = simulate(p1, A[i * tm:i * tm + bs], B1, functional=True)
            r2 = simulate(p2, requant(r1.C, ch.shift), B2, functional=True)
            C2[i * tm:i * tm + bs] = r2.C
    s1, _, s2, _ = plans[blocks[0]]
    return dict(cycles=cyc, bytes_rd=brd, bytes_wr=bwr, mem_stall=stall, compute_busy=busy, C2=C2,
                tm=tm, nblocks=len(blocks), inter_words=inter_words(ch, hw, tm),
                sched1=s1.key(), sched2=s2.key())


def tm_candidates(ch: Chain, hw: HwConfig):
    cap = min(ch.M, hw.max_dim)
    c = {t for t in (1, 2, 4, 8, 16, 32, 64, 128) if t <= cap}
    c |= {cdiv(ch.M, n) for n in (1, 2, 3, 4, 6, 8, 12, 16)}
    return sorted(t for t in c if 1 <= t <= cap and hw.a_depth - inter_words(ch, hw, t) >= 1)


def best_fused(ch: Chain, hw: HwConfig, max_cands=6):
    best, allr = None, []
    cands = tm_candidates(ch, hw)
    if len(cands) > max_cands:                       # thin evenly, always keep smallest and largest
        cands = sorted({cands[round(i * (len(cands) - 1) / (max_cands - 1))] for i in range(max_cands)})
    for tm in cands:
        r = run_fused(ch, hw, tm)
        allr.append((tm, r["cycles"], r["bytes_rd"]))
        if best is None or (r["cycles"], tm) < (best["cycles"], best["tm"]):
            best = r
    if best is None:
        raise NoFitError("no fused configuration fits")
    best["all"] = allr
    return best


def chains():
    return [Chain("attn_s64_d32", 64, 32, 64, 32), Chain("attn_s128_d64", 128, 64, 128, 64),
            Chain("attn_s256_d64", 256, 64, 256, 64),
            Chain("mlp_b1_784_256_128", 1, 784, 256, 128), Chain("mlp_b32_784_256_128", 32, 784, 256, 128),
            Chain("mlp_b128_784_256_128", 128, 784, 256, 128), Chain("mlp_b64_128_512_128", 64, 128, 512, 128),
            Chain("wideN1_b256_64_1024_64", 256, 64, 1024, 64)]
