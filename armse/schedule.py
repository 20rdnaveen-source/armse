"""Schedules and descriptor-program generation (decoupled DMA / compute programs).

A Schedule is (tile shape, loop order, buffering).  build_program() lowers it to two in-order
programs that the RTL executes:

  * DMA program    : LOAD_A / LOAD_B / WRITEBACK jobs, each waiting for `dep_cp`  completed computes
  * compute program: tile MAC jobs,                 each waiting for `dep_dma` completed DMA jobs

Legality (slot reuse, output-slot reuse) is encoded entirely in those two counters, so the hardware
only needs two counters and two comparators.  The simulator proves legality (deadlock + hazard check).
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from .config import (HwConfig, Workload, ScheduleError, UnsupportedOp, cdiv)

LA, LB, WB = 0, 1, 2
KIND_NAME = {LA: "LOAD_A", LB: "LOAD_B", WB: "WRITEBACK"}


@dataclass(frozen=True)
class Schedule:
    tm: int
    tn: int
    tk: int
    order: str = "MN"       # loop nest outer->inner:  MN = for m, for n, for k ; NM = for n, for m, for k
    nbuf_in: int = 1        # slots in A and B buffers (1 = single, 2 = ping-pong)
    nbuf_out: int = 1       # slots in output buffer
    label: str = ""

    def key(self):
        return (self.tm, self.tn, self.tk, self.order, self.nbuf_in, self.nbuf_out)


@dataclass
class DmaOp:
    kind: int
    dep_cp: int
    rows: int
    cols: int
    ext_addr: int
    ext_stride: int
    base: int
    stride: int
    tile: Tuple[int, int, int]
    slot: int
    ebytes: int = 1        # external bytes per column element (loads 1, writeback = out_bytes)
    seg: int = 0           # segment id (fused-overlap programs only)
    ebits: int = 0         # >0: packed external element width in bits (compressed B loads), else ebytes applies


@dataclass
class CmpOp:
    tm: int
    tn: int
    tk: int
    first: bool
    dep_dma: int
    a_base: int
    b_base: int
    o_base: int
    tile: Tuple[int, int, int]
    a_slot: int
    b_slot: int
    o_slot: int
    a_dma: int      # index of the DMA op that produced the A tile used here
    b_dma: int
    q: int          # output-tile ordinal
    extra: int = 0  # extra compute-engine cycles (requantisation pass in fused mode)
    seg: int = 0    # segment id (fused-overlap programs only)


@dataclass
class Layout:
    a_off: int
    b_off: int
    c_off: int
    ext_bytes: int


@dataclass
class Program:
    wl: Workload
    hw: HwConfig
    sch: Schedule
    dma: List[DmaOp] = field(default_factory=list)
    cmp: List[CmpOp] = field(default_factory=list)
    layout: Optional[Layout] = None
    wb_of: dict = field(default_factory=dict)     # q -> dma index of its writeback
    no_wb: bool = False                           # fused phase 1: outputs stay on chip (requantised)
    a_resident: bool = False                      # fused phase 2: A operand already on chip


def _al4(x):
    return (x + 3) & ~3


def make_layout(wl: Workload) -> Layout:
    a = 0
    b = _al4(wl.M * wl.K)
    c = _al4(b + wl.K * wl.N)
    return Layout(a, b, c, c + wl.out_bytes * wl.M * wl.N + 8)


def check_op(wl: Workload):
    if wl.op != "gemm":
        raise UnsupportedOp(f"unsupported operator '{wl.op}' (only lowered GEMM is implemented)")
    if min(wl.M, wl.N, wl.K) < 0:
        raise ScheduleError("negative tensor dimension")


def slot_words(hw: HwConfig, sch: Schedule):
    return hw.a_depth // sch.nbuf_in, hw.b_depth // sch.nbuf_in, hw.o_words // sch.nbuf_out


def violations(wl: Workload, hw: HwConfig, sch: Schedule) -> List[str]:
    v = []
    if sch.order not in ("MN", "NM"):
        v.append("order must be MN or NM")
    if sch.nbuf_in not in (1, 2) or sch.nbuf_out not in (1, 2):
        v.append("nbuf must be 1 or 2")
    if min(sch.tm, sch.tn, sch.tk) < 1:
        v.append("tile dims must be >= 1")
        return v
    if max(sch.tm, sch.tn, sch.tk) > hw.max_dim:
        v.append(f"tile dim exceeds descriptor field limit {hw.max_dim}")
    if sch.tm > wl.M or sch.tn > wl.N or sch.tk > wl.K:
        v.append("tile larger than tensor dimension")
    if v:
        return v
    kg = cdiv(sch.tk, hw.lanes)
    sa, sb, so = slot_words(hw, sch)
    iters = cdiv(wl.M, sch.tm) * cdiv(wl.N, sch.tn) * cdiv(wl.K, sch.tk)
    if 3 * iters > hw.max_entries:
        v.append("program too long for the dependency counters")
    if sch.tm * kg > sa:
        v.append(f"A tile needs {sch.tm * kg} words/bank, slot has {sa}")
    if sch.tn * kg > sb:
        v.append(f"B tile needs {sch.tn * kg} words/bank, slot has {sb}")
    if sch.tm * sch.tn > so:
        v.append(f"O tile needs {sch.tm * sch.tn} words, slot has {so}")
    return v


def is_legal(wl, hw, sch) -> bool:
    return not violations(wl, hw, sch)


def iteration_order(nm, nn, nk, order):
    if order == "MN":
        return [(m, n, k) for m in range(nm) for n in range(nn) for k in range(nk)]
    return [(m, n, k) for n in range(nn) for m in range(nm) for k in range(nk)]


def build_program(wl: Workload, hw: HwConfig, sch: Schedule, a_resident: bool = False,
                  no_wb: bool = False) -> Program:
    """a_resident / no_wb are MODEL-ONLY fusion modes (no RTL support): see armse/fused.py."""
    check_op(wl)
    if wl.out_bytes not in (1, 4):
        raise ScheduleError("out_bytes must be 1 or 4")
    if wl.weight_bits not in (4, 8):
        raise ScheduleError("weight_bits must be 4 or 8")
    lay = make_layout(wl)
    prog = Program(wl, hw, sch, layout=lay, no_wb=no_wb, a_resident=a_resident)
    M, N, K = wl.M, wl.N, wl.K
    if M == 0 or N == 0 or K == 0:
        return prog                       # empty workload: no work, C stays zero
    v = violations(wl, hw, sch)
    if v:
        raise ScheduleError("; ".join(v))
    tm, tn, tk = sch.tm, sch.tn, sch.tk
    nm, nn, nk = cdiv(M, tm), cdiv(N, tn), cdiv(K, tk)
    sa, sb, so = slot_words(hw, sch)
    size = lambda i, t, D: min(t, D - i * t)

    a_id = b_id = None
    aslot = bslot = -1
    last_a = [-1] * sch.nbuf_in
    last_b = [-1] * sch.nbuf_in
    a_idx = b_idx = -1
    q = -1
    cur = None                            # (q, mi, ni, rows, cols, oslot)

    def emit_wb(c, dep):
        qq, mi, ni, rows, cols, oslot = c
        ob = wl.out_bytes
        prog.dma.append(DmaOp(WB, dep, rows, cols,
                              lay.c_off + ob * (mi * tm * N + ni * tn), ob * N,
                              oslot * so, cols, (mi, ni, -1), oslot, ob))
        prog.wb_of[qq] = len(prog.dma) - 1

    for it, (mi, ni, ki) in enumerate(iteration_order(nm, nn, nk, sch.order)):
        mr, nc, kk = size(mi, tm, M), size(ni, tn, N), size(ki, tk, K)
        kg = cdiv(kk, hw.lanes)
        first = ki == 0
        if a_resident:
            aslot, a_idx, a_id = 0, -1, (mi, ki)
        elif (mi, ki) != a_id:
            aslot = 0 if aslot < 0 else (aslot + 1) % sch.nbuf_in
            prog.dma.append(DmaOp(LA, last_a[aslot] + 1, mr, kk,
                                  lay.a_off + mi * tm * K + ki * tk, K,
                                  aslot * sa, kg, (mi, ni, ki), aslot))
            a_idx, a_id = len(prog.dma) - 1, (mi, ki)
        if (ki, ni) != b_id:
            bslot = 0 if bslot < 0 else (bslot + 1) % sch.nbuf_in
            prog.dma.append(DmaOp(LB, last_b[bslot] + 1, kk, nc,
                                  lay.b_off + ki * tk * N + ni * tn, N,
                                  bslot * sb, kg, (mi, ni, ki), bslot, 1, 0, 0 if wl.weight_bits == 8 else wl.weight_bits))
            b_idx, b_id = len(prog.dma) - 1, (ki, ni)
        last_a[aslot] = it
        last_b[bslot] = it
        if first and it > 0 and not no_wb:
            emit_wb(cur, it)              # previous tile's last iteration is it-1 -> need `it` computes done
        odep = 0
        if first:
            q += 1
            oslot = q % sch.nbuf_out
            if q >= sch.nbuf_out and not no_wb:
                odep = prog.wb_of[q - sch.nbuf_out] + 1
            cur = (q, mi, ni, mr, nc, oslot)
        dep = max(a_idx, b_idx) + 1
        if first:
            dep = max(dep, odep)
        prog.cmp.append(CmpOp(mr, nc, kk, first, dep, aslot * sa, bslot * sb, cur[5] * so,
                              (mi, ni, ki), aslot, bslot, cur[5], a_idx, b_idx, cur[0],
                              cdiv(mr * nc, hw.lanes) if (no_wb and ki == nk - 1) else 0))
    if not no_wb:
        emit_wb(cur, len(prog.cmp))
    if len(prog.dma) > hw.max_entries or len(prog.cmp) > hw.max_entries:
        raise ScheduleError("program too long for the dependency counters")
    return prog
