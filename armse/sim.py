"""Event-driven executor for ARMSE programs.
 
Semantics (mirrors rtl/, see docs/ARCHITECTURE.md section 6):
  * one in-order DMA engine and one in-order compute engine
  * a job starts at the first cycle where its dependency counter is visible and the engine is free
  * a job's completion becomes visible `duration` cycles after it starts
Durations:
  DMA     = job_ovh + rows * (row_fix + beats_per_row)
  compute = tm * tn * ceil(tk/lanes) * conflict_factor + cmp_ovh
The simulator also (a) detects deadlock, (b) checks buffer hazards from the produced timing, and
(c) optionally executes the data movement functionally with numpy so results can be compared with
an independent reference.
"""
from dataclasses import dataclass
from math import gcd
from typing import Optional
import numpy as np
from .config import (HwConfig, cdiv, DeadlockError, HazardError)
from .schedule import Program, LA, LB, WB
 
 
def dma_beats(hw: HwConfig, op) -> int:
    if op.ebits:                                   # packed (compressed) elements
        return cdiv(op.cols * op.ebits, 8 * hw.bus_bytes)
    return cdiv(op.cols * op.ebytes, hw.bus_bytes)
 
 
def load_bytes(op) -> int:
    """External bytes of a load job (compressed loads also carry a 2-byte scale per column)."""
    if op.ebits:
        return cdiv(op.rows * op.cols * op.ebits, 8) + 2 * op.cols
    return op.rows * op.cols
 
 
def dma_dur(hw: HwConfig, op) -> int:
    extra = cdiv(2 * op.cols, hw.bus_bytes) if op.ebits else 0          # scale vector fetched with the tile
    return hw.job_ovh + op.rows * (hw.row_fix + dma_beats(hw, op)) + extra
 
 
def conflict_factor(hw: HwConfig, tn: int) -> int:
    """Serialisation of a B-lane read for the naive (row-major [k][j]) layout: gcd(tn, banks)."""
    return gcd(tn, hw.lanes) if hw.bank_mode == "naive" else 1
 
 
def cmp_issue(hw: HwConfig, op) -> int:
    return op.tm * op.tn * cdiv(op.tk, hw.lanes) * conflict_factor(hw, op.tn)
 
 
def cmp_dur(hw: HwConfig, op) -> int:
    return cmp_issue(hw, op) + hw.cmp_ovh + op.extra
 
 
@dataclass
class SimResult:
    cycles: int
    ds: list
    de: list
    cs: list
    ce: list
    metrics: dict
    C: Optional[np.ndarray] = None
 
 
def schedule_times(prog: Program):
    hw = prog.hw
    nd, nc = len(prog.dma), len(prog.cmp)
    ds, de, cs, ce = [0] * nd, [0] * nd, [0] * nc, [0] * nc
    di = ci = 0
    dfree = cfree = hw.start_off
    blk = hw.prog_fetch_block
    fetch = 0
    if blk and nd:                      # one burst per block: blk DMA entries + proportional compute entries
        fetch = hw.job_ovh + hw.row_fix + cdiv(16 * (blk + cdiv(blk * nc, nd)), hw.bus_bytes)
    while di < nd or ci < nc:
        moved = False
        if di < nd:
            op = prog.dma[di]
            dep = op.dep_cp
            if dep <= ci:
                t = ce[dep - 1] if dep > 0 else 0
                s = dfree if dfree > t else t
                ds[di], de[di] = s, s + dma_dur(hw, op) + (fetch if blk and di % blk == 0 else 0)
                dfree = de[di]
                di += 1
                moved = True
        if ci < nc:
            op = prog.cmp[ci]
            dep = op.dep_dma
            if dep <= di:
                t = de[dep - 1] if dep > 0 else 0
                s = cfree if cfree > t else t
                cs[ci], ce[ci] = s, s + cmp_dur(hw, op)
                cfree = ce[ci]
                ci += 1
                moved = True
        if not moved:
            raise DeadlockError(f"deadlock: dma index {di}/{nd}, compute index {ci}/{nc}")
    return ds, de, cs, ce
 
 
def check_hazards(prog: Program, ds, de, cs, ce):
    """Verify buffer ownership from timing. Raises HazardError on the first violation."""
    nxt = {}
    for kind in (LA, LB):
        last = {}
        for i, op in enumerate(prog.dma):
            if op.kind == kind:
                if op.slot in last:
                    nxt[last[op.slot]] = i
                last[op.slot] = i
    for j, c in enumerate(prog.cmp):
        for idx in (c.a_dma, c.b_dma):
            if idx < 0:
                continue                                  # resident operand (fused mode)
            if cs[j] < de[idx]:
                raise HazardError(f"compute {j} starts before its operand load {idx} completed")
            n = nxt.get(idx)
            if n is not None and ds[n] < ce[j]:
                raise HazardError(f"load {n} overwrites slot still read by compute {j}")
    first, lastc = {}, {}
    for j, c in enumerate(prog.cmp):
        first.setdefault(c.q, j)
        lastc[c.q] = j
    nb = prog.sch.nbuf_out
    for q, w in prog.wb_of.items():
        if ds[w] < ce[lastc[q]]:
            raise HazardError(f"writeback of tile {q} starts before its last compute ended")
        q2 = q + nb
        if q2 in first and cs[first[q2]] < de[w]:
            raise HazardError(f"tile {q2} computes into output slot before writeback of tile {q} done")
 
 
def metrics_of(prog: Program, ds, de, cs, ce) -> dict:
    hw, wl = prog.hw, prog.wl
    nd, nc = len(prog.dma), len(prog.cmp)
    if nd == 0 and nc == 0:
        return dict(cycles=0, compute_busy=0, compute_issue=0, mem_stall=0, tail=0, dma_busy=0,
                    dma_sync=0, dma_latency=0, dma_data=0, bytes_rd=0, bytes_wr=0, transfers=0,
                    macs=0, mac_util=0.0, bw_sustained=0.0, buf_util=0.0, program_bytes=0,
                    bank_serial_cycles=0, loads=0)
    total = max(de[-1] if nd else 0, ce[-1] if nc else 0)
    stall, prev = 0, hw.start_off
    for j in range(nc):
        stall += cs[j] - prev
        prev = ce[j]
    brd = sum(load_bytes(op) for op in prog.dma if op.kind != WB)
    bwr = sum(op.ebytes * op.rows * op.cols for op in prog.dma if op.kind == WB)
    issue = sum(cmp_issue(hw, c) for c in prog.cmp)
    ideal_issue = sum(c.tm * c.tn * cdiv(c.tk, hw.lanes) for c in prog.cmp)
    s = prog.sch
    kg = cdiv(s.tk, hw.lanes)
    used = hw.lanes * (s.tm * kg * s.nbuf_in + s.tn * kg * s.nbuf_in) + 4 * s.tm * s.tn * s.nbuf_out
    dbusy = sum(dma_dur(hw, op) for op in prog.dma)
    return dict(
        cycles=total,
        compute_busy=sum(cmp_dur(hw, c) for c in prog.cmp),
        compute_issue=issue,
        mem_stall=stall,
        tail=total - ce[-1],
        dma_busy=dbusy,
        dma_sync=hw.job_ovh * nd,
        dma_latency=hw.row_fix * sum(op.rows for op in prog.dma),
        dma_data=sum(op.rows * dma_beats(hw, op) for op in prog.dma),
        bytes_rd=brd, bytes_wr=bwr, transfers=nd,
        loads=sum(1 for op in prog.dma if op.kind != WB),
        macs=wl.macs,
        mac_util=wl.macs / (hw.lanes * total) if total else 0.0,
        bw_sustained=(brd + bwr) / total if total else 0.0,
        buf_util=used / hw.onchip_bytes if hw.onchip_bytes else 0.0,
        program_bytes=16 * (nd + nc),
        bank_serial_cycles=issue - ideal_issue,
    )
 
 
def functional_run(prog: Program, A: np.ndarray, B: np.ndarray, ds, de, cs, ce) -> np.ndarray:
    """Replay data movement in time order: loads land at completion, computes read at start and
    write at end, writebacks read at start.  Wrong dependencies give wrong numerical results."""
    wl, sch = prog.wl, prog.sch
    tm, tn, tk = sch.tm, sch.tn, sch.tk
    C = np.zeros((wl.M, wl.N), dtype=np.int32)
    bufA, bufB, bufO = {}, {}, {}
    ev = []
    for i, op in enumerate(prog.dma):
        if op.kind == WB:
            ev.append((ds[i], 1, i, "W"))
        else:
            ev.append((de[i], 0, i, "L"))
    for j in range(len(prog.cmp)):
        ev.append((cs[j], 1, j, "Cs"))
        ev.append((ce[j], 0, j, "Ce"))
    ev.sort(key=lambda e: (e[0], e[1], e[3], e[2]))
    pend = {}
    for _, _, i, kind in ev:
        if kind == "L":
            op = prog.dma[i]
            mi, ni, ki = op.tile
            if op.kind == LA:
                m0, k0 = mi * tm, ki * tk
                bufA[op.slot] = A[m0:m0 + op.rows, k0:k0 + op.cols].astype(np.int32)
            else:
                k0, n0 = ki * tk, ni * tn
                bufB[op.slot] = B[k0:k0 + op.rows, n0:n0 + op.cols].astype(np.int32)
        elif kind == "Cs":
            c = prog.cmp[i]
            if c.a_dma < 0:                               # resident A operand (fused phase 2)
                mi_, _, ki_ = c.tile
                atile = A[mi_ * tm:mi_ * tm + c.tm, ki_ * tk:ki_ * tk + c.tk].astype(np.int32)
            else:
                atile = bufA[c.a_slot]
            part = atile @ bufB[c.b_slot]
            base = None if c.first else bufO[c.o_slot].copy()
            pend[i] = (part, base)
        elif kind == "Ce":
            c = prog.cmp[i]
            part, base = pend.pop(i)
            bufO[c.o_slot] = part if base is None else base + part
            if prog.no_wb and c.tile[2] == cdiv(wl.K, tk) - 1:   # outputs stay on chip: capture them
                mi_, ni_, _ = c.tile
                C[mi_ * tm:mi_ * tm + c.tm, ni_ * tn:ni_ * tn + c.tn] = bufO[c.o_slot]
        else:
            op = prog.dma[i]
            mi, ni, _ = op.tile
            m0, n0 = mi * tm, ni * tn
            C[m0:m0 + op.rows, n0:n0 + op.cols] = bufO[op.slot]
    return C
 
 
def simulate(prog: Program, A=None, B=None, functional=False, check=True) -> SimResult:
    ds, de, cs, ce = schedule_times(prog)
    if check:
        check_hazards(prog, ds, de, cs, ce)
    m = metrics_of(prog, ds, de, cs, ce)
    C = None
    if functional:
        if A is None or B is None:
            raise ValueError("functional simulation needs A and B")
        C = (np.zeros((prog.wl.M, prog.wl.N), dtype=np.int32)
             if not prog.cmp else functional_run(prog, A, B, ds, de, cs, ce))
    return SimResult(m["cycles"], ds, de, cs, ce, m, C)
 
 
def reference(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Independent reference: plain int64 matmul reduced to int32 (two's-complement wrap)."""
    return (A.astype(np.int64) @ B.astype(np.int64)).astype(np.int32)
 
 
def random_operands(wl, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.integers(-128, 128, size=(wl.M, wl.K), dtype=np.int64).astype(np.int8)
    B = rng.integers(-128, 128, size=(wl.K, wl.N), dtype=np.int64).astype(np.int8)
    return A, B
 