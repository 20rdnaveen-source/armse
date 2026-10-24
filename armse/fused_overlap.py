"""Fused GEMM chains WITH cross-phase / cross-block overlap (MODEL ONLY, extends armse/fused.py).

fused.py ran every phase of every row block as a separate program behind a barrier.  Here ONE continuous
program is built: a chain of segments [block0.phase1, block0.phase2, block1.phase1, ...] that share the slot
state (A/B/O ping-pong slots, last-use tracking, output-tile counter), so the DMA engine can prefetch the
next segment's weights while the current segment computes.  Writebacks are queued after the next segment's
first loads (same rule as inside one segment), so they never block prefetch.
Constraint: all segments use the same (nbuf_in, nbuf_out) so the slot layout is common.
Legality is proven by the same deadlock + hazard checkers; numerical results by a segment-aware replay."""
from dataclasses import dataclass
import numpy as np
from .config import HwConfig, Workload, NoFitError, cdiv
from .schedule import (Schedule, Program, DmaOp, CmpOp, LA, LB, WB, make_layout, iteration_order, slot_words,
                       violations)
from .sim import schedule_times, check_hazards, metrics_of
from .fused import Chain, requant, inter_words, _best, chain_reference
from .sim import reference


@dataclass
class Seg:
    wl: Workload
    hw: HwConfig
    sch: Schedule
    a_resident: bool
    no_wb: bool
    row0: int          # first row of this block in the full M dimension
    phase: int         # 1 or 2


def build_chain_program(segs, hw_main: HwConfig) -> Program:
    nin, nout = segs[0].sch.nbuf_in, segs[0].sch.nbuf_out
    if any(sg.sch.nbuf_in != nin or sg.sch.nbuf_out != nout for sg in segs):
        raise ValueError("all segments must share nbuf_in / nbuf_out")
    prog = Program(segs[0].wl, hw_main, segs[0].sch, layout=make_layout(segs[0].wl))
    aslot = bslot = -1
    last_a, last_b = [-1] * nin, [-1] * nin
    a_idx = b_idx = -1
    q = -1
    wb_idx = {}                      # tile ordinal -> dma index (only tiles that are written back)
    pending = None                   # (qq, seg_id, mi, ni, rows, cols, oslot, ob, tm, tn) awaiting its writeback
    git = 0                          # global compute index
    for sid, sg in enumerate(segs):
        wl, sch, hw = sg.wl, sg.sch, sg.hw
        if violations(wl, hw, sch):
            raise NoFitError("; ".join(violations(wl, hw, sch)))
        M, N, K = wl.M, wl.N, wl.K
        tm, tn, tk = sch.tm, sch.tn, sch.tk
        nm, nn, nk = cdiv(M, tm), cdiv(N, tn), cdiv(K, tk)
        sa, sb, so = slot_words(hw, sch)
        a_id = b_id = None
        lay = make_layout(wl)
        for (mi, ni, ki) in iteration_order(nm, nn, nk, sch.order):
            mr, nc, kk = min(tm, M - mi * tm), min(tn, N - ni * tn), min(tk, K - ki * tk)
            kg = cdiv(kk, hw.lanes)
            first = ki == 0
            if sg.a_resident:
                aslot, a_idx, a_id = 0, -1, (mi, ki)
            elif (mi, ki) != a_id:
                aslot = 0 if aslot < 0 else (aslot + 1) % nin
                prog.dma.append(DmaOp(LA, last_a[aslot] + 1, mr, kk, lay.a_off + mi * tm * K + ki * tk, K,
                                      aslot * sa, kg, (mi, ni, ki), aslot, 1, sid))
                a_idx, a_id = len(prog.dma) - 1, (mi, ki)
            if (ki, ni) != b_id:
                bslot = 0 if bslot < 0 else (bslot + 1) % nin
                prog.dma.append(DmaOp(LB, last_b[bslot] + 1, kk, nc, lay.b_off + ki * tk * N + ni * tn, N,
                                      bslot * sb, kg, (mi, ni, ki), bslot, 1, sid))
                b_idx, b_id = len(prog.dma) - 1, (ki, ni)
            if not sg.a_resident:
                last_a[aslot] = git
            last_b[bslot] = git
            if first and pending is not None:                 # queue previous tile's writeback behind these loads
                qq, ps, pmi, pni, prow, pcol, pos, pob, ptm, ptn = pending
                prog.dma.append(DmaOp(WB, git, prow, pcol, 0, pob * segs[ps].wl.N, pos * (hw_main.o_words // nout),
                                      pcol, (pmi, pni, -1), pos, pob, ps))
                wb_idx[qq] = len(prog.dma) - 1
                pending = None
            odep = 0
            if first:
                q += 1
                oslot = q % nout
                if q >= nout and (q - nout) in wb_idx:
                    odep = wb_idx[q - nout] + 1
                cur = (q, oslot)
                tile_info = (q, sid, mi, ni, mr, nc, oslot)
            dep = max(a_idx, b_idx) + 1
            if first:
                dep = max(dep, odep)
            last_k = ki == nk - 1
            extra = cdiv(mr * nc, hw.lanes) if (sg.no_wb and last_k) else 0
            prog.cmp.append(CmpOp(mr, nc, kk, first, dep, aslot * sa if not sg.a_resident else 0, bslot * sb,
                                  cur[1] * so, (mi, ni, ki), aslot, bslot, cur[1], a_idx, b_idx, cur[0], extra, sid))
            git += 1
            if last_k and not sg.no_wb:
                qq, s_, mi_, ni_, mr_, nc_, os_ = tile_info
                pending = (qq, s_, mi_, ni_, mr_, nc_, os_, wl.out_bytes, tm, tn)
    if pending is not None:
        qq, ps, pmi, pni, prow, pcol, pos, pob, ptm, ptn = pending
        prog.dma.append(DmaOp(WB, git, prow, pcol, 0, pob * segs[ps].wl.N, pos * (hw_main.o_words // nout),
                              pcol, (pmi, pni, -1), pos, pob, ps))
        wb_idx[qq] = len(prog.dma) - 1
    prog.wb_of = wb_idx
    prog.sch = Schedule(1, 1, 1, "MN", nin, nout)                # only nbuf is used by the hazard checker
    return prog


def functional_chain(prog, segs, ch: Chain, data):
    """Segment-aware replay -> C2 (M x N2). Loads land at completion, computes read at start / write at end."""
    A, B1, B2 = data
    ds, de, cs, ce = schedule_times(prog)
    C2 = np.zeros((ch.M, ch.N2), np.int32)
    C1 = {}
    bufA, bufB, bufO, pend = {}, {}, {}, {}
    ev = []
    for i, op in enumerate(prog.dma):
        ev.append((ds[i], 1, i, "W") if op.kind == WB else (de[i], 0, i, "L"))
    for j in range(len(prog.cmp)):
        ev += [(cs[j], 1, j, "Cs"), (ce[j], 0, j, "Ce")]
    ev.sort(key=lambda e: (e[0], e[1], e[3], e[2]))

    def a_of(sid):                      # only called when the A operand is actually needed
        sg = segs[sid]
        if sg.phase == 1:
            return A[sg.row0:sg.row0 + sg.wl.M]
        return requant(C1[sid - 1], ch.shift)

    def b_of(sid):
        return B1 if segs[sid].phase == 1 else B2

    for _, _, i, kind in ev:
        if kind == "L":
            op = prog.dma[i]
            sg = segs[op.seg]
            mi, ni, ki = op.tile
            tm, tn, tk = sg.sch.tm, sg.sch.tn, sg.sch.tk
            if op.kind == LA:
                bufA[op.slot] = a_of(op.seg)[mi * tm:mi * tm + op.rows, ki * tk:ki * tk + op.cols].astype(np.int32)
            else:
                bufB[op.slot] = b_of(op.seg)[ki * tk:ki * tk + op.rows, ni * tn:ni * tn + op.cols].astype(np.int32)
        elif kind == "Cs":
            c = prog.cmp[i]
            sg = segs[c.seg]
            mi, ni, ki = c.tile
            tm, tk = sg.sch.tm, sg.sch.tk
            if c.a_dma < 0:
                atile = a_of(c.seg)[mi * tm:mi * tm + c.tm, ki * tk:ki * tk + c.tk].astype(np.int32)
            else:
                atile = bufA[c.a_slot]
            pend[i] = (atile @ bufB[c.b_slot], None if c.first else bufO[c.o_slot].copy())
        elif kind == "Ce":
            c = prog.cmp[i]
            sg = segs[c.seg]
            part, base = pend.pop(i)
            bufO[c.o_slot] = part if base is None else base + part
            if sg.no_wb and c.tile[2] == cdiv(sg.wl.K, sg.sch.tk) - 1:
                mi, ni, _ = c.tile
                C1.setdefault(c.seg, np.zeros((sg.wl.M, sg.wl.N), np.int32))[
                    mi * sg.sch.tm:mi * sg.sch.tm + c.tm, ni * sg.sch.tn:ni * sg.sch.tn + c.tn] = bufO[c.o_slot]
        else:
            op = prog.dma[i]
            sg = segs[op.seg]
            mi, ni, _ = op.tile
            r0, c0 = sg.row0 + mi * sg.sch.tm, ni * sg.sch.tn
            C2[r0:r0 + op.rows, c0:c0 + op.cols] = bufO[op.slot]
    return C2


def _segments(ch, hw, tm, nin, nout):
    blocks = [tm] * (ch.M // tm) + ([ch.M % tm] if ch.M % tm else [])
    iw = inter_words(ch, hw, tm)
    if hw.a_depth - iw < 1:
        raise NoFitError("intermediate block does not fit in the A buffer")
    hw1 = hw.replace(a_depth=hw.a_depth - iw)
    hw2 = hw.replace(a_depth=10 ** 6)
    pick, segs = {}, []
    for bs in blocks:
        if bs not in pick:
            w1 = Workload(ch.name + "_o1", bs, ch.N1, ch.K1, "chain", out_bytes=1)
            w2 = Workload(ch.name + "_o2", bs, ch.N2, ch.N1, "chain")
            f = lambda s: s.nbuf_in == nin and s.nbuf_out == nout
            s1 = _best(w1, hw1, dict(no_wb=True), filt=f)
            s2 = _best(w2, hw2, dict(a_resident=True), filt=lambda s: f(s) and (s.tk % hw.lanes == 0 or s.tk == ch.N1))
            pick[bs] = (w1, s1, w2, s2)
    row = 0
    for bs in blocks:
        w1, s1, w2, s2 = pick[bs]
        segs.append(Seg(w1, hw1, s1, False, True, row, 1))
        segs.append(Seg(w2, hw2, s2, True, False, row, 2))
        row += bs
    return segs


def run_overlap(ch: Chain, hw: HwConfig, tm: int, nin: int, nout: int, data=None):
    segs = _segments(ch, hw, tm, nin, nout)
    prog = build_chain_program(segs, hw)
    ds, de, cs, ce = schedule_times(prog)
    check_hazards(prog, ds, de, cs, ce)
    m = metrics_of(prog, ds, de, cs, ce)
    out = dict(cycles=m["cycles"], bytes_rd=m["bytes_rd"], bytes_wr=m["bytes_wr"], mem_stall=m["mem_stall"],
               tm=tm, nbuf=f"{nin}/{nout}", nblocks=len(segs) // 2, C2=None)
    if data is not None:
        out["C2"] = functional_chain(prog, segs, ch, data)
    return out


def best_overlap(ch: Chain, hw: HwConfig, max_cands=6):
    from .fused import tm_candidates
    cands = tm_candidates(ch, hw)
    if len(cands) > max_cands:
        cands = sorted({cands[round(i * (len(cands) - 1) / (max_cands - 1))] for i in range(max_cands)})
    best = None
    for tm in cands:
        for nin, nout in ((1, 1), (2, 1), (2, 2), (1, 2)):
            try:
                r = run_overlap(ch, hw, tm, nin, nout)
            except NoFitError:
                continue
            if best is None or (r["cycles"], tm, r["nbuf"]) < (best["cycles"], best["tm"], best["nbuf"]):
                best = r
    if best is None:
        raise NoFitError("no overlapped fused configuration fits")
    return best
