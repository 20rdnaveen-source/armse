"""Channel-parallel depthwise convolution mapping (MODEL ONLY).

out[m, c] = sum_k A3[m, k, c] * W[k, c]    m = output pixel, k = kernel tap (KK = kh*kw), c = channel
Lanes are spread over CHANNELS (not over K): every issue cycle one (pixel, tap) pair is processed for `lanes`
channels, so utilisation depends on the channel tile, not on KK.
Tile (tm pixels x tc channels): A tile = tm x (KK*tc) bytes (im2col per channel, window reuse NOT modelled),
weight tile = KK x tc, output tile = tm x tc INT32.
Reuse of the simulator: a CmpOp is stored as (tm, tn:=KK, tk:=tc) so cmp_issue = tm*KK*ceil(tc/lanes).
The Workload is Workload(name, M=pixels, N=channels, K=KK, 'depthwise_cp') so macs = M*N*K is the true MAC count."""
import itertools
import numpy as np
from .config import HwConfig, Workload, NoFitError, ScheduleError, cdiv
from .schedule import Schedule, Program, DmaOp, CmpOp, LA, LB, WB, make_layout, iteration_order
from .sim import schedule_times, check_hazards, metrics_of


def dw_violations(wl: Workload, hw: HwConfig, tm, tc, nin, nout):
    v = []
    if not (1 <= tm <= wl.M and 1 <= tc <= wl.N and tm <= hw.max_dim and tc <= hw.max_dim):
        v.append("bad tile")
        return v
    if tm * cdiv(wl.K * tc, hw.lanes) > hw.a_depth // nin:
        v.append("A tile too large")
    if wl.K * cdiv(tc, hw.lanes) > hw.b_depth // nin:
        v.append("B tile too large")
    if tm * tc > hw.o_words // nout:
        v.append("O tile too large")
    if 3 * cdiv(wl.M, tm) * cdiv(wl.N, tc) > hw.max_entries:
        v.append("program too long")
    return v


def build_dw_program(wl: Workload, hw: HwConfig, tm, tc, order="MN", nin=2, nout=2) -> Program:
    if wl.family != "depthwise_cp":
        raise ScheduleError("not a channel-parallel depthwise workload")
    v = dw_violations(wl, hw, tm, tc, nin, nout)
    if v:
        raise ScheduleError("; ".join(v))
    KK, M, C = wl.K, wl.M, wl.N
    sch = Schedule(tm, tc, KK, order, nin, nout, "dw_cp")
    prog = Program(wl, hw, sch, layout=make_layout(Workload(wl.name, M, 1, KK)))
    nm, nc_t = cdiv(M, tm), cdiv(C, tc)
    sa, sb, so = hw.a_depth // nin, hw.b_depth // nin, hw.o_words // nout
    last_a, last_b = [-1] * nin, [-1] * nin
    aslot = bslot = -1
    a_id = b_id = None
    a_idx = b_idx = -1
    pending = None
    for it, (mi, ni, _) in enumerate(iteration_order(nm, nc_t, 1, order)):
        mr, ncw = min(tm, M - mi * tm), min(tc, C - ni * tc)
        if (mi, ni) != a_id:
            aslot = 0 if aslot < 0 else (aslot + 1) % nin
            prog.dma.append(DmaOp(LA, last_a[aslot] + 1, mr, KK * ncw, 0, KK * C, aslot * sa, 0, (mi, ni, 0), aslot))
            a_idx, a_id = len(prog.dma) - 1, (mi, ni)
        if ni != b_id:
            bslot = 0 if bslot < 0 else (bslot + 1) % nin
            prog.dma.append(DmaOp(LB, last_b[bslot] + 1, KK, ncw, 0, C, bslot * sb, 0, (mi, ni, 0), bslot))
            b_idx, b_id = len(prog.dma) - 1, ni
        last_a[aslot] = it
        last_b[bslot] = it
        if pending is not None:
            q_, mi_, ni_, r_, c_, os_ = pending
            prog.dma.append(DmaOp(WB, it, r_, c_, 0, 4 * C, os_ * so, 0, (mi_, ni_, -1), os_, 4))
            prog.wb_of[q_] = len(prog.dma) - 1
            pending = None
        q = it                                                  # every iteration is its own output tile (nk = 1)
        oslot = q % nout
        odep = prog.wb_of[q - nout] + 1 if q >= nout else 0
        dep = max(max(a_idx, b_idx) + 1, odep)
        prog.cmp.append(CmpOp(mr, KK, ncw, True, dep, aslot * sa, bslot * sb, oslot * so, (mi, ni, 0), aslot, bslot,
                              oslot, a_idx, b_idx, q))
        pending = (q, mi, ni, mr, ncw, oslot)
    q_, mi_, ni_, r_, c_, os_ = pending
    prog.dma.append(DmaOp(WB, len(prog.cmp), r_, c_, 0, 4 * C, os_ * so, 0, (mi_, ni_, -1), os_, 4))
    prog.wb_of[q_] = len(prog.dma) - 1
    return prog


def run_dw(wl, hw, tm, tc, order="MN", nin=2, nout=2, data=None):
    prog = build_dw_program(wl, hw, tm, tc, order, nin, nout)
    ds, de, cs, ce = schedule_times(prog)
    check_hazards(prog, ds, de, cs, ce)
    m = metrics_of(prog, ds, de, cs, ce)
    out = dict(metrics=m, C=None, tile=(tm, tc), order=order, nbuf=(nin, nout))
    if data is not None:
        out["C"] = functional_dw(prog, data[0], data[1], (ds, de, cs, ce))
    return out


def functional_dw(prog, A3, W, times):
    """Replay: A3 shape (M, KK, C) int8, W shape (KK, C) int8 -> out (M, C) int32."""
    ds, de, cs, ce = times
    sch, wl = prog.sch, prog.wl
    tm, tc = sch.tm, sch.tn
    out = np.zeros((wl.M, wl.N), np.int32)
    bufA, bufB, bufO, pend, ev = {}, {}, {}, {}, []
    for i, op in enumerate(prog.dma):
        ev.append((ds[i], 1, i, "W") if op.kind == WB else (de[i], 0, i, "L"))
    for j in range(len(prog.cmp)):
        ev += [(cs[j], 1, j, "Cs"), (ce[j], 0, j, "Ce")]
    ev.sort(key=lambda e: (e[0], e[1], e[3], e[2]))
    for _, _, i, kind in ev:
        if kind == "L":
            op = prog.dma[i]
            mi, ni, _ = op.tile
            if op.kind == LA:
                bufA[op.slot] = A3[mi * tm:mi * tm + op.rows, :, ni * tc:ni * tc + op.cols // wl.K].astype(np.int32)
            else:
                bufB[op.slot] = W[:, ni * tc:ni * tc + op.cols].astype(np.int32)
        elif kind == "Cs":
            c = prog.cmp[i]
            pend[i] = np.einsum("mkc,kc->mc", bufA[c.a_slot], bufB[c.b_slot])
        elif kind == "Ce":
            c = prog.cmp[i]
            bufO[c.o_slot] = pend.pop(i)
        else:
            op = prog.dma[i]
            mi, ni, _ = op.tile
            out[mi * tm:mi * tm + op.rows, ni * tc:ni * tc + op.cols] = bufO[op.slot]
    return out


def dw_reference(A3, W):
    return np.einsum("mkc,kc->mc", A3.astype(np.int64), W.astype(np.int64)).astype(np.int32)


def best_dw(wl, hw, k_cands=None):
    """Exhaustive search by SIMULATION over (tm, tc, order, buffering). Small spaces only."""
    tms = sorted({t for t in (1, 2, 4, 8, 16, 32, 64, 128, 255, wl.M) if t <= min(wl.M, hw.max_dim)} |
                 {cdiv(wl.M, n) for n in (1, 2, 3, 4, 6, 8, 12, 16) if cdiv(wl.M, n) <= hw.max_dim})
    tcs = sorted({t for t in (hw.lanes, 2 * hw.lanes, 4 * hw.lanes, 8 * hw.lanes, wl.N, min(wl.N, hw.max_dim)) if 1 <= t <= wl.N} |
                 {c for c in (1, 2, 4) if c <= wl.N})
    best = None
    for tm, tc, order, (nin, nout) in itertools.product(tms, tcs, ("MN", "NM"), ((1, 1), (2, 1), (2, 2), (1, 2))):
        if dw_violations(wl, hw, tm, tc, nin, nout):
            continue
        r = run_dw(wl, hw, tm, tc, order, nin, nout)
        key = (r["metrics"]["cycles"], tm, tc, order, nin, nout)
        if best is None or key < best[0]:
            best = (key, r)
    if best is None:
        raise NoFitError("no legal depthwise tile")
    return best[1]
