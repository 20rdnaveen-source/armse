"""Closed-form memory-cost estimator (the scheduler's cost function).

Assumptions (all exposed, all validated against the event simulator in tests/ and bench):
  * traffic is exact: the loop-nest reuse rule is expressed through two 'traversal' counts
  * DMA and compute durations are exact sums over tile-size classes
  * total cycles is an *approximation*: overlap is modelled by  max(DMA-steady, compute) with the
    unoverlappable first load and last writeback added; its error is measured, not assumed.
"""
from .config import HwConfig, Workload, cdiv
from .schedule import Schedule
from .sim import conflict_factor


def classes(D: int, t: int):
    f, r = divmod(D, t)
    out = []
    if f:
        out.append((t, f))
    if r:
        out.append((r, 1))
    return out


def estimate(wl: Workload, hw: HwConfig, sch: Schedule) -> dict:
    M, N, K = wl.M, wl.N, wl.K
    if min(M, N, K) == 0:
        return dict(bytes_rd=0, bytes_wr=0, transfers=0, t_dma=0, t_cmp=0, cycles=0,
                    dur_A=0, dur_B=0, dur_WB=0)
    tm, tn, tk = sch.tm, sch.tn, sch.tk
    nm, nn, nk = cdiv(M, tm), cdiv(N, tn), cdiv(K, tk)
    if nk > 1:
        tA, tB = nn, nm
    elif sch.order == "MN":
        tA, tB = 1, (nm if nn > 1 else 1)
    else:
        tA, tB = (nn if nm > 1 else 1), 1
    jo, rf, bus, L = hw.job_ovh, hw.row_fix, hw.bus_bytes, hw.lanes
    kcl, ncl = classes(K, tk), classes(N, tn)
    SA = sum(c * cdiv(s, bus) for s, c in kcl)
    SB = sum(c * cdiv(s, bus) for s, c in ncl)
    ob = wl.out_bytes
    SW = sum(c * cdiv(ob * s, bus) for s, c in ncl)
    G = sum(c * cdiv(s, L) for s, c in kcl)
    durA = nm * nk * jo + rf * M * nk + M * SA                 # one traversal of all A tiles
    durB = nk * nn * jo + rf * K * nn + K * SB                 # one traversal of all B tiles
    durW = nm * nn * jo + rf * M * nn + M * SW                 # all writebacks
    t_dma = tA * durA + tB * durB + durW
    t_cmp = (M * G * sum(c * s * conflict_factor(hw, s) for s, c in ncl)
             + hw.cmp_ovh * nm * nn * nk)
    # unoverlappable pieces
    r0, c0 = min(tm, M), min(tk, K)
    first_ld = (jo + r0 * (rf + cdiv(c0, bus))) + (jo + c0 * (rf + cdiv(min(tn, N), bus)))
    lm, ln = M - (nm - 1) * tm, N - (nn - 1) * tn
    last_wb = jo + lm * (rf + cdiv(ob * ln, bus))
    if sch.nbuf_in == 1:
        cyc = hw.start_off + (t_dma - durW) + t_cmp + (durW if sch.nbuf_out == 1 else last_wb)
    else:
        tc = t_cmp + ((durW - last_wb) if sch.nbuf_out == 1 else 0)
        steady = max(t_dma - first_ld - last_wb, tc)
        cyc = hw.start_off + first_ld + steady + last_wb
    return dict(bytes_rd=tA * M * K + tB * K * N, bytes_wr=ob * M * N,
                transfers=tA * nm * nk + tB * nk * nn + nm * nn,
                t_dma=t_dma, t_cmp=t_cmp, cycles=int(cyc),
                dur_A=durA, dur_B=durB, dur_WB=durW, tA=tA, tB=tB,
                first_ld=first_ld, last_wb=last_wb, iters=nm * nn * nk, nm=nm, nn=nn, nk=nk)
