"""Baselines A/B/C and the deterministic adaptive scheduler D (+ feedback variant).

All policies run on the SAME HwConfig (same physical buffers, same bus, same compute lanes).
D's cost function is armse.analytic.estimate (documented there); D_fb additionally executes the
top-K estimator candidates in the simulator and keeps the one with the fewest measured cycles
(this models 'observed execution behaviour' feedback).
"""
import itertools
from .config import HwConfig, Workload, NoFitError, cdiv
from .schedule import Schedule, violations, build_program, check_op
from .analytic import estimate
from .sim import simulate

NBUFS = ((1, 1), (2, 1), (2, 2), (1, 2))


def _clip(wl, t):
    return min(t, wl.M), min(t, wl.N), min(t, wl.K)


def fixed_square_size(hw: HwConfig, nbuf_in, nbuf_out):
    """Largest power-of-two T whose TxTxT tile fits the physical buffers (design-time constant)."""
    for t in (128, 64, 32, 16, 8, 4, 2, 1):
        kg = cdiv(t, hw.lanes)
        if (t * kg <= hw.a_depth // nbuf_in and t * kg <= hw.b_depth // nbuf_in
                and t * t <= hw.o_words // nbuf_out):
            return t
    raise NoFitError("hardware cannot hold even a 1x1x1 tile")


def _minimal(wl, hw):
    """Fallback schedule: smallest legal tiles, single buffered."""
    for tk in (min(wl.K, hw.max_dim), min(wl.K, 64), min(wl.K, 16), 1):
        s = Schedule(1, 1, tk, "MN", 1, 1, "fallback")
        if not violations(wl, hw, s):
            return s
    raise NoFitError("no legal schedule: buffers too small for a 1x1 tile")


def empty_ok(wl):
    check_op(wl)
    return min(wl.M, wl.N, wl.K) == 0


def policy_A(wl, hw):
    """A: direct / minimally buffered: one output element at a time, no operand reuse."""
    if empty_ok(wl):
        return Schedule(1, 1, 1, label="A_direct")
    tk = min(wl.K, hw.max_dim, hw.lanes * hw.a_depth, hw.lanes * hw.b_depth)
    while tk > 1 and violations(wl, hw, Schedule(1, 1, tk)):
        tk -= 1
    s = Schedule(1, 1, tk, "MN", 1, 1, "A_direct")
    if violations(wl, hw, s):
        raise NoFitError("direct baseline does not fit")
    return s


def _fixed(wl, hw, nin, nout, label):
    if empty_ok(wl):
        return Schedule(1, 1, 1, label=label)
    t = fixed_square_size(hw, nin, nout)
    tm, tn, tk = _clip(wl, t)
    s = Schedule(tm, tn, tk, "MN", nin, nout, label)
    return s if not violations(wl, hw, s) else _minimal(wl, hw)


def policy_B(wl, hw):
    """B: fixed square tiling, single buffered."""
    return _fixed(wl, hw, 1, 1, "B_fixed_single")


def policy_C(wl, hw):
    """C: fixed square tiling + ping-pong double buffering of operands and outputs."""
    return _fixed(wl, hw, 2, 2, "C_fixed_double")


def dim_candidates(D, hw, is_k=False):
    cap = min(D, hw.max_dim)
    c = {cap}
    t = 1
    while t <= cap:
        c.add(t)
        t *= 2
    for n in range(1, 17):                    # balanced splits minimise ragged edge tiles
        v = cdiv(D, n)
        if v <= cap:
            c.add(v)
            if is_k:
                r = cdiv(v, hw.lanes) * hw.lanes
                if r <= cap:
                    c.add(r)
    if is_k:
        for m in range(1, cap // hw.lanes + 1):
            c.add(m * hw.lanes) if m * hw.lanes <= cap and m <= 8 else None
    return sorted(x for x in c if 1 <= x <= cap)


def candidates(wl, hw, orders=("MN", "NM"), nbufs=NBUFS, square_only=False):
    if square_only:
        ts = sorted(set(dim_candidates(min(wl.M, wl.N, wl.K), hw)))
        shapes = [(t, t, t) for t in ts]
        shapes += [_clip(wl, t) for t in (1, 2, 4, 8, 16, 32, 64, 128)]
        shapes = sorted(set(shapes))
    else:
        shapes = itertools.product(dim_candidates(wl.M, hw), dim_candidates(wl.N, hw),
                                   dim_candidates(wl.K, hw, True))
    for (tm, tn, tk) in shapes:
        for order in orders:
            for ni, no in nbufs:
                s = Schedule(tm, tn, tk, order, ni, no)
                if not violations(wl, hw, s):
                    yield s


def rank(wl, hw, cands):
    scored = []
    for s in cands:
        e = estimate(wl, hw, s)
        scored.append(((e["cycles"], e["bytes_rd"] + e["bytes_wr"], s.nbuf_in + s.nbuf_out,
                        s.order, s.tm, s.tn, s.tk), s))
    scored.sort(key=lambda x: x[0])
    return scored


_RANK_CACHE = {}


def ranked(wl, hw, **kw):
    key = (wl, hw, tuple(sorted((k, v) for k, v in kw.items())))
    if key not in _RANK_CACHE:
        _RANK_CACHE[key] = rank(wl, hw, candidates(wl, hw, **kw))
    return _RANK_CACHE[key]


def policy_D(wl, hw, **kw):
    """D: deterministic workload-aware scheduler, ranked by the closed-form cost model."""
    if empty_ok(wl):
        return Schedule(1, 1, 1, label="D_adaptive")
    r = ranked(wl, hw, **kw)
    if not r:
        return _minimal(wl, hw)               # fallback when nothing fits
    s = r[0][1]
    return Schedule(*s.key(), label="D_adaptive")


def policy_D_fb(wl, hw, k=8):
    """D_fb: simulate the estimator's top-k candidates, keep the best measured one."""
    if empty_ok(wl):
        return Schedule(1, 1, 1, label="D_feedback")
    r = ranked(wl, hw)
    if not r:
        return _minimal(wl, hw)
    best = min(r[:k], key=lambda x: (simulate(build_program(wl, hw, x[1])).cycles, x[0]))
    return Schedule(*best[1].key(), label="D_feedback")


def policy_C_best_square(wl, hw, topn=16):
    """Strong fixed-shape baseline: among SQUARE-tile schedules (tile size x order x buffering),
    pick the best by SIMULATION. To bound run time the simulated set is the estimator's top-`topn`
    square candidates plus the fixed B and C schedules, so it is never worse than B or C."""
    if empty_ok(wl):
        return Schedule(1, 1, 1, label="C_best_square")
    cands = [s for _, s in rank(wl, hw, candidates(wl, hw, square_only=True))[:topn]]
    for f in (policy_B, policy_C):
        s = f(wl, hw)
        if not violations(wl, hw, s):
            cands.append(s)
    if not cands:
        return _minimal(wl, hw)
    best = min(cands, key=lambda s: (simulate(build_program(wl, hw, s)).cycles, s.key()))
    return Schedule(*best.key(), label="C_best_square")


def policy_D_v2(wl, hw, force=False):
    """D_v2: deterministic scheduler ranked by the CALIBRATED closed-form estimator (armse/estfit.py). The correction
    coefficients were fitted on simulator samples of ONE hardware configuration; for any other configuration this falls
    back to the uncorrected estimator (policy_D) unless force=True (used only to measure transfer). No simulation is
    used at scheduling time."""
    import json
    import os
    from .estfit import predict
    if empty_ok(wl):
        return Schedule(1, 1, 1, label="D_v2")
    cfg = json.load(open(os.path.join(os.path.dirname(__file__), "est_v2_coef.json")))
    c = cfg["hw"]
    if not force and (hw.lanes, hw.bus_bytes, hw.row_lat, hw.a_depth, hw.b_depth, hw.o_words) != (
            c["lanes"], c["bus"], c["row_lat"], c["a_depth"], c["b_depth"], c["o_words"]):
        return policy_D(wl, hw)                  # uncalibrated hardware (force=True applies the coefficients anyway)
    best, bs = None, None
    for s in candidates(wl, hw):
        e = estimate(wl, hw, s)
        key = (predict(e, s.nbuf_in, s.nbuf_out, cfg["coef"]), e["bytes_rd"] + e["bytes_wr"], s.nbuf_in + s.nbuf_out,
               s.order, s.tm, s.tn, s.tk)
        if best is None or key < best:
            best, bs = key, s
    if bs is None:
        return _minimal(wl, hw)
    return Schedule(*bs.key(), label="D_v2")


POLICIES = {
    "A_direct": policy_A,
    "B_fixed_single": policy_B,
    "C_fixed_double": policy_C,
    "C_best_square": policy_C_best_square,
    "D_adaptive": policy_D,
    "D_feedback": policy_D_fb,
}

ABLATIONS = {
    "D_full": lambda wl, hw: policy_D(wl, hw),
    "D_no_shape (square tiles)": lambda wl, hw: policy_D(wl, hw, square_only=True),
    "D_no_order (MN only)": lambda wl, hw: policy_D(wl, hw, orders=("MN",)),
    "D_no_bufalloc ((2,2) only)": lambda wl, hw: policy_D(wl, hw, nbufs=((2, 2),)),
    "D_no_overlap ((1,1) only)": lambda wl, hw: policy_D(wl, hw, nbufs=((1, 1),)),
}
