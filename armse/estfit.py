"""Estimator study: collect (estimate, simulated cycles) samples, fit simple corrections, evaluate leave-one-WORKLOAD-out.
python3 -m armse.estfit [outdir]   -> results/estimator_samples.csv, results/ESTIMATOR_STUDY.md   (MODEL ONLY)"""
import random
import sys
import numpy as np
import pandas as pd
from .config import HwConfig
from .workloads import suite
from .scheduler import candidates, rank
from .schedule import build_program
from .sim import simulate
from .analytic import estimate
from .bench import md, gm


def collect(hw, seed=7, ntop=40, nrand=40, max_iters=20000):
    rng, rows = random.Random(seed), []
    for wl in suite():
        r = rank(wl, hw, candidates(wl, hw))
        pool = [s for _, s in r]
        top = pool[:ntop]
        rest = [s for s in pool[ntop:] if estimate(wl, hw, s)["iters"] <= max_iters]
        pick = top + rng.sample(rest, min(nrand, len(rest)))
        for k, s in enumerate(pick):
            e = estimate(wl, hw, s)
            true = simulate(build_program(wl, hw, s)).cycles
            rows.append(dict(workload=wl.name, family=wl.family, tm=s.tm, tn=s.tn, tk=s.tk, order=s.order,
                             nbi=s.nbuf_in, nbo=s.nbuf_out, from_top=k < len(top), true=true, p1=e["cycles"],
                             t_dma=e["t_dma"], t_cmp=e["t_cmp"], first_ld=e["first_ld"], last_wb=e["last_wb"],
                             dur_W=e["dur_WB"], iters=e["iters"]))
        print("collected", wl.name, flush=True)
    return pd.DataFrame(rows)


def features(d):
    """Linear features, all in cycles. Class one-hots allow different overlap behaviour per buffering mode."""
    X = pd.DataFrame(index=d.index)
    X["p1"] = d.p1
    for a, b in ((1, 1), (2, 1), (2, 2), (1, 2)):
        m = ((d.nbi == a) & (d.nbo == b)).astype(float)
        X[f"p1_c{a}{b}"] = d.p1 * m
        X[f"mn_c{a}{b}"] = np.minimum(d.t_dma, d.t_cmp) * m          # the part that does NOT overlap perfectly
    X["iters"] = d.iters
    return X


def fit(X, y):
    w, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
    return w


def regret_by_choice(d, col, group="workload"):
    """true cycles of the candidate with min predicted `col`, relative to the best true cycles in the sampled pool."""
    out = {}
    for w, g in d.groupby(group):
        pick = g.loc[g[col].idxmin()]
        out[w] = pick.true / g.true.min()
    return pd.Series(out)


CLASSES = ((1, 1), (2, 1), (2, 2), (1, 2))
CLAMP = (0.85, 1.25)          # keep the corrected estimate within [0.85, 1.25] x v1: guards extrapolation outside the fitted region


def row_features(e, nbi, nbo):
    x = [e["cycles"]]
    for a, b in CLASSES:
        m = 1.0 if (nbi, nbo) == (a, b) else 0.0
        x += [e["cycles"] * m, min(e["t_dma"], e["t_cmp"]) * m]
    x.append(e["iters"])
    return x


def predict(e, nbi, nbo, coef):
    p = float(np.dot(row_features(e, nbi, nbo), coef))
    return min(max(p, CLAMP[0] * e["cycles"]), CLAMP[1] * e["cycles"])


def evaluate_picks(hw, d, suite_csv="results/suite_results.csv"):
    """End-to-end: for each workload, fit on the OTHER workloads, rank ALL legal candidates with the corrected
    closed form (no simulation), simulate only the chosen one."""
    X, y = features(d), d.true.astype(float)
    ref = pd.read_csv(suite_csv)
    rows = []
    for wl in suite():
        coef = fit(X[d.workload != wl.name], y[d.workload != wl.name])
        best, bs = None, None
        for s in candidates(wl, hw):
            e = estimate(wl, hw, s)
            p = predict(e, s.nbuf_in, s.nbuf_out, coef)
            key = (p, e["bytes_rd"] + e["bytes_wr"], s.nbuf_in + s.nbuf_out, s.order, s.tm, s.tn, s.tk)
            if best is None or key < best:
                best, bs = key, s
        c2 = simulate(build_program(wl, hw, bs)).cycles
        g = ref[ref.workload == wl.name].set_index("policy").cycles
        rows.append(dict(workload=wl.name, D_v2_loo=c2, D_v1=g["D_adaptive"], D_fb=g["D_feedback"],
                         best_square=g["C_best_square"], C_fixed_double=g["C_fixed_double"]))
        print("picked", wl.name, flush=True)
    return pd.DataFrame(rows), fit(X, y)


def main(out="results", reuse=True):
    import os
    hw = HwConfig()
    if reuse and os.path.exists(f"{out}/estimator_samples.csv"):
        d = pd.read_csv(f"{out}/estimator_samples.csv")
    else:
        d = collect(hw)
        d.to_csv(f"{out}/estimator_samples.csv", index=False)
    X, y = features(d), d.true.astype(float)
    d["p2"] = np.nan
    for w in d.workload.unique():                                  # leave-one-workload-out
        tr = d.workload != w
        coef = fit(X[tr], y[tr])
        d.loc[~tr, "p2"] = X[~tr].values @ coef
    d["err1"] = (d.p1 - d.true) / d.true
    d["err2"] = (d.p2 - d.true) / d.true
    r1, r2, rt = regret_by_choice(d, "p1"), regret_by_choice(d, "p2"), regret_by_choice(d, "true")
    top = d[d.from_top]
    hit = lambda col: np.mean([int(g.loc[g[col].idxmin()].true <= g.true.min() * 1.0) for _, g in d.groupby("workload")])
    summ = pd.DataFrame([
        dict(estimator="v1 (closed form)", mean_abs_err_pct=100 * d.err1.abs().mean(), p90_abs_err_pct=100 * d.err1.abs().quantile(.9),
             mean_abs_err_pct_top40=100 * top.err1.abs().mean(), mean_regret_pct=100 * (r1.mean() - 1), worst_regret_pct=100 * (r1.max() - 1),
             exact_best_pick=int(sum(r1 == 1.0))),
        dict(estimator="v2 (class-wise linear correction, LOO)", mean_abs_err_pct=100 * d.err2.abs().mean(),
             p90_abs_err_pct=100 * d.err2.abs().quantile(.9), mean_abs_err_pct_top40=100 * top.err2.abs().mean(),
             mean_regret_pct=100 * (r2.mean() - 1), worst_regret_pct=100 * (r2.max() - 1), exact_best_pick=int(sum(r2 == 1.0)))])
    rep = ["# Estimator study (MODEL ONLY, `python3 -m armse.estfit`)\n",
           f"Samples: {len(d)} (workload, schedule) pairs over {d.workload.nunique()} workloads (top-40 by v1 + up to 40 random legal per workload, "
           "programs <= 20k iterations). `regret` = true cycles of the candidate each estimator picks, relative to the best true cycles "
           "in the SAMPLED pool of that workload (so it is a lower bound on the regret vs the full candidate space). v2 coefficients are "
           "fitted on all other workloads and evaluated on the held-out one (leave-one-workload-out).\n", md(summ, "{:.2f}")]
    pk, coef_all = evaluate_picks(hw, d)
    pk.to_csv(f"{out}/estimator_picks.csv", index=False)
    import json
    json.dump(dict(features=["p1"] + [f"{n}_c{a}{b}" for a, b in CLASSES for n in ("p1", "mn")] + ["iters"],
                   coef=[float(c) for c in coef_all], clamp=list(CLAMP), hw=dict(lanes=hw.lanes, bus=hw.bus_bytes, row_lat=hw.row_lat, a_depth=hw.a_depth, b_depth=hw.b_depth, o_words=hw.o_words)),
              open(f"{out}/estimator_v2_coef.json", "w"), indent=1)
    import shutil
    shutil.copy(f"{out}/estimator_v2_coef.json", os.path.join(os.path.dirname(__file__), "est_v2_coef.json"))
    cmpt = pd.DataFrame([dict(policy=n, vs_best_square=gm(pk[n] / pk.best_square), vs_C_fixed_double=gm(pk[n] / pk.C_fixed_double),
                              wins_vs_best_square=int((pk[n] < pk.best_square).sum()), ties=int((pk[n] == pk.best_square).sum()),
                              losses=int((pk[n] > pk.best_square).sum()))
                         for n in ("D_v1", "D_v2_loo", "D_fb")])
    rep += ["\n## End-to-end: chosen schedule's SIMULATED cycles over the 30-workload suite (no simulator used to choose D_v1 / D_v2)\n",
            "D_v2_loo: coefficients fitted WITHOUT the evaluated workload; the estimator ranks all legal candidates, only the winner is simulated. "
            "D_fb simulates the estimator's top-8 (uses the simulator). The coefficient file in `estimator_v2_coef.json` is fitted on all "
            "workloads and is specific to the default hardware configuration.\n", md(cmpt, "{:.3f}")]
    open(f"{out}/ESTIMATOR_STUDY.md", "w").write("\n".join(rep))
    print(summ.round(2).to_string()); print(cmpt.round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
