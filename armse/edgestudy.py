"""python3 -m armse.edgestudy [outdir] -> results/EDGE_WORKLOADS.md (+csv)   (MODEL ONLY)
1. KV-cache decode with a GROWING cache: static (compile-time) vs runtime-adaptive vs hybrid (bucketed) scheduling.
2. Depthwise convolution: compute underutilisation of a lane-parallel-over-K datapath.
3. Tensor-lifetime planning: naive vs peak-live vs planned arena for an MLP and a transformer block."""
import sys
import time
import pandas as pd
from .config import HwConfig, Workload
from .schedule import Schedule, build_program, violations
from .sim import simulate
from .scheduler import policy_C, policy_C_best_square, policy_D_fb, POLICIES, _RANK_CACHE
from .workloads import kv_decode, attention, depthwise
from . import memplan
from .bench import md, gm


def clip(wl, s):
    return Schedule(min(s.tm, wl.M), min(s.tn, wl.N), min(s.tk, wl.K), s.order, s.nbuf_in, s.nbuf_out)


def pow2_ge(x):
    p = 1
    while p < x:
        p *= 2
    return p


def kv_study(hw, d=64, shapes=range(32, 1025, 32), factory=None, static_lens=(1024, 256)):
    factory = factory or kv_decode
    shapes = list(shapes)
    pol, search_s, nsearch = {}, {}, {}

    def timed(fn, *a):
        _RANK_CACHE.clear()
        t = time.perf_counter()
        r = fn(*a)
        return r, time.perf_counter() - t

    static = {}
    for Ls in static_lens:                                      # compile-time schedules for a chosen length
        for idx, w in enumerate(factory("s", Ls, d)):
            static[(Ls, idx)] = policy_D_fb(w, hw)
    k1, k2 = f"static_L{static_lens[0]}", f"static_L{static_lens[1]}"
    hybrid_cache, rows = {}, []
    tot = {k: 0 for k in ("C_fixed_double", "C_best_square", k1, k2, "runtime_D_fb", "hybrid_pow2")}
    worst = {k: 0.0 for k in tot}
    ss = {k: 0.0 for k in tot}
    ns = {k: 0 for k in tot}
    ns[k1] = ns[k2] = 2
    for L in shapes:
        per = {k: 0 for k in tot}
        for suf, w in enumerate(factory("kv", L, d)):
            cands = {}
            s, dt = timed(policy_C, w, hw); cands["C_fixed_double"] = s; ss["C_fixed_double"] += dt
            s, dt = timed(policy_C_best_square, w, hw); cands["C_best_square"] = s; ss["C_best_square"] += dt; ns["C_best_square"] += 1
            cands[k1] = clip(w, static[(static_lens[0], suf)])
            cands[k2] = clip(w, static[(static_lens[1], suf)])
            s, dt = timed(policy_D_fb, w, hw); cands["runtime_D_fb"] = s; ss["runtime_D_fb"] += dt; ns["runtime_D_fb"] += 1
            B = pow2_ge(L)
            if (B, suf) not in hybrid_cache:
                wb = factory("b", B, d)[suf]
                s, dt = timed(policy_D_fb, wb, hw)
                hybrid_cache[(B, suf)] = s; ss["hybrid_pow2"] += dt; ns["hybrid_pow2"] += 1
            cands["hybrid_pow2"] = clip(w, hybrid_cache[(B, suf)])
            for k, s in cands.items():
                assert not violations(w, hw, s), (k, L, s)
                per[k] += simulate(build_program(w, hw, s)).cycles
        for k in tot:
            tot[k] += per[k]
        rows.append(dict(L=L, **per))
    df = pd.DataFrame(rows)
    for k in tot:
        worst[k] = float((df[k] / df["runtime_D_fb"]).max())
    summ = pd.DataFrame([dict(policy=k, total_cycles=tot[k], vs_runtime_D_fb=tot[k] / tot["runtime_D_fb"],
                              worst_step_vs_runtime=worst[k], searches=ns[k], search_seconds=ss[k]) for k in tot])
    return df, summ


def depthwise_study(hw):
    rows = []
    for name, (H, W, C, k) in {"dw3x3_28_c64": (28, 28, 64, 3), "dw3x3_14_c128": (14, 14, 128, 3),
                               "dw5x5_28_c32": (28, 28, 32, 5)}.items():
        wl, ch = depthwise(name, H, W, C, k)
        for pn in ("C_fixed_double", "D_feedback"):
            s = POLICIES[pn](wl, hw)
            m = simulate(build_program(wl, hw, s)).metrics
            rows.append(dict(workload=name, channels=ch, policy=pn, tile=f"{s.tm}x{s.tn}x{s.tk}",
                             cycles_per_channel=m["cycles"], total_cycles=m["cycles"] * ch, mac_util=m["mac_util"],
                             mem_stall_frac=m["mem_stall"] / m["cycles"]))
    return pd.DataFrame(rows)


def planning_study():
    rows = []
    cases = {"mlp_b1_784_256_128_10": memplan.mlp_activations(1, [784, 256, 128, 10]),
             "mlp_b32_784_256_128_10": memplan.mlp_activations(32, [784, 256, 128, 10]),
             "xformer_s64_d64": memplan.transformer_block(64, 64), "xformer_s128_d64": memplan.transformer_block(128, 64),
             "xformer_s256_d64": memplan.transformer_block(256, 64), "xformer_s64_d64_inplace": memplan.transformer_block(64, 64, inplace=True)}
    for n, ts in cases.items():
        off, arena = memplan.plan(ts)
        assert memplan.verify(ts, off)
        rows.append(dict(graph=n, naive_bytes=memplan.naive_total(ts), peak_live_bytes=memplan.peak_live(ts),
                         planned_arena_bytes=arena, saving_vs_naive=1 - arena / memplan.naive_total(ts),
                         fits_8KB_onchip=arena <= 8192))
    return pd.DataFrame(rows)


def main(out="results"):
    hw = HwConfig()
    t0 = time.time()
    kv, kvs = kv_study(hw)
    kv.to_csv(f"{out}/kv_decode_steps.csv", index=False)
    pf, pfs = kv_study(hw, 64, range(16, 257, 16), attention, (256, 64))
    pf.to_csv(f"{out}/prefill_steps.csv", index=False)
    dw = depthwise_study(hw)
    pl = planning_study()
    dw.to_csv(f"{out}/depthwise.csv", index=False)
    pl.to_csv(f"{out}/memplan.csv", index=False)
    rep = ["# Edge workloads, dynamic shapes and memory planning (MODEL ONLY; `python3 -m armse.edgestudy`)\n",
           "## 1. KV-cache decode, cache growing 32 -> 1024 (d = 64, INT8, softmax not modelled)\n",
           "Each step = QK^T GEMV + PV GEMV over the current cache length. Static schedules are tuned once (D_feedback) for a "
           "chosen cache length and clipped to smaller shapes; runtime re-plans every step; hybrid plans once per power-of-two "
           "bucket (tuned at the bucket's upper bound) and clips. Search seconds = host Python time (not hardware).\n",
           md(kvs, "{:.3f}"),
           "\n### 1b. Attention PREFILL with varying sequence length (QK^T: M=N=L, K=64; PV: M=L, N=64, K=L), L = 16 -> 256\n",
           md(pfs, "{:.3f}"),
           "\n## 2. Depthwise convolution (one channel lowered to GEMM M=OH*OW, N=1, K=k*k; x channels)\n",
           md(dw, "{:.3f}"),
           "\nThe lane-parallel-over-K datapath cannot fill its lanes when K = k*k is 9 or 25 and N = 1: low MAC utilisation is a "
           "property of this mapping, not of the schedule. A channel-parallel mapping is not modelled.\n",
           "\n## 3. Tensor-lifetime planning (activations only, INT8; transformer lifetimes are illustrative hand-set intervals)\n",
           md(pl, "{:.3f}")]
    open(f"{out}/EDGE_WORKLOADS.md", "w").write("\n".join(rep))
    print(kvs.to_string()); print(pfs.to_string()); print(dw.round(3).to_string()); print(pl.round(3).to_string()); print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
