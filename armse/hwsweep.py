"""python3 -m armse.hwsweep [outdir] -> results/HW_SWEEP.md (+csv)   (MODEL ONLY)
Generality check: 27 hardware configurations (lanes x buffer-capacity scale x burst latency, bus = 4 B) on 12 workloads.
Also: does the D_v2 correction fitted at the default configuration transfer to other configurations (force=True)?"""
import sys
import time
import pandas as pd
from .config import HwConfig, NoFitError
from .workloads import suite
from .scheduler import policy_C, policy_C_best_square, policy_D, policy_D_fb, policy_D_v2, _RANK_CACHE
from .schedule import build_program
from .sim import simulate
from .bench import md, gm

NAMES = ("gemm_sq64", "gemm_sq128", "gemv_1x256x256", "tall_512x16x64", "wide_16x512x64", "deepK_32x32x1024",
         "ragged_101x67x131", "mlp_b32_L0", "mlp_b1_L0", "conv3x3_16to32_28", "attn_s128_d64_QKt", "stress_512x512x64")


def cycles(wl, hw, s):
    return simulate(build_program(wl, hw, s)).cycles


def main(out="results"):
    base = HwConfig()
    W = [w for w in suite() if w.name in NAMES]
    rows, t0 = [], time.time()
    for lanes in (4, 8, 16):
        for cap in (0.5, 1.0, 2.0):
            for lat in (6, 12, 24):
                hw = base.replace(lanes=lanes, row_lat=lat).scaled(cap)
                cyc = {k: [] for k in ("C_fixed_double", "C_best_square", "D_adaptive", "D_v2_forced", "D_feedback")}
                try:
                    for wl in W:
                        _RANK_CACHE.clear()
                        cyc["C_fixed_double"].append(cycles(wl, hw, policy_C(wl, hw)))
                        cyc["C_best_square"].append(cycles(wl, hw, policy_C_best_square(wl, hw)))
                        cyc["D_adaptive"].append(cycles(wl, hw, policy_D(wl, hw)))
                        cyc["D_v2_forced"].append(cycles(wl, hw, policy_D_v2(wl, hw, force=True)))
                        cyc["D_feedback"].append(cycles(wl, hw, policy_D_fb(wl, hw)))
                except NoFitError:
                    continue
                r = lambda a, b: gm([x / y for x, y in zip(cyc[a], cyc[b])])
                wins = lambda a, b: sum(x < y for x, y in zip(cyc[a], cyc[b]))
                rows.append(dict(lanes=lanes, cap_scale=cap, row_lat=lat, balance_ops_per_B=2 * lanes / hw.bus_bytes,
                                 onchip_B=hw.onchip_bytes, Dfb_vs_best_square=r("D_feedback", "C_best_square"),
                                 Dfb_vs_C_double=r("D_feedback", "C_fixed_double"), Dv1_vs_best_square=r("D_adaptive", "C_best_square"),
                                 Dv2forced_vs_best_square=r("D_v2_forced", "C_best_square"),
                                 Dfb_wins_vs_best=wins("D_feedback", "C_best_square"), Dfb_losses_vs_best=wins("C_best_square", "D_feedback")))
        print("lanes", lanes, f"{time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/hw_sweep.csv", index=False)
    s = lambda c: f"{df[c].min():.3f} / {gm(df[c]):.3f} / {df[c].max():.3f}"
    by_lanes = df.groupby("lanes")[["Dfb_vs_best_square", "Dfb_vs_C_double", "Dv1_vs_best_square", "Dv2forced_vs_best_square"]].agg(gm).reset_index()
    by_cap = df.groupby("cap_scale")[["Dfb_vs_best_square", "Dfb_vs_C_double"]].agg(gm).reset_index()
    by_lat = df.groupby("row_lat")[["Dfb_vs_best_square", "Dfb_vs_C_double"]].agg(gm).reset_index()
    worse = df[df.Dfb_vs_best_square > 1.0]
    d1 = df[(df.lanes == 8) & (df.cap_scale == 1.0) & (df.row_lat == 12)]
    rep = ["# Hardware-configuration sweep (MODEL ONLY, `python3 -m armse.hwsweep`)\n",
           f"{len(df)} configurations: lanes {{4,8,16}} x buffer-capacity scale {{0.5,1,2}} x burst latency {{6,12,24}} cycles, bus 4 B/cycle, "
           f"{len(W)} workloads each. Ratios are geomean cycles (min / geomean / max over configurations):\n",
           f"* D_feedback / C_best_square: {s('Dfb_vs_best_square')}",
           f"* D_feedback / C_fixed_double: {s('Dfb_vs_C_double')}",
           f"* D_adaptive (raw estimator) / C_best_square: {s('Dv1_vs_best_square')}",
           f"* D_v2 with the DEFAULT-configuration coefficients forced onto every configuration / C_best_square: {s('Dv2forced_vs_best_square')}",
           f"\nD_feedback is slower than C_best_square (ratio > 1) in {len(worse)} of {len(df)} configurations.\n",
           "## By lanes\n", md(by_lanes, "{:.3f}"), "\n## By buffer-capacity scale\n", md(by_cap, "{:.3f}"),
           "\n## By burst latency\n", md(by_lat, "{:.3f}"), "\n## All configurations\n", md(df, "{:.3f}")]
    open(f"{out}/HW_SWEEP.md", "w").write("\n".join(rep))
    print("\n".join(rep[2:8])); print(by_lanes.round(3).to_string()); print(by_cap.round(3).to_string()); print(by_lat.round(3).to_string())
    print("default-config row:", d1[["Dfb_vs_best_square", "Dv1_vs_best_square", "Dv2forced_vs_best_square"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
