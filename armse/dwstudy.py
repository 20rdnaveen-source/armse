"""python3 -m armse.dwstudy [outdir] -> results/DEPTHWISE_MAPPING.md  (MODEL ONLY)
K-parallel (depthwise lowered to GEMM, N=1, one channel at a time) vs channel-parallel mapping."""
import sys
import pandas as pd
from .config import HwConfig, Workload
from .workloads import depthwise
from .scheduler import policy_D_fb
from .schedule import build_program
from .sim import simulate
from .depthwise import best_dw
from .bench import md, gm


def main(out="results"):
    hw, rows = HwConfig(), []
    for name, (H, W, C, k) in {"dw3x3_28_c64": (28, 28, 64, 3), "dw3x3_14_c128": (14, 14, 128, 3),
                               "dw5x5_28_c32": (28, 28, 32, 5), "dw3x3_56_c24": (56, 56, 24, 3)}.items():
        g, ch = depthwise(name, H, W, C, k)
        s = policy_D_fb(g, hw)
        m = simulate(build_program(g, hw, s)).metrics
        wl = Workload(name, g.M, C, g.K, "depthwise_cp")
        r = best_dw(wl, hw)
        d = r["metrics"]
        rows.append(dict(workload=name, pixels=g.M, channels=C, taps=g.K, kpar_cycles=m["cycles"] * ch, kpar_mac_util=m["mac_util"],
                         cp_cycles=d["cycles"], cp_mac_util=d["mac_util"], speedup=m["cycles"] * ch / d["cycles"],
                         cp_tile=f"{r['tile'][0]}x{r['tile'][1]}", cp_order=r["order"], cp_nbuf=f"{r['nbuf'][0]}/{r['nbuf'][1]}",
                         cp_bytes_rd=d["bytes_rd"], kpar_bytes_rd=m["bytes_rd"] * ch, cp_mem_stall_frac=d["mem_stall"] / d["cycles"]))
        print(name, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/depthwise_mapping.csv", index=False)
    rep = ["# Channel-parallel depthwise mapping (MODEL ONLY, `python3 -m armse.dwstudy`)\n",
           "K-parallel = the earlier lowering (one channel at a time, M=OH*OW, N=1, K=k*k) scheduled by D_feedback, x channels. "
           "Channel-parallel = lanes spread over channels; tile (pixels x channels) searched exhaustively BY SIMULATION over tile, order "
           "and buffering (small space). Same on-chip memory, same bus. im2col per channel is counted (no window reuse). Numerics are "
           "checked against a numpy depthwise reference in the tests.\n", md(df, "{:.3f}"),
           f"\nGeomean speedup of channel-parallel over K-parallel: {gm(df.speedup):.2f}x. MAC utilisation: "
           f"{df.kpar_mac_util.mean():.3f} -> {df.cp_mac_util.mean():.3f} (mean). The remaining gap to 1.0 is memory stall: "
           "depthwise has very low arithmetic intensity (K = 9 or 25 MACs per loaded byte-row), so it stays memory-bound.\n"]
    open(f"{out}/DEPTHWISE_MAPPING.md", "w").write("\n".join(rep))
    print(df[["workload", "kpar_cycles", "cp_cycles", "speedup", "kpar_mac_util", "cp_mac_util", "cp_tile", "cp_mem_stall_frac"]].round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
