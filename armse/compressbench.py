"""python3 -m armse.compressbench [outdir] -> results/COMPRESSION_CANDIDATE.md (+csv)   (MODEL ONLY)
Candidate 'compression-aware buffering': B (weights) stored externally as packed INT4 + a 16-bit scale per column per
tile load; unpacked to INT8 on arrival, so on-chip capacity and compute are unchanged. The SAME schedule chosen for
INT8 is reused (upper bound of the benefit without re-tuning). ACCURACY IMPACT OF INT4 IS NOT EVALUATED here."""
import sys
from dataclasses import replace
import pandas as pd
from .config import HwConfig
from .workloads import suite
from .scheduler import policy_D_fb
from .schedule import build_program
from .sim import simulate
from . import energy
from .bench import md, gm


def main(out="results"):
    hw, rows = HwConfig(), []
    for wl in suite():
        s = policy_D_fb(wl, hw)
        m8 = simulate(build_program(wl, hw, s)).metrics
        m4 = simulate(build_program(replace(wl, weight_bits=4), hw, s)).metrics
        e8, e4 = energy.level2(m8, hw)["e_total_pj"], energy.level2(m4, hw)["e_total_pj"]
        wbytes = wl.K * wl.N
        rows.append(dict(workload=wl.name, family=wl.family, weight_share_of_reads=wbytes / max(1, m8["bytes_rd"]) if False else None,
                         cycles_int8=m8["cycles"], cycles_int4=m4["cycles"], cycle_ratio=m4["cycles"] / m8["cycles"],
                         bytes_int8=m8["bytes_rd"] + m8["bytes_wr"], bytes_int4=m4["bytes_rd"] + m4["bytes_wr"],
                         bytes_ratio=(m4["bytes_rd"] + m4["bytes_wr"]) / (m8["bytes_rd"] + m8["bytes_wr"]),
                         l2_energy_ratio=e4 / e8, mem_stall_frac_int8=m8["mem_stall"] / m8["cycles"]))
    df = pd.DataFrame(rows).drop(columns=["weight_share_of_reads"])
    df.to_csv(f"{out}/compression_candidate.csv", index=False)
    fam = df.groupby("family").agg(cycle_ratio=("cycle_ratio", gm), bytes_ratio=("bytes_ratio", gm),
                                   l2_energy_ratio=("l2_energy_ratio", gm)).reset_index()
    mb = df[df.workload.isin(["gemv_1x256x256", "gemv_256x1x256", "mlp_b1_L0", "mlp_b1_L1", "mlp_b1_L2"])]
    rep = ["# Compression-aware candidate: packed INT4 weights (MODEL ONLY, `python3 -m armse.compressbench`)\n",
           "Same schedule as D_feedback for INT8 (no re-tuning). B loads carry 4 bits per weight plus a 16-bit scale per column per tile "
           "load; activations/outputs unchanged; buffers and MACs unchanged. **Accuracy loss from INT4 is NOT modelled; "
           "Level-2 energy uses illustrative constants.**\n",
           f"Geomean over {len(df)} workloads: cycles {gm(df.cycle_ratio):.3f}x, total external bytes {gm(df.bytes_ratio):.3f}x, "
           f"L2 energy {gm(df.l2_energy_ratio):.3f}x (all relative to INT8). Memory-bound subset (GEMV + batch-1 MLP): cycles "
           f"{gm(mb.cycle_ratio):.3f}x, bytes {gm(mb.bytes_ratio):.3f}x.\n", "## By family\n", md(fam, "{:.3f}"),
           "\n## Per workload\n", md(df, "{:.3f}")]
    open(f"{out}/COMPRESSION_CANDIDATE.md", "w").write("\n".join(rep))
    print(fam.round(3).to_string()); print("all", gm(df.cycle_ratio), gm(df.bytes_ratio), gm(df.l2_energy_ratio)); print("membound", gm(mb.cycle_ratio), gm(mb.bytes_ratio))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
