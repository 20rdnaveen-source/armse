"""python3 -m armse.progbench [outdir] -> results/program_compression.csv + PROGRAM_COMPRESSION.md"""
import sys
import pandas as pd
from .config import HwConfig
from .workloads import suite
from .scheduler import policy_C, policy_D_fb
from .schedule import build_program
from .progcompress import analyse
from .bench import md, gm


def main(out="results"):
    hw, rows = HwConfig(), []
    for wl in suite():
        for name, fn in (("C_fixed_double", policy_C), ("D_feedback", policy_D_fb)):
            a = analyse(build_program(wl, hw, fn(wl, hw)))
            rows.append(dict(workload=wl.name, policy=name, raw_entries=a["raw"], compressed_entries=a["compressed"],
                             ratio=a["raw"] / a["compressed"], raw_bytes=16 * a["raw"], compressed_bytes=16 * a["compressed"]))
        print(wl.name, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/program_compression.csv", index=False)
    s = df.groupby("policy").agg(geomean_ratio=("ratio", gm), min_ratio=("ratio", "min"), max_ratio=("ratio", "max"),
                                 mean_raw_bytes=("raw_bytes", "mean"), mean_compressed_bytes=("compressed_bytes", "mean"),
                                 max_raw_bytes=("raw_bytes", "max"), max_compressed_bytes=("compressed_bytes", "max")).reset_index()
    rep = ["# Descriptor-program compressibility (potential of a hardware loop generator; MODEL ONLY)\n",
           f"Lossless greedy repeat-with-constant-delta encoding (decode verified on every program). On-chip DATA memory = {hw.onchip_bytes} B. "
           "16 bytes per descriptor entry.\n", md(s, "{:.2f}"), "\n## Per workload\n", md(df, "{:.2f}")]
    open(f"{out}/PROGRAM_COMPRESSION.md", "w").write("\n".join(rep))
    print(s.to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
