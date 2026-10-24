"""python3 -m armse.overlapbench [outdir] -> results/fusion_overlap.csv + FUSION_OVERLAP.md   (MODEL ONLY)
Compares unfused / fused-with-barriers (results/fusion.csv from fusionbench) / fused-with-overlap."""
import sys
import time
import pandas as pd
from .config import HwConfig, NoFitError
from .fused import chains, chain_data, chain_reference
from .fused_overlap import best_overlap, run_overlap
from .bench import md, gm


def main(out="results"):
    base = pd.read_csv(f"{out}/fusion.csv")
    t0, rows, ok, n = time.time(), [], 0, 0
    for _, b in base.iterrows():
        ch = next(c for c in chains() if c.name == b.chain)
        depth = int(b.a_buf_bytes // 8)
        hw = HwConfig(a_depth=depth)
        row = dict(chain=ch.name, a_buf_bytes=int(b.a_buf_bytes), unfused_cycles=b.unfused_cycles,
                   barrier_cycles=b.get("fused_cycles"), unfused_bytes=b.unfused_bytes, barrier_bytes=b.get("fused_bytes"))
        try:
            o = best_overlap(ch, hw)
            data = chain_data(ch, 1)
            v = run_overlap(ch, hw, o["tm"], *map(int, o["nbuf"].split("/")), data=data)
            n += 1
            ok += int((v["C2"] == chain_reference(*data, ch.shift)).all())
            row.update(overlap_cycles=o["cycles"], overlap_bytes=o["bytes_rd"] + o["bytes_wr"], tm=o["tm"], nbuf=o["nbuf"],
                       overlap_vs_unfused=o["cycles"] / b.unfused_cycles,
                       barrier_vs_unfused=(b.fused_cycles / b.unfused_cycles) if b.get("fused_cycles") == b.get("fused_cycles") else None,
                       status="fits")
        except NoFitError:
            row["status"] = "no fit"
        rows.append(row)
        print(row["chain"], row["a_buf_bytes"], row.get("overlap_vs_unfused"), f"{time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/fusion_overlap.csv", index=False)
    fit = df[df.status == "fits"]
    rep = ["# Fused chains with cross-phase/cross-block overlap (MODEL ONLY; `python3 -m armse.overlapbench`)\n",
           "Ratios are cycles relative to the unfused D_feedback pair (<1 = faster). Barrier = earlier study (separate programs). "
           f"Overlap = one continuous program sharing slot state, common nbuf across segments. Functional check: {ok}/{n} "
           "overlapped configurations bit-exact vs an independent numpy reference; all pass deadlock + hazard checks.\n",
           md(df[["chain", "a_buf_bytes", "unfused_cycles", "barrier_cycles", "overlap_cycles", "barrier_vs_unfused",
                  "overlap_vs_unfused", "unfused_bytes", "overlap_bytes", "tm", "nbuf", "status"]].fillna("-"), "{:.3f}")]
    if len(fit):
        rep.append(f"\nGeomean cycles vs unfused over configurations that fit: barrier {gm(fit.barrier_vs_unfused.dropna()):.3f}, "
                   f"overlap {gm(fit.overlap_vs_unfused):.3f}; overlap faster than unfused in "
                   f"{(fit.overlap_vs_unfused < 1).sum()}/{len(fit)} cases (barrier: {(fit.barrier_vs_unfused < 1).sum()}/{len(fit)}).\n")
    open(f"{out}/FUSION_OVERLAP.md", "w").write("\n".join(rep))
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
