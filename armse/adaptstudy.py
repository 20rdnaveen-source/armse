"""python3 -m armse.adaptstudy [outdir] -> results/ADAPTIVE_LOOP.md (+csv)  (MODEL ONLY)"""
import sys
import pandas as pd
from .config import HwConfig
from .workloads import suite
from .adaptive import run_sequence
from .bench import md, gm

LAYERS = ["conv3x3_16to32_28", "conv3x3_32to64_14", "conv1x1_64to64_14", "mlp_b32_L0", "mlp_b32_L1", "mlp_b32_L2",
          "attn_s64_d32_QKt", "attn_s64_d32_PV"]


def main(out="results"):
    W = {w.name: w for w in suite()}
    seq = [W[n] for n in LAYERS]
    nom = HwConfig()
    scenarios = {
        "no mismatch (control)": nom,
        "latency 12 -> 30 cycles": nom.replace(row_lat=30),
        "latency 12 -> 4 cycles": nom.replace(row_lat=4),
        "bus assumed 8 B, true 4 B": None,                       # nominal model optimistic about the bus
        "banking: true naive layout": nom.replace(bank_mode="naive"),
        "latency 30 AND naive banking": nom.replace(row_lat=30, bank_mode="naive"),
    }
    rows = []
    for name, true in scenarios.items():
        if name.startswith("bus"):
            n8 = nom.replace(bus_bytes=8)
            r = {m: run_sequence(seq, n8, nom, m) for m in ("static", "adaptive", "oracle")}
            hw_true, hw_nom = nom, n8
        else:
            r = {m: run_sequence(seq, nom, true, m) for m in ("static", "adaptive", "oracle")}
            hw_true, hw_nom = true, nom
        fm = r["adaptive"].final_model
        rows.append(dict(scenario=name, static_cycles=r["static"].cycles, adaptive_cycles=r["adaptive"].cycles,
                         oracle_cycles=r["oracle"].cycles, adaptive_vs_static=r["adaptive"].cycles / r["static"].cycles,
                         adaptive_vs_oracle=r["adaptive"].cycles / r["oracle"].cycles,
                         replans=r["adaptive"].replans, inferred=f"lat={fm.row_lat} bus={fm.bus_bytes} bank={fm.bank_mode}",
                         true=f"lat={hw_true.row_lat} bus={hw_true.bus_bytes} bank={hw_true.bank_mode}",
                         plan_s_static=r["static"].plan_seconds, plan_s_adaptive=r["adaptive"].plan_seconds))
        print(name, f"{rows[-1]['adaptive_vs_static']:.3f}", f"{rows[-1]['adaptive_vs_oracle']:.3f}", rows[-1]["inferred"], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/adaptive_loop.csv", index=False)
    rep = ["# Counter-driven runtime adaptation (mechanism E; MODEL ONLY; `python3 -m armse.adaptstudy`)\n",
           f"Sequence of {len(seq)} layers (CNN, MLP, attention GEMMs). The scheduler plans with a nominal hardware model; the layers execute on the TRUE "
           "hardware (simulated). Static = plan everything with the nominal model; Adaptive = after each layer infer latency / bus width / banking "
           "behaviour from counters (dma_busy, cmp_busy, n_xfer, bytes + host-known descriptors) and re-plan later layers; Oracle = plan with the true "
           "model. All three use the same raw estimator (no simulation to plan). Cycles are simulated on the true hardware; the first layer is "
           "always planned with the nominal model in Adaptive mode (its cost is included). Plan seconds are host Python time.\n",
           md(df, "{:.3f}"),
           "\nThe inference is exact here because the model's timing formulas are exact; on real hardware the counters would be noisier "
           "(contention, refresh, unmodelled stalls) and the inferred parameters would be estimates. The banking scenario assumes hardware whose "
           "read conflicts are visible only through longer compute time: the implemented RTL uses skewed banking and has none.\n"]
    open(f"{out}/ADAPTIVE_LOOP.md", "w").write("\n".join(rep))
    print(df[["scenario", "static_cycles", "adaptive_cycles", "oracle_cycles", "adaptive_vs_static", "adaptive_vs_oracle", "replans"]].round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
