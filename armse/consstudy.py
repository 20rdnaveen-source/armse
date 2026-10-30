"""python3 -m armse.consstudy [outdir] -> results/CONSTRAINED_OPERATION.md (+csv)  (MODEL ONLY)"""
import sys
import pandas as pd
from .config import HwConfig
from .workloads import suite
from .constrained import pool, measure, choose, choose_verified
from .bench import md, gm


def main(out="results"):
    hw, rows_d, rows_b, front = HwConfig(), [], [], []
    for wl in suite():
        c = measure(wl, hw, pool(wl, hw))
        t_min = min(x.true_cycles for x in c)
        fast = min(c, key=lambda x: (x.true_cycles, x.energy))              # cycle-optimal (what D_feedback targets)
        lean = min(c, key=lambda x: (x.energy, x.true_cycles))              # energy-optimal
        front.append(dict(workload=wl.name, cycles_fast=fast.true_cycles, energy_fast=fast.energy, cycles_lean=lean.true_cycles,
                          energy_lean=lean.energy, energy_saving=1 - lean.energy / fast.energy, cycle_cost=lean.true_cycles / fast.true_cycles))
        for slack in (0.9, 1.0, 1.05, 1.1, 1.25, 1.5):
            dl = slack * t_min
            for margin in (0.0, 0.05, 0.10):
                p, fb = choose(c, deadline=dl, margin=margin)
                rows_d.append(dict(workload=wl.name, slack=slack, margin=margin, fallback=fb, violated=p.true_cycles > dl,
                                   energy_vs_fast=p.energy / fast.energy, checks=0))
            pv, fbv, nchk = choose_verified(c, deadline=dl)
            rows_d.append(dict(workload=wl.name, slack=slack, margin=-1.0, fallback=fbv, violated=pv.true_cycles > dl,
                               energy_vs_fast=pv.energy / fast.energy, checks=nchk))
        e_min = lean.energy
        for f in (1.0, 1.05, 1.1, 1.25, 1.5, 2.0):
            b = f * e_min
            p, fb = choose(c, budget=b)
            rows_b.append(dict(workload=wl.name, budget_factor=f, fallback=fb, violated=p.energy > b, cycles_vs_fast=p.true_cycles / fast.true_cycles))
        print(wl.name, flush=True)
    fr, dd, bb = pd.DataFrame(front), pd.DataFrame(rows_d), pd.DataFrame(rows_b)
    fr.to_csv(f"{out}/constrained_frontier.csv", index=False)
    dd.to_csv(f"{out}/constrained_deadline.csv", index=False)
    bb.to_csv(f"{out}/constrained_budget.csv", index=False)
    dd["policy"] = dd.margin.map(lambda m: "estimator + verification (simulator)" if m < 0 else f"estimator only, margin {m:.2f}")
    d_sum = dd.groupby(["slack", "policy"]).agg(violation_rate=("violated", "mean"), fallback_rate=("fallback", "mean"),
                                                energy_vs_cycle_optimal=("energy_vs_fast", gm), mean_verification_sims=("checks", "mean")).reset_index()
    b_sum = bb.groupby("budget_factor").agg(violation_rate=("violated", "mean"), fallback_rate=("fallback", "mean"),
                                            cycles_vs_cycle_optimal=("cycles_vs_fast", gm)).reset_index()
    rep = ["# Resource-bounded operation (mechanism F; MODEL ONLY; `python3 -m armse.consstudy`)\n",
           "Energy = Level-2 analytical model with illustrative constants (NOT measured); traffic, hence energy, is exact in the model. "
           "Per workload a pool of ~90 candidate schedules (best by estimated cycles + best by traffic) is simulated. `slack` = deadline / "
           "fastest achievable cycles in the pool (0.9 = infeasible by construction). The controller chooses with the ESTIMATOR and a safety "
           "`margin` on the deadline; violations are judged on SIMULATED cycles. Fallback = nothing predicted feasible -> fastest schedule, "
           "constraint not guaranteed. `estimator + verification` simulates candidates in objective order and commits to the first that meets "
           "the deadline in simulation (exact model), else the fastest verified schedule.\n",
           "## Latency-energy trade-off (cycle-optimal vs energy-optimal schedule in the pool)\n",
           f"Energy saving of the energy-optimal schedule: geomean {gm(1 - fr.energy_saving):.3f}x energy (mean saving {100 * fr.energy_saving.mean():.1f} %, "
           f"max {100 * fr.energy_saving.max():.1f} %); it costs {gm(fr.cycle_cost):.3f}x cycles (max {fr.cycle_cost.max():.2f}x).\n",
           "## Deadline mode (minimise energy subject to the deadline)\n", md(d_sum, "{:.3f}"),
           "\n## Energy-budget mode (minimise cycles subject to the budget; budget = factor x minimum achievable energy)\n", md(b_sum, "{:.3f}"),
           "\n## Per-workload frontier\n", md(fr, "{:.3f}")]
    open(f"{out}/CONSTRAINED_OPERATION.md", "w").write("\n".join(rep))
    print(d_sum.round(3).to_string()); print(b_sum.round(3).to_string()); print(fr[["energy_saving", "cycle_cost"]].describe().round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
