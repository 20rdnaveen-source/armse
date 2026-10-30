# Estimator study (MODEL ONLY, `python3 -m armse.estfit`)

Samples: 2400 (workload, schedule) pairs over 30 workloads (top-40 by v1 + up to 40 random legal per workload, programs <= 20k iterations). `regret` = true cycles of the candidate each estimator picks, relative to the best true cycles in the SAMPLED pool of that workload (so it is a lower bound on the regret vs the full candidate space). v2 coefficients are fitted on all other workloads and evaluated on the held-out one (leave-one-workload-out).

| estimator | mean_abs_err_pct | p90_abs_err_pct | mean_abs_err_pct_top40 | mean_regret_pct | worst_regret_pct | exact_best_pick |
|---|---|---|---|---|---|---|
| v1 (closed form) | 2.68 | 9.34 | 4.60 | 4.18 | 12.38 | 9 |
| v2 (class-wise linear correction, LOO) | 2.26 | 6.18 | 3.47 | 1.75 | 9.89 | 10 |

## End-to-end: chosen schedule's SIMULATED cycles over the 30-workload suite (no simulator used to choose D_v1 / D_v2)

D_v2_loo: coefficients fitted WITHOUT the evaluated workload; the estimator ranks all legal candidates, only the winner is simulated. D_fb simulates the estimator's top-8 (uses the simulator). The coefficient file in `estimator_v2_coef.json` is fitted on all workloads and is specific to the default hardware configuration.

| policy | vs_best_square | vs_C_fixed_double | wins_vs_best_square | ties | losses |
|---|---|---|---|---|---|
| D_v1 | 0.888 | 0.749 | 26 | 0 | 4 |
| D_v2_loo | 0.870 | 0.734 | 28 | 0 | 2 |
| D_fb | 0.869 | 0.733 | 29 | 0 | 1 |