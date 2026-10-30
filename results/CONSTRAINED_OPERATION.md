# Resource-bounded operation (mechanism F; MODEL ONLY; `python3 -m armse.consstudy`)

Energy = Level-2 analytical model with illustrative constants (NOT measured); traffic, hence energy, is exact in the model. Per workload a pool of ~90 candidate schedules (best by estimated cycles + best by traffic) is simulated. `slack` = deadline / fastest achievable cycles in the pool (0.9 = infeasible by construction). The controller chooses with the ESTIMATOR and a safety `margin` on the deadline; violations are judged on SIMULATED cycles. Fallback = nothing predicted feasible -> fastest schedule, constraint not guaranteed. `estimator + verification` simulates candidates in objective order and commits to the first that meets the deadline in simulation (exact model), else the fastest verified schedule.

## Latency-energy trade-off (cycle-optimal vs energy-optimal schedule in the pool)

Energy saving of the energy-optimal schedule: geomean 0.800x energy (mean saving 18.9 %, max 40.6 %); it costs 1.052x cycles (max 1.29x).

## Deadline mode (minimise energy subject to the deadline)

| slack | policy | violation_rate | fallback_rate | energy_vs_cycle_optimal | mean_verification_sims |
|---|---|---|---|---|---|
| 0.900 | estimator + verification (simulator) | 1.000 | 1.000 | 1.000 | 90.000 |
| 0.900 | estimator only, margin 0.00 | 1.000 | 1.000 | 1.095 | 0.000 |
| 0.900 | estimator only, margin 0.05 | 1.000 | 1.000 | 1.095 | 0.000 |
| 0.900 | estimator only, margin 0.10 | 1.000 | 1.000 | 1.095 | 0.000 |
| 1.000 | estimator + verification (simulator) | 0.000 | 0.000 | 1.000 | 10.133 |
| 1.000 | estimator only, margin 0.00 | 0.633 | 0.000 | 0.955 | 0.000 |
| 1.000 | estimator only, margin 0.05 | 0.700 | 0.933 | 1.095 | 0.000 |
| 1.000 | estimator only, margin 0.10 | 0.700 | 1.000 | 1.095 | 0.000 |
| 1.050 | estimator + verification (simulator) | 0.000 | 0.000 | 0.879 | 2.633 |
| 1.050 | estimator only, margin 0.00 | 0.133 | 0.000 | 0.870 | 0.000 |
| 1.050 | estimator only, margin 0.05 | 0.267 | 0.433 | 1.015 | 0.000 |
| 1.050 | estimator only, margin 0.10 | 0.367 | 0.967 | 1.095 | 0.000 |
| 1.100 | estimator + verification (simulator) | 0.000 | 0.000 | 0.833 | 1.100 |
| 1.100 | estimator only, margin 0.00 | 0.067 | 0.000 | 0.828 | 0.000 |
| 1.100 | estimator only, margin 0.05 | 0.033 | 0.000 | 0.870 | 0.000 |
| 1.100 | estimator only, margin 0.10 | 0.067 | 0.600 | 1.044 | 0.000 |
| 1.250 | estimator + verification (simulator) | 0.000 | 0.000 | 0.800 | 1.000 |
| 1.250 | estimator only, margin 0.00 | 0.000 | 0.000 | 0.800 | 0.000 |
| 1.250 | estimator only, margin 0.05 | 0.000 | 0.000 | 0.800 | 0.000 |
| 1.250 | estimator only, margin 0.10 | 0.000 | 0.000 | 0.815 | 0.000 |
| 1.500 | estimator + verification (simulator) | 0.000 | 0.000 | 0.800 | 1.000 |
| 1.500 | estimator only, margin 0.00 | 0.000 | 0.000 | 0.800 | 0.000 |
| 1.500 | estimator only, margin 0.05 | 0.000 | 0.000 | 0.800 | 0.000 |
| 1.500 | estimator only, margin 0.10 | 0.000 | 0.000 | 0.800 | 0.000 |

## Energy-budget mode (minimise cycles subject to the budget; budget = factor x minimum achievable energy)

| budget_factor | violation_rate | fallback_rate | cycles_vs_cycle_optimal |
|---|---|---|---|
| 1.000 | 0.000 | 0.000 | 1.054 |
| 1.050 | 0.000 | 0.000 | 1.045 |
| 1.100 | 0.000 | 0.000 | 1.044 |
| 1.250 | 0.000 | 0.000 | 1.047 |
| 1.500 | 0.000 | 0.000 | 1.031 |
| 2.000 | 0.000 | 0.000 | 1.037 |

## Per-workload frontier

| workload | cycles_fast | energy_fast | cycles_lean | energy_lean | energy_saving | cycle_cost |
|---|---|---|---|---|---|---|
| gemm_sq16 | 1082 | 262988.800 | 1082 | 262988.800 | 0.000 | 1.000 |
| gemm_sq32 | 4969 | 1118310.400 | 4969 | 1118310.400 | 0.000 | 1.000 |
| gemm_sq64 | 34340 | 8306483.200 | 39439 | 5664563.200 | 0.318 | 1.148 |
| gemm_sq128 | 264313 | 55966105.600 | 286613 | 40114585.600 | 0.283 | 1.084 |
| gemm_sq192 | 888199 | 159256166.400 | 938861 | 123590246.400 | 0.224 | 1.057 |
| gemm_sq256 | 2102983 | 363515084.800 | 2192837 | 278973644.800 | 0.233 | 1.043 |
| gemv_1x256x256 | 24675 | 11079500.800 | 24675 | 11079500.800 | 0.000 | 1.000 |
| gemv_256x1x256 | 68597 | 11327180.800 | 88577 | 11079500.800 | 0.022 | 1.291 |
| tall_512x16x64 | 67351 | 12815206.400 | 67351 | 12815206.400 | 0.000 | 1.000 |
| wide_16x512x64 | 67263 | 15292006.400 | 67687 | 12815206.400 | 0.162 | 1.006 |
| deepK_32x32x1024 | 132915 | 26037452.800 | 133655 | 15469772.800 | 0.406 | 1.006 |
| ragged_37x29x53 | 8934 | 2224609.450 | 9730 | 1481085.700 | 0.334 | 1.089 |
| ragged_101x67x131 | 120176 | 24925724.350 | 125535 | 17849268.100 | 0.284 | 1.045 |
| ragged_250x255x249 | 2046277 | 349208280.000 | 2133275 | 267299730.000 | 0.235 | 1.043 |
| mlp_b1_L0 | 74703 | 33593051.200 | 74703 | 33593051.200 | 0.000 | 1.000 |
| mlp_b1_L1 | 12404 | 5539750.400 | 12404 | 5539750.400 | 0.000 | 1.000 |
| mlp_b1_L2 | 2181 | 238624.000 | 2181 | 238624.000 | 0.000 | 1.000 |
| mlp_b32_L0 | 806151 | 128344678.400 | 816501 | 95981158.400 | 0.252 | 1.013 |
| mlp_b32_L1 | 132959 | 22719692.800 | 137981 | 17435852.800 | 0.233 | 1.038 |
| mlp_b32_L2 | 7007 | 1237568.000 | 7007 | 1237568.000 | 0.000 | 1.000 |
| conv3x3_16to32_28 | 454295 | 85301081.600 | 488631 | 67468121.600 | 0.209 | 1.076 |
| conv1x1_64to64_14 | 103319 | 23911244.800 | 114751 | 15985484.800 | 0.331 | 1.111 |
| conv5x5_3to8_32 | 83988 | 20211950.000 | 83988 | 20211950.000 | 0.000 | 1.000 |
| conv3x3_32to64_14 | 454195 | 82474201.600 | 470965 | 61669081.600 | 0.252 | 1.037 |
| attn_s64_d32_QKt | 17762 | 5463961.600 | 19007 | 3812761.600 | 0.302 | 1.070 |
| attn_s64_d32_PV | 17916 | 4153241.600 | 19751 | 2832281.600 | 0.318 | 1.102 |
| attn_s128_d64_QKt | 133607 | 34546892.800 | 151757 | 21337292.800 | 0.382 | 1.136 |
| attn_s128_d64_PV | 132921 | 27983052.800 | 143749 | 20057292.800 | 0.283 | 1.081 |
| stress_512x512x64 | 2102983 | 489344204.800 | 2360051 | 325545164.800 | 0.335 | 1.122 |
| stress_64x64x2048 | 1052359 | 163407462.400 | 1056717 | 121136742.400 | 0.259 | 1.004 |