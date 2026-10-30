# Counter-driven runtime adaptation (mechanism E; MODEL ONLY; `python3 -m armse.adaptstudy`)

Sequence of 8 layers (CNN, MLP, attention GEMMs). The scheduler plans with a nominal hardware model; the layers execute on the TRUE hardware (simulated). Static = plan everything with the nominal model; Adaptive = after each layer infer latency / bus width / banking behaviour from counters (dma_busy, cmp_busy, n_xfer, bytes + host-known descriptors) and re-plan later layers; Oracle = plan with the true model. All three use the same raw estimator (no simulation to plan). Cycles are simulated on the true hardware; the first layer is always planned with the nominal model in Adaptive mode (its cost is included). Plan seconds are host Python time.

| scenario | static_cycles | adaptive_cycles | oracle_cycles | adaptive_vs_static | adaptive_vs_oracle | replans | inferred | true | plan_s_static | plan_s_adaptive |
|---|---|---|---|---|---|---|---|---|---|---|
| no mismatch (control) | 2042499 | 2042499 | 2042499 | 1.000 | 1.000 | 0 | lat=12 bus=4 bank=skew | lat=12 bus=4 bank=skew | 2.945 | 0.000 |
| latency 12 -> 30 cycles | 2416170 | 2306228 | 2132878 | 0.954 | 1.081 | 1 | lat=30 bus=4 bank=skew | lat=30 bus=4 bank=skew | 0.000 | 2.717 |
| latency 12 -> 4 cycles | 1993999 | 1999112 | 1998896 | 1.003 | 1.000 | 1 | lat=4 bus=4 bank=skew | lat=4 bus=4 bank=skew | 0.000 | 2.691 |
| bus assumed 8 B, true 4 B | 2018398 | 2021973 | 2042499 | 1.002 | 0.990 | 2 | lat=12 bus=4 bank=skew | lat=12 bus=4 bank=skew | 3.139 | 0.472 |
| banking: true naive layout | 15786156 | 5255426 | 2140062 | 0.333 | 2.456 | 1 | lat=12 bus=4 bank=naive | lat=12 bus=4 bank=naive | 0.000 | 2.669 |
| latency 30 AND naive banking | 15795305 | 5732671 | 2717803 | 0.363 | 2.109 | 1 | lat=30 bus=4 bank=naive | lat=30 bus=4 bank=naive | 0.000 | 2.799 |

The inference is exact here because the model's timing formulas are exact; on real hardware the counters would be noisier (contention, refresh, unmodelled stalls) and the inferred parameters would be estimates. The banking scenario assumes hardware whose read conflicts are visible only through longer compute time: the implemented RTL uses skewed banking and has none.
