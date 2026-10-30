# ARMSE-X proposal: coverage check against what this project has built

Source: the "ARMSE-X: Adaptive, Resource-Bounded Memory Orchestration for Edge AI" proposal text. Status legend: **DONE** = built and verified at the stated level; **PARTIAL** = some of it built; **FUTURE** = not built, planned; **NOT ADDED** = not built and not currently planned. "Model" = Python simulator (verified against an independent reference); "RTL" = Verilog (7 fixed cases simulated in Icarus by the project owner; 40 random cases pending). No synthesis, board or literature verification exists.

## The central question in the proposal
*Does jointly choosing memory policies beat strong, individually optimised fixed policies?*
Our evidence (model, 30 workloads, default configuration): the adaptive scheduler (tile shape x loop order x buffering chosen jointly under a fixed capacity) needs 0.869x (simulator-assisted) / 0.870x (calibrated estimator) the cycles of the strongest square-tiling baseline. **But the ablation shows the joint selection is mostly one dimension:** removing tile-shape freedom costs 0.749 -> 0.844 (vs fixed double buffering); removing loop-order choice (0.747) or buffer-allocation choice (0.752) changes essentially nothing on this suite; removing overlap (single buffering) costs 0.945. So the data currently supports "tile-shape adaptation + overlap", not "joint multi-policy selection" as such.

## The six mechanisms
| Mech. | Proposal | Status | What exists | What is missing |
|---|---|---|---|---|
| A | Capacity-aware tensor lifetime management | **PARTIAL (model only)** | `armse/memplan.py`: lifetime-interval planner reaching the peak-live bound on 6 graphs (12-60 % less than naive), verified non-overlapping; used in `edgestudy` | illustrative hand-set lifetimes (not from a real graph); activations only; no spill/eviction model; not coupled to the scheduler; no RTL; "keep frequently reused data local" not modelled |
| B | Reuse-aware, capacity-constrained scheduling | **DONE (model + RTL execution)** | D / D_feedback / D_v2: cost-model search over tile, order, buffering under fixed capacity; legality proven by simulator; RTL runs the programs | joint optimisation across several tensors/layers; partial-sum vs operand capacity trade-offs only through fixed buffer split |
| C | Conflict-aware bank placement | **PARTIAL** | skewed mapping in RTL (zero conflicts, exhaustive test); conflict-aware estimator; naive-vs-skew study (up to ~8x) | no dynamic bank assignment; no layout-conversion cost; not coordinated with lifetimes; static mapping only |
| D | Stall-aware transfer scheduling | **DONE (model + RTL)** | decoupled DMA/compute, two-counter dependencies, double buffering chosen only when estimated to help, stall and DMA-time attribution | arbitration among several masters; bandwidth contention; lookahead deeper than one slot |
| E | Runtime adaptation using hardware counters | **PARTIAL (closed loop built in the model; no hardware controller)** | RTL counters; `armse/adaptive.py`: after each layer infers latency / bus width / banking from counters + descriptors and re-plans (exact in the model; tests); result: large gain only for large persistent model error (naive banking 0.333x vs static), 0-5 % for mild latency/bus errors; hybrid bucketed re-planning for dynamic shapes | counters on real hardware (noise, contention); a hardware or firmware controller; adaptation at tile (not layer) granularity; first-layer misplanning cost remains (2.1-2.5x oracle in the banking scenarios) |
| F | Safe operation under hard resource limits | **PARTIAL (built in the model; nothing in hardware)** | capacity/field/program-length feasibility; fallback to minimal tile; `NoFitError`; `armse/constrained.py`: latency-deadline and energy-budget planning, estimator-proposes / exact-model-verifies selection (0 % violations for feasible deadlines in the study, 10.1 verification sims at slack 1.0), known-good fallback flagged when infeasible; latency-energy frontier (energy-optimal saves 18.9 % for 1.053x cycles) | worst-case-execution-time guarantees on real hardware (variability, contention); RTL watchdog / illegal-descriptor traps; energy measured, not modelled; fallback-frequency data on real workloads |

## Compression / sparsity / quantisation track
| Item | Status |
|---|---|
| Quantisation (INT4 weights) | **PARTIAL (model only)**: traffic/cycle/energy effect measured (cycles 0.973x, bytes 0.801x); **accuracy NOT evaluated**; no RTL |
| Lossless compression | **NOT ADDED** (only descriptor-program compression studied; weak, 1.37x) |
| Structured/unstructured sparsity, zero skipping | **NOT ADDED** |
| Layout-aware intermediate storage | **NOT ADDED** (fusion keeps the intermediate on chip, model only, mostly a loss at <= 32 KB) |

## Evaluation plan in the proposal
| Requirement | Status |
|---|---|
| Simple streaming baseline | **DONE** (A_direct) |
| Fixed tiling; static reuse + double buffering | **DONE** (B, C) |
| Best feasible fixed policy per workload | **DONE** (C_best_square, square tiles only; non-square offline-best corresponds to D_feedback) |
| Runtime adaptation vs offline optimum | **PARTIAL**: static-vs-runtime-vs-hybrid study (KV-decode, prefill) plus a counter-driven closed loop under model mismatch (`results/ADAPTIVE_LOOP.md`); no hardware counters, one sequence |
| Ablations of each mechanism | **PARTIAL**: scheduling features ablated; lifetimes, bank placement, compression not ablated inside one combined system |
| SCALE-Sim v3 and Timeloop/Accelergy comparison | **NOT ADDED** (no network in the authoring environment; these are the missing "strongest established simulator" baselines) |
| Metrics: latency, external bytes, stalls, SRAM use | **DONE** (model) |
| Metrics: energy | **PARTIAL**: Level-1/2 only, illustrative constants |
| Metrics: LUT / FF / timing | **FUTURE** (Yosys script written, not run; no Fmax without a device flow) |
| Simulator vs synthesis vs hardware reported separately | **DONE** (every document labels the evidence level) |

## Roadmap stages in the proposal
1 Research and baseline: **PARTIAL** (baselines done; no verified literature review) | 2 Memory-system simulator: **DONE** | 3 Static optimisation: **DONE** | 4 Adaptive controller: **PARTIAL** (simulator-in-the-loop and calibrated estimator; no hardware counter loop) | 5 RTL: **PARTIAL** (scheduler is host software; buffers, banking, DMA, counters in RTL) | 6 Validation/integration: **PARTIAL** (RTL simulated on 7 cases; no synthesis, board, or integration with the existing accelerator, whose files have not been received).

## Honest reading
Built and measured: B, D, most of C, parts of A/E/F, the baselines and the model-level evidence. Not built: closed-loop *hardware* adaptation (E, built in the model only), deadline/energy-constrained operation in hardware (F, built in the model only), eviction-aware local-memory management and coupling of lifetimes to scheduling (A), sparsity and lossless compression, the SCALE-Sim/Timeloop comparison, and everything that needs a device or board. The proposal's novelty claim is unestablished (see `docs/RESEARCH_DECISION.md`); the closest prior work must be checked before any originality claim.
