# ARMSE: A Workload-Aware Memory Scheduling Study for INT8 FPGA-Class Inference Accelerators

**Status of this document:** technical report of a *simulation-level* study. All performance numbers are produced by the project's own Python simulator or its cycle-level RTL transcription; the Verilog RTL was simulated by the project owner with Icarus Verilog (7 fixed cases pass; see section 11). **No synthesis, board, bandwidth or energy measurement has been performed.** Related-work statements come from the author's recall (no literature search was possible) and must be verified before citation. No novelty claim is made.

---

## Abstract

Moving weights, activations and partial sums between memory levels limits quantised neural-network inference on small FPGA-class accelerators. We built ARMSE (Adaptive Reuse and Memory Scheduling Engine): a parameterised local-memory and transfer architecture (Verilog RTL: skewed banked buffers, DMA, MAC-lane compute engine, two dependency-counter descriptor sequencers), a deterministic host-side scheduler that selects tile shape, loop order and buffer allocation under a closed-form cost model, and a cycle-level simulator with deadlock and hazard checking. On 30 INT8 GEMM-lowered workloads and one hardware configuration, the adaptive scheduler with simulator feedback needs 0.73x the cycles of a fixed double-buffered baseline and 0.87x those of the strongest fixed-shape baseline (29 wins, 1 loss), with the gain coming mainly from tile-shape freedom. The gain is not a traffic gain (adaptive schedules read more bytes than the best square tiling), analytical energy is lower than the fixed baseline but higher than the best square tiling, 25 of 30 workloads are compute-bound, and operator fusion of two-GEMM chains is mostly a loss at on-chip capacities of 2-32 KB. Results are model-level and conditional on stated assumptions.

---

## 1. Problem and hypothesis

**Problem.** Data movement, not arithmetic, often dominates latency and energy when weights and intermediate tensors exceed on-chip memory.

**Hypothesis (to be tested, not assumed).** A workload-aware memory subsystem that selects tensor tiling, buffer allocation, data reuse and transfer scheduling from operator shape, memory capacity and observed behaviour reduces memory traffic and compute stalls versus an appropriate fixed-policy baseline under equivalent hardware resources.

**Falsifiable form used.** For fixed on-chip capacity and external bandwidth, a deterministic cost-model-driven schedule reduces *cycles* by at least 10 % versus the best fixed-shape double-buffered policy on at least one workload class, after including scheduler overhead and resource cost, with bit-exact results.

---

## 2. Related work (from recall; verify before citing)

Mapping/dataflow search with analytical cost models (MAESTRO, Timeloop, Interstellar), fixed-dataflow accelerators (TPU weight-stationary systolic array; Eyeriss row-stationary), flexible-interconnect designs (MAERI), cross-layer fusion (fused-layer CNN accelerators; FlashAttention-style IO-aware tiling on GPUs), FPGA compile-time dataflow toolflows (FINN, hls4ml, DNNBuilder), software-managed scratchpads, double buffering and banked/skewed memories, compression and near-memory computing. **Consequence: choosing tile shape, order and buffering by a cost-model search is established prior art.** ARMSE does not claim to introduce it. See `docs/RELATED_WORK.md`.

---

## 3. System

### 3.1 What is hardware, software and model
* **RTL (Verilog):** `armse_skew_mem` (skewed banked operand buffer), `armse_obuf`, `armse_dma`, `armse_compute`, `armse_top`.
* **Host software (Python):** schedule enumeration, cost function, legality, descriptor-program generation.
* **Models (Python):** event-driven simulator, closed-form estimator, bank/energy analysis, RTL transcription, fusion extension.

### 3.2 Architecture
* **Decoupled access/execute:** a DMA program (LOAD_A, LOAD_B, WRITEBACK) and a compute program, each entry waiting on a completed-job counter of the other engine (`DEP`). Single versus double buffering is purely a different set of `DEP` values.
* **Memory:** external memory (one outstanding burst, `row_lat` cycles to first beat, `bus_bytes` per beat) -> A and B buffers (NB byte-wide banks, 1R+1W each) and an INT32 output/partial-sum buffer.
* **Skewed bank mapping:** element (r, c) -> bank (r + c) mod NB, word r*stride + c/NB. B is stored transposed. Lane reads and 4-element DMA writes are conflict-free by construction (exhaustively tested for NB in {4, 8, 16}).
* **Compute:** NB parallel MAC lanes along K; one (i, j) output pair per issue group; INT32 accumulation across K tiles in the output buffer.
* **Timing contract:** DMA job = 2 + rows*(row_lat + 1 + beats); compute job = tm*tn*ceil(tk/NB) + 4; first start at cycle 1 after `go`.

### 3.3 Scheduler
Deterministic: validate -> enumerate (tile sizes per dimension x 2 loop orders x 4 buffering pairs) -> reject illegal (capacity, descriptor field width, program length) -> rank by estimated cycles (ties by bytes, buffering, order, tile) -> fallback to the minimal 1x1xK tile or `NoFitError`. **D_feedback** simulates the estimator's top-8 and keeps the best measured. Cost-function assumptions are in `armse/analytic.py` (traffic and busy cycles exact; overlap approximated).

### 3.4 Baselines (identical physical buffers, bus and lanes)
* **A_direct:** one output element at a time, no reuse.
* **B_fixed_single:** fixed 32^3 tile, single-buffered.
* **C_fixed_double:** fixed 16^3 tile, ping-pong buffered.
* **C_best_square:** best square-tile schedule per workload by simulation (over the estimator's top-16 square candidates plus B and C). Included so that the comparison is not against a weak baseline.

---

## 4. Methodology

**Hardware configuration (headline):** 8 lanes, 4 B/cycle bus, 12-cycle burst latency, A buffer 2 KB, B buffer 2 KB, output buffer 4 KB (8 KB on-chip data). Fixed square tile: 32 (single), 16 (double).

**Workloads (30):** square GEMMs 16-256, GEMV (two orientations), tall/wide/deep-K skinny shapes, ragged shapes, MLP layers (batch 1 and 32), convolutions lowered by im2col, attention QK^T and PV GEMMs, capacity-stress shapes.

**Metrics:** cycles, compute-busy, memory-stall, DMA-busy (sync / latency / data beats), bytes read and written, transfers, MAC utilisation, buffer utilisation, descriptor-program size, host scheduler time. Energy: Level 1 (activity) and Level 2 (analytical, illustrative constants) only.

**Verification performed:** independent int64 numpy reference for 144 workload x policy runs (0 mismatches) plus 150 random cases; deadlock and buffer-hazard checking for every program (checker shown to catch deliberately broken programs); estimator exactness tests (600 random cases); RTL transcription on 7 fixed and 40 random cases (bit-exact data, exact cycles, bytes, transfers, busy counters, zero conflicts); RTL simulated in Icarus Verilog on the 7 fixed cases by the project owner.

---

## 5. Results (model only)

### 5.1 Cycles (30 workloads, geomean)
| Policy | vs C_fixed_double | vs C_best_square | wins / ties / losses vs C_best_square |
|---|---|---|---|
| A_direct | 59.64x | 70.70x | 0 / 0 / 30 |
| B_fixed_single | 1.079x | 1.279x | 0 / 1 / 29 |
| C_fixed_double | 1.000x | 1.185x | 0 / 2 / 28 |
| C_best_square | 0.844x | 1.000x | - |
| **D_adaptive** | **0.749x** | **0.888x** | 26 / 0 / 4 |
| **D_feedback** | **0.733x** | **0.869x** | 29 / 0 / 1 |

By family (relative to C_best_square): GEMV 0.704 (D), MLP 0.81-0.82, conv 0.84-0.87, GEMM 0.86-0.89, attention 0.89-0.93, stress 0.92, ragged 0.96-0.97, skinny 1.02-1.04.

### 5.2 External read traffic (geomean, vs C_fixed_double / vs compulsory)
A 6.95x / 23.2x; B 0.61x / 2.03x; C_best_square 0.63x / 2.11x; D_adaptive 0.94x / 3.12x; D_feedback 0.83x / 2.76x. **Adaptive schedules read more than the best square tiling.**

### 5.3 Ablation (estimator-selected D, cycles vs C_fixed_double)
Full 0.749; no tile-shape freedom (square only) 0.844; no loop-order choice 0.747; no buffer-allocation choice 0.752; no overlap (single buffering only) 0.945 (with read bytes 0.540 vs 0.938). **Tile-shape freedom is the main contributor; order and allocation choice add about nothing on this suite.**

### 5.4 Estimator calibration
Traffic exact. Cycle error of the estimator on the schedules each policy selects: D_adaptive 5.1 % mean / 14.6 % max; D_feedback 2.8 % / 14.6 %; C_fixed_double 1.0 % / 7.6 %. Estimator top-1 equalled the best of its own top-8 (by simulation) on 12/30 workloads.

### 5.5 Capacity and latency/bandwidth sensitivity (8-workload subset)
D_feedback / C_best_square by capacity scale: 0.25x 0.907, 0.5x 0.816, 1x 0.873, 2x 0.937, 4x 0.963. D_feedback / C_fixed_double: 0.304, 0.779, 0.739, 0.915, 0.915. Burst latency 4 / 12 / 32 cycles: D_fb / C = 0.897 / 0.739 / 0.398. Bus 4-32 B/cycle: D_fb / best-square 0.84-0.87.

### 5.6 Bound classification and runtime attribution
Machine balance 4 ops/B. **25 of 30 workloads are compute-bound; 5 are memory-bound** (both GEMVs and the three batch-1 MLP layers). Under D_feedback, compute is busy 80.9 % of cycles on average (median 94.6 %) with 18.0 % memory stall (median 5.2 %). DMA-busy time splits into synchronisation 0.8 %, per-burst latency 54.8 %, data beats 44.4 % (means): **latency, not bandwidth, dominates DMA time at this configuration.** Avoidable reads (beyond compulsory) average 57.2 % of reads for C and 48.8 % for D_feedback.

### 5.7 Banking
Skewed mapping: zero extra cycles. With a naive row-major B layout, a schedule chosen without knowing about conflicts runs 1.0-8.0x slower (e.g. gemm_sq64 38,771 -> 263,161 cycles; mlp_b32_L0 806,151 -> 6,425,863); a conflict-aware estimator recovers most (gemm_sq64 40,889). A synthetic 8-bank stride study shows naive linear mapping conflicting for 32 of 64 strides (worst 7 extra cycles).

### 5.8 Energy (Level 2, analytical, illustrative constants - NOT measured)
D_feedback total energy vs C_fixed_double: 0.909 / 0.898 / 0.891 at external-memory energy scale 0.5 / 1 / 2; vs C_best_square: 1.156 / 1.187 / 1.208. Adaptive scheduling saves energy against the fixed baseline but costs more than the best square tiling because it moves more bytes.

### 5.9 Cost of the architecture
Same on-chip data memory for every policy (8 KB). Unrolled descriptor programs: mean 11.4 KB (D_adaptive), 39.2 KB (C_fixed_double), 5.8 KB (C_best_square), 1.2 MB (A_direct); maximum 57 KB for D. Host scheduling: 0.37-0.40 s mean in Python for D (not a hardware cost). LUT/FF/DSP/BRAM, Fmax and timing closure: **not measured.**
* Program compression (lossless repeat-with-delta, decode verified): only 1.37x geomean for D (1.32x for C); 16x on one regular workload. A simple hardware loop generator is therefore not clearly worthwhile.
* Descriptor streaming from external memory (model): D_feedback slows 1.3 % geomean (median 0.6 %, worst 11.6 %) at block size 8; C_fixed_double 3.3 %; D/C remains 0.719. Not implemented in RTL.

### 5.10 Fusion of GEMM chains (model only)
Row-block fusion keeps the requantised INT8 intermediate on chip. Over 20 chain x buffer-size cases (2-32 KB A buffer): barrier version geomean 1.096x cycles vs unfused (faster in 5/20); with cross-phase overlap 1.084x (faster in 7/20). Best: attention (seq 64, d 32) 0.951x at 32 KB. Dominant cost: B1 and B2 re-streamed once per row block. Bit-exact verification of all fused configurations.

---

## 6. Answers to the research questions

1. **Compute-bound vs memory-bound?** 25/30 compute-bound, 5 memory-bound (GEMVs, batch-1 MLP layers) at 4 ops/B balance (section 5.6).
2. **Which transfers are avoidable through reuse?** On average 48.8 % (D_feedback) to 57.2 % (C) of external reads exceed the compulsory minimum; best square tiling avoids more than adaptive scheduling.
3. **How much can be retained on chip?** Limited by 8 KB: buffer utilisation averages 0.59 (D) of capacity; retention happens only when a tile id repeats between consecutive iterations (e.g. a full-K tile).
4. **Which access patterns cause bank conflicts / underutilisation?** Column-wise (B) lane reads under row-major layout (gcd(tn, NB) serialisation); fixed by skewed mapping (section 5.7).
5. **Where does runtime go?** Compute-busy 80.9 %, memory stall 18.0 %; DMA time dominated by burst latency (54.8 %) over data beats (44.4 %).
6. **When does adaptive scheduling beat simpler fixed strategies?** Largest on memory-bound shapes (GEMV 0.70x), MLP/conv, and at higher burst latency or reduced capacity; vanishes at high capacity (0.96x at 4x) and for skinny/ragged shapes.
7. **Additional cost?** Larger descriptor programs than the best square baseline, host-side search time, and (unmeasured) logic. Streaming programs from external memory costs about 1-3 % cycles in the model.
8. **Where does it lose?** wide_16x512x64 (both D variants); deepK_32x32x1024, mlp_b1_L2 and mlp_b32_L1 (D_adaptive only); compute-bound workloads; large buffers; and in traffic and analytical energy against the best square tiling.

---

## 7. Established techniques versus what this project adds

* **Established (not claimed as new):** tiling, loop-order choice, double buffering, banked and skewed memories, descriptor-driven DMA, analytical-cost-model mapping search, operator fusion.
* **What this project contributes as engineering:** a small, fully reproducible, verified reference stack (RTL + model + transcription + tests) in which these techniques are compared against strong fixed baselines under identical resources, with explicit loss analysis, calibration of a simple estimator against a cycle-level simulator, and measured negative results (fusion, descriptor compression).
* **Whether any piece is a publishable contribution is undetermined** and requires a real literature search plus hardware evidence.

---

## 8. Threats to validity and limitations

1. Single headline hardware configuration; sensitivity covered only on an 8-workload subset.
2. Model-level results; cycle constants verified against RTL simulation on 7 cases (Icarus) and 47 cases (transcription), not on hardware.
3. im2col convolution counts lowered-matrix traffic (no sliding-window reuse); attention without softmax; MLP without activation.
4. C_best_square and D_feedback use simulator-in-the-loop information that a deployed system would obtain by profiling.
5. Idealised external memory (one outstanding burst, unaligned rows allowed, no errors, sustained = peak).
6. Fixed per-buffer capacity (no dynamic repartitioning); single DMA channel.
7. Level-2 energy uses illustrative constants typed from memory; no Level-3 measurement.
8. No synthesis: resource and timing cost unknown. Two memories use asynchronous reads (LUT-RAM when synthesised).
9. Fusion and descriptor streaming exist only in the model.
10. Related-work content is unverified recall.

---

## 9. Reproducibility guide

Environment: Python 3.9+ with numpy, pandas, matplotlib (Windows PowerShell commands shown; `make` equivalents exist).
```
python -m unittest discover -s tests -t .                  # 39 tests
python -m armse.bench                                      # main benchmark  -> results/REPORT.md (+CSV, PNG)
python -m armse.fusionbench                                # fusion (barrier)
python -m armse.overlapbench                               # fusion (overlap)
python -m armse.progbench                                  # descriptor compression
python -m armse.progstream                                 # descriptor streaming
python -m armse.rtlexport . --random 40                    # regenerate RTL vectors
python -m armse.rtlmodel rtl_vectors                       # RTL transcription check
powershell -ExecutionPolicy Bypass -File scripts\run_rtl_tests.ps1   # Icarus Verilog (needs iverilog)
powershell -ExecutionPolicy Bypass -File scripts\run_synth.ps1       # Yosys estimate (needs yosys; not yet run)
```
All benchmarks are deterministic (fixed seeds, deterministic tie-breaking). Scheduler wall-clock figures vary by machine.

---

## 10. Future work (see ARMSE_MASTER_PROMPT_v2.md section 8)
Synthesis and timing on a chosen device part (no board needed); measured bandwidth and power once a board exists; RTL descriptor streaming; estimator improvement with held-out workloads; direct convolution and larger transformer shapes; hardware requantiser and resident-operand stride for fusion; a real literature search and novelty assessment; integration with the owner's existing accelerator only after explicit approval.

---

## 11. Verification status at the time of writing

| Item | Status |
|---|---|
| Python model, scheduler, simulator, tests (39) | executed, passing |
| Functional correctness vs independent reference | executed, 0 mismatches |
| RTL transcription (7 fixed + 40 random) | executed, exact agreement |
| Verilog in Icarus (skew-memory test + 7 fixed cases) | executed by project owner, all PASS, cycle delta 0 |
| Verilog in Icarus, 40 random cases | not yet run |
| CI workflow, Verilator lint | not yet run |
| Yosys resource estimate | script written, not run |
| Synthesis for a device, Fmax, board, energy | not done |

## 12. Corrections log
* Earlier text said "24/30 workloads are compute-bound"; the result table gives **25 compute-bound, 5 memory-bound**. Corrected here and in README and the master prompt.
* An earlier generated sentence implied a hardware loop generator would remove most descriptor memory; the compression study shows only ~1.4x with simple repeat-with-delta encoding. Corrected in `armse/bench.py` and `results/REPORT.md`.
