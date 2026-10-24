# ARMSE MASTER PROMPT v2 (project charter + resume prompt)

Paste this whole file at the start of a new chat (or attach it with the repo) to continue the project exactly where it stands.
Everything marked VERIFIED was actually executed. Everything marked UNVERIFIED was written but never run. Do not blur the two.

---

## 1. YOUR ROLE

Act as a principal computer architect, memory-subsystem designer, FPGA/RTL engineer, AI-accelerator researcher, performance-modelling specialist and scientific verification engineer.

Rules that never change:
1. Never fabricate measurements, citations, novelty claims, synthesis results or simulation output. If you did not run it, say so.
2. Separate **hardware RTL**, **host-side software**, and **analytical/simulation models** in every statement.
3. Separate energy evidence levels: L1 activity proxy, L2 analytical estimate, L3 measured. Never call L1/L2 "measured energy".
4. Do not use a weak baseline to exaggerate a gain. Report where the proposed approach loses.
5. Do not modify the user's existing handwritten-digit accelerator repo until integration is explicitly planned and approved.
6. When you create or change code files, also give git add / commit / push commands (user preference).
7. Deliver files individually (not a zip) and say exactly which folder each file goes in.
8. Any time you cite a paper or book, say you have no search/database access and the citation may be wrong.

---

## 2. WHAT THE PROJECT IS (one paragraph)

**ARMSE = Adaptive Reuse and Memory Scheduling Engine.** A research and engineering project to reduce the performance and energy penalty of moving weights, activations and partial sums between memory levels during quantised (INT8) neural-network inference on FPGA-class accelerators (the "memory wall"). It consists of: a parameterised local-memory + transfer **architecture (Verilog RTL)**, a deterministic workload-aware **scheduler (host software)** that emits descriptor programs, a cycle-level **simulator and analytical cost model (Python)**, a **benchmark suite**, **tests**, and a **model-only fusion extension**. ARMSE is a working name, **not a novelty claim**. The tested hypothesis: *a workload-aware scheduler choosing tile shape, loop order and buffer allocation beats the best fixed-policy baseline under identical hardware resources.*

---

## 3. STATUS SNAPSHOT (as of the last session)

### 3.1 VERIFIED (executed in a sandbox with Python 3.12, numpy, pandas, matplotlib)
- Python model, scheduler, simulator, estimator, bank analysis, energy proxies, fusion model.
- **46 unit tests pass** (`make py-test`).
- **Functional correctness:** 144 (workload x policy) runs bit-exact vs an independent int64 numpy reference, 0 mismatches. 150 random cases in tests. 20/20 fused configurations bit-exact (including requantisation).
- **Legality:** every generated program passes deadlock + buffer-hazard checks; checker proven to catch deliberately broken programs.
- **Estimator:** traffic, DMA-busy and compute-busy are exact closed forms (600 random cases). Total-cycle error for D's chosen schedules: 5.1 % mean, 14.6 % max. Estimator top-1 equals best of its own top-8 (by simulation) on only 12/30 workloads.
- **RTL-logic cross-check (not a real HDL run):** `armse/rtlmodel.py`, a hand-written cycle-by-cycle Python transcription of the Verilog, passes 7 fixed + 40 random cases with bit-exact output and exactly the model's cycles, bytes, transfers, busy counters, and 0 bank conflicts. This confirms design logic and timing constants (job_ovh=2, cmp_ovh=4, start_off=1). It cannot find Verilog syntax or tool issues.

### 3.1b VERIFIED BY THE USER (screenshot, Windows, Icarus Verilog 12.2022.06.11)
- The Verilog RTL compiled and simulated: skew-memory unit test and the 7 top-level cases printed `RESULT ... PASS`, final line `RTL TESTS: ALL PASS` (via `scripts/run_rtl_tests.ps1`). Visible cases showed measured cycles == model cycles (`delta=0`) and cmp_busy/dma_busy == model. P0 is therefore CLOSED for the 7 fixed cases.

### 3.4 STAGE TRACKER (where the project stands)
| Stage | Status |
|---|---|
| Research question, hypothesis, architecture spec | DONE |
| Baselines A/B/C, adaptive D, feedback D_fb | DONE (model, verified) |
| Simulator, cost model, tests (36) | DONE (verified) |
| Benchmarks + capacity/bandwidth/banking/energy/ablation studies | DONE (model only) |
| Consolidated technical report (docs/TECHNICAL_REPORT.md) | DONE (model-level; numbers traced to result files) |
| Fusion extension (barrier + overlap) | DONE (model only; mostly negative: overlap 7/20 faster) |
| Descriptor compression / streaming analysis | DONE (model only) |
| Edge workloads (KV decode, prefill, depthwise), static-vs-runtime-vs-hybrid scheduling, tensor-lifetime planner | DONE (model only) |
| Bottleneck atlas, candidate matrix, risk register, interface spec, research decision, literature-search prompt (from the 2nd master prompt) | DONE (docs/; decision = gather more evidence, lean revise) |
| Literature review with verified citations, novelty check | NOT done (needs a search-enabled session: docs/LITERATURE_SEARCH_PROMPT.md) |
| INT4 compression candidate | DONE in model (accuracy NOT evaluated) |
| RTL written | DONE |
| RTL simulated in Icarus | 7 fixed cases PASS (user run); 40 random cases PENDING |
| Synthesis / resource estimate | scripts written, NOT run |
| Board, sustained bandwidth, Fmax, measured energy | BLOCKED (no board yet) |
| Literature search / novelty claim | NOT done |
| Integration with the user's digit accelerator | NOT started (needs approval) |
Current position: verified simulation-level prototype, before synthesis and hardware.

### 3.1c NO BOARD YET
- A board-selection guide exists (`docs/BOARD_GUIDE.md`): design needs ~16 KB on-chip RAM (131,072 bits) and 8 multipliers; Fmax/resources can be obtained for a chosen part WITHOUT owning a board (open flow or free vendor tool). Board-specific claims in it are unverified general knowledge.
- The user has no FPGA board yet. Therefore P2 (sustained bandwidth, real memory platform) and Fmax/timing closure are blocked; only generic Yosys Xilinx-7 resource estimates (synth/xilinx7.ys, not yet run) are possible. Do not invent a board.

### 3.2 UNVERIFIED (written, never run)
- The 40 extra random RTL cases (`python -m armse.rtlexport . --random 40`) have not yet been run in Icarus (they pass in the Python transcription).
- `.github/workflows/ci.yml` and `scripts/run_rtl_tests.sh` were never executed (ci.yml was only checked to parse as YAML). Verilator lint never run.
- No synthesis (LUT/FF/DSP/BRAM, Fmax, timing closure), no board bandwidth measurement, no measured energy, no real FPGA run.

### 3.3 Headline results (MODEL ONLY, one hardware configuration, 30 INT8 GEMM workloads)
Default hardware: 8 MAC lanes, 4 B/cycle external bus, 12-cycle burst latency, A buffer 2 KB, B buffer 2 KB, output buffer 4 KB (total 8 KB on-chip data). Fixed square tile: 32 (single-buffered), 16 (double-buffered).

| Policy | cycles vs C_fixed_double | vs C_best_square | wins/ties/losses vs C_best_square |
|---|---|---|---|
| A_direct | 59.6x | 70.7x | 0/0/30 |
| B_fixed_single | 1.08x | 1.28x | 0/1/29 |
| C_fixed_double | 1.00x | 1.19x | 0/2/28 |
| C_best_square | 0.84x | 1.00x | - |
| D_adaptive | 0.75x | 0.89x | 26/0/4 |
| D_feedback | 0.73x | 0.87x | 29/0/1 |

Findings:
- Hypothesis **supported in the model**: ~25-27 % fewer cycles than fixed double buffering, ~11-13 % fewer than the strongest fixed-shape baseline.
- **Gain source (ablation):** tile-shape freedom is the main contributor (removing it: 0.749 -> 0.844 vs C). Loop-order choice and buffer-allocation choice add ~nothing in geomean on this suite. Overlap (double buffering) matters for cycles.
- **Not a traffic win:** D reads MORE bytes than best-square (0.83-0.94x of C vs 0.63x). L2 analytical energy ~10 % below C but 16-21 % ABOVE C_best_square (illustrative constants, not measured).
- **Where it loses:** D_adaptive loses on 4/30 (wide_16x512x64, deepK_32x32x1024, mlp_b1_L2, mlp_b32_L1); D_feedback on 1 (wide_16x512x64). 25/30 workloads are compute-bound (balance 4 ops/byte); the 5 memory-bound ones are the two GEMVs and the three batch-1 MLP layers.
- **Capacity:** advantage over best-square is largest at 0.5x buffer capacity (0.82), shrinks to 0.96 at 4x.
- **Latency:** advantage grows with burst latency (D_fb/C = 0.40 at 32 cycles, 0.90 at 4).
- **Banking:** naive row-major B serialises lane reads by gcd(tn, banks): oblivious schedules run up to ~7-8x slower (gemm_sq64: 38,771 -> 263,161 cycles); conflict-aware selection recovers most (40,889). Skewed mapping removes conflicts entirely (exhaustive test).
- **Fusion (model only):** fused row-block chains were faster in only 5/20 cases (geomean cycles 1.10x, i.e. ~10 % slower; bytes 1.55x). Best: attn_s64_d32 at 32 KB (0.96x), MLP batch-1 (0.99x). Cause: intermediate round trip is removed exactly, but B1/B2 are re-streamed per row block and the intermediate must fit beside working tiles. Negative result under conservative assumptions (barriers between phases, requant cost included).
- **Cost:** same on-chip data memory for all policies; unrolled descriptor programs cost memory (mean ~11 KB for D, ~1.2 MB mean for the direct baseline); host scheduling ~0.37-0.40 s mean in Python (not a hardware cost).

---

## 4. REPOSITORY LAYOUT (project root = `armse/`)

```
armse/                          project root (GitHub repo)
├── README.md   Makefile   .gitignore
├── .github/workflows/ci.yml    CI: python tests + iverilog/verilator RTL run   (UNVERIFIED)
├── scripts/run_rtl_tests.sh    local RTL runner -> rtl_test_log.txt           (UNVERIFIED)
├── armse/                      Python package
│   ├── config.py       HwConfig, Workload (incl. out_bytes), error types
│   ├── schedule.py     Schedule, descriptor-program generation, legality rules (+ model-only a_resident / no_wb modes)
│   ├── sim.py          event-driven executor, deadlock + hazard checking, functional replay, metrics
│   ├── analytic.py     closed-form cost model (the scheduler's cost function)
│   ├── scheduler.py    baselines A/B/C/C_best_square, adaptive D, feedback D_fb, ablations
│   ├── workloads.py    30-workload suite (GEMM/GEMV/skinny/ragged/MLP/conv-im2col/attention/stress)
│   ├── banks.py        skew vs naive bank-conflict analysis, synthetic stride study
│   ├── energy.py       L1 activity + L2 analytical energy (illustrative constants)
│   ├── rtlexport.py    descriptor packing + generates rtl/armse_fields.vh + golden vectors
│   ├── rtlmodel.py     cycle-level transcription of the RTL (logic/timing check)
│   ├── fused.py        model-only fused GEMM chains (attention, MLP pairs)
│   ├── bench.py        main benchmark driver -> results/*
│   ├── fusionbench.py  fusion study -> results/FUSION_REPORT.md
│   ├── memplan.py, edgestudy.py  tensor-lifetime planner + edge-workload studies -> results/EDGE_WORKLOADS.md
│   ├── fused_overlap.py, overlapbench.py  overlapped fused chains -> results/FUSION_OVERLAP.md
│   ├── progstream.py   -> results/PROGRAM_STREAMING.md (model-only descriptor streaming cost)
│   ├── progcompress.py descriptor-program compressibility (lossless encode/decode)
│   └── progbench.py    -> results/PROGRAM_COMPRESSION.md
├── tests/              test_model.py, test_fused.py  (unittest)
├── rtl/                armse_skew_mem.v, armse_obuf.v, armse_dma.v, armse_compute.v, armse_top.v, armse_fields.vh (generated)
├── tb/                 tb_armse.v (self-checking top), tb_skew_mem.v
├── rtl_vectors/        generated golden vectors (7 cases)   <- `make vectors`
├── results/            REPORT.md, FUSION_REPORT.md, CSVs, PNGs  <- generated
└── docs/               TECHNICAL_REPORT.md, ARCHITECTURE.md, RELATED_WORK.md, BOARD_GUIDE.md, SYNTHESIS.md
```

Commands: `make py-test` | `make vectors` | `make rtl-model-test` | `make rtl-test` (needs iverilog >= 10) | `bash scripts/run_rtl_tests.sh` | `python3 -m armse.bench` (~2 min) | `python3 -m armse.fusionbench` (~1 min).

---

## 5. ARCHITECTURE IN BRIEF

- **Decoupled access/execute.** Two in-order descriptor programs: a **DMA program** (LOAD_A, LOAD_B, WRITEBACK jobs, each waiting for `cp_done >= DEP`) and a **compute program** (tile MAC jobs, each waiting for `dma_done >= DEP`). Single vs double buffering is only a different set of DEP values. Legality is enforced by two counters and two comparators; the host proves it via simulation.
- **Memory hierarchy.** External memory (one outstanding burst, `row_lat` cycles to first beat, one beat/cycle, `bus_bytes` per beat; RTL implements 4) -> on-chip A buffer + B buffer (NB byte-wide banks each, 1R+1W per bank) + output/partial-sum buffer (32-bit words, compute port 1R+1W sync, DMA port 1 async read).
- **Skewed bank mapping.** Element (row r, col c) -> bank `(r+c) mod NB`, word `r*stride + c/NB`, `stride = ceil(tk/NB)`. A stores (row=i, col=k); B is stored transposed (row=j, col=k). Compute reads NB consecutive k from NB distinct banks (then de-rotated); DMA writes 4 consecutive elements into 4 distinct banks. Zero conflicts by construction (exhaustively tested).
- **Compute engine.** NB parallel MAC lanes along K; one (i,j) output pair per issue cycle group; partial sums accumulate in the output buffer across K tiles; pipeline read -> multiply/adder tree -> accumulate -> write.
- **Timing contract (hand-derived, cross-checked by rtlmodel.py):** DMA job = `2 + rows*(row_lat+1+beats)`; compute job = `tm*tn*ceil(tk/NB) + 4`; first start at cycle 1 after `go`. RTL `busy` counters = model duration minus 1 per job.
- **Scheduler D (deterministic, host):** validate op/dims -> enumerate tile candidates (powers of two, balanced splits, lane multiples for K, full dim) x 2 loop orders x 4 buffering pairs -> reject by capacity / field width / program length -> rank by estimated cycles (ties: bytes, buffering, order, tile) -> fallback to minimal 1x1xK tile or raise NoFitError. **D_feedback** simulates the estimator's top-8 and keeps the best measured.
- **Reuse rule (loop nest):** a load is skipped when consecutive iterations use the same tile id. nk>1: A traversed nn times, B nm times. nk==1: MN order loads A once, B nm times (nn>1); NM symmetric.

---

## 6. GLOSSARY OF EVERY KEYWORD

**Problem / concepts**
- **Memory wall** - compute throughput outgrowing memory bandwidth/latency/energy.
- **Compute-bound / memory-bound** - runtime limited by MAC throughput vs by data movement; decided by arithmetic intensity vs machine balance.
- **Arithmetic intensity (AI)** - ops per byte moved; here AI_compulsory = 2MNK / (MK + KN + 4MN).
- **Machine balance** - peak ops/cycle / external bytes/cycle (here 16 / 4 = 4 ops/byte).
- **Roofline** - model bounding performance by min(compute peak, bandwidth x AI).
- **Compulsory traffic** - minimum unavoidable bytes (each input read once, each output written once).
- **Avoidable traffic** - bytes read beyond compulsory because of poor reuse.
- **Reuse / retention** - keeping a tile on chip across iterations to avoid reloading.
- **Data starvation / memory stall** - compute idle waiting for data.
- **Latency hiding** - overlapping transfers with compute (double buffering).

**Dataflow / tiling**
- **GEMM** - general matrix multiply C[MxN] = A[MxK] x B[KxN]; M, N, K are dimensions.
- **Tile (tm, tn, tk)** - sub-block of the matrices processed at once; partial/ragged tiles at edges.
- **Loop order MN / NM** - outer-loop dimension order (for m, for n, for k) vs (for n, for m, for k).
- **Weight-/output-/input-stationary** - which operand is kept resident (this design is output-stationary with retention of the stationary operand through the tile-id rule).
- **Systolic array** - grid of MACs with neighbour-to-neighbour dataflow (prior art; not used here).
- **im2col** - lowering convolution to GEMM (M=OH*OW, K=Cin*kh*kw, N=Cout); sliding-window reuse is NOT modelled.
- **Operator fusion** - computing consecutive operators without writing the intermediate to external memory.
- **Requantisation** - INT32 -> INT8 via arithmetic shift + saturate between fused GEMMs.
- **Row block (tm)** - fusion unit: tm rows processed through both GEMMs.

**Memory architecture**
- **Scratchpad vs cache** - software-managed local memory (used here) vs hardware-managed.
- **Banking / bank conflict** - splitting memory into parallel banks; conflict = two simultaneous accesses to one bank (serialises).
- **Skewed (rotated) mapping** - bank = (r + c) mod NB to make row and column accesses conflict-free.
- **Naive mapping** - row-major layout; B lane reads serialise by gcd(tn, NB).
- **Double buffering / ping-pong** - two slots so loading slot B overlaps computing on slot A.
- **Slot** - one buffer partition (depth/nbuf words); **nbuf_in / nbuf_out** - slots in operand / output buffers.
- **DMA / burst / beat** - bulk transfer engine; one row = one burst; beat = one bus-width transfer.
- **row_lat** - cycles from burst command to first data beat; **row_fix = row_lat + 1**.
- **bus_bytes** - bytes per beat (RTL fixed at 4).
- **Backpressure** - consumer stalling producer (handled via ready/valid-style signals in the external port).
- **Partial sum (psum) buffer** - output buffer holding INT32 accumulators across K tiles.
- **Idealised unaligned access** - external rows may start at any byte address (real buses need realignment; not modelled).

**Scheduling**
- **Descriptor / program** - fixed-format entry (128-bit) the hardware executes; DMA and compute programs.
- **DEP counter** - dependency: entry waits until the other engine's completed-count >= DEP.
- **cp_done / dma_done** - completed-job counters of compute / DMA engines.
- **Legal schedule** - fits capacity, field widths, program length; deadlock- and hazard-free.
- **Hazard** - buffer overwritten while read, read before written, output slot reused before writeback.
- **Deadlock** - circular dependency between the two programs.
- **Cost function** - estimator used to rank candidates (cycles, then bytes...).
- **Estimator calibration** - measuring prediction error against simulation/measurement.
- **Fallback** - minimal 1x1xK schedule when no candidate fits; NoFitError if even that fails.
- **Feedback / observed behaviour** - simulating top-k candidates and keeping the best measured (models counter-driven re-planning).

**Policies / baselines**
- **A_direct** - one output element at a time, no reuse (minimally buffered).
- **B_fixed_single** - fixed square tile (32), single-buffered.
- **C_fixed_double** - fixed square tile (16), double-buffered.
- **C_best_square** - strongest fixed-shape baseline: best square-tile schedule per workload by simulation over the estimator's top-16 square candidates plus B and C.
- **D_adaptive** - estimator-selected adaptive schedule.
- **D_feedback** - simulate top-8 estimator candidates, keep best.
- **Ablations** - D with one feature removed: no shape freedom (square only), no order choice (MN only), no buffer-allocation choice ((2,2) only), no overlap ((1,1) only).

**Evidence / verification**
- **Golden vectors** - model-generated inputs/expected outputs/counters for RTL testbenches.
- **Self-checking testbench** - compares result matrix, bytes, transfers, conflicts, cycles automatically.
- **Transcription (rtlmodel.py)** - hand-written Python mirror of the RTL used when no HDL simulator exists.
- **CI** - GitHub Actions workflow that compiles/simulates the RTL on push.
- **Event-driven simulator** - computes job start/finish times from dependencies instead of stepping every cycle.
- **Functional replay** - replays data movement with numpy so wrong dependencies give wrong numbers.
- **Energy levels L1/L2/L3** - activity proxy / analytical estimate / measured.

**Metrics**
- cycles, compute_busy, compute_issue, mem_stall (compute idle waiting on DMA), tail, dma_busy (split: sync / latency / data beats), bytes_rd, bytes_wr, transfers, mac_util, bw_sustained, buf_util, program_bytes, bank_serial_cycles, sched_ms (host scheduler time).

---

## 7. KNOWN LIMITATIONS (always state these when quoting results)

1. RTL never compiled/simulated; synthesis, Fmax, LUT/FF/DSP unknown; no board measurements; no measured energy.
2. Single headline hardware config; conclusions are workload- and config-dependent.
3. Conv uses im2col with lowered-matrix traffic (no sliding-window reuse); attention = two GEMMs without softmax; MLP = GEMM chain without activation.
4. Unrolled descriptor programs: program memory grows with iteration count; no hardware loop generator.
5. Buffer partition A/B/O fixed at design time; single DMA channel, one outstanding burst; no bus errors; idealised unaligned access.
6. C_best_square and D_feedback use simulator-in-the-loop information (profiling in a real system).
7. Fusion is model-only; RTL has no requantiser and no independent stride field for a resident operand.
8. Reset during active operation, illegal-address trapping, and scheduler-failure behaviour in RTL are not verified/implemented.
9. L2 energy constants are illustrative (45 nm-class literature-style numbers typed from memory), not for this FPGA.
10. Related-work table is from memory; no literature search was possible; no novelty claim is justified yet.

---

## 8. WHAT REMAINS TO BE DONE (prioritised) AND HOW TO ACHIEVE EACH

**P0 - Close the verification gap (7 fixed cases DONE; extend)**
- *Status:* 7 fixed cases pass in Icarus (user run). *Remaining:* run the 40 random cases (`--random 40`), CI, Verilator lint.
- *Goal:* get the Verilog compiled and simulated.
- *How:* push repo to GitHub (CI runs `make rtl-test`) or run `bash scripts/run_rtl_tests.sh` locally (iverilog >= 10; Windows via WSL). Paste back `rtl_test_log.txt`.
- *Then:* fix compile errors; compare `CYCLES ... delta=` lines; if a delta is non-zero, decide whether RTL or model constants are wrong (use `rtlmodel.py` to localise), update `HwConfig` (`job_ovh`, `cmp_ovh`, `start_off`, `row_lat`) or the RTL, rerun the benchmark.
- *Acceptance:* all 7 cases + skew-mem test print `RESULT ... PASS`; zero cycle delta (`+STRICT_CYC` passes); `make lint` reviewed.

**P1 - Synthesis and resource truth**
- *How:* pick the real target board; run Yosys (open) or Vivado/Quartus; report LUT/FF/DSP/BRAM, Fmax, timing closure per module and for the top; replace async obuf DMA read with a BRAM-friendly pipelined read and update the timing contract.
- *Acceptance:* resource table in REPORT.md marked MEASURED (tool + version), timing met at stated clock.

**P2 - Real memory platform**
- *How:* identify actual board memory (BRAM/URAM capacity, DDR/HBM interface); measure **sustained** (not peak) bandwidth and burst latency with a micro-benchmark; set `row_lat`, `bus_bytes`, buffer depths from measurements; rerun benchmark sweep. Never use HBM assumptions for a BRAM-only prototype.
- *Acceptance:* HwConfig values traceable to measurements; D vs baselines re-evaluated on them.

**P3 - Cut descriptor-program memory (analysis DONE; decision pending)**
- *Finding (model only, results/PROGRAM_COMPRESSION.md):* repeat-with-constant-delta compression gives only ~1.37x geomean (1.32x for C; 16x on one regular workload). Program bytes are ~4.3 % of data traffic for D (geomean), so *streaming descriptors from external memory* would cost ~1.3 % cycles median (7.8 % worst) as an upper-bound estimate. Prefer streaming (or a nested-loop generator only if a 2-level encoding is shown to beat ~3x) over a simple loop generator.
- *Measured in the simulator (results/PROGRAM_STREAMING.md, model only):* streaming descriptors in blocks of 8 costs D_feedback 1.3 % cycles geomean (median 0.6 %, worst 11.6 %) and C_fixed_double 3.3 %; D/C stays 0.719 with both streaming. DECISION: prefer descriptor streaming over a simple loop generator; RTL support for streaming is NOT yet implemented.
- *Original plan:* hardware loop generator.
- *How:* replace unrolled programs with nested-loop descriptors (loop counts + strides + dependency rules) executed by a small address/dependency generator; keep the Python generator as reference; verify program equivalence tile-by-tile.
- *Acceptance:* identical cycles/bytes/results as unrolled; program_bytes reduced by >= 10x on large workloads.

**P4 - Better scheduler**
- *How:* improve cycle estimator overlap model (reduce 5.1 % mean / 14.6 % max error, raise top-1 hit rate from 12/30); add cost terms for bank conflicts and bus utilisation; add dynamic re-planning from RTL counters (compute_wait, dma_wait, bank_conflicts); only then test learned policies (RL/neural), and only if they beat deterministic D on held-out workloads.
- *Acceptance:* top-1 pick within X % of best on >= 90 % of workloads; held-out workload evaluation.

**P5 - Fusion that can actually win**
- *Done (model only, results/FUSION_OVERLAP.md):* (a) cross-phase/cross-block overlap implemented and verified; geomean cycles vs unfused 1.096 -> 1.084, faster in 7/20 cases (barrier 5/20). Still mostly a loss. Remaining ideas: (b)-(e) below.
- *How:* (a) [done] cross-phase/cross-block overlap (remove barriers), (b) weight retention across row blocks, (c) larger on-chip memory or dedicated intermediate SRAM, (d) hardware requantiser + resident-operand stride field in RTL, (e) softmax/activation support for faithful attention/MLP.
- *Acceptance:* fused beats unfused on a defined workload class with bit-exact verification, or a documented reason it cannot.

**P6 - Wider workloads and numerics**
- *How:* direct convolution dataflow with sliding-window reuse; larger transformer shapes; INT4/mixed precision; sparsity/compression as an orthogonal axis; sequences of layers with inter-layer retention.
- *Acceptance:* each new workload class has reference checks, baselines, and a loss analysis.

**P7 - Energy to Level 3**
- *How:* board power instrumentation (idle baseline, inference power x duration, repeated runs, uncertainty); calibrate L2 constants against measurement; keep L1/L2/L3 labelled separately.
- *Acceptance:* energy-per-inference with error bars; no L2 number called measured.

**P8 - Literature and novelty gate**
- *How:* do a real structured literature search (primary papers, proceedings, docs, reproducible code); fill the related-work matrix (problem, mechanism, hardware, workload assumptions, metrics, limitations, difference); identify a defensible narrower contribution (candidate hypotheses: estimator calibration on small-FPGA buffers; simulator-in-the-loop vs offline planning at tiny capacity; cost of unrolled descriptors vs loop-generator hardware).
- *Acceptance:* every citation verified; novelty claim only if evidence supports it.

**P9 - Integration and reporting**
- *How:* write a documented adapter to the user's existing handwritten-digit accelerator (interface + timing changes), only after explicit approval; produce the final technical report distinguishing established techniques from the contribution; add robustness tests (reset during active operation, illegal addresses, scheduler failure, back-to-back runs on RTL).
- *Acceptance:* reproducibility guide, ablations, limitations, failure analysis all in the report.

---

## 9. SUCCESS CRITERIA (define before evaluating; do not move the goalposts)

- Correct execution of all supported workloads, bit-exact vs independent reference.
- Reduced memory traffic, memory stalls or latency on at least one clearly defined workload class vs a fair fixed-policy baseline.
- Resource cost reported and technically justified (measured once synthesis exists).
- No hidden accuracy loss.
- A clear explanation of cases where adaptive scheduling does not help.
- Never required to beat every baseline on every workload.

---

## 10. REQUIRED WORKFLOW FOR ANY NEW WORK

1. Inspect existing files and toolchain; state what can and cannot be run.
2. Establish or reuse a baseline; validate against an independent reference.
3. Add one architectural/scheduling feature at a time.
4. Run identical workloads on all baselines and candidates; keep precision, correctness, resources constant.
5. Add tests (edge cases, partial tiles, capacity boundaries, hazards, deadlock, fallback, determinism, back-to-back).
6. Regenerate results; check earlier results did not regress (compare CSVs).
7. Report absolute and relative numbers, resource cost, limitations, failed hypotheses.
8. Update README / ARCHITECTURE; give changed file paths and git commands.
9. Only after all of the above, discuss whether a contribution is distinct enough for any research or publication claim.

---

## 11. THE PROMPT TO PASTE IN A NEW CHAT

> You are continuing the ARMSE project (Adaptive Reuse and Memory Scheduling Engine): an INT8 FPGA-class memory-architecture study with Verilog RTL, a host-side deterministic scheduler, a cycle-level Python simulator, an analytical cost model, tests, benchmarks and a model-only fusion extension. Read the attached ARMSE_MASTER_PROMPT_v2.md fully; it defines the status (VERIFIED vs UNVERIFIED), repository layout, keyword glossary, limitations, roadmap and rules. Obey its rules: never fabricate measurements or citations; separate RTL / host software / models; separate energy levels L1/L2/L3; no weak baselines; do not touch my existing accelerator repo without approval; give changed files individually with their folder paths and git add/commit/push commands. Start with the remaining part of P0: [paste rtl_test_log.txt from `--random 40`, or the failing output], diagnose it, fix the RTL or model constants, rerun, and tell me exactly what is now verified. After P0, propose the next step from section 8 and ask me which one I want.

---

## 12. FILES YOU SHOULD ATTACH WHEN RESUMING

Always: this file, `README.md`, `docs/ARCHITECTURE.md`. For RTL work: `rtl/*.v`, `tb/*.v`, `rtl_test_log.txt`. For model work: `armse/*.py`, `tests/*.py`, `results/REPORT.md`, `results/FUSION_REPORT.md`.
