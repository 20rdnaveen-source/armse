# ARMSE - Adaptive Reuse and Memory Scheduling Engine

Memory-wall mitigation study for FPGA-class INT8 inference accelerators: a parameterised
local-memory + transfer architecture (RTL), a deterministic workload-aware scheduler (host), a
cycle-level simulator, an analytical cost model, tests and a benchmark suite.
ARMSE is a working name; **no novelty claim is made** (see `docs/RELATED_WORK.md`).

## Verification status (read this first)

| Component | Status |
|---|---|
| Python model, scheduler, simulator, estimator, bank/energy analysis | **Executed here**: 28 unit tests pass (`make py-test`) |
| Functional correctness of tiled execution | 144 (workload x policy) runs bit-exact vs an independent int64 numpy reference, 0 mismatches; plus 150 random cases in tests |
| Schedule legality | every generated program passes deadlock + buffer-hazard checking; the checker is shown to catch deliberately broken programs |
| Estimator | traffic/DMA-busy/compute-busy exact (600 random cases); cycles error measured (report section 3) |
| **RTL (`rtl/*.v`) and testbenches (`tb/*.v`)** | **Written but NOT compiled or simulated.** The authoring environment had no Icarus/Verilator/Yosys and no network. Only structural sanity checks (balanced `begin/end`, parentheses) were run. Expect to fix compile-level issues. (The design *logic* is cross-checked by the transcription below.) |
| RTL logic and timing, via `armse/rtlmodel.py` (cycle-by-cycle Python transcription of the Verilog, hand-written) | **Executed here**: 7 fixed + 40 random cases give bit-exact results and *exactly* the model's cycles, bytes, transfers and busy counters, with 0 bank conflicts (`make rtl-model-test`). Verifies the design logic and the timing constants; it is a transcription, so it cannot catch Verilog syntax/tool issues or transcription slips, and X-propagation is not modelled |
| Synthesis (LUT/FF/DSP/BRAM, Fmax), board bandwidth, measured energy | **not done** |

The RTL testbenches are self-checking against golden vectors produced by the verified Python model, so
the first time you run them they will either confirm or refute the RTL against that model.

## Layout
```
armse/          config, schedule (program generation), sim, analytic, scheduler, workloads, banks, energy, rtlexport, bench
tests/          unittest suite (python3 -m unittest discover -s tests -t .)
armse/rtlmodel.py cycle-level transcription of the RTL (logic/timing check without an HDL simulator)
rtl/            armse_skew_mem, armse_obuf, armse_dma, armse_compute, armse_top, armse_fields.vh (generated)
tb/             tb_armse (self-checking top-level), tb_skew_mem (unit)
rtl_vectors/    generated golden vectors for 7 RTL test cases
results/        REPORT.md + CSVs + PNGs produced by `python3 -m armse.bench`
docs/           ARCHITECTURE.md, RELATED_WORK.md
```

## Reproduce
```
make py-test                 # 28 tests
python3 -m armse.bench       # ~100 s, regenerates results/ (deterministic)
make vectors                 # regenerates rtl/armse_fields.vh and rtl_vectors/
make rtl-model-test          # cycle-level transcription of the RTL (no HDL tool needed)
make rtl-test                # needs iverilog >= 10; runs skew-mem unit test + 7 top-level cases
```
What to expect from `make rtl-test` if the RTL matches the model: each case prints `RESULT tNN: PASS` and a
`CYCLES ... delta=` line. PASS requires bit-exact C, exact bytes read/written, exact transfer count and zero
bank conflicts. A non-zero cycle delta is *informational* (use `+STRICT_CYC` to make it fatal) and should be
resolved by calibrating `job_ovh / cmp_ovh / start_off` in `HwConfig`.

## Results (model only; default hardware: 8 lanes, 4 B/cycle bus, 12-cycle burst latency, 2 KB A + 2 KB B + 4 KB O buffers)

30 INT8 GEMM workloads (GEMM, GEMV, skinny, ragged, MLP, conv via im2col, attention QK/PV, capacity stress).
Baselines share identical physical buffers: A direct (no reuse), B fixed 32^3 single-buffered, C fixed 16^3 double-buffered,
C_best_square (best square-tile schedule per workload by simulation; strong fixed-shape baseline), D (estimator-selected adaptive schedule), D_feedback (simulate estimator's top-8).

| Policy | geomean cycles vs C | vs C_best_square | wins / ties / losses vs C_best_square | geomean read bytes vs C |
|---|---|---|---|---|
| A_direct | 59.6x | 70.7x | 0 / 0 / 30 | 6.95x |
| B_fixed_single | 1.08x | 1.28x | 0 / 1 / 29 | 0.61x |
| C_fixed_double | 1.00x | 1.19x | 0 / 2 / 28 | 1.00x |
| C_best_square | 0.84x | 1.00x | - | 0.63x |
| **D_adaptive** | **0.75x** | **0.89x** | 26 / 0 / 4 | 0.94x |
| **D_feedback** | **0.73x** | **0.87x** | 29 / 0 / 1 | 0.83x |

What the data does and does not say (all in the model, one hardware configuration):

* The hypothesis is **supported in the model**: adaptive schedules reduce cycles versus fixed double buffering by ~25-27 % and versus the strongest fixed-shape baseline by ~11-13 % (geomean), largest on memory-bound GEMV/MLP-batch-1 (0.70x vs best square).
* **Where the gain comes from (ablation):** tile-shape freedom is the main contributor (removing it: 0.749 -> 0.844 vs C). Loop-order choice (0.747) and buffer-allocation choice (0.752) contribute ~nothing in geomean on this suite. Removing overlap (single buffering) costs cycles (0.945) but cuts read bytes.
* **It is not a traffic win.** D reads *more* bytes than the best square tiling (0.83-0.94x of C vs 0.63x) because it trades traffic for latency hiding; analytical (Level-2, illustrative constants) energy is ~10 % below C but 16-21 % **above** C_best_square. Not measured energy.
* **Where it loses:** D_adaptive is slower than C_best_square on 4/30 workloads (wide_16x512x64, deepK_32x32x1024, mlp_b1_L2, mlp_b32_L1); D_feedback on 1 (wide_16x512x64). 24/30 workloads are compute-bound by arithmetic intensity (balance 4 ops/B), where there is little memory time left to remove.
* **Capacity:** the advantage over best-square is largest at 0.5x capacity (0.82) and shrinks to 0.96 at 4x; versus fixed double buffering it is 0.30 at 0.25x capacity (fixed tile too large to double-buffer effectively) and 0.92 at 4x.
* **Latency/bandwidth:** advantage grows with burst latency (D_fb/C = 0.40 at 32 cycles, 0.90 at 4 cycles).
* **Banking:** with naive row-major B, schedules chosen without knowing about conflicts run 1.0-8x slower (e.g. gemm_sq64: 38,771 -> 263,161 cycles); conflict-aware selection recovers most (40,889). The skewed mapping removes the effect entirely (exhaustive test).
* **Estimator:** bytes exact; cycle error for D's chosen schedules 5.1 % mean / 14.6 % max; its top-1 pick was the best of its own top-8 on only 12/30 workloads, which is why D_feedback helps.
* **Cost:** same on-chip data memory for all policies. Unrolled descriptor programs cost memory (mean 11 KB for D, 1.2 MB mean for the direct baseline) and host scheduling is 0.26 s mean in Python. Hardware LUT/FF/DSP cost is unknown (no synthesis).

Caveats that bound these conclusions: single hardware configuration for the headline table; im2col conv without window reuse; D_feedback and C_best_square use simulator-in-the-loop information that a deployed system would obtain by profiling; the cycle model's constants are hand-derived from RTL that has not been simulated; no fusion, no sparsity, no measured power.

## Getting the RTL compiled and simulated (the one thing still open)
Option 1 - GitHub does it for you: push the repo; `.github/workflows/ci.yml` installs Icarus Verilog and Verilator, runs
`make rtl-test` (skew-memory unit test + 7 top-level cases) and uploads `rtl_test_log.txt`.
Option 2 - locally: `bash scripts/run_rtl_tests.sh` (needs `iverilog`; on Windows use WSL) writes `rtl_test_log.txt`.
Either way, send me the log and I will fix whatever it reports. `make rtl-test` exits non-zero on any compile error,
any `ERROR` line, or any case that does not print `RESULT ...: PASS`. Neither workflow has been executed yet.

## Git (add / commit / push)
```
git add .
git commit -m "Add ARMSE memory-architecture model, scheduler, simulator, RTL, tests and benchmarks"
git push origin main
```
