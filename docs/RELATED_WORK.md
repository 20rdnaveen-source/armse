# Related-work assessment (structured, NOT a substitute for a literature review)

**Citation caveat:** this environment had no search or database access. Everything below is from model knowledge; titles, venues and details may be wrong or incomplete. Verify every entry before citing it. No novelty claim is made.

| Approach | Problem addressed | Mechanism | Hardware | Reported metrics (to verify) | Limitations | Difference from ARMSE |
|---|---|---|---|---|---|---|
| TPU (Jouppi et al., ISCA 2017) | DNN inference throughput/energy | Weight-stationary systolic array, large unified buffer, DMA | ASIC | throughput / perf-per-W vs CPU/GPU | fixed dataflow, large on-chip memory | fixed-dataflow baseline analogue; ARMSE targets small-FPGA buffers and per-workload schedules |
| Eyeriss (Chen et al., ISCA 2016 / JSSC 2017) | data-movement energy | row-stationary dataflow, memory hierarchy | ASIC | energy per op, DRAM accesses | one dataflow family | dataflow selection is prior art |
| MAESTRO (Kwon et al., MICRO 2019), Timeloop (Parashar et al., ISPASS 2019), Interstellar (Yang et al., ASPLOS 2020) | mapping / dataflow design-space exploration | analytical cost models + search over tiling and loop order under capacity limits | models (no RTL) | predicted latency/energy | model accuracy vs hardware rarely closed | **closest prior art**: deterministic tile/order search with a cost model is established; ARMSE's cost model is far simpler |
| MAERI (Kwon et al., ASPLOS 2018) | irregular dataflows | reconfigurable distribution/reduction trees | ASIC/FPGA-style | utilisation | focuses on compute mapping | adapts compute, not memory scheduling |
| Fused-layer CNN accelerators (Alwani et al., MICRO 2016) | intermediate tensor traffic | cross-layer tiling/fusion | FPGA | DRAM traffic reduction | CNN-specific | fusion is NOT implemented in ARMSE (listed limitation) |
| FINN (Umuroglu et al., FPGA 2017), hls4ml, DNNBuilder | FPGA inference | per-network compile-time dataflow with on-chip weights | FPGA | throughput, resources | per-network re-synthesis | ARMSE keeps hardware fixed and changes only the program |
| FlashAttention (Dao et al., 2022) | attention memory traffic | IO-aware tiling keeping softmax intermediates on chip | GPU | HBM traffic, speed | GPU | principle (retain intermediates) not implemented here |
| Deep Compression, EIE (Han et al.) | weight footprint / traffic | pruning, quantisation, compression | software/ASIC | compression ratio | accuracy trade-offs | orthogonal; INT8 only here |
| Scratchpad vs cache (embedded-systems literature, e.g. Banakar et al.) | predictable low-energy local memory | software-managed scratchpads | any | energy/area vs cache | needs compiler support | ARMSE is a scratchpad design |
| Double buffering, banked/skewed memories, DMA | latency hiding, conflict-free parallel access | classic techniques | any | - | - | established; used here as building blocks |
| PIM / near-memory (ISAAC, PRIME, Newton, UPMEM-class) | memory wall at the source | compute in/near memory | ASIC/DRAM | throughput/energy | new technology | out of scope for a BRAM prototype |

## Where a contribution could still be defensible (hypotheses, unproven)
1. Calibrating a *simple* analytical estimator against cycle-accurate simulation/RTL on small-FPGA-scale buffers and reporting where it mis-ranks (this repo quantifies it in the model: bytes exact, cycles 5.1 % mean / 14.6 % max error for D's picks, top-1 correct on 12/30 workloads).
2. Simulator-in-the-loop re-planning ("observed behaviour") versus offline estimator-only planning at tiny capacities.
3. The cost/benefit of unrolled descriptor programs versus loop-generator hardware.
Whether any of these is novel requires a real literature search.
