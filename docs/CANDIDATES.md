# Candidate architecture / strategy comparison (evidence from this project only)

Every number below comes from the project's simulator or RTL transcription (model-level, one hardware configuration: 8 lanes, 4 B/cycle, 12-cycle burst latency, 8 KB on-chip data). "Strongest competing approach" entries name *kinds* of prior work from recall; they are unverified and not benchmarked here (see `docs/LITERATURE_SEARCH_PROMPT.md`).

| | C1 Adaptive tiling / buffer scheduling | C2 Lifetime planner + operator fusion | C3 Compression-aware buffering (INT4 weights) | C4 Skewed banked operand buffers | C5 Descriptor streaming (infrastructure) |
|---|---|---|---|---|---|
| Target problem | Memory stalls and reload traffic from poor tile shape / buffering | Intermediate-tensor storage and traffic | External weight traffic | Lane-parallel read conflicts on column access | On-chip descriptor memory |
| Core mechanism | Cost-model search over tile shape x loop order x buffering; feedback variant simulates top-8 | Greedy lifetime-interval allocation; row-block fusion keeping the INT8 intermediate on chip | B stored as packed INT4 + 16-bit scale per column per tile load; unpacked to INT8 on arrival | bank = (r + c) mod NB with de-rotation | Fetch descriptor blocks over the external bus |
| Hardware | Two counters + comparators, decoupled DMA/compute (implemented in RTL) | Requantiser + resident-operand stride (NOT in RTL) | Unpacker + scale handling (NOT in RTL) | Rotator + per-element address generation (in RTL) | Fetch FSM, bus arbitration (NOT in RTL) |
| Software | Host scheduler (Python) | Host planner | Offline quantiser | none | host program placement |
| **Measured effect** | cycles 0.749x (D) / 0.733x (D_fb) vs fixed double-buffering; 0.888x / 0.869x vs best square tiling (26-29 wins, 0 ties, 1-4 losses of 30) | planner: 12-60 % less activation memory than naive allocation (illustrative lifetimes); fusion: 1.084x cycles vs unfused, faster in 7/20 cases, best 0.951x | cycles 0.973x geomean, 0.887x on memory-bound subset; external bytes 0.801x; L2 energy 0.826x | naive layout up to ~8x slower (gemm_sq64 6.8x, mlp_b32_L0 8.0x); skew: zero conflicts (exhaustive) | +1.3 % cycles (D_fb, median 0.6 %); D/C stays 0.719 |
| Cost / overhead | host search 0.37-0.40 s (Python); programs mean 11.4 KB for D; extra bytes read vs best square (0.83x vs 0.63x of C); L2 energy 1.187x best square | barriers or overlap needed; B1/B2 re-streamed per block | accuracy loss UNKNOWN; scale metadata; unpack logic unmeasured | logic unmeasured | 3.3 % (C) / 1.3 % (D) cycles; model-only |
| Verification difficulty | medium (legality proven by simulator; RTL checked in Icarus on 7 cases) | high (numerics + hazards; model-verified bit-exact) | medium-high (accuracy is a model/training question) | low (exhaustive pattern test) | medium-high (arbitration, deadlock) |
| Compatible with | C3, C4, C5 | C1 | C1 (interaction not measured: same schedule reused) | all | C1 |
| Strongest competing approach (unverified) | mapping-search tools with analytical cost models (Timeloop/MAESTRO/Interstellar class) | cross-layer fusion accelerators; IO-aware attention tiling | mixed-precision / weight-compression accelerators | padding or replication of operands | descriptor/command-queue streaming in commercial NPUs |
| Reject if | oracle best-square matches it on >= 90 % of workloads, or capacity is large (0.963x at 4x buffers) | fusion loses at the target capacity (it does at <= 32 KB) | accuracy loss above the application's budget | never (needed for column reads) | descriptor traffic > ~5 % of cycles |
| Status | **keep (core)** | planner: keep as tool; fusion: **defer** | **optional**, gated on accuracy evidence | **keep (mandatory)** | **keep as infrastructure option**, implement only if program memory becomes a constraint |

## Interaction check (do not stack blindly)
* C3 lowers memory-boundness, which shrinks C1's advantage where it was largest (GEMV/MLP); the combination was NOT re-tuned or measured.
* C2 fusion's cost (weight re-streaming) grows as buffers shrink, the opposite of C1's growing benefit at low capacity; they favour different regimes.
* C4 and C1 are independent: C1's estimator already includes the conflict penalty and avoids conflict-prone tile widths under naive banking, but recovery is partial: conflict-aware cycles are 0.96x-2.13x of skewed-bank cycles across the 8-workload banking study (median ~1.05x; tall_512x16x64 stays 2.13x, deepK 1.37x, gemv 1.24x). The skewed mapping remains the right fix.

## Preferred composite (subject to the research decision)
**C4 (mandatory) + C1 (core) with C5 and C3 as gated options; C2 limited to the planner.** Rationale: C4 removes an up-to-8x penalty at low cost; C1 gives a measured 11-13 % cycle gain over the strongest fixed-shape baseline on 30 workloads; C3 and C5 are cheap in the model but unproven in hardware/accuracy; fusion did not pay at the capacities studied.

## Not evaluated (listed so the gap is explicit)
Hardware-managed caches, near/in-memory compute, structured sparsity, direct convolution dataflows with sliding-window reuse, channel-parallel depthwise mapping, multi-model execution, any learned policy.
