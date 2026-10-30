# Research decision

**Decision: GATHER MORE EVIDENCE before deciding (outcome 4), with a lean toward REVISE (outcome 2).**

## Evidence for
* A verified, reproducible stack exists (46 unit tests, bit-exact against an independent reference, RTL simulated in Icarus on 7 cases with cycle-exact agreement to the model).
* On 30 workloads and one configuration, the adaptive scheduler needs 0.733x (D_feedback) / 0.749x (D_adaptive) the cycles of fixed double buffering and 0.869x / 0.888x those of the strongest fixed-shape baseline; the gain comes mainly from tile-shape freedom (ablation 0.749 vs 0.844 without it).
* The advantage holds across a 27-configuration hardware sweep (lanes, capacity, latency): D_feedback / best square 0.536-0.973, never above 1 (12 workloads, model only).
* The skewed bank mapping removes an up-to-8x penalty (naive layout) at no cycle cost.
* Honest negative results were measured: fusion mostly loses at <= 32 KB, descriptor compression is weak (1.37x), depthwise convolution on a lane-parallel-over-K datapath uses only 3.6-8.9 % of MACs (a channel-parallel mapping reaches 26-38 % in the model; no RTL).

## Evidence against / unresolved
* **Novelty is not established and is probably low for the core idea:** cost-model-driven tile/loop/buffer search is, from recall, established (mapping-search tools). No search could be performed.
* The adaptive schedules read more bytes and (analytically) use more energy than the best square tiling; the benefit is latency hiding, not traffic reduction.
* 25/30 workloads are compute-bound; benefit shrinks at larger buffers (0.963x at 4x).
* No synthesis, board, bandwidth or energy measurement exists; the 40 random RTL cases and CI are not yet run.

## What would change the decision
| Outcome | Trigger |
|---|---|
| Reject current concept | the literature search finds prior work with the same mechanisms *and* equal or better measured results on comparable workloads |
| Revise | prior art covers the scheduler but a narrower question remains defensible (e.g. estimator calibration on small-buffer FPGAs; simulator-in-the-loop re-planning; cost of unrolled descriptors) -> reposition as a verified reference study |
| Proceed | the search finds no equivalent combination *and* synthesis/board data confirm the cost is acceptable |

## Next evidence to collect (in order)
1. Literature/novelty search (`docs/LITERATURE_SEARCH_PROMPT.md`) in a session with web search.
2. 40 random RTL cases + CI run; then Yosys/vendor resource and Fmax estimates for a chosen part.
3. A board (or a vendor model) for sustained bandwidth and burst latency; re-run the benchmark with measured constants.
4. Accuracy study before any INT4 claim.
