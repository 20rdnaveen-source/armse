# Channel-parallel depthwise mapping (MODEL ONLY, `python3 -m armse.dwstudy`)

K-parallel = the earlier lowering (one channel at a time, M=OH*OW, N=1, K=k*k) scheduled by D_feedback, x channels. Channel-parallel = lanes spread over channels; tile (pixels x channels) searched exhaustively BY SIMULATION over tile, order and buffering (small space). Same on-chip memory, same bus. im2col per channel is counted (no window reuse). Numerics are checked against a numpy depthwise reference in the tests.

| workload | pixels | channels | taps | kpar_cycles | kpar_mac_util | cp_cycles | cp_mac_util | speedup | cp_tile | cp_order | cp_nbuf | cp_bytes_rd | kpar_bytes_rd | cp_mem_stall_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dw3x3_28_c64 | 676 | 64 | 9 | 1316032 | 0.037 | 159948 | 0.304 | 8.228 | 2x64 | MN | 1/2 | 389952 | 389952 | 0.686 |
| dw3x3_14_c128 | 144 | 128 | 9 | 578944 | 0.036 | 65495 | 0.317 | 8.840 | 1x128 | MN | 1/2 | 167040 | 167040 | 0.672 |
| dw5x5_28_c32 | 576 | 32 | 25 | 644768 | 0.089 | 151497 | 0.380 | 4.256 | 1x32 | MN | 2/2 | 461600 | 461600 | 0.604 |
| dw3x3_56_c24 | 2916 | 24 | 9 | 2110200 | 0.037 | 305118 | 0.258 | 6.916 | 8x24 | MN | 1/2 | 630072 | 630072 | 0.736 |

Geomean speedup of channel-parallel over K-parallel: 6.80x. MAC utilisation: 0.050 -> 0.315 (mean). The remaining gap to 1.0 is memory stall: depthwise has very low arithmetic intensity (K = 9 or 25 MACs per loaded byte-row), so it stays memory-bound.
