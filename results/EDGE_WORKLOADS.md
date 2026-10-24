# Edge workloads, dynamic shapes and memory planning (MODEL ONLY; `python3 -m armse.edgestudy`)

## 1. KV-cache decode, cache growing 32 -> 1024 (d = 64, INT8, softmax not modelled)

Each step = QK^T GEMV + PV GEMV over the current cache length. Static schedules are tuned once (D_feedback) for a chosen cache length and clipped to smaller shapes; runtime re-plans every step; hybrid plans once per power-of-two bucket (tuned at the bucket's upper bound) and clips. Search seconds = host Python time (not hardware).

| policy | total_cycles | vs_runtime_D_fb | worst_step_vs_runtime | searches | search_seconds |
|---|---|---|---|---|---|
| C_fixed_double | 2514336 | 2.639 | 2.680 | 0 | 0.001 |
| C_best_square | 1508072 | 1.583 | 1.604 | 64 | 0.725 |
| static_L1024 | 952752 | 1.000 | 1.020 | 2 | 0.000 |
| static_L256 | 952752 | 1.000 | 1.020 | 2 | 0.000 |
| runtime_D_fb | 952650 | 1.000 | 1.000 | 64 | 1.534 |
| hybrid_pow2 | 952605 | 1.000 | 1.000 | 12 | 0.218 |

### 1b. Attention PREFILL with varying sequence length (QK^T: M=N=L, K=64; PV: M=L, N=64, K=L), L = 16 -> 256

| policy | total_cycles | vs_runtime_D_fb | worst_step_vs_runtime | searches | search_seconds |
|---|---|---|---|---|---|
| C_fixed_double | 7524602 | 1.198 | 1.214 | 0 | 0.001 |
| C_best_square | 6909492 | 1.100 | 1.163 | 32 | 1.417 |
| static_L256 | 6375599 | 1.015 | 1.149 | 2 | 0.000 |
| static_L64 | 6584862 | 1.048 | 1.371 | 2 | 0.000 |
| runtime_D_fb | 6282164 | 1.000 | 1.000 | 32 | 11.886 |
| hybrid_pow2 | 6389874 | 1.017 | 1.079 | 10 | 2.434 |

## 2. Depthwise convolution (one channel lowered to GEMM M=OH*OW, N=1, K=k*k; x channels)

| workload | channels | policy | tile | cycles_per_channel | total_cycles | mac_util | mem_stall_frac |
|---|---|---|---|---|---|---|---|
| dw3x3_28_c64 | 64 | C_fixed_double | 16x1x9 | 20581 | 1317184 | 0.037 | 0.913 |
| dw3x3_28_c64 | 64 | D_feedback | 43x1x9 | 20563 | 1316032 | 0.037 | 0.884 |
| dw3x3_14_c128 | 128 | C_fixed_double | 16x1x9 | 4485 | 574080 | 0.036 | 0.835 |
| dw3x3_14_c128 | 128 | D_feedback | 15x1x9 | 4523 | 578944 | 0.036 | 0.857 |
| dw5x5_28_c32 | 32 | C_fixed_double | 16x1x16 | 40069 | 1282208 | 0.045 | 0.930 |
| dw5x5_28_c32 | 32 | D_feedback | 16x1x25 | 20149 | 644768 | 0.089 | 0.859 |

The lane-parallel-over-K datapath cannot fill its lanes when K = k*k is 9 or 25 and N = 1: low MAC utilisation is a property of this mapping, not of the schedule. A channel-parallel mapping is not modelled.


## 3. Tensor-lifetime planning (activations only, INT8; transformer lifetimes are illustrative hand-set intervals)

| graph | naive_bytes | peak_live_bytes | planned_arena_bytes | saving_vs_naive | fits_8KB_onchip |
|---|---|---|---|---|---|
| mlp_b1_784_256_128_10 | 1178 | 1040 | 1040 | 0.117 | True |
| mlp_b32_784_256_128_10 | 37696 | 33280 | 33280 | 0.117 | False |
| xformer_s64_d64 | 77824 | 36864 | 36864 | 0.526 | False |
| xformer_s128_d64 | 172032 | 73728 | 73728 | 0.571 | False |
| xformer_s256_d64 | 409600 | 163840 | 163840 | 0.600 | False |
| xformer_s64_d64_inplace | 77824 | 24576 | 24576 | 0.684 | False |