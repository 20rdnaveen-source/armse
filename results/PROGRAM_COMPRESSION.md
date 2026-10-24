# Descriptor-program compressibility (potential of a hardware loop generator; MODEL ONLY)

Lossless greedy repeat-with-constant-delta encoding (decode verified on every program). On-chip DATA memory = 8192 B. 16 bytes per descriptor entry.

| policy | geomean_ratio | min_ratio | max_ratio | mean_raw_bytes | mean_compressed_bytes | max_raw_bytes | max_compressed_bytes |
|---|---|---|---|---|---|---|---|
| C_fixed_double | 1.32 | 1.00 | 4.06 | 39181.33 | 29011.20 | 212992 | 155648 |
| D_feedback | 1.37 | 1.00 | 16.08 | 11108.27 | 8893.87 | 57344 | 45760 |

## Per workload

| workload | policy | raw_entries | compressed_entries | ratio | raw_bytes | compressed_bytes |
|---|---|---|---|---|---|---|
| gemm_sq16 | C_fixed_double | 4 | 4 | 1.00 | 64 | 64 |
| gemm_sq16 | D_feedback | 19 | 18 | 1.06 | 304 | 288 |
| gemm_sq32 | C_fixed_double | 28 | 27 | 1.04 | 448 | 432 |
| gemm_sq32 | D_feedback | 97 | 69 | 1.41 | 1552 | 1104 |
| gemm_sq64 | C_fixed_double | 208 | 154 | 1.35 | 3328 | 2464 |
| gemm_sq64 | D_feedback | 130 | 104 | 1.25 | 2080 | 1664 |
| gemm_sq128 | C_fixed_double | 1600 | 1375 | 1.16 | 25600 | 22000 |
| gemm_sq128 | D_feedback | 468 | 343 | 1.36 | 7488 | 5488 |
| gemm_sq192 | C_fixed_double | 5328 | 4176 | 1.28 | 85248 | 66816 |
| gemm_sq192 | D_feedback | 1280 | 1280 | 1.00 | 20480 | 20480 |
| gemm_sq256 | C_fixed_double | 12544 | 9472 | 1.32 | 200704 | 151552 |
| gemm_sq256 | D_feedback | 3146 | 2860 | 1.10 | 50336 | 45760 |
| gemv_1x256x256 | C_fixed_double | 784 | 592 | 1.32 | 12544 | 9472 |
| gemv_1x256x256 | D_feedback | 194 | 138 | 1.41 | 3104 | 2208 |
| gemv_256x1x256 | C_fixed_double | 784 | 193 | 4.06 | 12544 | 3088 |
| gemv_256x1x256 | D_feedback | 104 | 56 | 1.86 | 1664 | 896 |
| tall_512x16x64 | C_fixed_double | 416 | 298 | 1.40 | 6656 | 4768 |
| tall_512x16x64 | D_feedback | 193 | 12 | 16.08 | 3088 | 192 |
| wide_16x512x64 | C_fixed_double | 416 | 298 | 1.40 | 6656 | 4768 |
| wide_16x512x64 | D_feedback | 312 | 244 | 1.28 | 4992 | 3904 |
| deepK_32x32x1024 | C_fixed_double | 772 | 532 | 1.45 | 12352 | 8512 |
| deepK_32x32x1024 | D_feedback | 390 | 276 | 1.41 | 6240 | 4416 |
| ragged_37x29x53 | C_fixed_double | 78 | 72 | 1.08 | 1248 | 1152 |
| ragged_37x29x53 | D_feedback | 39 | 36 | 1.08 | 624 | 576 |
| ragged_101x67x131 | C_fixed_double | 980 | 840 | 1.17 | 15680 | 13440 |
| ragged_101x67x131 | D_feedback | 380 | 333 | 1.14 | 6080 | 5328 |
| ragged_250x255x249 | C_fixed_double | 12544 | 9728 | 1.29 | 200704 | 155648 |
| ragged_250x255x249 | D_feedback | 3200 | 2753 | 1.16 | 51200 | 44048 |
| mlp_b1_L0 | C_fixed_double | 2368 | 1648 | 1.44 | 37888 | 26368 |
| mlp_b1_L0 | D_feedback | 590 | 402 | 1.47 | 9440 | 6432 |
| mlp_b1_L1 | C_fixed_double | 392 | 296 | 1.32 | 6272 | 4736 |
| mlp_b1_L1 | D_feedback | 97 | 69 | 1.41 | 1552 | 1104 |
| mlp_b1_L2 | C_fixed_double | 25 | 21 | 1.19 | 400 | 336 |
| mlp_b1_L2 | D_feedback | 10 | 10 | 1.00 | 160 | 160 |
| mlp_b32_L0 | C_fixed_double | 4736 | 3296 | 1.44 | 75776 | 52736 |
| mlp_b32_L0 | D_feedback | 1216 | 896 | 1.36 | 19456 | 14336 |
| mlp_b32_L1 | C_fixed_double | 784 | 592 | 1.32 | 12544 | 9472 |
| mlp_b32_L1 | D_feedback | 490 | 370 | 1.32 | 7840 | 5920 |
| mlp_b32_L2 | C_fixed_double | 50 | 42 | 1.19 | 800 | 672 |
| mlp_b32_L2 | D_feedback | 19 | 18 | 1.06 | 304 | 288 |
| conv3x3_16to32_28 | C_fixed_double | 2744 | 2254 | 1.22 | 43904 | 36064 |
| conv3x3_16to32_28 | D_feedback | 931 | 809 | 1.15 | 14896 | 12944 |
| conv1x1_64to64_14 | C_fixed_double | 676 | 487 | 1.39 | 10816 | 7792 |
| conv1x1_64to64_14 | D_feedback | 224 | 169 | 1.33 | 3584 | 2704 |
| conv5x5_3to8_32 | C_fixed_double | 1024 | 1024 | 1.00 | 16384 | 16384 |
| conv5x5_3to8_32 | D_feedback | 385 | 261 | 1.48 | 6160 | 4176 |
| conv3x3_32to64_14 | C_fixed_double | 2860 | 2132 | 1.34 | 45760 | 34112 |
| conv3x3_32to64_14 | D_feedback | 728 | 598 | 1.22 | 11648 | 9568 |
| attn_s64_d32_QKt | C_fixed_double | 112 | 86 | 1.30 | 1792 | 1376 |
| attn_s64_d32_QKt | D_feedback | 62 | 51 | 1.22 | 992 | 816 |
| attn_s64_d32_PV | C_fixed_double | 104 | 82 | 1.27 | 1664 | 1312 |
| attn_s64_d32_PV | D_feedback | 78 | 68 | 1.15 | 1248 | 1088 |
| attn_s128_d64_QKt | C_fixed_double | 832 | 586 | 1.42 | 13312 | 9376 |
| attn_s128_d64_QKt | D_feedback | 468 | 417 | 1.12 | 7488 | 6672 |
| attn_s128_d64_PV | C_fixed_double | 800 | 687 | 1.16 | 12800 | 10992 |
| attn_s128_d64_PV | D_feedback | 450 | 386 | 1.17 | 7200 | 6176 |
| stress_512x512x64 | C_fixed_double | 13312 | 9226 | 1.44 | 212992 | 147616 |
| stress_512x512x64 | D_feedback | 3584 | 2566 | 1.40 | 57344 | 41056 |
| stress_64x64x2048 | C_fixed_double | 6160 | 4176 | 1.48 | 98560 | 66816 |
| stress_64x64x2048 | D_feedback | 1544 | 1064 | 1.45 | 24704 | 17024 |

## Program size versus data traffic (computed from results/suite_results.csv)
| policy | program bytes / data traffic (geomean) | max | extra cycles if the program were streamed from external memory at 4 B/cycle, as a fraction of total cycles (median / max) |
|---|---|---|---|
| C_fixed_double | 0.090 | 0.276 (mlp_b1_L2) | 0.021 / 0.044 |
| D_feedback | 0.043 | 0.253 (gemm_sq32) | 0.013 / 0.078 |

Upper-bound style estimate (assumes the descriptor bytes serialise with data on the same bus; no overlap, no prefetch modelling).
