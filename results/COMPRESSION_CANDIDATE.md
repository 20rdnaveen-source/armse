# Compression-aware candidate: packed INT4 weights (MODEL ONLY, `python3 -m armse.compressbench`)

Same schedule as D_feedback for INT8 (no re-tuning). B loads carry 4 bits per weight plus a 16-bit scale per column per tile load; activations/outputs unchanged; buffers and MACs unchanged. **Accuracy loss from INT4 is NOT modelled; Level-2 energy uses illustrative constants.**

Geomean over 30 workloads: cycles 0.973x, total external bytes 0.801x, L2 energy 0.826x (all relative to INT8). Memory-bound subset (GEMV + batch-1 MLP): cycles 0.887x, bytes 0.760x.

## By family

| family | cycle_ratio | bytes_ratio | l2_energy_ratio |
|---|---|---|---|
| attention | 0.988 | 0.842 | 0.863 |
| conv | 0.998 | 0.824 | 0.853 |
| gemm | 0.986 | 0.825 | 0.851 |
| gemv | 0.913 | 0.863 | 0.867 |
| mlp | 0.929 | 0.734 | 0.755 |
| ragged | 0.985 | 0.781 | 0.814 |
| skinny | 0.992 | 0.814 | 0.841 |
| stress | 1.000 | 0.768 | 0.810 |

## Per workload

| workload | family | cycles_int8 | cycles_int4 | cycle_ratio | bytes_int8 | bytes_int4 | bytes_ratio | l2_energy_ratio | mem_stall_frac_int8 |
|---|---|---|---|---|---|---|---|---|---|
| gemm_sq16 | gemm | 1082 | 1058 | 0.978 | 1536 | 1440 | 0.938 | 0.941 | 0.426 |
| gemm_sq32 | gemm | 4969 | 4857 | 0.977 | 6144 | 5696 | 0.927 | 0.935 | 0.140 |
| gemm_sq64 | gemm | 37019 | 35867 | 0.969 | 61440 | 46080 | 0.750 | 0.774 | 0.108 |
| gemm_sq128 | gemm | 265101 | 264541 | 0.998 | 278528 | 214016 | 0.768 | 0.805 | 0.008 |
| gemm_sq192 | gemm | 888639 | 888129 | 0.999 | 811008 | 645888 | 0.796 | 0.833 | 0.002 |
| gemm_sq256 | gemm | 2145301 | 2129389 | 0.993 | 1835008 | 1455616 | 0.793 | 0.832 | 0.020 |
| gemv_1x256x256 | gemv | 24675 | 20579 | 0.834 | 67072 | 50688 | 0.756 | 0.762 | 0.652 |
| gemv_256x1x256 | gemv | 68597 | 68629 | 1.000 | 68608 | 67648 | 0.986 | 0.986 | 0.872 |
| tall_512x16x64 | skinny | 67351 | 67231 | 0.998 | 66560 | 66080 | 0.993 | 0.994 | 0.020 |
| wide_16x512x64 | skinny | 73413 | 72017 | 0.981 | 110592 | 86016 | 0.778 | 0.801 | 0.096 |
| deepK_32x32x1024 | skinny | 133152 | 132903 | 0.998 | 135168 | 94272 | 0.697 | 0.747 | 0.008 |
| ragged_37x29x53 | ragged | 9818 | 9573 | 0.975 | 10864 | 9256 | 0.852 | 0.869 | 0.208 |
| ragged_101x67x131 | ragged | 123030 | 120755 | 0.982 | 141300 | 105460 | 0.746 | 0.781 | 0.060 |
| ragged_250x255x249 | ragged | 2046277 | 2045635 | 1.000 | 1768920 | 1326248 | 0.750 | 0.796 | 0.001 |
| mlp_b1_L0 | mlp | 74703 | 62159 | 0.832 | 203296 | 153120 | 0.753 | 0.759 | 0.652 |
| mlp_b1_L1 | mlp | 12404 | 10356 | 0.835 | 33536 | 25344 | 0.756 | 0.762 | 0.648 |
| mlp_b1_L2 | mlp | 2181 | 2068 | 0.948 | 1448 | 868 | 0.599 | 0.608 | 0.909 |
| mlp_b32_L0 | mlp | 806151 | 806039 | 1.000 | 634880 | 459776 | 0.724 | 0.780 | 0.001 |
| mlp_b32_L1 | mlp | 135677 | 134137 | 0.989 | 196608 | 135168 | 0.688 | 0.724 | 0.027 |
| mlp_b32_L2 | mlp | 7007 | 6912 | 0.986 | 6656 | 6136 | 0.922 | 0.932 | 0.160 |
| conv3x3_16to32_28 | conv | 454295 | 454215 | 1.000 | 439040 | 344960 | 0.786 | 0.822 | 0.002 |
| conv1x1_64to64_14 | conv | 103934 | 103486 | 0.996 | 140800 | 112128 | 0.796 | 0.821 | 0.032 |
| conv5x5_3to8_32 | conv | 83988 | 83917 | 0.999 | 110168 | 109884 | 0.997 | 0.998 | 0.016 |
| conv3x3_32to64_14 | conv | 458767 | 456751 | 0.996 | 402688 | 297856 | 0.740 | 0.787 | 0.013 |
| attn_s64_d32_QKt | attention | 18528 | 18304 | 0.988 | 22528 | 21632 | 0.960 | 0.965 | 0.099 |
| attn_s64_d32_PV | attention | 19089 | 18801 | 0.985 | 24576 | 19968 | 0.812 | 0.834 | 0.115 |
| attn_s128_d64_QKt | attention | 141631 | 139531 | 0.985 | 188416 | 151552 | 0.804 | 0.828 | 0.068 |
| attn_s128_d64_PV | attention | 138185 | 137321 | 0.994 | 139264 | 111616 | 0.801 | 0.833 | 0.045 |
| stress_512x512x64 | stress | 2102983 | 2102871 | 1.000 | 2621440 | 2162688 | 0.825 | 0.849 | 0.000 |
| stress_64x64x2048 | stress | 1052359 | 1052247 | 1.000 | 802816 | 573440 | 0.714 | 0.774 | 0.001 |