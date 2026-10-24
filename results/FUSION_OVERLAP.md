# Fused chains with cross-phase/cross-block overlap (MODEL ONLY; `python3 -m armse.overlapbench`)

Ratios are cycles relative to the unfused D_feedback pair (<1 = faster). Barrier = earlier study (separate programs). Overlap = one continuous program sharing slot state, common nbuf across segments. Functional check: 20/20 overlapped configurations bit-exact vs an independent numpy reference; all pass deadlock + hazard checks.

| chain | a_buf_bytes | unfused_cycles | barrier_cycles | overlap_cycles | barrier_vs_unfused | overlap_vs_unfused | unfused_bytes | overlap_bytes | tm | nbuf | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| attn_s64_d32 | 2048 | 36730 | 37537 | 35597 | 1.022 | 0.969 | 51200 | 43008 | 16 | 2/2 | fits |
| attn_s128_d64 | 2048 | 270608 | 282807 | 275325 | 1.045 | 1.017 | 303104 | 319488 | 8 | 2/2 | fits |
| attn_s256_d64 | 2048 | 1056871 | 1138816 | 1094691 | 1.078 | 1.036 | 950272 | 2211840 | 4 | 2/2 | fits |
| mlp_b1_784_256_128 | 2048 | 86915 | 86293 | 86144 | 0.993 | 0.991 | 236064 | 235552 | 1 | 2/1 | fits |
| mlp_b32_784_256_128 | 2048 | 941444 | 953040 | 946517 | 1.012 | 1.005 | 806912 | 1934336 | 4 | 2/2 | fits |
| mlp_b128_784_256_128 | 2048 | 3746354 | 3812160 | 3784517 | 1.018 | 1.010 | 2899968 | 7737344 | 4 | 2/2 | fits |
| mlp_b64_128_512_128 | 2048 | 1053710 | 1575104 | 1557637 | 1.495 | 1.478 | 851968 | 4259840 | 2 | 2/1 | fits |
| wideN1_b256_64_1024_64 | 2048 | 4205582 | 13936896 | 13864837 | 3.314 | 3.297 | 3473408 | 33751040 | 1 | 2/1 | fits |
| attn_s128_d64 | 8192 | 270608 | 279312 | 274157 | 1.032 | 1.013 | 303104 | 319488 | 16 | 2/2 | fits |
| mlp_b32_784_256_128 | 8192 | 941444 | 942452 | 940269 | 1.001 | 0.999 | 806912 | 749568 | 16 | 2/2 | fits |
| mlp_b64_128_512_128 | 8192 | 1053710 | 1070641 | 1062869 | 1.016 | 1.009 | 851968 | 1163264 | 8 | 2/2 | fits |
| wideN1_b256_64_1024_64 | 8192 | 4203855 | 4338112 | 4280105 | 1.032 | 1.018 | 2506752 | 8585216 | 4 | 2/1 | fits |
| attn_s64_d32 | 32768 | 36730 | 35234 | 34941 | 0.959 | 0.951 | 51200 | 47104 | 64 | 2/2 | fits |
| attn_s128_d64 | 32768 | 270608 | 270011 | 269381 | 0.998 | 0.995 | 303104 | 278528 | 128 | 2/2 | fits |
| attn_s256_d64 | 32768 | 1056871 | 1069968 | 1066849 | 1.012 | 1.009 | 950272 | 966656 | 86 | 2/2 | fits |
| mlp_b1_784_256_128 | 32768 | 86915 | 86293 | 86144 | 0.993 | 0.991 | 236064 | 235552 | 1 | 2/1 | fits |
| mlp_b32_784_256_128 | 32768 | 941444 | 940442 | 939975 | 0.999 | 0.998 | 806912 | 782336 | 32 | 2/2 | fits |
| mlp_b128_784_256_128 | 32768 | 3744554 | 3761768 | 3756849 | 1.005 | 1.003 | 2899968 | 2998272 | 16 | 2/2 | fits |
| mlp_b64_128_512_128 | 32768 | 1053710 | 1059512 | 1056697 | 1.006 | 1.003 | 851968 | 819200 | 32 | 2/2 | fits |
| wideN1_b256_64_1024_64 | 32768 | 4203855 | 4269857 | 4250311 | 1.016 | 1.011 | 2506752 | 4521984 | 8 | 2/2 | fits |

Geomean cycles vs unfused over configurations that fit: barrier 1.096, overlap 1.084; overlap faster than unfused in 7/20 cases (barrier: 5/20).
