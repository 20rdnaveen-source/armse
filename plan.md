# Plan for `digit_mlp_example`

Hardware model: lanes=8, bus=4 B/cycle, burst latency=12, A/B buffers 2048/2048 B, output 4096 B. All numbers are SIMULATED cycles (not measured).

| layer | MxNxK | bound | tile | order | buf | cycles | vs fixed double-buffering | vs best fixed tiling | stall | verified |
|---|---|---|---|---|---|---|---|---|---|---|
| fc1 | 1x128x784 | memory-bound | 1x128x8 | MN | 2/1 | 37418 | 0.325x | 0.545x | 65% | True |
| fc2 | 1x64x128 | memory-bound | 1x64x16 | MN | 2/1 | 4092 | 0.429x | 0.705x | 72% | True |
| fc3 | 1x10x64 | memory-bound | 1x10x32 | MN | 2/2 | 1144 | 0.988x | 1.000x | 90% | True |

**Total:** 42654 cycles; fixed double-buffering 125836 (0.339x), best fixed tiling 75544 (0.565x). Planning time 0.73 s (host).
Layers marked memory-bound benefit most; compute-bound layers have little memory time to remove.