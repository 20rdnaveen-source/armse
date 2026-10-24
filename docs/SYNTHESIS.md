# Synthesis / resource estimate (P1)

Status: scripts written, **not yet run**. Nothing here is a measured result until you run it and the numbers are recorded in REPORT.md with the tool version.

## Run
```
powershell -ExecutionPolicy Bypass -File scripts\run_synth.ps1      (Windows)
yosys -l synth/yosys_xilinx7.log -s synth/xilinx7.ys                 (Linux/WSL)
```
Output: `synth/yosys_xilinx7.log` and `synth/xilinx7_stat.txt` (cells per module: LUT*, FDRE/FDCE (flip-flops), RAMB18/RAMB36, RAM64X1D (LUTRAM), DSP48E1).

## What the numbers will and will not mean
* Yosys gives a **post-synthesis cell estimate**, not place-and-route. **No Fmax, no timing closure.** Those need Vivado/Quartus/nextpnr on a real device.
* The estimate targets generic Xilinx 7-series primitives. Your board may differ (Intel/Lattice/other); once you tell me the board I will add the matching flow.
* **Known pessimistic spots, by design of the current RTL:**
  * the output buffer's DMA port is an *asynchronous* read -> maps to LUTRAM, not BRAM;
  * the two descriptor program memories are *asynchronous* read arrays -> LUTRAM/registers (shrunk to 256 entries in the script). A BRAM version needs one extra pipeline stage, which changes the timing contract and the Python model constants.
  * `initial` blocks that zero memories are simulation conveniences.
* Resource counts should be reported per module (`armse_skew_mem` x2, `armse_obuf`, `armse_dma`, `armse_compute`) and for the top, next to the on-chip data capacity (8 KB default).

## Record results here after running
| Module | LUT | FF | BRAM18/36 | LUTRAM | DSP48 | Tool/version |
|---|---|---|---|---|---|---|
| armse_skew_mem (A) | | | | | | |
| armse_skew_mem (B) | | | | | | |
| armse_obuf | | | | | | |
| armse_dma | | | | | | |
| armse_compute | | | | | | |
| armse_top (PROG_DEPTH=256) | | | | | | |
