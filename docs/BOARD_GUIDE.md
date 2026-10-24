# Board / device selection guide for ARMSE

Written without any board or datasheet access. Everything below about *this design* is computed from the RTL; everything about *specific boards or tools* is general knowledge that you must verify against the vendor datasheet before buying.

## 1. What the current design needs (computed from the RTL defaults)

| Resource | Amount | Source |
|---|---|---|
| A buffer | 16,384 bits (8 banks x 256 x 8 b) | `armse_top` defaults |
| B buffer | 16,384 bits | same |
| Output buffer | 32,768 bits (1024 x 32 b) | same |
| Descriptor memories (testbench size, PROG_DEPTH=256, two memories x 128 b) | 65,536 bits | `rtl` / `synth/xilinx7.ys` |
| **Total on-chip RAM** | **131,072 bits = 16 KB** | sum |
| Descriptor memories at PROG_DEPTH=4096 | 1,048,576 bits = 128 KB | would dominate; model says streaming descriptors from external memory is cheaper |
| Multipliers | 8 (signed 8x8) + adder tree | `lanes = 8` |
| External memory | **not required for the tests**; needed only when tensors exceed on-chip RAM | model assumes one outstanding burst, 4 B/beat |

Conclusion: the design is tiny. Almost any mid-size FPGA has enough RAM and multipliers. The deciding factors are therefore **toolchain, external-memory access, and what you want to measure**, not size.

Caveat that matters for small devices: the output buffer's DMA read and the descriptor memories are *asynchronous reads*, which synthesis maps to LUT-RAM (logic) rather than block RAM. On a very small device this can use a large share of the LUTs. A synchronous-read variant needs one extra pipeline stage and a new timing contract (not implemented). Run `scripts/run_synth.ps1` first and look at the LUT count before choosing a device.

## 2. You can get real timing numbers WITHOUT owning a board

Fmax and post-place-and-route resources come from running the vendor/open tool for a chosen *device part*, not from hardware:
* **Open-source route:** Yosys + nextpnr for a supported family (the Lattice ECP5 and iCE40 flows are the mature ones) reports Fmax and utilisation for a part with no board.
* **Vendor route:** the free edition of the vendor IDE (e.g. Xilinx Vivado ML Standard, Intel Quartus Prime Lite) can implement a design for supported parts and give a timing report. Check the tool's supported-device list for the exact part.
Doing this first tells you which board families are worth buying. It is the cheapest way to close roadmap item P1.

## 3. What a board adds (and what it does not)

A board is needed for: **measured sustained external-memory bandwidth and burst latency** (roadmap P2), **measured power/energy** (P7, needs instrumentation), and an actual end-to-end run. The model's constants `row_lat`, `bus_bytes` and buffer depths should then be replaced by measured values and the benchmark rerun.

## 4. Board categories (verify every claim against the datasheet)

| Category | Typical examples | Good for | Watch out for |
|---|---|---|---|
| Open-flow friendly (Lattice ECP5 / iCE40 class) | community ECP5 boards (e.g. ULX3S-style), small iCE40 boards | Fully open synth + place-and-route, Fmax with no vendor licence | smallest iCE40 parts may be too small; check RAM and multiplier counts; external memory varies |
| Xilinx 7-series (Artix-7 class) | Digilent Arty A7, Basys 3 | matches the Yosys Xilinx script; free Vivado edition covers small parts (check part support); some boards have DDR3, some do not | boards without external DRAM cannot measure external-memory behaviour |
| SoC FPGA (Zynq-7000 class) | PYNQ-Z2, ZedBoard | ARM host + DDR on the same chip, real DMA to DDR, Python host control | more complex bring-up (AXI interfaces); the current RTL's simple command/read/write port would need an adapter |
| Intel/Altera SoC class | DE10-Nano-class boards | ARM host + DDR, Quartus Lite flow | different tool flow than the scripts in this repo |
| Gowin class | low-cost Tang-series boards | very low cost, open flow support exists for some parts | check maturity and documentation before relying on it |

## 5. Decision shortcuts
* **Cheapest path to a defensible Fmax/resource table:** pick a part, run the open or free vendor flow, no purchase yet.
* **Want to measure external-memory behaviour (the heart of this project):** the board must have external DRAM and a documented way to reach it from fabric. Prefer a board where you already know how to write a DDR/AXI test.
* **Want the least integration work:** a SoC board with an ARM host, accepting that the memory port needs an adapter.
* **Want fully open tools:** an ECP5-class board.

## 6. Checklist before buying
1. RAM bits >= 200 Kbit (this design, with margin) and >= 8 multipliers or enough LUTs to build them.
2. Toolchain you can actually run on your PC (OS support, licence, install size).
3. External DRAM present? Size, width, and a reference design that reaches it.
4. A way to load and observe results (UART/JTAG/USB, ARM host).
5. Power measurement possible (shunt/header) if you want Level-3 energy.
6. Local availability, price, and shipping time.
7. Run `scripts/run_synth.ps1` and compare LUT/FF/BRAM against the candidate device's datasheet.

## 7. What to send me once you have a candidate
Part number or board name, the vendor datasheet numbers for RAM/DSP/LUTs, and the tool you can install. I will then add the matching synthesis/implementation script and update the model constants only from measurements, never from guesses.
