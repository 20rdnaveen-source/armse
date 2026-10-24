# Interface specification (specification level)

"Implemented" = present in `rtl/` and exercised by `tb/` (7 fixed cases simulated in Icarus by the project owner; 40 random cases pending). "Proposed" = specification only; no RTL, no test.

## 1. Top-level ports of `armse_top` (implemented)
| Port | Dir | Width | Meaning |
|---|---|---|---|
| clk, rst | in | 1 | clock, synchronous reset (clears sequencers and counters) |
| go | in | 1 | one-cycle pulse = cycle 0; clears counters, starts both engines |
| n_dma, n_cmp | in | 20 | number of DMA / compute descriptors (held stable during the run) |
| h_we, h_sel, h_addr[15:0], h_wdata[127:0] | in | - | host program-write port (h_sel 0 = DMA program, 1 = compute program) |
| busy, done | out | 1 | running / finished (done is sticky until the next go) |
| cycles | out | 32 | cycle index at which both programs completed (go pulse = cycle 0) |
| x_cmd_valid / x_cmd_ready | out/in | 1 | burst command handshake |
| x_cmd_write, x_cmd_addr[31:0], x_cmd_beats[7:0] | out | - | direction, byte address, number of 4-byte beats |
| x_rvalid, x_rdata[31:0] | in | - | read beat (one per cycle after `row_lat` wait) |
| x_wvalid, x_wdata[31:0], x_wready | out/in/in | - | write beat handshake (transfer when both high) |
| cmp_busy, cmp_wait, dma_busy, dma_wait, bytes_rd, bytes_wr, n_xfer, bank_conflicts | out | 32 each | performance counters (section 4) |

Backpressure: only on the write beat (`x_wready`). Read beats are assumed always accepted (the DMA writes the local buffer every cycle). Reset during an active run is **not verified**.

## 2. Descriptor formats (implemented; generated from `armse/rtlexport.py`, round-trip tested)
Entries are 128-bit words; unused high bits are zero.

**DMA descriptor (110 bits):** TYPE[1:0] (0 LOAD_A, 1 LOAD_B, 2 WRITEBACK), DEP[21:2] (job starts when compute_done >= DEP), ROWS[29:22], COLS[37:30], EXT_ADDR[69:38], EXT_STRIDE[85:70] (bytes between rows), BASE[101:86] (buffer word offset incl. slot), STRIDE[109:102] (words per buffer row: ceil(tk/NB) for loads, tn for writeback).

**Compute descriptor (93 bits):** TM[7:0], TN[15:8], TK[23:16], FIRST[24] (start new output tile: partial sum = 0), DEP[44:25] (starts when dma_done >= DEP), ABASE[60:45], BBASE[76:61], OBASE[92:77].

Limits: tile fields <= 255; DEP and indices < 2^20; external addresses 32-bit; EXT_STRIDE < 65536.

## 3. Execution semantics (implemented)
Two in-order sequencers, one per program. A job starts at the first cycle at which its dependency counter (visible one cycle after the other engine's job ended) is satisfied. DMA job duration = 2 + rows*(row_lat + 1 + beats). Compute job duration = tm*tn*ceil(tk/NB) + 4. Legal programs (slot reuse, output-slot reuse, no circular dependency) are guaranteed only by the host-side generator and simulator; **the RTL does not check legality**.

## 4. Performance counters (implemented)
cmp_busy (cycles outside IDLE), cmp_wait (idle with a valid entry waiting on DMA), dma_busy, dma_wait (waiting on compute), bytes_rd / bytes_wr (valid bytes), n_xfer (DMA jobs), bank_conflicts (cycles with two same-bank writes; must be 0), cycles. Busy counters equal model duration minus one per job.

## 5. Proposed memory-mapped wrapper (NOT implemented)
| Offset | Name | Access | Description |
|---|---|---|---|
| 0x00 | CTRL | W | bit0 GO (self-clearing), bit1 SOFT_RESET |
| 0x04 | STATUS | R | bit0 BUSY, bit1 DONE, bit2 ERROR (sticky), bits[7:4] ERROR_CODE |
| 0x08 | N_DMA | RW | descriptor counts |
| 0x0C | N_CMP | RW | |
| 0x10 | PROG_BASE | RW | external base of the descriptor image (for streaming mode) |
| 0x14 | CFG | RW | bit0 STREAM_EN, bits[7:4] fetch block size |
| 0x20-0x3C | counters | R | cycles, cmp_busy, cmp_wait, dma_busy, dma_wait, bytes_rd, bytes_wr, n_xfer |
| 0x40 | CONFLICTS | R | bank_conflicts |
| 0x44 | IRQ_EN | RW | interrupt on DONE / ERROR |

Proposed ERROR_CODE values: 1 illegal descriptor field (out-of-range base/stride), 2 external bus error, 3 watchdog timeout (no job completes within a programmable window), 4 unsupported descriptor type. None of these checks exist in the RTL today.

## 6. Fallback and failure behaviour (specification; host-side unless noted)
| Situation | Behaviour |
|---|---|
| Tensor does not fit local memory | host emits tiled program; if even a 1x1xK tile does not fit: `NoFitError` (no program) |
| No reuse available | schedule degenerates to streaming tiles (equivalent to the direct baseline) |
| Bank conflicts | cannot occur with the skewed mapping; the counter would expose a design bug |
| Overlap impossible | host selects single-buffering (nbuf 1/1) |
| Dynamic shape changes | host re-plans, or reuses a bucketed schedule clipped to the shape (hybrid, section "edge workloads") |
| Metadata unavailable or wrong | RTL executes whatever program it is given; a wrong program can deadlock or corrupt data (no hardware watchdog) -> host simulator validation is mandatory; proposed watchdog (error 3) |
| Adaptive scheduler slower than static | host keeps both candidates and picks by estimate; D_feedback shows how counters would drive this |
| Scheduler deadline reached | host uses the cached bucket schedule or the fixed baseline schedule |
