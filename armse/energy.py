"""Energy evidence levels (docs/ARCHITECTURE.md section 9).

Level 1  activity proxies           : exact counts from the simulator (bytes, beats, MACs, buffer accesses)
Level 2  analytical estimate        : counts x per-event energies. The defaults below are ILLUSTRATIVE
                                      45 nm-class literature-style numbers typed from memory (Horowitz,
                                      ISSCC 2014, as commonly quoted) - NOT measured, NOT for this FPGA.
                                      Always report with the sensitivity sweep.
Level 3  measured energy            : NOT performed (requires board power instrumentation).
"""
from .config import cdiv

DEFAULT_PJ = dict(dram_per_32b=640.0, sram_per_32b=5.0, mac_int8=0.2 + 0.1)  # mult8 + add32 (approx.)


def level1(m, hw):
    sram_wr = m["bytes_rd"]                       # every loaded byte is written to a local buffer
    sram_rd = 2 * m["macs"]                       # one A byte + one B byte per MAC
    sram_o = 2 * (m["macs"] // max(1, hw.lanes)) * 4   # approx. partial-sum traffic (bytes), upper-bound-ish
    return dict(ext_bytes=m["bytes_rd"] + m["bytes_wr"], sram_bytes=sram_wr + sram_rd + sram_o,
                macs=m["macs"])


def level2(m, hw, pj=None, dram_scale=1.0):
    p = dict(DEFAULT_PJ if pj is None else pj)
    p["dram_per_32b"] *= dram_scale
    l1 = level1(m, hw)
    e_dram = l1["ext_bytes"] / 4 * p["dram_per_32b"]
    e_sram = l1["sram_bytes"] / 4 * p["sram_per_32b"]
    e_mac = l1["macs"] * p["mac_int8"]
    return dict(e_dram_pj=e_dram, e_sram_pj=e_sram, e_mac_pj=e_mac, e_total_pj=e_dram + e_sram + e_mac)
