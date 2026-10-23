"""Cycle-by-cycle Python transcription of rtl/*.v + tb/tb_armse.v (register-transfer behaviour).

Purpose: check the RTL *logic* (bank/address math, pipeline, handshakes, dependency counters) and the
hand-derived timing constants against the verified event model, without an HDL simulator.
It is a transcription written by hand from the Verilog: it can expose design/logic errors, it CANNOT find
Verilog syntax/semantic-tool errors, and X-propagation is not modelled.  Every register update uses the
values from before the clock edge (non-blocking semantics)."""
import os
from .rtlexport import DMA_L, CMP_L, unpack, rtl_hw

LA, LB, WB = 0, 1, 2
S_IDLE, S_CMD, S_LD, S_WR, S_DONE = 0, 1, 2, 3, 4          # DMA
C_IDLE, C_RUN, C_D1, C_D2, C_DONE = 0, 1, 2, 3, 4         # compute


def s32(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


def s8(v):
    v &= 0xFF
    return v - 256 if v & 0x80 else v


class Rtl:
    def __init__(self, NB, a_depth, b_depth, o_depth, row_lat, dprog, cprog, ext):
        self.NB, self.LOG = NB, NB.bit_length() - 1
        self.L = row_lat
        self.dprog = [unpack(DMA_L, w) for w in dprog]
        self.cprog = [unpack(CMP_L, w) for w in cprog]
        self.n_dma, self.n_cmp = len(dprog), len(cprog)
        self.amem = [[0] * a_depth for _ in range(NB)]
        self.bmem = [[0] * b_depth for _ in range(NB)]
        self.ard, self.brd = [0] * NB, [0] * NB
        self.apar = self.bpar = 0
        self.omem = [0] * o_depth
        self.c_rdata = 0
        self.xmem = bytearray(ext)
        self.conflicts = 0
        # ext memory model
        self.active = self.wr = False
        self.addr_cur = self.beats_left = self.cnt = 0
        # top
        self.run = self.fin = False
        self.cyc = 0
        # dma
        self.d = dict(state=S_IDLE, idx=0, done=0, typ=0, rows=0, cols=0, stride=0, row=0, beat=0, nbeats=0,
                      eaddr=0, estr=0, base=0, rb=0, jb=0, bytes_rd=0, bytes_wr=0, jobs=0, busy=0, wait=0)
        # compute
        self.c = dict(state=C_IDLE, idx=0, done=0, tm=0, tn=0, tk=0, kg=0, first=0, bbase=0, i=0, j=0, g=0,
                      a_row=0, b_col=0, o_ptr=0, v1=0, g0_1=0, gl_1=0, g1=0, optr1=0, v2=0, g0_2=0, gl_2=0,
                      optr2=0, dot=0, part=0, acc=0, busy=0, wait=0)

    # ------------------------------------------------------------------ one clock cycle
    def step(self, go):
        NB, LOG, d, c = self.NB, self.LOG, self.d, self.c
        # ---- combinational view of the current state ----
        active, wr, cnt, bl = self.active, self.wr, self.cnt, self.beats_left
        x_cmd_ready = not active
        x_rvalid = active and not wr and cnt == 0 and bl != 0
        x_wready = active and wr and cnt == 0 and bl != 0
        a = self.addr_cur
        x_rdata = (self.xmem[a] | self.xmem[a + 1] << 8 | self.xmem[a + 2] << 16 | self.xmem[a + 3] << 24)
        # DMA comb
        have_d = self.run and d["idx"] < self.n_dma
        e = self.dprog[d["idx"]] if d["idx"] < self.n_dma else None
        dep_ok_d = e is not None and c["done"] >= e["DEP"]
        start_d = d["state"] == S_IDLE and have_d and dep_ok_d
        col0 = (d["beat"] & 63) * 4
        rem = (d["cols"] - col0) & 0x1FF
        nvalid = 4 if rem >= 4 else rem & 15
        st = d["stride"]
        x_cmd_valid = d["state"] == S_CMD
        x_wvalid = d["state"] == S_WR
        od_raddr = ((d["rb"] & 0x3FFF) + d["beat"]) & 0x3FFF
        in_range = od_raddr < len(self.omem)
        x_wdata = self.omem[od_raddr] if in_range else 0     # Verilog: X, unused unless a write beat transfers
        wr_elems = []                                           # (buffer, bank, addr, byte)
        if d["state"] == S_LD and x_rvalid:
            banks = []
            for q in range(4):
                if q < nvalid:
                    elem = col0 + q
                    bank = (d["row"] + elem) % NB
                    banks.append(bank)
                    byte = (x_rdata >> (8 * q)) & 0xFF
                    if d["typ"] == LA:
                        wr_elems.append(("A", bank, (d["rb"] + (elem >> LOG)) & 0xFFF, byte))
                    elif d["typ"] == LB:
                        wr_elems.append(("B", bank, (d["jb"] + q * st + (d["row"] >> LOG)) & 0xFFF, byte))
            if len(set(banks)) != len(banks):
                self.conflicts += 1
        beat_xfer = (d["state"] == S_LD and x_rvalid) or (d["state"] == S_WR and x_wready)
        if d["state"] == S_WR and x_wready:
            assert in_range, f"writeback reads outside the output buffer: addr {od_raddr}"
        last_beat = d["beat"] == (d["nbeats"] - 1) & 0xFF
        last_row = d["row"] == (d["rows"] - 1) & 0xFF
        # compute comb
        have_c = self.run and c["idx"] < self.n_cmp
        ec = self.cprog[c["idx"]] if c["idx"] < self.n_cmp else None
        dep_ok_c = ec is not None and d["done"] >= ec["DEP"]
        start_c = c["state"] == C_IDLE and have_c and dep_ok_c
        issuing = c["state"] == C_RUN
        a_addr = (c["a_row"] + c["g"]) & 0xFFF
        b_addr = (c["b_col"] + c["g"]) & 0xFFF
        o_re = issuing and c["g"] == 0 and not c["first"]
        o_raddr = c["o_ptr"] & 0x3FFF
        # stage-1 dot product from registered bank outputs (de-rotated lanes)
        dot_c = 0
        for l in range(NB):
            k = c["g1"] * NB + l
            if k < c["tk"]:
                av = s8(self.ard[(l + self.apar) % NB])
                bv = s8(self.brd[(l + self.bpar) % NB])
                dot_c += av * bv
        dot_c = s32(dot_c)
        acc_next = s32((c["part"] if c["g0_2"] else c["acc"]) + c["dot"])
        o_we = bool(c["v2"] and c["gl_2"])
        o_waddr, o_wdata = c["optr2"] & 0x3FFF, acc_next

        # ---- next state ----
        nd, nc = dict(d), dict(c)
        # external memory model (tb)
        n_active, n_wr, n_addr, n_bl, n_cnt = active, wr, self.addr_cur, bl, cnt
        xw = []
        if x_cmd_valid and x_cmd_ready:
            sd = d
            cmd_addr = d["eaddr"]
            n_active, n_wr, n_addr, n_bl, n_cnt = True, d["typ"] == WB, cmd_addr, d["nbeats"], self.L
        elif active:
            if cnt != 0:
                n_cnt = cnt - 1
            elif x_rvalid or (x_wready and x_wvalid):
                if x_wready and x_wvalid:
                    for k in range(4):
                        xw.append((self.addr_cur + k, (x_wdata >> (8 * k)) & 0xFF))
                n_addr = (self.addr_cur + 4) & 0xFFFFFFFF
                n_bl = (bl - 1) & 0xFF
                if bl == 1:
                    n_active = False
        # DMA engine
        if go:
            nd.update(state=S_IDLE, idx=0, done=0, bytes_rd=0, bytes_wr=0, jobs=0, busy=0, wait=0)
        else:
            if d["state"] != S_IDLE:
                nd["busy"] += 1
            elif have_d and not dep_ok_d:
                nd["wait"] += 1
            s = d["state"]
            if s == S_IDLE:
                if start_d:
                    nbeats = e["COLS"] if e["TYPE"] == WB else (e["COLS"] + 3) >> 2
                    nd.update(typ=e["TYPE"], rows=e["ROWS"], cols=e["COLS"], stride=e["STRIDE"], nbeats=nbeats,
                              eaddr=e["EXT_ADDR"], estr=e["EXT_STRIDE"], base=e["BASE"], rb=e["BASE"],
                              jb=e["BASE"], row=0, beat=0, state=S_CMD)
            elif s == S_CMD:
                if x_cmd_ready:
                    nd.update(beat=0, state=S_WR if d["typ"] == WB else S_LD)
            elif s in (S_LD, S_WR):
                if beat_xfer:
                    if s == S_LD:
                        nd["bytes_rd"] += nvalid
                    else:
                        nd["bytes_wr"] += 4
                    if last_beat:
                        if last_row:
                            nd["state"] = S_DONE
                        else:
                            nd.update(row=d["row"] + 1, eaddr=(d["eaddr"] + d["estr"]) & 0xFFFFFFFF,
                                      rb=(d["rb"] + st) & 0xFFFF, jb=d["base"], beat=0, state=S_CMD)
                    else:
                        nd.update(beat=d["beat"] + 1, jb=(d["jb"] + 4 * st) & 0xFFFF)
            elif s == S_DONE:
                nd.update(done=d["done"] + 1, idx=d["idx"] + 1, jobs=d["jobs"] + 1, state=S_IDLE)
        # compute engine
        if go:
            nc.update(state=C_IDLE, idx=0, done=0, v1=0, v2=0, busy=0, wait=0)
        else:
            nc.update(v1=int(issuing), g0_1=int(c["g"] == 0), gl_1=int(c["g"] == (c["kg"] - 1) & 0xFF),
                      g1=c["g"], optr1=c["o_ptr"], v2=c["v1"], g0_2=c["g0_1"], gl_2=c["gl_1"],
                      optr2=c["optr1"], dot=dot_c, part=0 if c["first"] else self.c_rdata)
            if c["v2"]:
                nc["acc"] = acc_next
            if c["state"] != C_IDLE:
                nc["busy"] += 1
            elif have_c and not dep_ok_c:
                nc["wait"] += 1
            s = c["state"]
            if s == C_IDLE:
                if start_c:
                    kg = (ec["TK"] + NB - 1) >> LOG
                    nc.update(tm=ec["TM"], tn=ec["TN"], tk=ec["TK"], kg=kg, first=ec["FIRST"], bbase=ec["BBASE"],
                              i=0, j=0, g=0, a_row=ec["ABASE"], b_col=ec["BBASE"], o_ptr=ec["OBASE"], state=C_RUN)
            elif s == C_RUN:
                if c["g"] == (c["kg"] - 1) & 0xFF:
                    nc.update(g=0, o_ptr=(c["o_ptr"] + 1) & 0xFFFF)
                    if c["j"] == (c["tn"] - 1) & 0xFF:
                        nc.update(j=0, b_col=c["bbase"])
                        if c["i"] == (c["tm"] - 1) & 0xFF:
                            nc["state"] = C_D1
                        else:
                            nc.update(i=c["i"] + 1, a_row=(c["a_row"] + c["kg"]) & 0xFFFF)
                    else:
                        nc.update(j=c["j"] + 1, b_col=(c["b_col"] + c["kg"]) & 0xFFFF)
                else:
                    nc["g"] = c["g"] + 1
            elif s == C_D1:
                nc["state"] = C_D2
            elif s == C_D2:
                nc["state"] = C_DONE
            elif s == C_DONE:
                nc.update(done=c["done"] + 1, idx=c["idx"] + 1, state=C_IDLE)
        # memories (all reads use pre-edge contents)
        if issuing:
            self.ard = [self.amem[b][a_addr] for b in range(NB)]
            self.brd = [self.bmem[b][b_addr] for b in range(NB)]
            self.apar, self.bpar = c["i"] % NB, c["j"] % NB
        if o_re:
            self.c_rdata = self.omem[o_raddr]
        if o_we:
            self.omem[o_waddr] = o_wdata & 0xFFFFFFFF
        for buf, bank, addr, byte in wr_elems:
            (self.amem if buf == "A" else self.bmem)[bank][addr] = byte
        for adr, v in xw:
            self.xmem[adr] = v
        self.active, self.wr, self.addr_cur, self.beats_left, self.cnt = n_active, n_wr, n_addr, n_bl, n_cnt
        # top (cycle counter)
        done_w = self.run and d["done"] == self.n_dma and c["done"] == self.n_cmp
        if go:
            self.run, self.fin, self.cyc = True, False, 1
        elif self.run and not self.fin:
            if done_w:
                self.fin = True
            else:
                self.cyc += 1
        self.d, self.c = nd, nc


def run_case(case_dir, max_cycles=5_000_000):
    """Run one exported RTL test directory through the transcription; returns a result dict."""
    meta = {}
    for line in open(os.path.join(case_dir, "meta.vh")):
        p = line.split()
        if len(p) >= 3 and p[0] == "`define":
            meta[p[1]] = p[2].strip('"')
    rd = lambda f: [l.strip() for l in open(os.path.join(case_dir, f)) if l.strip()]
    ext = bytes(int(x, 16) for x in rd("ext.hex"))
    dprog = [int(x, 16) for x in rd("dprog.hex")]
    cprog = [int(x, 16) for x in rd("cprog.hex")]
    expc = [int(x, 16) for x in rd("expc.hex")]
    m = Rtl(int(meta["NB_P"]), int(meta["A_DEPTH_P"]), int(meta["B_DEPTH_P"]), int(meta["O_DEPTH_P"]),
            int(meta["ROW_LAT_P"]), dprog, cprog, ext)
    m.step(True)                                    # go pulse = cycle 0
    n = 0
    while not m.fin and n < max_cycles:
        m.step(False)
        n += 1
    c_off = int(meta["C_OFF"])
    errs = 0
    for i, w in enumerate(expc):
        got = int.from_bytes(m.xmem[c_off + 4 * i:c_off + 4 * i + 4], "little")
        errs += got != w
    return dict(name=meta["TEST_NAME"], finished=m.fin, data_errors=errs, cycles=m.cyc,
                exp_cycles=int(meta["EXP_CYCLES"]), bytes_rd=m.d["bytes_rd"], exp_bytes_rd=int(meta["EXP_BYTES_RD"]),
                bytes_wr=m.d["bytes_wr"], exp_bytes_wr=int(meta["EXP_BYTES_WR"]), xfers=m.d["jobs"],
                exp_xfers=int(meta["EXP_XFER"]), cmp_busy=m.c["busy"], exp_cmp_busy=int(meta["EXP_CMP_BUSY"]),
                dma_busy=m.d["busy"], exp_dma_busy=int(meta["EXP_DMA_BUSY"]), conflicts=m.conflicts)


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "rtl_vectors"
    bad = 0
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        r = run_case(p)
        ok = (r["finished"] and r["data_errors"] == 0 and r["bytes_rd"] == r["exp_bytes_rd"] and
              r["bytes_wr"] == r["exp_bytes_wr"] and r["xfers"] == r["exp_xfers"] and r["conflicts"] == 0)
        bad += not ok
        print(f"{name}: {'PASS' if ok else 'FAIL'} data_err={r['data_errors']} cycles rtl={r['cycles']} "
              f"model={r['exp_cycles']} delta={r['cycles'] - r['exp_cycles']} | cmp_busy {r['cmp_busy']}/{r['exp_cmp_busy']}"
              f" dma_busy {r['dma_busy']}/{r['exp_dma_busy']} conflicts={r['conflicts']}")
    sys.exit(1 if bad else 0)
