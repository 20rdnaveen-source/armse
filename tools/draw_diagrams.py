"""Draws docs/img/armse_block_diagram.png and docs/img/armse_flow.png (matplotlib). Run: python tools/draw_diagrams.py"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "img")
os.makedirs(OUT, exist_ok=True)
COL = dict(host="#dbeafe", hw="#dcfce7", mem="#fef3c7", ext="#e5e7eb", ctl="#fce7f3", cnt="#ede9fe")


def box(ax, x, y, w, h, text, c, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=c, ec="#374151", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, p, q, text="", style="-|>", color="#111827", off=(0, 0.12), fs=7.5, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=12, lw=1.3, color=color, linestyle=ls))
    if text:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, ha="center", va="center", fontsize=fs, color=color,
                bbox=dict(fc="white", ec="none", pad=0.6))


def poly(ax, pts, text="", tpos=None, color="#111827", both=False, fs=7.5, ls="-"):
    for p, q in zip(pts[:-2], pts[1:-1]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=1.3, ls=ls)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="<|-|>" if both else "-|>", mutation_scale=12, lw=1.3, color=color, linestyle=ls))
    if both and len(pts) == 2:
        pass
    if text and tpos:
        ax.text(tpos[0], tpos[1], text, ha="center", va="center", fontsize=fs, color=color, bbox=dict(fc="white", ec="none", pad=0.5))


def block_diagram():
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis("off")
    ax.text(8, 10.7, "ARMSE block diagram  (green/yellow/pink = RTL in this repo;  blue = host software, not RTL)", ha="center", fontsize=13.5, fontweight="bold")
    box(ax, 0.3, 6.2, 3.6, 3.2, "HOST SOFTWARE (Python)\n\nscheduler: tile x order x buffering\ncost model (+ calibrated D_v2)\nsimulator: deadlock / hazard check\ndescriptor generator\n\nmodel-only: counter-driven re-planning,\ndeadline / energy-bounded planner", COL["host"], 8.3)
    box(ax, 4.6, 8.3, 3.0, 1.1, "DMA program memory\n128-bit descriptors\nLOAD_A / LOAD_B / WRITEBACK", COL["ctl"], 8)
    box(ax, 11.6, 8.3, 3.3, 1.1, "compute program memory\n128-bit descriptors\ntm tn tk first DEP bases", COL["ctl"], 8)
    box(ax, 8.4, 7.9, 2.4, 1.0, "dependency counters\ndma_done   cp_done\n(20-bit)", COL["cnt"], 8)
    box(ax, 4.6, 5.4, 3.0, 2.2, "armse_dma\nsequencer + transfer engine\nIDLE > CMD > LD / WR > DONE\nbank + address generation\n(4 elements per beat)", COL["hw"], 8.2, True)
    box(ax, 11.6, 4.9, 3.3, 2.7, "armse_compute\nsequencer + NB MAC lanes\n8x8 signed multiplies + adder tree\nread > dot > accumulate > write\nlane mask for ragged tk", COL["hw"], 8.2, True)
    box(ax, 8.4, 6.5, 2.4, 1.1, "A buffer (armse_skew_mem)\nNB banks, bank = (r + c) mod NB", COL["mem"], 7.8)
    box(ax, 8.4, 5.2, 2.4, 1.1, "B buffer (armse_skew_mem)\nstored transposed, same mapping", COL["mem"], 7.8)
    box(ax, 8.4, 3.4, 6.5, 1.2, "output / partial-sum buffer (armse_obuf), 32-bit words\ncompute port: 1 read + 1 write (sync)     DMA port: 1 read (async)", COL["mem"], 8)
    box(ax, 4.6, 0.8, 10.3, 1.3, "EXTERNAL MEMORY  (testbench model here; DDR / AXI later)\none outstanding burst: command (valid / ready, write, addr, beats)  >  row_lat wait  >  4-byte beats", COL["ext"], 8.3)
    box(ax, 0.3, 4.2, 3.6, 1.5, "control / status\ngo pulse, n_dma, n_cmp\nbusy, done, cycles", COL["host"], 8.2)
    box(ax, 0.3, 0.8, 3.6, 3.0, "performance counters (outputs)\ncmp_busy / cmp_wait\ndma_busy / dma_wait\nbytes_rd / bytes_wr\nn_xfer (DMA jobs)\nbank_conflicts (must be 0)", COL["cnt"], 8.2)
    # host -> program memories (write port)
    poly(ax, [(3.9, 8.85), (4.6, 8.85)], "h_we / h_sel /\nh_addr / h_wdata", (4.25, 9.75))
    poly(ax, [(2.1, 9.4), (2.1, 10.15), (13.25, 10.15), (13.25, 9.4)], "same program-write port (h_sel = 1)", (8.0, 10.15))
    # program memory -> engines
    poly(ax, [(6.1, 8.3), (6.1, 7.6)], "entry[idx]", (5.3, 7.95))
    poly(ax, [(13.25, 8.3), (13.25, 7.6)], "entry[idx]", (14.1, 7.95))
    # counters <-> engines
    poly(ax, [(7.4, 7.6), (7.4, 8.15), (8.4, 8.15)], "", None)
    poly(ax, [(8.4, 8.0), (7.0, 8.0), (7.0, 7.6)], "", None)
    ax.text(8.0, 8.45, "dma_done >", fontsize=7.3, ha="center")
    ax.text(7.7, 7.78, "< cp_done", fontsize=7.3, ha="center")
    poly(ax, [(12.0, 7.6), (12.0, 8.15), (10.8, 8.15)], "", None)
    poly(ax, [(10.8, 8.0), (12.4, 8.0), (12.4, 7.6)], "", None)
    ax.text(11.2, 8.45, "< cp_done", fontsize=7.3, ha="center")
    ax.text(11.5, 7.78, "dma_done >", fontsize=7.3, ha="center")
    # data path
    poly(ax, [(7.6, 7.05), (8.4, 7.05)], "write", (8.0, 7.3))
    poly(ax, [(7.6, 5.75), (8.4, 5.75)], "write", (8.0, 6.0))
    poly(ax, [(10.8, 7.05), (11.6, 7.05)], "NB-lane\nreads", (11.2, 7.42))
    poly(ax, [(10.8, 5.75), (11.6, 5.75)], "", None)
    poly(ax, [(13.25, 4.9), (13.25, 4.6)], "", None, both=True)
    ax.text(15.45, 4.75, "psum read /\nresult write", fontsize=7.5, ha="center", va="center")
    poly(ax, [(8.4, 4.0), (6.2, 4.0), (6.2, 5.4)], "writeback data (async read)", (7.3, 3.75))
    poly(ax, [(5.4, 5.4), (5.4, 2.1)], "", None, color="#6b7280", both=True)
    ax.text(5.15, 3.5, "x_cmd_*  x_rdata  x_wdata  x_wready", fontsize=7.5, color="#6b7280", rotation=90, ha="center", va="center")
    ax.text(8, 0.3, "Legality (slot reuse, output-slot reuse, no circular dependency) is guaranteed by the host generator + simulator; the RTL does not check it.\n"
            "Each engine starts a job only when the other engine's completed-job counter >= the job's DEP field.", ha="center", fontsize=8.5, style="italic")
    fig.savefig(os.path.join(OUT, "armse_block_diagram.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def flow_diagram():
    fig, ax = plt.subplots(figsize=(15, 5.2))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    ax.text(7.5, 4.9, "Scheduling and execution flow", ha="center", fontsize=13, fontweight="bold")
    steps = [("workload\nGEMM M,N,K\n(conv/attn/MLP lowered)", COL["host"]), ("enumerate candidates\ntile x order x buffering", COL["host"]),
             ("reject illegal\ncapacity / field width /\nprogram length", COL["host"]), ("rank by cost model\n(closed form;\ncalibrated D_v2)", COL["host"]),
             ("verify (simulator)\ndeadlock + hazards\n[+ top-k feedback]", COL["host"]), ("emit descriptor\nprograms\n(DMA + compute)", COL["host"]),
             ("RTL executes\ntwo counters,\ndecoupled engines", COL["hw"])]
    x = 0.2
    w, gap = 1.8, 0.25
    for i, (t, c) in enumerate(steps):
        box(ax, x, 2.4, w, 1.6, t, c, 8.5)
        if i:
            arrow(ax, (x - gap, 3.2), (x, 3.2))
        x += w + gap
    box(ax, 4.2, 0.5, 4.4, 1.3, "no candidate fits  ->  minimal 1x1xK tile\nnothing legal  ->  NoFitError", "#fee2e2", 8.5)
    arrow(ax, (5.4, 2.4), (5.9, 1.8), "", color="#b91c1c")
    box(ax, 9.0, 0.5, 5.6, 1.3, "model-only extensions:\ncounters -> infer latency / bus / banking -> re-plan later layers\ndeadline / energy budget -> estimator proposes, exact model verifies,\nelse flagged fallback", COL["host"], 8.2)
    arrow(ax, (13.9, 2.4), (12.5, 1.8), "", ls="--")
    fig.savefig(os.path.join(OUT, "armse_flow.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    block_diagram()
    flow_diagram()
    print("ok")
