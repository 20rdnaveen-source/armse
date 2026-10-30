"""Draws docs/img/armse_final_state.png: the final state of the idea (what exists, at which evidence level)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "img")
os.makedirs(OUT, exist_ok=True)
G, B, Y, P, E, V = "#bbf7d0", "#bfdbfe", "#fde68a", "#fbcfe8", "#e5e7eb", "#ddd6fe"
EV = {"rtl": ("#16a34a", "RTL simulated"), "model": ("#2563eb", "model-verified"), "future": ("#9ca3af", "future / needs hardware")}


def box(ax, x, y, w, h, text, c, fs=8.5, bold=False, ec="#374151"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=c, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def tag(ax, x, y, kind):
    ax.add_patch(Circle((x, y), 0.14, fc=EV[kind][0], ec="white", lw=1.2, zorder=5))


def arrow(ax, p, q, both=False, color="#111827"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>" if both else "-|>", mutation_scale=13, lw=1.5, color=color))


fig, ax = plt.subplots(figsize=(18, 11.5))
ax.set_xlim(0, 18)
ax.set_ylim(0, 11.5)
ax.axis("off")
ax.text(9, 11.15, "ARMSE  -  Adaptive Reuse and Memory Scheduling Engine", ha="center", fontsize=19, fontweight="bold")
ax.text(9, 10.72, "FINAL STATE OF THE IDEA: a verified simulation-level prototype of a small-accelerator memory system  (not yet measured on a board)", ha="center", fontsize=10.5, style="italic")

# ---------------- memory system (left) ----------------
box(ax, 0.3, 8.5, 10.9, 1.5, "HOST SOFTWARE:  adaptive scheduler  -  chooses tile shape x loop order x single/double buffering per layer\n"
    "calibrated cost model  -  simulator-verified legality (no deadlock, no overwritten data)  -  emits descriptor programs", B, 9, True)
tag(ax, 0.55, 9.75, "model")
box(ax, 2.9, 7.5, 8.3, 0.75, "two completed-job counters (dma_done / cp_done):\neach engine starts a job only when the other has finished its dependency", V, 8.2)
tag(ax, 3.15, 8.0, "rtl")
box(ax, 0.3, 2.9, 1.9, 4.2, "EXTERNAL\nMEMORY\n\nDDR / AXI\nlater\n\n(testbench\nmodel today)", E, 8.5, True)
tag(ax, 0.55, 6.85, "future")
box(ax, 2.9, 3.9, 2.4, 3.0, "DMA ENGINE\n\nburst reads / writes\nbank + address\ngeneration\n4 elements per beat", G, 8.5, True)
tag(ax, 3.15, 6.65, "rtl")
box(ax, 6.0, 5.2, 2.5, 1.7, "A BUFFER\nskewed banks\nbank = (row + col) mod NB", Y, 8.3, True)
box(ax, 6.0, 3.9, 2.5, 1.0, "B BUFFER (transposed)\nsame skewed mapping", Y, 8.3, True)
tag(ax, 6.25, 6.65, "rtl")
tag(ax, 8.35, 4.75, "rtl")
box(ax, 9.1, 3.9, 2.1, 3.0, "MAC LANES\n\nNB parallel\n8x8 multiplies\n+ adder tree\n\nINT8 -> INT32", G, 8.5, True)
tag(ax, 9.35, 6.65, "rtl")
box(ax, 6.0, 2.5, 5.2, 1.0, "OUTPUT / PARTIAL-SUM BUFFER (32-bit)", Y, 8.5, True)
tag(ax, 6.25, 3.25, "rtl")
box(ax, 0.3, 0.9, 10.9, 1.0, "PERFORMANCE COUNTERS\ncycles  -  compute / DMA busy and wait  -  bytes read / written  -  transfers  -  bank conflicts (must be 0)", P, 8.0, True)
tag(ax, 0.55, 1.65, "rtl")
arrow(ax, (2.2, 5.4), (2.9, 5.4), both=True)
arrow(ax, (5.3, 6.2), (6.0, 6.2))
arrow(ax, (5.3, 4.4), (6.0, 4.4))
arrow(ax, (8.5, 6.0), (9.1, 6.0))
arrow(ax, (8.5, 4.4), (9.1, 4.8))
arrow(ax, (10.1, 3.9), (10.1, 3.5), both=True)
arrow(ax, (6.0, 3.0), (4.1, 3.0))
arrow(ax, (4.1, 3.0), (4.1, 3.9))
ax.text(7.6, 7.75, "", fontsize=1)
# model-only add-ons
ax.text(0.45, 2.3, "model-only add-ons (no RTL):  counter-driven re-planning  |  deadline / energy-bounded planning  |", fontsize=7.8, color="#2563eb", fontweight="bold", va="center")
ax.text(0.45, 2.05, "tensor-lifetime planner  |  fusion  |  INT4 weights  |  channel-parallel depthwise mapping", fontsize=7.8, color="#2563eb", fontweight="bold", va="center")
tag(ax, 0.2, 2.2, "model")

# ---------------- right panels ----------------
def panel(x, y, w, h, title, lines, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=c, ec="#374151", lw=1.3))
    ax.text(x + 0.2, y + h - 0.3, title, fontsize=11, fontweight="bold", va="center")
    for i, l in enumerate(lines):
        ax.text(x + 0.2, y + h - 0.72 - i * 0.37, l, fontsize=8.6, va="center")

panel(11.6, 7.15, 6.1, 3.0, "WHAT IT SOLVES", [
    "- accelerator idle while waiting for slow memory",
    "- tiles / buffering that do not fit each layer",
    "- bank conflicts on column reads (up to ~8x)",
    "- exposed transfer latency (overlapped)",
    "- safe, checkable schedules (no deadlock)",
    "- deterministic cycle counts (planner, model)"], "#ecfccb")
panel(11.6, 3.95, 6.1, 3.0, "WHERE IT APPLIES  (intended use)", [
    "- small INT8 inference accelerators on FPGA / ASIC",
    "- always-on, low-latency edge tasks: digit / keyword /",
    "  small CNN, MLP layers at batch 1, small attention",
    "- memory-bound layers (GEMV, batch-1 FC)",
    "- designs with a few KB of on-chip SRAM / BRAM",
    "- not yet run on any real device"], "#fef9c3")
panel(11.6, 0.9, 6.1, 2.85, "WHY USE IT", [
    "- fixed tiling wastes cycles on layers it was not tuned for",
    "- gain comes from per-layer tile shape + overlap",
    "- hardware stays small: two counters, no cache logic",
    "- every plan is proven legal before it runs",
    "- all results reproducible (62 tests, bit-exact checks)"], "#fee2e2")

# ---------------- key numbers ----------------
nums = [("0.73x", "cycles vs fixed\ndouble buffering"), ("0.87x", "cycles vs best\nfixed tiling"), ("~8x", "vs a naive\nbank layout"), ("7 / 7", "RTL cases pass in\nIcarus (your run)")]
for i, (big, small) in enumerate(nums):
    x = 0.3 + i * 2.75
    box(ax, x, 0.05, 2.6, 0.8, "", "#f3f4f6")
    ax.text(x + 0.15, 0.45, big, fontsize=13, fontweight="bold", va="center", color="#111827")
    ax.text(x + 1.3, 0.45, small, fontsize=7.2, va="center")
ax.text(11.6, 0.55, "Numbers: 30 INT8 GEMM workloads, one default hardware\nmodel, simulation. Gain is largest on memory-bound layers;\n25 of 30 workloads are compute-bound.", fontsize=7.6, va="center", style="italic")

# ---------------- legend ----------------
ax.text(0.3, 10.35 - 0.0, "", fontsize=1)
for i, (k, (c, t)) in enumerate(EV.items()):
    ax.add_patch(Circle((11.75 + i * 2.1, 10.45), 0.13, fc=c, ec="white"))
    ax.text(11.95 + i * 2.1, 10.45, t, fontsize=7.8, va="center")
fig.savefig(os.path.join(OUT, "armse_final_state.png"), dpi=130, bbox_inches="tight")
print("ok")
