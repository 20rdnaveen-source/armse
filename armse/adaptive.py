"""Counter-driven runtime adaptation (mechanism E) - MODEL ONLY.

The scheduler plans with a hardware MODEL (hw_plan). The real hardware (hw_true) may differ: burst latency, effective
bus width, or banking behaviour. After each layer the controller reads the counters the RTL exposes (dma_busy, cmp_busy,
n_xfer, bytes) plus facts the host already knows from the descriptors (rows, jobs, beats), infers the true parameters,
and re-plans the NEXT layers with the corrected model. No simulator is used to plan; the same raw estimator is used by
the static, adaptive and oracle modes so the comparison isolates the value of knowing the hardware parameters.
Inference (exact in this model because the timing formulas are exact):
  dma_busy = job_ovh*jobs + row_fix*rows + beats(bus)   ->  row_fix per candidate bus; the bus whose row_fix is
  consistent across >= 2 executed layers is chosen (single layer: nominal bus assumed)
  cmp_busy / expected_skew_busy > 1.02  ->  banking is behaving like the naive layout"""
import time
from dataclasses import dataclass, field
from .config import HwConfig, Workload, cdiv
from .schedule import build_program
from .sim import simulate, cmp_issue
from .scheduler import policy_D


def counters(prog, hw_true: HwConfig):
    """What the RTL counters would report for this program on the TRUE hardware."""
    m = simulate(build_program(prog.wl, hw_true, prog.sch), check=False).metrics
    return dict(dma_busy=m["dma_busy"], cmp_busy=m["compute_busy"], n_xfer=m["transfers"],
                bytes=m["bytes_rd"] + m["bytes_wr"], cycles=m["cycles"])


def _descr(prog, bus):
    R = sum(op.rows for op in prog.dma)
    nj = len(prog.dma)
    B = sum(op.rows * cdiv(op.cols * op.ebytes, bus) for op in prog.dma)
    return R, nj, B


def infer_hw(history, hw_nom: HwConfig) -> HwConfig:
    """history: list of (program-built-on-plan-hw, counters). Returns the corrected planning model."""
    if not history:
        return hw_nom
    best = None
    for bus in (4, 8, 16, 32, 64):
        rfs = []
        for prog, c in history:
            R, nj, B = _descr(prog, bus)
            rfs.append((c["dma_busy"] - hw_nom.job_ovh * nj - B) / R)
        spread = max(rfs) - min(rfs)
        if len(history) == 1 and bus != hw_nom.bus_bytes:
            continue                                         # one layer cannot separate latency from bus width
        key = (round(spread, 6), abs(bus - hw_nom.bus_bytes))
        if best is None or key < best[0]:
            best = (key, bus, sum(rfs) / len(rfs))
    if best is None or best[0][0] > 0.5 or best[2] < 1:
        bus, rf = hw_nom.bus_bytes, hw_nom.row_fix
    else:
        _, bus, rf = best
    expected = sum(cmp_issue(hw_nom.replace(bank_mode="skew"), c) + hw_nom.cmp_ovh for p, _ in history for c in p.cmp)
    observed = sum(c["cmp_busy"] for _, c in history)
    mode = "naive" if observed > 1.02 * expected else hw_nom.bank_mode
    return hw_nom.replace(bus_bytes=bus, row_lat=max(0, int(round(rf)) - 1), bank_mode=mode)


@dataclass
class SeqResult:
    cycles: int = 0
    per_layer: list = field(default_factory=list)
    replans: int = 0
    plan_seconds: float = 0.0
    final_model: HwConfig = None


def run_sequence(layers, hw_nom: HwConfig, hw_true: HwConfig, mode: str) -> SeqResult:
    """mode: 'static' (plan every layer with hw_nom), 'oracle' (plan with hw_true), 'adaptive' (re-plan from counters)."""
    res, hist, model = SeqResult(), [], hw_nom
    for wl in layers:
        t = time.perf_counter()
        plan_hw = {"static": hw_nom, "oracle": hw_true, "adaptive": model}[mode]
        sch = policy_D(wl, plan_hw)
        res.plan_seconds += time.perf_counter() - t
        prog = build_program(wl, plan_hw, sch)
        c = counters(prog, hw_true)
        res.cycles += c["cycles"]
        res.per_layer.append((wl.name, sch.key(), c["cycles"]))
        if mode == "adaptive":
            hist.append((prog, c))
            new = infer_hw(hist, hw_nom)
            if (new.row_lat, new.bus_bytes, new.bank_mode) != (model.row_lat, model.bus_bytes, model.bank_mode):
                res.replans += 1
            model = new
    res.final_model = model
    return res
