"""Network planner: JSON network description -> GEMM layers -> ARMSE schedules -> cycles / bytes / descriptors.

python -m armse.netplan examples/digit_mlp.json [--policy auto] [--export out_dir] [--report plan.md] [--clock-mhz 100]

Supported layers: fc (in, out, optional batch), conv2d (H, W, cin, cout, k, stride, pad), attention (seq, d: QK^T and PV GEMMs).
Activations, pooling, softmax, normalisation are NOT modelled (they are not memory-bound GEMMs); depthwise conv is not
planned here (see armse/depthwise.py). All cycle numbers are SIMULATED on the hardware model you pass; clock conversion is an
assumption, not a measurement."""
import argparse
import json
import os
import time
import numpy as np
from .config import HwConfig, Workload, NoFitError
from .workloads import conv2d
from .scheduler import policy_C, policy_C_best_square, policy_D, policy_D_fb, policy_D_v2
from .schedule import build_program
from .sim import simulate, reference, random_operands
from . import rtlexport as rx


def lower(spec):
    """Returns a list of (layer_name, Workload)."""
    out, batch = [], int(spec.get("batch", 1))
    for i, L in enumerate(spec["layers"]):
        name, t = L.get("name", f"L{i}"), L["type"]
        if t == "fc":
            out.append((name, Workload(name, int(L.get("batch", batch)), int(L["out"]), int(L["in"]), "fc")))
        elif t == "conv2d":
            w = conv2d(name, L["H"], L["W"], L["cin"], L["cout"], L["k"], L.get("stride", 1), L.get("pad", 0))
            out.append((name, Workload(name, w.M, w.N, w.K, "conv")))
        elif t == "attention":
            s, d = int(L["seq"]), int(L["d"])
            out.append((name + "_QKt", Workload(name + "_QKt", s, s, d, "attention")))
            out.append((name + "_PV", Workload(name + "_PV", s, d, s, "attention")))
        else:
            raise ValueError(f"unsupported layer type '{t}' (supported: fc, conv2d, attention)")
    return out


def choose(wl, hw, policy):
    if policy == "D_v2" or (policy == "auto"):
        return policy_D_v2(wl, hw), "D_v2" if policy_D_v2(wl, hw).label == "D_v2" and _calibrated(hw) else "D (raw estimator; hw not calibrated)"
    if policy == "D_fb":
        return policy_D_fb(wl, hw), "D_feedback"
    return policy_D(wl, hw), "D (raw estimator)"


def _calibrated(hw):
    cfg = json.load(open(os.path.join(os.path.dirname(__file__), "est_v2_coef.json")))["hw"]
    return (hw.lanes, hw.bus_bytes, hw.row_lat, hw.a_depth, hw.b_depth, hw.o_words) == (
        cfg["lanes"], cfg["bus"], cfg["row_lat"], cfg["a_depth"], cfg["b_depth"], cfg["o_words"])


def plan(spec, hw, policy="auto", verify=True, export=None, clock_mhz=None):
    rows, t0 = [], time.perf_counter()
    for name, wl in lower(spec):
        row = dict(layer=name, M=wl.M, N=wl.N, K=wl.K)
        try:
            s, used = choose(wl, hw, policy)
        except NoFitError as e:
            rows.append(dict(row, error=str(e)))
            continue
        m = simulate(build_program(wl, hw, s)).metrics
        cd = simulate(build_program(wl, hw, policy_C(wl, hw))).metrics
        cb = simulate(build_program(wl, hw, policy_C_best_square(wl, hw))).metrics
        ai = 2 * wl.macs / (wl.compulsory_in + wl.compulsory_out)
        row.update(policy=used, tile=f"{s.tm}x{s.tn}x{s.tk}", order=s.order, nbuf=f"{s.nbuf_in}/{s.nbuf_out}",
                   cycles=m["cycles"], fixed_double=cd["cycles"], best_square=cb["cycles"],
                   vs_fixed_double=m["cycles"] / cd["cycles"], vs_best_square=m["cycles"] / cb["cycles"],
                   bytes_rd=m["bytes_rd"], bytes_wr=m["bytes_wr"], mac_util=m["mac_util"],
                   bound="memory-bound" if ai <= 2 * hw.lanes / hw.bus_bytes else "compute-bound",
                   stall_frac=m["mem_stall"] / m["cycles"], program_bytes=m["program_bytes"])
        if verify and wl.macs <= 5_000_000:
            A, B = random_operands(wl, 3)
            r = simulate(build_program(wl, hw, s), A, B, functional=True)
            row["verified"] = bool((r.C == reference(A, B)).all())
        if export:
            prog = build_program(wl, hw, s)
            d = os.path.join(export, name)
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "dprog.hex"), "w").write("\n".join(f"{rx.pack_dma(o):032x}" for o in prog.dma) + "\n")
            open(os.path.join(d, "cprog.hex"), "w").write("\n".join(f"{rx.pack_cmp(o):032x}" for o in prog.cmp) + "\n")
            row["exported"] = d
        rows.append(row)
    tot = lambda k: sum(r.get(k, 0) for r in rows if "error" not in r)
    summary = dict(total_cycles=tot("cycles"), fixed_double=tot("fixed_double"), best_square=tot("best_square"),
                   plan_seconds=time.perf_counter() - t0)
    if clock_mhz:
        summary["assumed_latency_us"] = summary["total_cycles"] / clock_mhz
    return rows, summary


def to_markdown(spec, hw, rows, summary):
    L = [f"# Plan for `{spec.get('name', 'network')}`", "",
         f"Hardware model: lanes={hw.lanes}, bus={hw.bus_bytes} B/cycle, burst latency={hw.row_lat}, A/B buffers {hw.lanes * hw.a_depth}/{hw.lanes * hw.b_depth} B, "
         f"output {4 * hw.o_words} B. All numbers are SIMULATED cycles (not measured).", "",
         "| layer | MxNxK | bound | tile | order | buf | cycles | vs fixed double-buffering | vs best fixed tiling | stall | verified |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            L.append(f"| {r['layer']} | {r['M']}x{r['N']}x{r['K']} | - | **no legal schedule**: {r['error']} | | | | | | | |")
            continue
        L.append(f"| {r['layer']} | {r['M']}x{r['N']}x{r['K']} | {r['bound']} | {r['tile']} | {r['order']} | {r['nbuf']} | {r['cycles']} | "
                 f"{r['vs_fixed_double']:.3f}x | {r['vs_best_square']:.3f}x | {100 * r['stall_frac']:.0f}% | {r.get('verified', '-')} |")
    s = summary
    L += ["", f"**Total:** {s['total_cycles']} cycles; fixed double-buffering {s['fixed_double']} ({s['total_cycles'] / max(1, s['fixed_double']):.3f}x), "
          f"best fixed tiling {s['best_square']} ({s['total_cycles'] / max(1, s['best_square']):.3f}x). Planning time {s['plan_seconds']:.2f} s (host).",
          "Layers marked memory-bound benefit most; compute-bound layers have little memory time to remove."]
    if "assumed_latency_us" in s:
        L.append(f"At an ASSUMED clock this is {s['assumed_latency_us']:.1f} us per inference (cycles / MHz; not measured).")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("spec")
    ap.add_argument("--policy", default="auto", choices=["auto", "D_v2", "D", "D_fb"])
    ap.add_argument("--lanes", type=int, default=8)
    ap.add_argument("--bus", type=int, default=4)
    ap.add_argument("--row-lat", type=int, default=12)
    ap.add_argument("--a-depth", type=int, default=256)
    ap.add_argument("--b-depth", type=int, default=256)
    ap.add_argument("--o-words", type=int, default=1024)
    ap.add_argument("--export")
    ap.add_argument("--report")
    ap.add_argument("--clock-mhz", type=float)
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args(argv)
    spec = json.load(open(a.spec))
    hw = HwConfig(lanes=a.lanes, bus_bytes=a.bus, row_lat=a.row_lat, a_depth=a.a_depth, b_depth=a.b_depth, o_words=a.o_words)
    rows, summary = plan(spec, hw, a.policy, not a.no_verify, a.export, a.clock_mhz)
    md = to_markdown(spec, hw, rows, summary)
    print(md)
    if a.report:
        open(a.report, "w").write(md)
    return rows, summary


if __name__ == "__main__":
    main()
