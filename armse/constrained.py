"""Resource-bounded planning (mechanism F) - MODEL ONLY.

Hard constraints (checked before anything is optimised): buffer capacity / field limits (schedule.violations).
Soft-hard constraints handled here: latency DEADLINE (cycles) and ENERGY budget (Level-2 analytical energy, illustrative
constants - NOT measured). The controller picks among feasible candidates by the closed-form estimator; if nothing is feasible
it returns the FASTEST known schedule and flags `fallback=True` (the deadline/budget cannot be guaranteed).
Determinism: programs are static, so cycles are repeatable; with the exact model the simulated cycles of a chosen program are
its worst case. The estimator (not the simulator) is used to choose, so `margin` guards against its error (<= ~15 %)."""
from dataclasses import dataclass
from typing import List, Optional
from .config import HwConfig, Workload
from .schedule import Schedule, build_program
from .sim import simulate
from .analytic import estimate
from .scheduler import candidates, rank
from . import energy


def est_energy(wl: Workload, hw: HwConfig, e: dict, dram_scale: float = 1.0) -> float:
    return energy.level2(dict(bytes_rd=e["bytes_rd"], bytes_wr=e["bytes_wr"], macs=wl.macs), hw, dram_scale=dram_scale)["e_total_pj"]


@dataclass
class Cand:
    sch: Schedule
    est_cycles: int
    energy: float
    true_cycles: Optional[int] = None


def pool(wl: Workload, hw: HwConfig, ncycle=60, nbytes=30, max_iters=20000) -> List[Cand]:
    """Candidate pool: best by estimated cycles plus best by traffic (so energy-lean schedules are represented)."""
    r = rank(wl, hw, candidates(wl, hw))
    pick = {s.key(): s for _, s in r[:ncycle]}
    by_bytes = sorted(r, key=lambda x: (x[0][1], x[0][0]))
    for _, s in by_bytes:
        if len(pick) >= ncycle + nbytes:
            break
        if estimate(wl, hw, s)["iters"] <= max_iters:
            pick.setdefault(s.key(), s)
    out = []
    for s in pick.values():
        e = estimate(wl, hw, s)
        out.append(Cand(s, e["cycles"], est_energy(wl, hw, e)))
    return out


def measure(wl: Workload, hw: HwConfig, cands: List[Cand]):
    for c in cands:
        c.true_cycles = simulate(build_program(wl, hw, c.sch)).cycles
    return cands


def choose(cands: List[Cand], deadline: Optional[float] = None, budget: Optional[float] = None, margin: float = 0.0):
    """Returns (candidate, fallback_flag). Selection uses ESTIMATED cycles with a safety margin on the deadline."""
    feas = [c for c in cands if (deadline is None or c.est_cycles <= deadline * (1 - margin))
            and (budget is None or c.energy <= budget)]
    key_min_energy = lambda c: (c.energy, c.est_cycles)
    key_min_cycles = lambda c: (c.est_cycles, c.energy)
    if feas:
        if deadline is not None and budget is None:
            return min(feas, key=key_min_energy), False          # meet the deadline with the least energy
        return min(feas, key=key_min_cycles), False              # budget (or both): fastest feasible
    return min(cands, key=key_min_cycles), True                  # known-good fallback: fastest schedule, constraint NOT guaranteed


def choose_verified(cands: List[Cand], deadline: Optional[float] = None, budget: Optional[float] = None):
    """Estimator proposes, the exact simulator VERIFIES (cands must carry true_cycles). Candidates are tried in objective order
    (estimator-feasible ones first); the first whose SIMULATED cycles meet the deadline wins. If none does, the schedule with the
    fewest simulated cycles is the known-good fallback (flag True: the deadline cannot be met). Returns (cand, fallback, n_checks).
    In this model the simulator is exact, so a verified choice cannot violate the deadline unless it is infeasible."""
    ok_budget = [c for c in cands if budget is None or c.energy <= budget]
    if not ok_budget:
        return min(cands, key=lambda c: (c.true_cycles, c.energy)), True, 0
    obj = (lambda c: (c.energy, c.est_cycles)) if (deadline is not None and budget is None) else (lambda c: (c.est_cycles, c.energy))
    est_ok = [c for c in ok_budget if deadline is None or c.est_cycles <= deadline]
    order = sorted(est_ok, key=obj) + sorted([c for c in ok_budget if c not in est_ok], key=lambda c: c.est_cycles)
    for n, c in enumerate(order, 1):
        if deadline is None or c.true_cycles <= deadline:
            return c, False, n
    return min(ok_budget, key=lambda c: (c.true_cycles, c.energy)), True, len(order)
