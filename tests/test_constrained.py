import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from armse.config import HwConfig, Workload
from armse.schedule import build_program
from armse.sim import simulate
from armse.analytic import estimate
from armse import energy
from armse.constrained import pool, measure, choose, choose_verified, est_energy, Cand
from armse.schedule import Schedule


class TestConstrained(unittest.TestCase):
    hw = HwConfig()

    def test_estimated_energy_equals_simulated_energy(self):
        rng = random.Random(2)
        wl = Workload("w", 61, 47, 83)
        for c in pool(wl, self.hw, 10, 6)[:12]:
            m = simulate(build_program(wl, self.hw, c.sch)).metrics
            self.assertAlmostEqual(c.energy, energy.level2(m, self.hw)["e_total_pj"], places=3)

    def test_choose_respects_constraints_and_flags_fallback(self):
        wl = Workload("w", 64, 64, 64)
        c = measure(wl, self.hw, pool(wl, self.hw, 20, 10))
        t_min = min(x.est_cycles for x in c)
        p, fb = choose(c, deadline=2 * t_min)
        self.assertFalse(fb)
        self.assertLessEqual(p.est_cycles, 2 * t_min)
        self.assertEqual(p.energy, min(x.energy for x in c if x.est_cycles <= 2 * t_min))       # least energy among feasible
        p, fb = choose(c, deadline=0.5 * t_min)                                                  # impossible -> known-good fallback
        self.assertTrue(fb)
        self.assertEqual(p.est_cycles, t_min)
        e_min = min(x.energy for x in c)
        p, fb = choose(c, budget=1.2 * e_min)
        self.assertFalse(fb)
        self.assertLessEqual(p.energy, 1.2 * e_min)
        p, fb = choose(c, budget=0.5 * e_min)
        self.assertTrue(fb)

    def test_margin_is_more_conservative(self):
        cs = [Cand(Schedule(1, 1, 1), 100, 5.0), Cand(Schedule(2, 2, 2), 95, 9.0), Cand(Schedule(3, 3, 3), 120, 1.0)]
        self.assertEqual(choose(cs, deadline=125, margin=0.0)[0].energy, 1.0)                    # cheapest feasible
        self.assertEqual(choose(cs, deadline=125, margin=0.10)[0].energy, 5.0)                   # 120 > 112.5 -> excluded

    def test_verified_choice_never_violates_a_feasible_deadline(self):
        for wl in (Workload("a", 64, 64, 64), Workload("b", 37, 29, 53), Workload("c", 1, 200, 120)):
            c = measure(wl, self.hw, pool(wl, self.hw, 25, 10))
            t_min = min(x.true_cycles for x in c)
            for slack in (1.0, 1.02, 1.1, 1.5):
                p, fb, n = choose_verified(c, deadline=slack * t_min)
                self.assertFalse(fb)
                self.assertLessEqual(p.true_cycles, slack * t_min)
                self.assertGreaterEqual(n, 1)
            p, fb, n = choose_verified(c, deadline=0.9 * t_min)                                   # infeasible -> verified fastest
            self.assertTrue(fb)
            self.assertEqual(p.true_cycles, t_min)

    def test_execution_is_deterministic(self):
        wl = Workload("w", 40, 40, 40)
        s = pool(wl, self.hw, 5, 3)[0].sch
        self.assertEqual(simulate(build_program(wl, self.hw, s)).cycles, simulate(build_program(wl, self.hw, s)).cycles)


if __name__ == "__main__":
    unittest.main()
