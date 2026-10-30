import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from armse.config import HwConfig, Workload, ScheduleError
from armse.depthwise import build_dw_program, run_dw, dw_reference, dw_violations, best_dw


def data(wl, seed):
    rng = np.random.default_rng(seed)
    A3 = rng.integers(-128, 128, size=(wl.M, wl.K, wl.N), dtype=np.int64).astype(np.int8)
    W = rng.integers(-128, 128, size=(wl.K, wl.N), dtype=np.int64).astype(np.int8)
    return A3, W


class TestDepthwise(unittest.TestCase):
    hw = HwConfig()

    def test_random_tiles_bit_exact_and_hazard_free(self):
        rng = random.Random(4)
        n = 0
        while n < 80:
            wl = Workload("d", rng.randint(1, 60), rng.randint(1, 40), rng.choice([9, 25, 4, 1]), "depthwise_cp")
            tm, tc = rng.randint(1, wl.M), rng.randint(1, wl.N)
            nin, nout, order = rng.choice([1, 2]), rng.choice([1, 2]), rng.choice(["MN", "NM"])
            if dw_violations(wl, self.hw, tm, tc, nin, nout):
                continue
            d = data(wl, n)
            r = run_dw(wl, self.hw, tm, tc, order, nin, nout, d)                  # hazard + deadlock checks run inside
            np.testing.assert_array_equal(r["C"], dw_reference(*d), err_msg=str((wl, tm, tc, order, nin, nout)))
            n += 1

    def test_utilisation_bound_and_beats_k_parallel_for_3x3(self):
        from armse.workloads import depthwise
        from armse.scheduler import policy_D
        from armse.schedule import build_program
        from armse.sim import simulate
        g, ch = depthwise("x", 14, 14, 64, 3)
        kp = simulate(build_program(g, self.hw, policy_D(g, self.hw))).cycles * ch
        cp = best_dw(Workload("x", g.M, 64, 9, "depthwise_cp"), self.hw)
        self.assertLessEqual(cp["metrics"]["mac_util"], 1.0)
        self.assertLess(cp["metrics"]["cycles"], kp)

    def test_rejects_wrong_family_and_oversize(self):
        with self.assertRaises(ScheduleError):
            build_dw_program(Workload("g", 8, 8, 9), self.hw, 2, 2)
        wl = Workload("d", 100, 64, 25, "depthwise_cp")
        self.assertTrue(dw_violations(wl, HwConfig(a_depth=4), 100, 64, 1, 1))


if __name__ == "__main__":
    unittest.main()
