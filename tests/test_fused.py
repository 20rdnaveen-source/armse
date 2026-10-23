import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from armse.config import HwConfig, Workload, NoFitError
from armse.schedule import Schedule, build_program
from armse.sim import simulate, reference, random_operands
from armse.fused import *


class TestFused(unittest.TestCase):
    hw = HwConfig(a_depth=1024)

    def test_no_wb_and_out_bytes(self):
        wl = Workload("g", 13, 11, 29, out_bytes=1)
        A, B = random_operands(wl, 3)
        s = Schedule(5, 4, 8, "MN", 2, 1)
        r = simulate(build_program(wl, self.hw, s, no_wb=True), A, B, functional=True)
        np.testing.assert_array_equal(r.C, reference(A, B))             # outputs captured on chip
        self.assertEqual(r.metrics["bytes_wr"], 0)
        r2 = simulate(build_program(wl, self.hw, s))
        self.assertEqual(r2.metrics["bytes_wr"], 13 * 11)               # int8 writeback = 1 byte/element
        with self.assertRaises(Exception):
            build_program(Workload("g", 4, 4, 4, out_bytes=2), self.hw, Schedule(1, 1, 1))

    def test_resident_operand_matches_reference(self):
        wl = Workload("r", 9, 10, 32)
        A, B = random_operands(wl, 4)
        s = Schedule(3, 4, 16, "NM", 2, 2)
        r = simulate(build_program(wl, self.hw.replace(a_depth=10 ** 6), s, a_resident=True), A, B, functional=True)
        np.testing.assert_array_equal(r.C, reference(A, B))

    def test_fused_and_unfused_equal_independent_reference(self):
        for ch in (Chain("c1", 20, 24, 24, 10), Chain("c2", 17, 33, 40, 9, shift=6), Chain("c3", 1, 19, 16, 5),
                   Chain("c4", 30, 40, 64, 12, shift=10)):
            data = chain_data(ch, 5)
            ref = chain_reference(*data, ch.shift)
            u = run_unfused(ch, self.hw, data)
            np.testing.assert_array_equal(u["C2"], ref, err_msg=ch.name)
            for tm in (1, 3, 7, ch.M):
                if tm > ch.M or self.hw.a_depth - inter_words(ch, self.hw, tm) < 1:
                    continue
                f = run_fused(ch, self.hw, tm, data)
                np.testing.assert_array_equal(f["C2"], ref, err_msg=f"{ch.name} tm={tm}")

    def test_fused_eliminates_the_intermediate_write(self):
        # Exact property: the M*N1 int8 intermediate is never written to external memory when fused.
        # (Total traffic is NOT guaranteed lower: both searches minimise cycles and may re-stream weights.)
        ch = Chain("one", 16, 32, 64, 32)
        hw = HwConfig(a_depth=2048)
        f = run_fused(ch, hw, 16)
        u = run_unfused(ch, hw)
        self.assertEqual(f["bytes_wr"], 4 * ch.M * ch.N2)
        self.assertEqual(u["bytes_wr"] - f["bytes_wr"], ch.M * ch.N1)

    def test_no_fit(self):
        ch = Chain("big", 64, 16, 4096, 16)
        with self.assertRaises(NoFitError):
            run_fused(ch, HwConfig(a_depth=64), 8)

    def test_requant_semantics(self):
        x = np.array([-100000, -256, -1, 0, 255, 256, 100000], np.int32)
        np.testing.assert_array_equal(requant(x, 8), np.array([-128, -1, -1, 0, 0, 1, 127], np.int8))


if __name__ == "__main__":
    unittest.main()
