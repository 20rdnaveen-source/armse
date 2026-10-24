import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from armse.config import HwConfig
from armse.fused import Chain, chain_data, chain_reference
from armse.fused_overlap import run_overlap, _segments, build_chain_program
from armse.sim import schedule_times, check_hazards


class TestOverlap(unittest.TestCase):
    hw = HwConfig(a_depth=1024)

    def test_bit_exact_vs_reference_and_hazard_free(self):
        for ch in (Chain("o1", 20, 24, 24, 10), Chain("o2", 17, 33, 40, 9, shift=6), Chain("o3", 1, 19, 16, 5),
                   Chain("o4", 30, 40, 64, 12, shift=10)):
            data = chain_data(ch, 7)
            ref = chain_reference(*data, ch.shift)
            for tm in (1, 4, 7, ch.M):
                if tm > ch.M:
                    continue
                for nin, nout in ((1, 1), (2, 1), (2, 2), (1, 2)):
                    try:
                        r = run_overlap(ch, self.hw, tm, nin, nout, data)   # hazard check runs inside
                    except Exception as e:                                   # only no-fit is acceptable
                        self.assertEqual(type(e).__name__, "NoFitError", (ch.name, tm, nin, nout, e))
                        continue
                    np.testing.assert_array_equal(r["C2"], ref, err_msg=f"{ch.name} tm={tm} nbuf={nin}/{nout}")

    def test_overlap_not_slower_than_barrier_for_same_schedules(self):
        # same segments, run (a) as one overlapped program, (b) as independent barrier programs
        from armse.schedule import build_program
        from armse.sim import simulate
        ch = Chain("cmp", 32, 32, 64, 32)
        segs = _segments(ch, self.hw, 8, 2, 2)
        prog = build_chain_program(segs, self.hw)
        over = max(schedule_times(prog)[1][-1], schedule_times(prog)[3][-1])
        sep = sum(simulate(build_program(s.wl, s.hw, s.sch, a_resident=s.a_resident, no_wb=s.no_wb)).cycles for s in segs)
        self.assertLessEqual(over, sep)

    def test_mixed_nbuf_rejected(self):
        ch = Chain("m", 16, 16, 16, 8)
        segs = _segments(ch, self.hw, 8, 2, 2)
        segs[1].sch = type(segs[1].sch)(segs[1].sch.tm, segs[1].sch.tn, segs[1].sch.tk, "MN", 1, 1)
        with self.assertRaises(ValueError):
            build_chain_program(segs, self.hw)


if __name__ == "__main__":
    unittest.main()
