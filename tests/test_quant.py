import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from armse.quant import requant, SETS, sample_inputs, export


def scalar(x, M, s, zp, relu):
    p = x * M
    r = (p + (1 << (s - 1))) >> s if s > 0 else p
    y = r + zp
    lo = zp if relu else -128
    return max(lo, min(127, y))


class TestQuant(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(int(requant([5], 1, 1, 0, 0)[0]), 3)            # (5+1)>>1
        self.assertEqual(int(requant([-5], 1, 1, 0, 0)[0]), -2)          # (-5+1)>>1 = -2 (floor)
        self.assertEqual(int(requant([1000], 1, 0, 0, 0)[0]), 127)       # saturate high
        self.assertEqual(int(requant([-1000], 1, 0, 0, 0)[0]), -128)     # saturate low
        self.assertEqual(int(requant([-1000], 1, 0, 0, 1)[0]), 0)        # ReLU at zero point 0
        self.assertEqual(int(requant([-1000], 1, 0, 10, 1)[0]), 10)      # ReLU clamps at the zero point
        self.assertEqual(int(requant([2 ** 31 - 1], 65535, 31, 0, 0)[0]), 127)

    def test_vectorised_equals_scalar_for_all_sets(self):
        for i, (M, s, zp, relu) in enumerate(SETS):
            x = sample_inputs(300, i)
            y = requant(x, M, s, zp, relu)
            for a, b in zip(x[:120], y[:120]):
                self.assertEqual(int(b), scalar(int(a), M, s, zp, relu))
            if relu:
                self.assertGreaterEqual(int(y.min()), max(-128, zp))

    def test_monotonic_in_input(self):
        x = np.arange(-70000, 70000, 37)
        for M, s, zp, relu in SETS:
            y = requant(x, M, s, zp, relu).astype(int)
            self.assertTrue((np.diff(y) >= 0).all())

    def test_export_files(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(export(d), len(SETS))
            p = os.path.join(d, "post_vectors", "p00")
            self.assertEqual(len(open(os.path.join(p, "in.hex")).read().split()), len(open(os.path.join(p, "exp.hex")).read().split()))
            self.assertIn("M_P", open(os.path.join(p, "meta.vh")).read())


if __name__ == "__main__":
    unittest.main()
