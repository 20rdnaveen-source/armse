import os
import random
import sys
import unittest
 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from armse import memplan
from armse.config import HwConfig, Workload
from armse.schedule import Schedule, build_program, violations
from armse.sim import simulate, reference, random_operands
from armse.workloads import kv_decode, depthwise
from armse.edgestudy import clip, pow2_ge
from armse.scheduler import policy_D
 
 
class TestMemplan(unittest.TestCase):
    def test_random_graphs_valid_and_bounded(self):
        rng = random.Random(3)
        for _ in range(300):
            ts = []
            for i in range(rng.randint(0, 25)):
                b = rng.randint(0, 20)
                ts.append(memplan.Tensor(f"t{i}", rng.randint(1, 100), b, b + rng.randint(1, 8)))
            for align in (1, 4, 16):
                off, arena = memplan.plan(ts, align)
                self.assertTrue(memplan.verify(ts, off, align))
                self.assertGreaterEqual(arena, memplan.peak_live(ts))          # peak live is a lower bound (align=1 tight)
                if align == 1:
                    self.assertLessEqual(arena, memplan.naive_total(ts))
 
    def test_verify_catches_overlap_and_known_cases(self):
        a, b = memplan.Tensor("a", 10, 0, 5), memplan.Tensor("b", 10, 3, 8)
        self.assertFalse(memplan.verify([a, b], {"a": 0, "b": 5}))             # b starts inside a: overlap
        self.assertTrue(memplan.verify([a, b], {"a": 0, "b": 10}))
        c = memplan.Tensor("c", 10, 5, 9)                                       # disjoint lifetime with a: may share bytes
        off, arena = memplan.plan([a, c])
        self.assertEqual(arena, 10)
        self.assertEqual(memplan.peak_live([]), 0)
        self.assertEqual(memplan.plan([]), ({}, 0))
 
    def test_inplace_never_increases_peak(self):
        for s, d in ((32, 32), (64, 64), (128, 64)):
            a = memplan.peak_live(memplan.transformer_block(s, d))
            b = memplan.peak_live(memplan.transformer_block(s, d, inplace=True))
            self.assertLessEqual(b, a)
 
 
class TestEdgeWorkloads(unittest.TestCase):
    hw = HwConfig()
 
    def test_kv_decode_shapes_and_correctness(self):
        qk, pv = kv_decode("k", 100, 64)
        self.assertEqual((qk.M, qk.N, qk.K, pv.M, pv.N, pv.K), (1, 100, 64, 1, 64, 100))
        for w in (qk, pv):
            A, B = random_operands(w, 1)
            r = simulate(build_program(w, self.hw, policy_D(w, self.hw)), A, B, functional=True)
            np.testing.assert_array_equal(r.C, reference(A, B))
 
    def test_depthwise_lowering_and_low_utilisation(self):
        w, c = depthwise("d", 28, 28, 64, 3, 1, 1)
        self.assertEqual((w.M, w.N, w.K, c), (784, 1, 9, 64))
        m = simulate(build_program(w, self.hw, policy_D(w, self.hw))).metrics
        self.assertLess(m["mac_util"], 0.5)
 
    def test_clipped_static_schedule_is_legal_and_correct(self):
        big = kv_decode("b", 1024, 64)[0]
        s = policy_D(big, self.hw)
        for L in (1, 7, 33, 200, 1024):
            w = kv_decode("x", L, 64)[0]
            c = clip(w, s)
            self.assertEqual(violations(w, self.hw, c), [])
            A, B = random_operands(w, L)
            r = simulate(build_program(w, self.hw, c), A, B, functional=True)
            np.testing.assert_array_equal(r.C, reference(A, B))
        self.assertEqual((pow2_ge(1), pow2_ge(33), pow2_ge(64), pow2_ge(65)), (1, 64, 64, 128))
 
 
if __name__ == "__main__":
    unittest.main()
 
 
class TestCompression(unittest.TestCase):
    def test_int4_weights_reduce_bytes_and_keep_results(self):
        from dataclasses import replace
        hw = HwConfig()
        wl = Workload("c", 20, 40, 64)
        s = Schedule(10, 20, 32, "MN", 2, 1)
        m8 = simulate(build_program(wl, hw, s)).metrics
        wl4 = replace(wl, weight_bits=4)
        A, B = random_operands(wl, 5)
        B = np.clip(B, -8, 7).astype(np.int8)                      # INT4-representable weights
        r4 = simulate(build_program(wl4, hw, s), A, B, functional=True)
        np.testing.assert_array_equal(r4.C, reference(A, B))        # numerics identical after unpack
        self.assertLess(r4.metrics["bytes_rd"], m8["bytes_rd"])
        self.assertEqual(r4.metrics["bytes_wr"], m8["bytes_wr"])
        with self.assertRaises(Exception):
            build_program(replace(wl, weight_bits=3), hw, s)
        self.assertEqual(simulate(build_program(replace(wl, weight_bits=8), hw, s)).metrics, m8)   # default unchanged
 
 
class TestEstimatorV2(unittest.TestCase):
    def test_v2_policy_legal_correct_and_falls_back(self):
        from armse.scheduler import policy_D_v2, policy_D
        hw = HwConfig()
        wl = Workload("v", 37, 29, 53)
        s = policy_D_v2(wl, hw)
        self.assertEqual(violations(wl, hw, s), [])
        A, B = random_operands(wl, 9)
        r = simulate(build_program(wl, hw, s), A, B, functional=True)
        np.testing.assert_array_equal(r.C, reference(A, B))
        self.assertEqual(policy_D_v2(wl, hw).key(), s.key())                      # deterministic
        other = hw.replace(row_lat=5)                                              # uncalibrated hardware -> v1 behaviour
        self.assertEqual(policy_D_v2(wl, other).key(), policy_D(wl, other).key())
        self.assertEqual(policy_D_v2(Workload("e", 0, 4, 4), hw).label, "D_v2")   # empty workload handled
 
    def test_prediction_is_clamped(self):
        from armse.estfit import predict, CLAMP
        e = dict(cycles=1000, t_dma=500, t_cmp=700, iters=10)
        self.assertEqual(predict(e, 2, 2, [1e9] * 10), CLAMP[1] * 1000)
        self.assertEqual(predict(e, 2, 2, [-1e9] * 10), CLAMP[0] * 1000)
 
 
class TestForcedTransfer(unittest.TestCase):
    def test_forced_v2_on_other_hardware_is_legal(self):
        from armse.scheduler import policy_D_v2
        hw = HwConfig(lanes=16, row_lat=24)
        wl = Workload("t", 50, 40, 70)
        s = policy_D_v2(wl, hw, force=True)
        self.assertEqual(violations(wl, hw, s), [])
        A, B = random_operands(wl, 3)
        np.testing.assert_array_equal(simulate(build_program(wl, hw, s), A, B, functional=True).C, reference(A, B))
 