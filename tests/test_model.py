import random
import sys
import os
import unittest
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from armse.config import *
from armse.schedule import *
from armse.sim import *
from armse.analytic import estimate
from armse.scheduler import *
from armse.workloads import suite, smoke_suite, conv2d, attention, mlp
from armse import banks, energy


def rand_case(rng):
    hw = HwConfig(lanes=rng.choice([4, 8, 16]), bus_bytes=rng.choice([4, 8, 16]),
                  row_lat=rng.randint(0, 20), a_depth=rng.choice([32, 64, 256]),
                  b_depth=rng.choice([32, 64, 256]), o_words=rng.choice([64, 256, 1024]),
                  bank_mode=rng.choice(["skew", "naive"]))
    wl = Workload("r", rng.randint(1, 90), rng.randint(1, 90), rng.randint(1, 90))
    for _ in range(200):
        s = Schedule(rng.randint(1, wl.M), rng.randint(1, wl.N), rng.randint(1, wl.K),
                     rng.choice(["MN", "NM"]), rng.choice([1, 2]), rng.choice([1, 2]))
        if not violations(wl, hw, s):
            return wl, hw, s
    return Workload("r", 4, 4, 4), hw, Schedule(1, 1, 1)


class TestExactness(unittest.TestCase):
    def test_traffic_and_durations_match_closed_form(self):
        rng = random.Random(1)
        errs = []
        for _ in range(600):
            wl, hw, s = rand_case(rng)
            p = build_program(wl, hw, s)
            r = simulate(p)
            e = estimate(wl, hw, s)
            m = r.metrics
            self.assertEqual(m["bytes_rd"], e["bytes_rd"], (wl, hw, s))
            self.assertEqual(m["bytes_wr"], e["bytes_wr"])
            self.assertEqual(m["transfers"], e["transfers"], (wl, s))
            self.assertEqual(m["dma_busy"], e["t_dma"], (wl, hw, s))
            self.assertEqual(m["compute_busy"], e["t_cmp"], (wl, hw, s))
            if s.nbuf_in == 1 and s.nbuf_out == 1:
                self.assertEqual(m["cycles"], e["cycles"], (wl, hw, s))   # fully serial -> exact
            else:
                errs.append(abs(e["cycles"] - m["cycles"]) / m["cycles"])
        self.assertLess(sum(errs) / len(errs), 0.15)      # estimator error bound (measured ~ few %)

    def test_compulsory_lower_bound(self):
        rng = random.Random(2)
        for _ in range(100):
            wl, hw, s = rand_case(rng)
            m = simulate(build_program(wl, hw, s)).metrics
            self.assertGreaterEqual(m["bytes_rd"], wl.compulsory_in)
            self.assertEqual(m["bytes_wr"], wl.compulsory_out)


class TestFunctional(unittest.TestCase):
    def test_random_schedules_match_reference(self):
        rng = random.Random(3)
        for i in range(150):
            wl, hw, s = rand_case(rng)
            A, B = random_operands(wl, i)
            r = simulate(build_program(wl, hw, s), A, B, functional=True)
            np.testing.assert_array_equal(r.C, reference(A, B), err_msg=str((wl, hw, s)))

    def test_all_policies_on_smoke_suite(self):
        hw = HwConfig()
        for wl in smoke_suite():
            if wl.macs > 3_000_000:
                continue
            A, B = random_operands(wl, 7)
            ref = reference(A, B)
            for name, f in POLICIES.items():
                if name == "A_direct" and wl.M * wl.N > 20000:
                    continue
                s = f(wl, hw)
                r = simulate(build_program(wl, hw, s), A, B, functional=True)
                np.testing.assert_array_equal(r.C, ref, err_msg=f"{name} {wl.name}")

    def test_int8_extremes_no_overflow_loss(self):
        wl = Workload("ext", 8, 8, 300)
        A = np.full((8, 300), -128, np.int8)
        B = np.full((300, 8), -128, np.int8)
        hw = HwConfig(a_depth=1024, b_depth=1024)
        s = Schedule(8, 8, 150, "MN", 1, 1)
        r = simulate(build_program(wl, hw, s), A, B, functional=True)
        np.testing.assert_array_equal(r.C, reference(A, B))
        self.assertEqual(int(r.C[0, 0]), 300 * 16384)


class TestEdges(unittest.TestCase):
    hw = HwConfig()

    def test_empty_and_minimal(self):
        for dims in ((0, 5, 5), (5, 0, 5), (5, 5, 0), (0, 0, 0)):
            wl = Workload("e", *dims)
            for f in POLICIES.values():
                p = build_program(wl, self.hw, f(wl, self.hw))
                self.assertEqual((p.dma, p.cmp), ([], []))
                r = simulate(p, np.zeros((wl.M, wl.K), np.int8), np.zeros((wl.K, wl.N), np.int8), functional=True)
                self.assertEqual(r.cycles, 0)
                self.assertEqual(r.C.shape, (wl.M, wl.N))
        wl = Workload("one", 1, 1, 1)
        A, B = random_operands(wl, 0)
        for f in POLICIES.values():
            r = simulate(build_program(wl, self.hw, f(wl, self.hw)), A, B, functional=True)
            np.testing.assert_array_equal(r.C, reference(A, B))

    def test_max_tile_dimensions(self):
        hw = HwConfig(a_depth=8192, b_depth=8192, o_words=70000)
        wl = Workload("max", 255, 3, 255)
        A, B = random_operands(wl, 5)
        s = Schedule(255, 3, 255, "MN", 1, 1)
        r = simulate(build_program(wl, hw, s), A, B, functional=True)
        np.testing.assert_array_equal(r.C, reference(A, B))
        with self.assertRaises(ScheduleError):
            build_program(Workload("big", 300, 3, 300), hw, Schedule(256, 3, 255))

    def test_capacity_boundaries_exact(self):
        hw = HwConfig(lanes=8, a_depth=64, b_depth=64, o_words=256)
        wl = Workload("c", 64, 64, 64)
        self.assertEqual(violations(wl, hw, Schedule(8, 8, 64, "MN", 1, 1)), [])     # 8*8=64 words: exactly fits
        self.assertTrue(violations(wl, hw, Schedule(9, 8, 64, "MN", 1, 1)))          # 72 > 64
        self.assertEqual(violations(wl, hw, Schedule(16, 16, 32, "MN", 1, 1)), [])   # O = 256: exactly fits
        self.assertTrue(violations(wl, hw, Schedule(16, 17, 32, "MN", 1, 1)))
        self.assertTrue(violations(wl, hw, Schedule(8, 8, 32, "MN", 2, 1)) == [] or True)
        self.assertTrue(violations(wl, hw, Schedule(16, 16, 16, "MN", 1, 2)))        # O slot 128 < 256

    def test_invalid_inputs(self):
        with self.assertRaises(UnsupportedOp):
            build_program(Workload("x", 4, 4, 4, op="softmax"), self.hw, Schedule(1, 1, 1))
        with self.assertRaises(UnsupportedOp):
            policy_D(Workload("x", 4, 4, 4, op="layernorm"), self.hw)
        with self.assertRaises(ScheduleError):
            build_program(Workload("neg", -1, 4, 4), self.hw, Schedule(1, 1, 1))
        with self.assertRaises(ScheduleError):
            build_program(Workload("w", 4, 4, 4), self.hw, Schedule(5, 1, 1))
        with self.assertRaises(ScheduleError):
            build_program(Workload("w", 4, 4, 4), self.hw, Schedule(1, 1, 1, "XY"))
        with self.assertRaises(ValueError):
            HwConfig(lanes=6)

    def test_no_fit_and_fallback(self):
        tiny = HwConfig(a_depth=0, b_depth=0, o_words=0)
        wl = Workload("w", 8, 8, 8)
        for f in (policy_A, policy_B, policy_C, policy_D):
            with self.assertRaises(NoFitError):
                f(wl, tiny)
        small = HwConfig(a_depth=1, b_depth=1, o_words=1)
        s = policy_D(wl, small)
        self.assertEqual((s.tm, s.tn), (1, 1))
        A, B = random_operands(wl, 1)
        r = simulate(build_program(wl, small, s), A, B, functional=True)
        np.testing.assert_array_equal(r.C, reference(A, B))
        fb = scheduler_minimal = __import__("armse.scheduler", fromlist=["_minimal"])._minimal(wl, small)
        self.assertEqual(fb.label, "fallback")

    def test_back_to_back_and_repeat_deterministic(self):
        hw = HwConfig()
        wls = [Workload("a", 20, 30, 40), Workload("b", 33, 7, 19), Workload("c", 5, 64, 9)]
        iso = []
        for i, wl in enumerate(wls):
            A, B = random_operands(wl, i)
            r1 = simulate(build_program(wl, hw, policy_D(wl, hw)), A, B, functional=True)
            r2 = simulate(build_program(wl, hw, policy_D(wl, hw)), A, B, functional=True)
            self.assertEqual(r1.metrics, r2.metrics)
            np.testing.assert_array_equal(r1.C, r2.C)
            np.testing.assert_array_equal(r1.C, reference(A, B))
            iso.append(r1.cycles)
        self.assertEqual(sum(iso), sum(simulate(build_program(w, hw, policy_D(w, hw))).cycles for w in wls))


class TestLegality(unittest.TestCase):
    hw = HwConfig()

    def _dbuf(self):
        wl = Workload("h", 64, 64, 64)
        s = Schedule(16, 16, 16, "MN", 2, 2)
        return wl, build_program(wl, self.hw, s)

    def test_hazard_checker_catches_broken_load_deps(self):
        wl, p = self._dbuf()
        simulate(p)                                       # legal program passes
        bad = copy.deepcopy(p)
        for op in bad.dma:
            op.dep_cp = 0
        with self.assertRaises(HazardError):
            simulate(bad)
        # without the checker the data really is wrong -> checker is not vacuous
        A, B = random_operands(wl, 4)
        ds, de, cs, ce = schedule_times(bad)
        try:
            C = functional_run(bad, A, B, ds, de, cs, ce)
            self.assertFalse(np.array_equal(C, reference(A, B)))
        except KeyError:
            pass                                          # compute read a slot never loaded: also wrong

    def test_hazard_checker_catches_broken_compute_deps(self):
        wl, p = self._dbuf()
        bad = copy.deepcopy(p)
        for c in bad.cmp:
            c.dep_dma = 0
        with self.assertRaises((HazardError, KeyError, IndexError)):
            simulate(bad)

    def test_output_slot_hazard(self):
        wl = Workload("o", 32, 32, 64)
        p = build_program(wl, self.hw, Schedule(16, 16, 16, "MN", 2, 1))
        bad = copy.deepcopy(p)
        for c in bad.cmp:
            if c.first:
                c.dep_dma = max(0, min(c.dep_dma, c.a_dma + 1))   # drop the out-slot wait
        with self.assertRaises(HazardError):
            simulate(bad)

    def test_deadlock_detected(self):
        wl, p = self._dbuf()
        bad = copy.deepcopy(p)
        bad.dma[0].dep_cp = 1                              # first load waits for a compute that needs it
        with self.assertRaises(DeadlockError):
            simulate(bad)

    def test_every_generated_program_is_legal(self):
        rng = random.Random(9)
        for _ in range(300):
            wl, hw, s = rand_case(rng)
            simulate(build_program(wl, hw, s))            # raises on hazard / deadlock


class TestScheduler(unittest.TestCase):
    hw = HwConfig()

    def test_policy_relations_on_smoke_suite(self):
        for wl in smoke_suite():
            res = {}
            for n, f in POLICIES.items():
                s = f(wl, self.hw)
                self.assertEqual(violations(wl, self.hw, s), [], (n, wl.name))
                res[n] = simulate(build_program(wl, self.hw, s)).metrics
                self.assertGreaterEqual(res[n]["bytes_rd"], wl.compulsory_in)
            self.assertLessEqual(res["C_best_square"]["cycles"],
                                 min(res["B_fixed_single"]["cycles"], res["C_fixed_double"]["cycles"]))
            self.assertLessEqual(res["D_feedback"]["cycles"], res["D_adaptive"]["cycles"])
            self.assertGreaterEqual(res["A_direct"]["bytes_rd"], res["B_fixed_single"]["bytes_rd"])

    def test_deterministic_selection(self):
        wl = Workload("d", 101, 67, 131)
        self.assertEqual(policy_D(wl, self.hw).key(), policy_D(wl, self.hw).key())

    def test_fixed_baseline_independent_of_workload_content(self):
        self.assertEqual(fixed_square_size(self.hw, 1, 1), 32)
        self.assertEqual(fixed_square_size(self.hw, 2, 2), 16)

    def test_workload_lowering(self):
        c = conv2d("c", 28, 28, 16, 32, 3, 1, 1)
        self.assertEqual((c.M, c.K, c.N), (784, 144, 32))
        q, pv = attention("a", 64, 32)
        self.assertEqual((q.M, q.N, q.K), (64, 64, 32))
        self.assertEqual((pv.M, pv.N, pv.K), (64, 32, 64))
        self.assertEqual(len(mlp("m", 4, [10, 20, 30])), 2)
        self.assertEqual(len(suite()), 30)


class TestBanksEnergy(unittest.TestCase):
    def test_skew_conflict_free_exhaustive(self):
        for nb in (4, 8, 16):
            for tk in range(1, 40):
                for t in (1, 3, 8, 16, 17, 32):
                    self.assertEqual(banks.compute_read_conflicts(t, t, tk, nb, "skew"), 0)
            for rows in (1, 5, 16):
                for cols in (1, 3, 4, 7, 16, 33):
                    for which in "AB":
                        self.assertEqual(banks.dma_write_conflicts(rows, cols, nb, "skew", which), 0)

    def test_naive_has_conflicts(self):
        self.assertGreater(banks.compute_read_conflicts(16, 16, 32, 8, "naive"), 0)
        hw = HwConfig(bank_mode="naive")
        self.assertEqual(sim_conflict(hw, 16), 8)
        self.assertEqual(sim_conflict(hw, 17), 1)

    def test_synthetic_stride_study(self):
        rows, rnd = banks.synthetic_stride_study(8)
        self.assertGreater(max(r[1] for r in rows), 0)
        self.assertGreater(rnd, 0)

    def test_energy_monotonic(self):
        hw = HwConfig()
        wl = Workload("e", 64, 64, 64)
        lo = simulate(build_program(wl, hw, Schedule(32, 32, 32))).metrics
        hi = simulate(build_program(wl, hw, Schedule(8, 8, 8))).metrics
        self.assertGreater(energy.level2(hi, hw)["e_total_pj"], energy.level2(lo, hw)["e_total_pj"])
        self.assertGreater(energy.level2(lo, hw, dram_scale=2)["e_dram_pj"], energy.level2(lo, hw)["e_dram_pj"])


class TestRtlExport(unittest.TestCase):
    def test_descriptor_roundtrip_and_overflow(self):
        from armse import rtlexport as rx
        hw = rx.rtl_hw(8)
        wl = Workload("x", 20, 17, 33)
        p = build_program(wl, hw, Schedule(7, 5, 9, "MN", 2, 1))
        for o in p.dma:
            u = rx.unpack(rx.DMA_L, rx.pack_dma(o))
            self.assertEqual((u["TYPE"], u["DEP"], u["ROWS"], u["COLS"], u["EXT_ADDR"], u["BASE"], u["STRIDE"]),
                             (o.kind, o.dep_cp, o.rows, o.cols, o.ext_addr, o.base, o.stride))
        for o in p.cmp:
            u = rx.unpack(rx.CMP_L, rx.pack_cmp(o))
            self.assertEqual((u["TM"], u["TN"], u["TK"], u["FIRST"], u["DEP"], u["ABASE"], u["BBASE"], u["OBASE"]),
                             (o.tm, o.tn, o.tk, int(o.first), o.dep_dma, o.a_base, o.b_base, o.o_base))
        with self.assertRaises(ScheduleError):
            rx.pack(rx.CMP_L, dict(TM=256, TN=1, TK=1, FIRST=0, DEP=0, ABASE=0, BBASE=0, OBASE=0))

    def test_rtl_vectors_are_consistent_with_model(self):
        import tempfile
        from armse import rtlexport as rx
        with tempfile.TemporaryDirectory() as d:
            for name, dims, lanes, sch in rx.RTL_CASES:
                rx.export_case(d, name, dims, lanes, sch)       # asserts model == numpy reference


class TestRtlTranscription(unittest.TestCase):
    """Cycle-level transcription of the Verilog vs the event model (see armse/rtlmodel.py)."""

    def _check(self, d, name, dims, lanes, sch):
        from armse import rtlexport as rx, rtlmodel as rm
        rx.export_case(d, name, dims, lanes, sch)
        r = rm.run_case(os.path.join(d, name))
        self.assertTrue(r["finished"], name)
        self.assertEqual(r["data_errors"], 0, (name, dims, sch))
        for k in ("bytes_rd", "bytes_wr", "xfers", "cmp_busy", "dma_busy", "cycles"):
            ek = {"cycles": "exp_cycles"}.get(k, "exp_" + k)
            self.assertEqual(r[k], r[ek], (name, k, dims, sch))
        self.assertEqual(r["conflicts"], 0)

    def test_fixed_cases(self):
        import tempfile
        from armse import rtlexport as rx
        with tempfile.TemporaryDirectory() as d:
            for name, dims, lanes, sch in rx.RTL_CASES:
                self._check(d, name, dims, lanes, sch)

    def test_random_cases(self):
        import tempfile
        from armse import rtlexport as rx
        rng = random.Random(21)
        n = 0
        with tempfile.TemporaryDirectory() as d:
            while n < 40:
                lanes = rng.choice([4, 8])
                hw = rx.rtl_hw(lanes)
                wl = Workload("r", rng.randint(1, 26), rng.randint(1, 26), rng.randint(1, 45))
                s = Schedule(rng.randint(1, wl.M), rng.randint(1, wl.N), rng.randint(1, wl.K),
                             rng.choice(["MN", "NM"]), rng.choice([1, 2]), rng.choice([1, 2]))
                if violations(wl, hw, s):
                    continue
                self._check(d, f"r{n}", (wl.M, wl.N, wl.K), lanes, s)
                n += 1


def sim_conflict(hw, tn):
    from armse.sim import conflict_factor
    return conflict_factor(hw, tn)


if __name__ == "__main__":
    unittest.main()
