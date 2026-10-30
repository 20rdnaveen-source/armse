import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from armse.config import HwConfig, Workload
from armse.scheduler import policy_D
from armse.schedule import build_program
from armse.adaptive import counters, infer_hw, run_sequence

LAYERS = [Workload("a", 64, 64, 64), Workload("b", 100, 33, 70), Workload("c", 1, 200, 120), Workload("d", 48, 48, 200)]


def hist(hw_plan, hw_true, layers):
    h = []
    for wl in layers:
        p = build_program(wl, hw_plan, policy_D(wl, hw_plan))
        h.append((p, counters(p, hw_true)))
    return h


class TestAdaptive(unittest.TestCase):
    nom = HwConfig()

    def test_latency_recovered_exactly_from_one_layer(self):
        for lat in (0, 4, 30, 55):
            est = infer_hw(hist(self.nom, self.nom.replace(row_lat=lat), LAYERS[:1]), self.nom)
            self.assertEqual((est.row_lat, est.bus_bytes, est.bank_mode), (lat, 4, "skew"))

    def test_no_mismatch_changes_nothing(self):
        est = infer_hw(hist(self.nom, self.nom, LAYERS), self.nom)
        self.assertEqual((est.row_lat, est.bus_bytes, est.bank_mode), (12, 4, "skew"))

    def test_bus_width_and_latency_separated_with_two_layers(self):
        n8 = self.nom.replace(bus_bytes=8)
        true = self.nom.replace(row_lat=20)                        # real bus 4 B, latency 20; model believes bus 8, latency 12
        est = infer_hw(hist(n8, true, LAYERS[:2]), n8)
        self.assertEqual((est.bus_bytes, est.row_lat), (4, 20))

    def test_naive_banking_detected(self):
        est = infer_hw(hist(self.nom, self.nom.replace(bank_mode="naive"), [Workload("w", 64, 64, 64)]), self.nom)
        self.assertEqual(est.bank_mode, "naive")

    def test_adaptive_never_beats_oracle_and_matches_static_without_mismatch(self):
        true = self.nom.replace(row_lat=40)
        s, a, o = (run_sequence(LAYERS, self.nom, true, m) for m in ("static", "adaptive", "oracle"))
        self.assertLessEqual(o.cycles, a.cycles + 0)               # oracle plans every layer with the true model
        self.assertLessEqual(a.cycles, s.cycles)
        s0, a0 = run_sequence(LAYERS, self.nom, self.nom, "static"), run_sequence(LAYERS, self.nom, self.nom, "adaptive")
        self.assertEqual((s0.cycles, a0.replans), (a0.cycles, 0))


if __name__ == "__main__":
    unittest.main()
