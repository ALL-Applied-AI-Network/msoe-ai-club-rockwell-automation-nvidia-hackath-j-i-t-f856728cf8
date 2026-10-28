"""Tests for the Isaac-independent fill-order sampler."""

from __future__ import annotations

import random
import sys
import unittest
from collections import Counter
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from fill_logic import OCCUPIED_CLASSES, SLOT_NAMES, sample_fill_scene


class FillLogicTests(unittest.TestCase):
    def test_prefix_and_not_yet_filled_semantics(self) -> None:
        rng = random.Random(41)
        for _ in range(2_000):
            scene = sample_fill_scene(rng)
            self.assertEqual(set(scene.labels), set(SLOT_NAMES))
            self.assertEqual(
                scene.not_yet_filled, list(SLOT_NAMES[scene.expected_through :])
            )
            for index, slot in enumerate(SLOT_NAMES, start=1):
                if index > scene.expected_through:
                    self.assertEqual(scene.labels[slot], "no_vial")
                elif slot == scene.missed_position:
                    self.assertEqual(scene.labels[slot], "no_vial")
                    self.assertNotIn(slot, scene.not_yet_filled)
                else:
                    self.assertIn(scene.labels[slot], OCCUPIED_CLASSES)

    def test_sampler_is_deterministic_for_a_seed(self) -> None:
        left_rng = random.Random(17)
        right_rng = random.Random(17)
        left = [sample_fill_scene(left_rng) for _ in range(20)]
        right = [sample_fill_scene(right_rng) for _ in range(20)]
        self.assertEqual(left, right)

    def test_distribution_has_endpoints_and_favors_later_stages(self) -> None:
        rng = random.Random(2026)
        counts = Counter()
        missed = 0
        eligible = 0
        occupied_classes = Counter()
        for _ in range(20_000):
            scene = sample_fill_scene(rng)
            counts[scene.expected_through] += 1
            if scene.expected_through >= 2:
                eligible += 1
                missed += scene.missed_position is not None
            for index, slot in enumerate(SLOT_NAMES, start=1):
                if index <= scene.expected_through and slot != scene.missed_position:
                    occupied_classes[scene.labels[slot]] += 1

        self.assertGreater(counts[0], 0)
        self.assertGreater(counts[30], 0)
        self.assertGreater(sum(counts[n] for n in range(10, 31)), 3 * sum(counts[n] for n in range(10)))
        self.assertGreater(missed / eligible, 0.02)
        self.assertLess(missed / eligible, 0.04)

        total = sum(occupied_classes.values())
        expected = {
            "pass": 0.80,
            "missing_cap": 0.07,
            "crooked_cap": 0.07,
            "damaged_vial": 0.06,
        }
        for label, probability in expected.items():
            self.assertAlmostEqual(
                occupied_classes[label] / total, probability, delta=0.01
            )


if __name__ == "__main__":
    unittest.main()
