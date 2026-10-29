"""Tests for the Isaac-independent fill-order sampler."""

from __future__ import annotations

import hashlib
import json
import random
import sys
import unittest
from collections import Counter
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from fill_logic import (
    ALL_SLOT_NAMES,
    FillScene,
    G_SLOT_NAMES,
    OCCUPIED_CLASSES,
    QUOTA_DAMAGE_TYPES,
    QUOTA_DEFECT_CLASSES,
    SLOT_NAMES,
    S_SLOT_NAMES,
    apply_quota_assignments,
    build_quota_schedule,
    sample_fill_scene,
)


class FillLogicTests(unittest.TestCase):
    def test_quota_schedule_balances_classes_and_damage_types(self) -> None:
        for image_count in (8, 13, 100):
            schedule = build_quota_schedule(image_count, 3)
            for prefix in ("G", "S"):
                assignments = [
                    assignment
                    for image_assignments in schedule[prefix]
                    for assignment in image_assignments
                ]
                class_counts = Counter(
                    assignment["class"] for assignment in assignments
                )
                self.assertEqual(set(class_counts), set(QUOTA_DEFECT_CLASSES))
                self.assertLessEqual(
                    max(class_counts.values()) - min(class_counts.values()), 1
                )
                damage_counts = Counter(
                    assignment["damage_type"]
                    for assignment in assignments
                    if assignment["class"] == "damaged_vial"
                )
                self.assertEqual(set(damage_counts), set(QUOTA_DAMAGE_TYPES))
                self.assertLessEqual(
                    max(damage_counts.values()) - min(damage_counts.values()), 1
                )

    def test_quota_defects_stay_inside_filled_prefix(self) -> None:
        for prefix, slot_names in (("G", G_SLOT_NAMES), ("S", S_SLOT_NAMES)):
            source = FillScene(
                labels={slot: "no_vial" for slot in slot_names},
                not_yet_filled=list(slot_names),
                expected_through=0,
                missed_position=None,
            )
            assignments = build_quota_schedule(1, 3)[prefix][0]
            scene, damage_types = apply_quota_assignments(
                source,
                prefix=prefix,
                assignments=assignments,
                rng=random.Random(1234),
            )
            self.assertEqual(scene.expected_through, 3)
            self.assertIsNone(scene.missed_position)
            self.assertEqual(scene.not_yet_filled, list(slot_names[3:]))
            defective = {
                slot
                for slot, label in scene.labels.items()
                if label in QUOTA_DEFECT_CLASSES
            }
            self.assertEqual(len(defective), 3)
            self.assertTrue(defective.issubset(set(slot_names[:3])))
            self.assertTrue(
                all(scene.labels[slot] == "no_vial" for slot in slot_names[3:])
            )
            self.assertTrue(set(damage_types).issubset(defective))

    def test_fixture_slot_names_are_complete_and_disjoint(self) -> None:
        self.assertEqual(len(G_SLOT_NAMES), 30)
        self.assertEqual(len(S_SLOT_NAMES), 30)
        self.assertEqual(len(ALL_SLOT_NAMES), 60)
        self.assertEqual(len(set(ALL_SLOT_NAMES)), 60)
        self.assertEqual(G_SLOT_NAMES[0], "G01")
        self.assertEqual(G_SLOT_NAMES[-1], "G30")
        self.assertEqual(S_SLOT_NAMES[0], "S01")
        self.assertEqual(S_SLOT_NAMES[-1], "S30")

    def test_s_fixture_sampling_uses_s_names(self) -> None:
        scene = sample_fill_scene(random.Random(29), prefix="S")
        self.assertEqual(tuple(scene.labels), S_SLOT_NAMES)
        self.assertTrue(all(name.startswith("S") for name in scene.labels))

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

    def test_quota_mode_off_legacy_sampler_snapshot(self) -> None:
        rng = random.Random(314159)
        scenes = [
            sample_fill_scene(rng, prefix=prefix)
            for _ in range(4)
            for prefix in ("G", "S")
        ]
        payload = [
            {
                "labels": scene.labels,
                "not_yet_filled": scene.not_yet_filled,
                "expected_through": scene.expected_through,
                "missed_position": scene.missed_position,
            }
            for scene in scenes
        ]
        digest = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            digest,
            "c70ec68587f8cf276e9476184e89b25ce6a1e19b496e7df59d7e1dada0f8b7c2",
        )

    def test_class_balance_forces_each_requested_defect(self) -> None:
        rng = random.Random(73)
        required = {"damaged_vial", "crooked_cap", "missing_cap"}
        for _ in range(500):
            scene = sample_fill_scene(rng, class_balance=True)
            occupied_labels = {
                scene.labels[slot]
                for slot in SLOT_NAMES[: scene.expected_through]
            }
            self.assertTrue(required.issubset(occupied_labels))
            self.assertIsNone(scene.missed_position)

    def test_zero_damage_weight_disables_damaged_vials(self) -> None:
        rng = random.Random(91)
        for class_balance in (False, True):
            for _ in range(500):
                scene = sample_fill_scene(
                    rng, class_balance=class_balance, damage_weight=0.0
                )
                self.assertNotIn("damaged_vial", scene.labels.values())

    def test_negative_damage_weight_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sample_fill_scene(random.Random(1), damage_weight=-0.1)

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
