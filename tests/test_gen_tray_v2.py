"""Isaac-independent tests for dual-fixture generator geometry."""

from __future__ import annotations

import argparse
import json
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from gen_tray_v2 import (
    DEFAULT_FLOOR_EMISSIVE,
    QUOTA_SLOT_SEED_OFFSET,
    approximate_uncovered_crop_corners,
    build_run_summary,
    clone_random_generator,
    compute_labeler_plane_coverage,
    default_light_scale,
    floor_box_spec,
    floor_material_inputs,
    high_x_backdrop_spec,
    labeler_grid_center,
    local_matrix_for_world_box,
    panel_crop_overruns,
    parse_args,
    parse_nonnegative_rgb_triplet,
    project_slab_corners_to_fisheye,
    sample_floor_emissive,
    sample_light_settings,
    translated_crop_centers,
)
from fill_logic import apply_quota_assignments, build_quota_schedule, sample_fill_scene


class GeneratorGeometryTests(unittest.TestCase):
    def test_golden_cli_flag_defaults_off_and_can_be_enabled(self) -> None:
        with patch.object(sys, "argv", ["gen_tray_v2.py"]):
            self.assertFalse(parse_args().golden)
        with patch.object(sys, "argv", ["gen_tray_v2.py", "--golden"]):
            self.assertTrue(parse_args().golden)

    def test_run_summary_counts_labels_and_damage_types(self) -> None:
        summary = build_run_summary(
            [
                {
                    "labels": {
                        "G01": "damaged_vial",
                        "G02": "pass",
                        "S01": "missing_cap",
                        "S02": "no_vial",
                    },
                    "variations": {
                        "G01": {"damage": {"kind": "bend"}},
                        "G02": {},
                        "S01": {},
                        "S02": {},
                    },
                }
            ]
        )
        self.assertEqual(summary["G"]["damaged_vial"], 1)
        self.assertEqual(summary["G"]["pass"], 1)
        self.assertEqual(summary["G"]["damage_types"]["bend"], 1)
        self.assertEqual(summary["S"]["missing_cap"], 1)
        self.assertEqual(summary["S"]["no_vial"], 1)

    def test_quota_mode_preserves_lighting_and_floor_samples(self) -> None:
        seed = 311
        image_seed = 987654321
        off_rng = random.Random(image_seed)
        quota_rng = random.Random(image_seed)
        off_scenes = {
            prefix: sample_fill_scene(off_rng, prefix=prefix)
            for prefix in ("G", "S")
        }
        quota_scenes = {
            prefix: sample_fill_scene(quota_rng, prefix=prefix)
            for prefix in ("G", "S")
        }
        self.assertEqual(off_scenes, quota_scenes)

        for rng in (off_rng, quota_rng):
            rng.randint(-15, 15)
            rng.randint(-15, 15)
            for _ in range(3):
                rng.uniform(-0.5, 0.5)

        slot_rng = random.Random(seed + QUOTA_SLOT_SEED_OFFSET)
        schedule = build_quota_schedule(1, 3)
        for prefix in ("G", "S"):
            apply_quota_assignments(
                quota_scenes[prefix],
                prefix=prefix,
                assignments=schedule[prefix][0],
                rng=slot_rng,
            )

        quota_variation_rng = clone_random_generator(quota_rng)
        for _ in range(40):
            quota_variation_rng.random()
        for _ in range(40):
            off_rng.random()
            quota_rng.random()

        off_lights = [
            sample_light_settings(off_rng, limits, "real", 1.35)
            for limits in ((0.7, 1.8), (0.7, 2.5))
        ]
        quota_lights = [
            sample_light_settings(quota_rng, limits, "real", 1.35)
            for limits in ((0.7, 1.8), (0.7, 2.5))
        ]
        self.assertEqual(off_lights, quota_lights)
        self.assertEqual(
            sample_floor_emissive(DEFAULT_FLOOR_EMISSIVE, 0.25, seed, 0),
            sample_floor_emissive(DEFAULT_FLOOR_EMISSIVE, 0.25, seed, 0),
        )

    def test_floor_emissive_is_omitted_when_zero_and_authored_when_nonzero(self) -> None:
        zero_inputs = floor_material_inputs(
            (0.36, 0.31, 0.22), (0.0, 0.0, 0.0)
        )
        self.assertNotIn("emissiveColor", zero_inputs)

        emissive = DEFAULT_FLOOR_EMISSIVE
        self.assertEqual(emissive, (0.15, 0.115, 0.07))
        tuned_inputs = floor_material_inputs((0.36, 0.31, 0.22), emissive)
        self.assertEqual(tuned_inputs["emissiveColor"], emissive)
        self.assertEqual(
            parse_nonnegative_rgb_triplet("0.8,0.592,0.28"),
            (0.8, 0.592, 0.28),
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_nonnegative_rgb_triplet("0.1,-0.2,0.3")

    def test_floor_jitter_is_deterministic_and_does_not_advance_scene_rng(self) -> None:
        scene_with_floor = random.Random(24680)
        scene_without_floor = random.Random(24680)
        before_with = [scene_with_floor.random() for _ in range(8)]
        before_without = [scene_without_floor.random() for _ in range(8)]
        self.assertEqual(before_with, before_without)

        sampled = sample_floor_emissive(DEFAULT_FLOOR_EMISSIVE, 0.25, 11, 0)
        repeated = sample_floor_emissive(DEFAULT_FLOOR_EMISSIVE, 0.25, 11, 0)
        self.assertEqual(sampled, repeated)
        self.assertGreaterEqual(sampled["brightness_factor"], 0.75)
        self.assertLessEqual(sampled["brightness_factor"], 1.25)
        self.assertGreaterEqual(sampled["red_factor"], 0.95)
        self.assertLessEqual(sampled["red_factor"], 1.05)
        self.assertGreaterEqual(sampled["blue_factor"], 0.95)
        self.assertLessEqual(sampled["blue_factor"], 1.05)

        after_with = [scene_with_floor.random() for _ in range(16)]
        after_without = [scene_without_floor.random() for _ in range(16)]
        self.assertEqual(after_with, after_without)

        disabled = sample_floor_emissive(DEFAULT_FLOOR_EMISSIVE, 0.0, 11, 0)
        self.assertEqual(disabled["emissive"], list(DEFAULT_FLOOR_EMISSIVE))
        self.assertEqual(disabled["brightness_factor"], 1.0)
        self.assertEqual(disabled["red_factor"], 1.0)
        self.assertEqual(disabled["blue_factor"], 1.0)

    @staticmethod
    def _real_labeler_geometry() -> dict:
        fixtures = {}
        for prefix, grid, offsets in (
            (
                "G",
                ((750, 398), (1078, 406), (744, 710), (1067, 715)),
                (-38, -47, 42, 43),
            ),
            (
                "S",
                ((1175, 628), (1480, 630), (1162, 928), (1466, 921)),
                (-28, -34, 143, 96),
            ),
        ):
            fixtures[prefix] = {
                "crop_offsets": offsets,
                "centers": {
                    f"{prefix}{position:02d}": labeler_grid_center(grid, position)
                    for position in range(1, 31)
                },
            }
        return {"fixtures": fixtures}

    def test_floor_box_math_with_scaled_translated_parent(self) -> None:
        camera = json.loads(
            (Path(__file__).resolve().parents[1] / "calib" / "camera_fit.json").read_text(
                encoding="utf-8"
            )
        )
        geometry = self._real_labeler_geometry()
        spec, coverage = floor_box_spec(camera, geometry, -0.095)
        np.testing.assert_allclose(
            spec["minimum"],
            [-4.152786419561394, -2.78785739385453, -0.105],
            atol=1.0e-9,
        )
        np.testing.assert_allclose(
            spec["maximum"],
            [3.124335986706717, 1.6787092679907525, -0.100],
            atol=1.0e-9,
        )
        self.assertAlmostEqual(coverage["plane_z"], -0.100)

        parent_world = np.eye(4, dtype=np.float64)
        parent_world[:3, :3] *= 0.0254
        parent_world[3, :3] = [0.25, -1.5, 0.75]
        local = local_matrix_for_world_box(
            parent_world, spec["center"], spec["size"]
        )
        composed_world = local @ parent_world
        local_corners = np.array(
            [
                [x_value, y_value, z_value, 1.0]
                for x_value in (-0.5, 0.5)
                for y_value in (-0.5, 0.5)
                for z_value in (-0.5, 0.5)
            ]
        )
        world_corners = local_corners @ composed_world
        np.testing.assert_allclose(
            world_corners[:, :3].min(axis=0), spec["minimum"], atol=1.0e-9
        )
        np.testing.assert_allclose(
            world_corners[:, :3].max(axis=0), spec["maximum"], atol=1.0e-9
        )
        uncovered = approximate_uncovered_crop_corners(
            camera,
            geometry,
            (-0.920, -0.209, -2.019, -0.038),
            None,
            spec,
        )
        self.assertEqual(uncovered, [])

    def test_legacy_light_sampling_is_unchanged_for_fixed_seed(self) -> None:
        sample = sample_light_settings(
            random.Random(12345), (0.7, 1.8), "legacy", 1.0
        )
        self.assertAlmostEqual(sample["intensity_factor"], 1.1582818597998752)
        self.assertAlmostEqual(sample["color_temperature_k"], 4220.338338914136)
        np.testing.assert_allclose(
            sample["color_rgb"],
            [1.0, 0.9860165207402994, 0.894795978279939],
            atol=0.0,
        )
        np.testing.assert_allclose(
            sample["position_jitter_m"],
            [-0.011842947946037186, -0.027570478585943317, 0.005940735185597512],
            atol=0.0,
        )
        self.assertEqual(default_light_scale("legacy"), 1.0)

    def test_real_light_sampling_is_neutral_and_within_temperature_range(self) -> None:
        rng = random.Random(6789)
        for _ in range(500):
            sample = sample_light_settings(rng, (0.7, 2.5), "real", 1.35)
            self.assertGreaterEqual(sample["color_temperature_k"], 5200.0)
            self.assertLessEqual(sample["color_temperature_k"], 6500.0)
            self.assertEqual(sample["color_rgb"], [1.0, 1.0, 1.0])
            self.assertGreaterEqual(sample["intensity_factor"], 0.7)
            self.assertLessEqual(sample["intensity_factor"], 2.5)
            self.assertAlmostEqual(
                sample["intensity_multiplier"],
                sample["intensity_factor"] * 1.35,
            )
        self.assertEqual(default_light_scale("real"), 1.35)

    def test_world_box_local_matrix_inverts_scaled_translated_parent(self) -> None:
        parent_world = np.eye(4, dtype=np.float64)
        parent_world[0, 0] = 0.0254
        parent_world[1, 1] = 0.0254
        parent_world[2, 2] = 0.0254
        parent_world[3, :3] = [1.25, -0.75, 0.40]
        center = np.array([-0.123996, -0.635520, 1.013000])
        size = np.array([0.172007, 0.612522, 0.005000])

        local = local_matrix_for_world_box(parent_world, center, size)
        composed_world = local @ parent_world
        expected_world = np.eye(4, dtype=np.float64)
        expected_world[0, 0] = size[0]
        expected_world[1, 1] = size[1]
        expected_world[2, 2] = size[2]
        expected_world[3, :3] = center
        np.testing.assert_allclose(composed_world, expected_world, atol=1.0e-12)

        local_corners = np.array(
            [
                [x_value, y_value, z_value, 1.0]
                for x_value in (-0.5, 0.5)
                for y_value in (-0.5, 0.5)
                for z_value in (-0.5, 0.5)
            ]
        )
        world_corners = local_corners @ composed_world
        np.testing.assert_allclose(
            world_corners[:, :3].min(axis=0), center - size / 2.0, atol=1.0e-12
        )
        np.testing.assert_allclose(
            world_corners[:, :3].max(axis=0), center + size / 2.0, atol=1.0e-12
        )

    def test_labeler_grid_uses_columns_from_right(self) -> None:
        grid = ((0.0, 0.0), (50.0, 0.0), (0.0, 40.0), (50.0, 40.0))
        self.assertEqual(labeler_grid_center(grid, 1), (50.0, 0.0))
        self.assertEqual(labeler_grid_center(grid, 5), (50.0, 40.0))
        self.assertEqual(labeler_grid_center(grid, 26), (0.0, 0.0))
        self.assertEqual(labeler_grid_center(grid, 30), (0.0, 40.0))

    def test_translated_crop_centers_cover_both_fixtures(self) -> None:
        geometry = {
            "fixtures": {
                prefix: {
                    "centers": {
                        f"{prefix}{position:02d}": (float(position), 100.0)
                        for position in range(1, 31)
                    }
                }
                for prefix in ("G", "S")
            }
        }
        centers = translated_crop_centers(geometry, (7, -11))
        self.assertEqual(len(centers), 60)
        self.assertEqual(centers["G01"], [8.0, 89.0])
        self.assertEqual(centers["S30"], [37.0, 89.0])

    def test_fisheye_plane_coverage_and_high_x_backdrop(self) -> None:
        camera = json.loads(
            (Path(__file__).resolve().parents[1] / "calib" / "camera_fit.json").read_text(
                encoding="utf-8"
            )
        )
        fixtures = self._real_labeler_geometry()["fixtures"]
        coverage = compute_labeler_plane_coverage(
            camera,
            {"fixtures": fixtures},
            1.016,
            frame_samples_per_edge=513,
            crop_samples_per_edge=17,
        )

        self.assertGreater(
            coverage["image_left_world"][0], coverage["image_right_world"][0]
        )
        self.assertAlmostEqual(coverage["frame_minimum"][0], -1.691893, places=5)
        self.assertAlmostEqual(coverage["frame_maximum"][0], 0.802783, places=5)
        self.assertAlmostEqual(coverage["crops_minimum"][0], -1.282877, places=5)
        self.assertAlmostEqual(coverage["crops_maximum"][0], -0.057993, places=5)
        self.assertAlmostEqual(coverage["crops_minimum"][1], -0.926676, places=5)
        self.assertAlmostEqual(coverage["crops_maximum"][1], -0.018429, places=5)

        overruns = panel_crop_overruns(coverage)
        self.assertEqual(overruns["high_x"], [f"G{position:02d}" for position in range(21, 31)])
        self.assertEqual(overruns["low_x"], [f"S{position:02d}" for position in range(1, 26)])
        self.assertEqual(overruns["high_y"], ["S20", "S25", "S30"])
        self.assertEqual(overruns["low_y"], [])

        spec = high_x_backdrop_spec(coverage)
        self.assertEqual(spec["offending_slots"], overruns["high_x"])
        self.assertAlmostEqual(spec["minimum"][0], -0.210, places=6)
        self.assertAlmostEqual(spec["maximum"][0], -0.037993, places=5)
        self.assertAlmostEqual(spec["minimum"][1], -0.941781, places=5)
        self.assertAlmostEqual(spec["maximum"][1], -0.329259, places=5)
        self.assertAlmostEqual(spec["size"][2], 0.005, places=7)

        corners = [
            [x_value, y_value, z_value]
            for x_value in (spec["minimum"][0], spec["maximum"][0])
            for y_value in (spec["minimum"][1], spec["maximum"][1])
            for z_value in (spec["minimum"][2], spec["maximum"][2])
        ]
        footprint = project_slab_corners_to_fisheye(camera, corners)
        self.assertEqual(footprint["pixels"].shape, (8, 2))
        self.assertAlmostEqual(footprint["pixel_minimum"][0], 668.509673, places=5)
        self.assertAlmostEqual(footprint["pixel_minimum"][1], 309.092711, places=5)
        self.assertAlmostEqual(footprint["pixel_maximum"][0], 817.250797, places=5)
        self.assertAlmostEqual(footprint["pixel_maximum"][1], 784.740171, places=5)


if __name__ == "__main__":
    unittest.main()
