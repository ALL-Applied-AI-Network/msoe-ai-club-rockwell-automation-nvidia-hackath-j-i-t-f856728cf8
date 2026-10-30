"""Generate fill-order tray images in Isaac Sim 5.1.0 headless mode."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import random
import re
import subprocess
import sys
import traceback
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fill_logic import (
    ALL_SLOT_NAMES,
    G_SLOT_NAMES,
    QUOTA_DAMAGE_TYPES,
    QUOTA_DEFECT_CLASSES,
    S_SLOT_NAMES,
    apply_quota_assignments,
    build_golden_fill_scene,
    build_quota_schedule,
    sample_fill_scene,
)
from fisheye import (
    PRINCIPAL_POINT,
    fisheye_pixels_to_pinhole_normalized,
    invert_theta,
    jitter_camera_pose,
    project_world,
)


CAMERA_PATH = Path("/job/in/cameras/camera_fit.json")
LABELER_CALIBRATION_PATH = Path("/job/in/tools/calibration.json")
STAGE_PATH = "/job/in/stations/tray_inspection/tray_inspection.usdc"
FIXTURE_SLOT_NAMES = {"G": G_SLOT_NAMES, "S": S_SLOT_NAMES}
EXPECTED_CROP_OFFSETS = {
    "G": (-38, -47, 42, 43),
    "S": (-28, -34, 143, 96),
}
MAX_SEATED_CAP_OFFSET_M = 0.0015
DAMAGE_KINDS = ("crush", "bend", "lean_crush", "broken_top")
DAMAGE_WEIGHTS = (0.30, 0.20, 0.25, 0.25)
MAX_DAMAGE_ATTEMPTS = 10
MIN_CRUSH_SHIFT_M = 0.003
MIN_BEND_SHIFT_M = 0.006
MIN_LEAN_DEGREES = 10.0
PANEL_BOUNDS_XY = (-0.920, -0.209, -2.019, -0.038)
PANEL_SURFACE_Z = 1.016
PANEL_MATERIAL_PATH = "/TrayInspection/_materials/tw_table_paint"
BACKDROP_PRIM_PATH = "/TrayInspection/GeneratorBackdropExtension"
BACKDROP_OVERLAP_M = 0.001
BACKDROP_MARGIN_M = 0.020
BACKDROP_THICKNESS_M = 0.005
BACKDROP_SURFACE_DROP_M = 0.0005
LIGHT_PROFILES = ("legacy", "real")
FLOOR_PRIM_PATH = "/TrayInspection/GeneratorFloorBackdrop"
FLOOR_MATERIAL_PATH = "/TrayInspection/_materials/GeneratorFloorMaterial"
FLOOR_MARGIN_M = 0.300
FLOOR_CLEARANCE_M = 0.005
FLOOR_THICKNESS_M = 0.005
DEFAULT_FLOOR_ALBEDO = (0.36, 0.31, 0.22)
DEFAULT_FLOOR_EMISSIVE = (0.15, 0.115, 0.07)
FLOOR_JITTER_SEED_OFFSET = 0x464C4F4F52
QUOTA_SLOT_SEED_OFFSET = 0x51554F5441


def parse_ordered_choices(
    value: str,
    allowed: tuple[str, ...],
    option_name: str,
) -> tuple[str, ...]:
    """Parse a non-empty, duplicate-free comma list from an allowed set."""
    choices = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = [choice for choice in choices if choice not in allowed]
    if not choices or invalid or len(set(choices)) != len(choices):
        allowed_text = ",".join(allowed)
        raise argparse.ArgumentTypeError(
            f"{option_name} must be a non-empty, duplicate-free comma list from "
            f"{allowed_text}"
        )
    return choices


def parse_quota_classes(value: str) -> tuple[str, ...]:
    return parse_ordered_choices(value, QUOTA_DEFECT_CLASSES, "--quota-classes")


def parse_damage_types(value: str) -> tuple[str, ...]:
    return parse_ordered_choices(value, QUOTA_DAMAGE_TYPES, "--damage-types")


def projected_points_crop_check(
    center_pixel,
    tip_pixel,
    crop_center,
    crop_offsets,
) -> dict[str, object]:
    """Report whether projected tube center and tip lie inside one crop."""
    x0 = float(crop_center[0]) + float(crop_offsets[0])
    y0 = float(crop_center[1]) + float(crop_offsets[1])
    x1 = float(crop_center[0]) + float(crop_offsets[2])
    y1 = float(crop_center[1]) + float(crop_offsets[3])

    def inside(point) -> bool:
        return x0 <= float(point[0]) <= x1 and y0 <= float(point[1]) <= y1

    center_inside = inside(center_pixel)
    tip_inside = inside(tip_pixel)
    return {
        "center_px": [float(center_pixel[0]), float(center_pixel[1])],
        "tip_px": [float(tip_pixel[0]), float(tip_pixel[1])],
        "crop_rect_px": [x0, y0, x1, y1],
        "center_inside": center_inside,
        "tip_inside": tip_inside,
        "inside": center_inside and tip_inside,
    }


def labeler_grid_center(grid, position: int) -> tuple[float, float]:
    if not 1 <= position <= 30:
        raise ValueError("position must be from 1 through 30")
    column, row = divmod(position - 1, 5)
    u = 1.0 - column / 5.0
    v = row / 4.0
    top_left, top_right, bottom_left, bottom_right = grid
    x = (1.0 - v) * (
        (1.0 - u) * top_left[0] + u * top_right[0]
    ) + v * ((1.0 - u) * bottom_left[0] + u * bottom_right[0])
    y = (1.0 - v) * (
        (1.0 - u) * top_left[1] + u * top_right[1]
    ) + v * ((1.0 - u) * bottom_left[1] + u * bottom_right[1])
    return float(x), float(y)


def load_labeler_geometry(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("image_size") != [1920, 1080]:
        raise ValueError("Labeler calibration must target 1920x1080 images")
    profiles = document.get("profiles", [])
    profile = next(
        (candidate for candidate in profiles if candidate.get("id") == "view_001"),
        None,
    )
    if profile is None:
        raise ValueError("Labeler calibration is missing profile 'view_001'")
    geometry = {}
    for prefix, area_name in (("G", "good"), ("S", "scrap")):
        area = profile["areas"][area_name]
        offsets = tuple(int(value) for value in area["crop_offsets"])
        if offsets != EXPECTED_CROP_OFFSETS[prefix]:
            raise ValueError(
                f"Unexpected {prefix} crop offsets {offsets}; "
                f"expected {EXPECTED_CROP_OFFSETS[prefix]}"
            )
        geometry[prefix] = {
            "area": area_name,
            "grid": area["grid"],
            "crop_offsets": offsets,
            "centers": {
                f"{prefix}{position:02d}": labeler_grid_center(
                    area["grid"], position
                )
                for position in range(1, 31)
            },
        }
    return {"profile": profile["id"], "fixtures": geometry}


def translated_crop_centers(
    geometry: dict[str, object], offset: tuple[int, int] | list[int]
) -> dict[str, list[float]]:
    if len(offset) != 2:
        raise ValueError("crop translation must contain dx and dy")
    centers = {
        name: [
            geometry["fixtures"][name[0]]["centers"][name][0] + offset[0],
            geometry["fixtures"][name[0]]["centers"][name][1] + offset[1],
        ]
        for name in ALL_SLOT_NAMES
    }
    if len(centers) != 60:
        raise AssertionError("Expected 60 translated G/S crop centers")
    return centers


def rectangle_perimeter_pixels(
    x0: float, y0: float, x1: float, y1: float, samples_per_edge: int
):
    """Return evenly sampled pixels around a closed axis-aligned rectangle."""
    import numpy as np

    if samples_per_edge < 2:
        raise ValueError("samples_per_edge must be at least 2")
    horizontal = np.linspace(float(x0), float(x1), samples_per_edge)
    vertical = np.linspace(float(y0), float(y1), samples_per_edge)
    return np.vstack(
        (
            np.column_stack((horizontal, np.full(samples_per_edge, float(y0)))),
            np.column_stack((horizontal, np.full(samples_per_edge, float(y1)))),
            np.column_stack((np.full(samples_per_edge, float(x0)), vertical)),
            np.column_stack((np.full(samples_per_edge, float(x1)), vertical)),
        )
    )


def fisheye_pixels_on_world_plane(camera: dict, pixels, plane_z: float):
    """Intersect fitted-camera rays for fisheye pixels with a world-Z plane."""
    import numpy as np

    focal = float(camera["focal_px"])
    normalized = fisheye_pixels_to_pinhole_normalized(
        np.asarray(pixels, dtype=np.float64),
        focal,
        float(camera["k1"]),
        float(camera["k2"]),
    )
    rotation_world_to_cv = np.asarray(camera["R_world_to_cv"], dtype=np.float64)
    camera_position = np.asarray(camera["position_m"], dtype=np.float64)
    camera_rays = np.column_stack(
        (normalized, np.ones(normalized.shape[0], dtype=np.float64))
    )
    world_rays = (rotation_world_to_cv.T @ camera_rays.T).T
    if np.any(np.abs(world_rays[:, 2]) < 1.0e-12):
        raise ValueError("One or more camera rays are parallel to the requested plane")
    distances = (float(plane_z) - camera_position[2]) / world_rays[:, 2]
    if np.any(distances <= 0.0):
        raise ValueError("The requested plane lies behind one or more camera rays")
    return camera_position + distances[:, None] * world_rays


def compute_labeler_plane_coverage(
    camera: dict,
    labeler_geometry: dict,
    plane_z: float,
    *,
    translation_limit_px: int = 15,
    frame_samples_per_edge: int = 2049,
    crop_samples_per_edge: int = 33,
) -> dict[str, object]:
    """Project the image frame and every jittered labeler crop onto a Z plane."""
    import numpy as np

    width, height = (int(value) for value in camera["image_size"])
    frame_pixels = rectangle_perimeter_pixels(
        0.0, 0.0, width - 1.0, height - 1.0, frame_samples_per_edge
    )
    frame_world = fisheye_pixels_on_world_plane(camera, frame_pixels, plane_z)
    image_midpoints = fisheye_pixels_on_world_plane(
        camera,
        np.asarray([[0.0, height / 2.0], [width - 1.0, height / 2.0]]),
        plane_z,
    )

    slot_bounds = {}
    all_crop_points = []
    jitter_values = (-int(translation_limit_px), int(translation_limit_px))
    for slot_name in ALL_SLOT_NAMES:
        fixture = slot_name[0]
        center = labeler_geometry["fixtures"][fixture]["centers"][slot_name]
        offsets = labeler_geometry["fixtures"][fixture]["crop_offsets"]
        slot_points = []
        for dx in jitter_values:
            for dy in jitter_values:
                pixels = rectangle_perimeter_pixels(
                    center[0] + dx + offsets[0],
                    center[1] + dy + offsets[1],
                    center[0] + dx + offsets[2],
                    center[1] + dy + offsets[3],
                    crop_samples_per_edge,
                )
                slot_points.append(
                    fisheye_pixels_on_world_plane(camera, pixels, plane_z)
                )
        combined = np.vstack(slot_points)
        all_crop_points.append(combined)
        slot_bounds[slot_name] = {
            "minimum": combined.min(axis=0),
            "maximum": combined.max(axis=0),
        }

    all_crops = np.vstack(all_crop_points)
    return {
        "plane_z": float(plane_z),
        "image_left_world": image_midpoints[0],
        "image_right_world": image_midpoints[1],
        "frame_minimum": frame_world.min(axis=0),
        "frame_maximum": frame_world.max(axis=0),
        "crops_minimum": all_crops.min(axis=0),
        "crops_maximum": all_crops.max(axis=0),
        "slots": slot_bounds,
    }


def panel_crop_overruns(
    coverage: dict[str, object],
    panel_bounds_xy: tuple[float, float, float, float] = PANEL_BOUNDS_XY,
) -> dict[str, list[str]]:
    """List labeler crops that project beyond each edge of the white panel."""
    panel_x_min, panel_x_max, panel_y_min, panel_y_max = panel_bounds_xy
    overruns = {"low_x": [], "high_x": [], "low_y": [], "high_y": []}
    for slot_name, bounds in coverage["slots"].items():
        minimum = bounds["minimum"]
        maximum = bounds["maximum"]
        if minimum[0] < panel_x_min:
            overruns["low_x"].append(slot_name)
        if maximum[0] > panel_x_max:
            overruns["high_x"].append(slot_name)
        if minimum[1] < panel_y_min:
            overruns["low_y"].append(slot_name)
        if maximum[1] > panel_y_max:
            overruns["high_y"].append(slot_name)
    return overruns


def high_x_backdrop_spec(
    coverage: dict[str, object],
    panel_bounds_xy: tuple[float, float, float, float] = PANEL_BOUNDS_XY,
    *,
    overlap_m: float = BACKDROP_OVERLAP_M,
    safety_margin_m: float = BACKDROP_MARGIN_M,
    thickness_m: float = BACKDROP_THICKNESS_M,
    surface_drop_m: float = BACKDROP_SURFACE_DROP_M,
) -> dict[str, object]:
    """Size one slab for crops that extend beyond the panel's high-X edge."""
    panel_x_max = float(panel_bounds_xy[1])
    offending = [
        (slot_name, bounds)
        for slot_name, bounds in coverage["slots"].items()
        if float(bounds["maximum"][0]) > panel_x_max
    ]
    if not offending:
        raise ValueError("No labeler crop extends beyond the panel's high-X edge")
    x_min = panel_x_max - float(overlap_m)
    x_max = max(float(bounds["maximum"][0]) for _, bounds in offending) + float(
        safety_margin_m
    )
    y_min = min(float(bounds["minimum"][1]) for _, bounds in offending) - float(
        safety_margin_m
    )
    y_max = max(float(bounds["maximum"][1]) for _, bounds in offending) + float(
        safety_margin_m
    )
    top_z = PANEL_SURFACE_Z - float(surface_drop_m)
    bottom_z = top_z - float(thickness_m)
    return {
        "prim_path": BACKDROP_PRIM_PATH,
        "material_path": PANEL_MATERIAL_PATH,
        "offending_slots": [slot_name for slot_name, _ in offending],
        "minimum": [x_min, y_min, bottom_z],
        "maximum": [x_max, y_max, top_z],
        "center": [
            (x_min + x_max) / 2.0,
            (y_min + y_max) / 2.0,
            (bottom_z + top_z) / 2.0,
        ],
        "size": [x_max - x_min, y_max - y_min, top_z - bottom_z],
    }


def project_slab_corners_to_fisheye(camera: dict, world_corners) -> dict[str, object]:
    """Project exactly eight slab corners with the fitted fisheye camera."""
    import numpy as np

    corners = np.asarray(world_corners, dtype=np.float64)
    if corners.shape != (8, 3):
        raise ValueError(f"world_corners must have shape (8, 3), got {corners.shape}")
    pixels = project_world(
        corners,
        np.asarray(camera["R_world_to_cv"], dtype=np.float64),
        np.asarray(camera["position_m"], dtype=np.float64),
        float(camera["focal_px"]),
        float(camera["k1"]),
        float(camera["k2"]),
    )
    return {
        "world_minimum": corners.min(axis=0),
        "world_maximum": corners.max(axis=0),
        "pixels": pixels,
        "pixel_minimum": pixels.min(axis=0),
        "pixel_maximum": pixels.max(axis=0),
    }


def local_matrix_for_world_box(parent_world_matrix, center, size):
    """Return a row-vector USD local matrix for an axis-aligned world box."""
    import numpy as np

    parent = np.asarray(parent_world_matrix, dtype=np.float64)
    center_values = np.asarray(center, dtype=np.float64)
    size_values = np.asarray(size, dtype=np.float64)
    if parent.shape != (4, 4):
        raise ValueError(f"parent_world_matrix must have shape (4, 4), got {parent.shape}")
    if center_values.shape != (3,) or size_values.shape != (3,):
        raise ValueError("center and size must each contain exactly three values")
    if np.any(size_values <= 0.0):
        raise ValueError("world box sizes must be positive")
    desired_world = np.eye(4, dtype=np.float64)
    desired_world[0, 0] = size_values[0]
    desired_world[1, 1] = size_values[1]
    desired_world[2, 2] = size_values[2]
    desired_world[3, :3] = center_values
    return desired_world @ np.linalg.inv(parent)


def default_light_scale(profile: str) -> float:
    """Return the profile-specific default intensity multiplier."""
    if profile == "legacy":
        return 1.0
    if profile == "real":
        return 1.35
    raise ValueError(f"Unknown light profile: {profile!r}")


def sample_light_settings(
    scene_rng: random.Random,
    intensity_range: tuple[float, float],
    profile: str,
    light_scale: float,
) -> dict[str, object]:
    """Sample one light while preserving the legacy RNG sequence exactly."""
    if profile not in LIGHT_PROFILES:
        raise ValueError(f"Unknown light profile: {profile!r}")
    if light_scale <= 0.0:
        raise ValueError("light_scale must be positive")
    factor = scene_rng.uniform(*intensity_range)
    if profile == "legacy":
        kelvin = scene_rng.uniform(4200.0, 6200.0)
        color = [
            1.0,
            scene_rng.uniform(0.92, 1.0),
            scene_rng.uniform(0.85, 1.0),
        ]
    else:
        kelvin = scene_rng.uniform(5200.0, 6500.0)
        color = [1.0, 1.0, 1.0]
    jitter = [scene_rng.uniform(-0.045, 0.045) for _ in range(3)]
    return {
        "profile": profile,
        "intensity_factor": factor,
        "intensity_scale": float(light_scale),
        "intensity_multiplier": factor * float(light_scale),
        "color_temperature_k": kelvin,
        "color_rgb": color,
        "position_jitter_m": jitter,
    }


def parse_rgb_triplet(value: str) -> tuple[float, float, float]:
    """Parse a comma-separated linear RGB triplet for argparse."""
    try:
        values = tuple(float(component.strip()) for component in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("RGB values must be numbers") from exc
    if len(values) != 3 or any(component < 0.0 or component > 1.0 for component in values):
        raise argparse.ArgumentTypeError(
            "RGB must contain exactly three comma-separated values from 0 to 1"
        )
    return values


def parse_nonnegative_rgb_triplet(value: str) -> tuple[float, float, float]:
    """Parse three comma-separated non-negative RGB values for argparse."""
    try:
        values = tuple(float(component.strip()) for component in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("RGB values must be numbers") from exc
    if len(values) != 3 or any(component < 0.0 for component in values):
        raise argparse.ArgumentTypeError(
            "RGB must contain exactly three comma-separated non-negative values"
        )
    return values


def floor_material_inputs(albedo, emissive) -> dict[str, object]:
    """Describe authored floor shader inputs, omitting zero emissive by default."""
    albedo_values = tuple(float(value) for value in albedo)
    emissive_values = tuple(float(value) for value in emissive)
    if len(albedo_values) != 3 or len(emissive_values) != 3:
        raise ValueError("floor albedo and emissive must each contain three values")
    if any(value < 0.0 for value in emissive_values):
        raise ValueError("floor emissive values must be non-negative")
    inputs = {
        "diffuseColor": albedo_values,
        "roughness": 0.9,
        "metallic": 0.0,
    }
    if any(value > 0.0 for value in emissive_values):
        inputs["emissiveColor"] = emissive_values
    return inputs


def sample_floor_emissive(
    base_emissive,
    jitter_fraction: float,
    seed: int,
    image_index: int,
) -> dict[str, object]:
    """Sample floor emission from an RNG isolated from all scene randomness."""
    base = tuple(float(value) for value in base_emissive)
    if len(base) != 3 or any(value < 0.0 for value in base):
        raise ValueError("base_emissive must contain three non-negative values")
    fraction = float(jitter_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("jitter_fraction must be from 0 to 1")
    if image_index < 0:
        raise ValueError("image_index must be non-negative")
    if fraction == 0.0:
        brightness_factor = 1.0
        red_factor = 1.0
        blue_factor = 1.0
    else:
        floor_rng = random.Random(int(seed) + FLOOR_JITTER_SEED_OFFSET + image_index)
        brightness_factor = floor_rng.uniform(1.0 - fraction, 1.0 + fraction)
        red_factor = floor_rng.uniform(0.95, 1.05)
        blue_factor = floor_rng.uniform(0.95, 1.05)
    emissive = [
        base[0] * brightness_factor * red_factor,
        base[1] * brightness_factor,
        base[2] * brightness_factor * blue_factor,
    ]
    return {
        "base_emissive": list(base),
        "emissive": emissive,
        "brightness_factor": brightness_factor,
        "red_factor": red_factor,
        "blue_factor": blue_factor,
        "jitter_fraction": fraction,
        "seed": int(seed) + FLOOR_JITTER_SEED_OFFSET + image_index,
        "image_index": image_index,
    }


def clone_random_generator(source: random.Random) -> random.Random:
    """Clone RNG state without advancing the source generator."""
    clone = random.Random()
    clone.setstate(source.getstate())
    return clone


def build_run_summary(image_records) -> dict[str, dict[str, object]]:
    """Count final labels and realized damage types for each fixture."""
    summary = {
        prefix: {
            "pass": 0,
            "missing_cap": 0,
            "crooked_cap": 0,
            "damaged_vial": 0,
            "damage_types": {damage_type: 0 for damage_type in QUOTA_DAMAGE_TYPES},
            "no_vial": 0,
        }
        for prefix in ("G", "S")
    }
    for record in image_records:
        for slot, label in record["labels"].items():
            fixture_summary = summary[slot[0]]
            fixture_summary[label] += 1
            if label == "damaged_vial":
                damage_type = record["variations"][slot]["damage"]["kind"]
                damage_counts = fixture_summary["damage_types"]
                damage_counts[damage_type] = damage_counts.get(damage_type, 0) + 1
    return summary


def print_run_summary(summary: dict[str, dict[str, object]]) -> None:
    """Print a compact per-fixture final-label summary."""
    print(
        "fixture pass missing_cap crooked_cap damaged_vial "
        "bend crush lean_crush broken_top no_vial",
        flush=True,
    )
    for prefix in ("G", "S"):
        counts = summary[prefix]
        damage_counts = counts["damage_types"]
        print(
            f"{prefix} {counts['pass']} {counts['missing_cap']} "
            f"{counts['crooked_cap']} {counts['damaged_vial']} "
            f"{damage_counts['bend']} {damage_counts['crush']} "
            f"{damage_counts['lean_crush']} {damage_counts['broken_top']} "
            f"{counts['no_vial']}",
            flush=True,
        )


def floor_box_spec(
    camera: dict,
    labeler_geometry: dict,
    station_minimum_z: float,
    *,
    margin_m: float = FLOOR_MARGIN_M,
    clearance_m: float = FLOOR_CLEARANCE_M,
    thickness_m: float = FLOOR_THICKNESS_M,
) -> tuple[dict[str, object], dict[str, object]]:
    """Size a floor below station geometry from the fisheye frame footprint."""
    top_z = float(station_minimum_z) - float(clearance_m)
    coverage = compute_labeler_plane_coverage(camera, labeler_geometry, top_z)
    x_min = float(coverage["frame_minimum"][0]) - float(margin_m)
    x_max = float(coverage["frame_maximum"][0]) + float(margin_m)
    y_min = float(coverage["frame_minimum"][1]) - float(margin_m)
    y_max = float(coverage["frame_maximum"][1]) + float(margin_m)
    bottom_z = top_z - float(thickness_m)
    spec = {
        "prim_path": FLOOR_PRIM_PATH,
        "material_path": FLOOR_MATERIAL_PATH,
        "minimum": [x_min, y_min, bottom_z],
        "maximum": [x_max, y_max, top_z],
        "center": [
            (x_min + x_max) / 2.0,
            (y_min + y_max) / 2.0,
            (bottom_z + top_z) / 2.0,
        ],
        "size": [x_max - x_min, y_max - y_min, top_z - bottom_z],
        "station_minimum_z": float(station_minimum_z),
        "clearance_m": float(clearance_m),
        "margin_m": float(margin_m),
    }
    return spec, coverage


def approximate_uncovered_crop_corners(
    camera: dict,
    labeler_geometry: dict,
    panel_bounds_xy: tuple[float, float, float, float],
    slab_spec: dict[str, object] | None,
    floor_specification: dict[str, object] | None,
    *,
    translation_limit_px: int = 15,
    authored_pixel_boxes: tuple[tuple[object, object], ...] = (),
) -> list[str]:
    """Approximate crop coverage by known planar extents and projected AABBs."""
    import numpy as np

    panel_x_min, panel_x_max, panel_y_min, panel_y_max = panel_bounds_xy
    uncovered = []
    for slot_name in ALL_SLOT_NAMES:
        fixture = slot_name[0]
        center = labeler_geometry["fixtures"][fixture]["centers"][slot_name]
        offsets = labeler_geometry["fixtures"][fixture]["crop_offsets"]
        pixels = []
        for dx in (-translation_limit_px, translation_limit_px):
            for dy in (-translation_limit_px, translation_limit_px):
                x0 = center[0] + dx + offsets[0]
                x1 = center[0] + dx + offsets[2]
                y0 = center[1] + dy + offsets[1]
                y1 = center[1] + dy + offsets[3]
                pixels.extend(((x0, y0), (x0, y1), (x1, y0), (x1, y1)))
        pixel_array = np.asarray(pixels, dtype=np.float64)
        panel_points = fisheye_pixels_on_world_plane(
            camera, pixel_array, PANEL_SURFACE_Z
        )
        covered = (
            (panel_points[:, 0] >= panel_x_min)
            & (panel_points[:, 0] <= panel_x_max)
            & (panel_points[:, 1] >= panel_y_min)
            & (panel_points[:, 1] <= panel_y_max)
        )
        if slab_spec is not None:
            slab_points = fisheye_pixels_on_world_plane(
                camera, pixel_array, float(slab_spec["maximum"][2])
            )
            covered |= (
                (slab_points[:, 0] >= float(slab_spec["minimum"][0]))
                & (slab_points[:, 0] <= float(slab_spec["maximum"][0]))
                & (slab_points[:, 1] >= float(slab_spec["minimum"][1]))
                & (slab_points[:, 1] <= float(slab_spec["maximum"][1]))
            )
        if floor_specification is not None:
            floor_points = fisheye_pixels_on_world_plane(
                camera, pixel_array, float(floor_specification["maximum"][2])
            )
            covered |= (
                (floor_points[:, 0] >= float(floor_specification["minimum"][0]))
                & (floor_points[:, 0] <= float(floor_specification["maximum"][0]))
                & (floor_points[:, 1] >= float(floor_specification["minimum"][1]))
                & (floor_points[:, 1] <= float(floor_specification["maximum"][1]))
            )
        for pixel_minimum, pixel_maximum in authored_pixel_boxes:
            minimum = np.asarray(pixel_minimum, dtype=np.float64)
            maximum = np.asarray(pixel_maximum, dtype=np.float64)
            covered |= np.all(pixel_array >= minimum, axis=1) & np.all(
                pixel_array <= maximum, axis=1
            )
        if not bool(np.all(covered)):
            uncovered.append(slot_name)
    return uncovered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prefix",
        default="",
        help="Prefix for every generated filename, including labels.json.",
    )
    parser.add_argument("--out", type=Path, default=Path("/job/out"))
    parser.add_argument("--subframes", type=int, default=24)
    parser.add_argument(
        "--damage-weight",
        type=float,
        default=1.0,
        help="Multiplier for damaged-vial sampling; zero disables it.",
    )
    parser.add_argument(
        "--crops-only",
        action="store_true",
        help="Write crops and labels plus only the first three full images.",
    )
    parser.add_argument(
        "--camera-view",
        default="top",
        help="Camera view to render (only 'top' is implemented).",
    )
    parser.add_argument(
        "--defect-boost",
        type=float,
        default=1.0,
        help="Multiply vial deformation magnitudes for inspection renders.",
    )
    parser.add_argument(
        "--class-balance",
        action="store_true",
        help="Force damaged, crooked-cap, and missing-cap examples per image.",
    )
    parser.add_argument(
        "--quota-mode",
        action="store_true",
        help="Use deterministic run-level class and damage-type quotas.",
    )
    parser.add_argument(
        "--quota-classes",
        type=parse_quota_classes,
        default=QUOTA_DEFECT_CLASSES,
        metavar="CLASS,...",
        help=(
            "Quota classes from missing_cap,crooked_cap,damaged_vial "
            "(default: all)."
        ),
    )
    parser.add_argument(
        "--damage-types",
        type=parse_damage_types,
        default=QUOTA_DAMAGE_TYPES,
        metavar="TYPE,...",
        help=(
            "Damage types from bend,crush,lean_crush,broken_top "
            "(default: all)."
        ),
    )
    parser.add_argument(
        "--golden",
        action="store_true",
        help="Render fixed all-pass, fully filled reference images.",
    )
    parser.add_argument(
        "--defects-per-fixture",
        type=int,
        default=3,
        metavar="K",
        help="Exact scheduled defects per fixture per image in quota mode (default: 3).",
    )
    parser.add_argument(
        "--width-scale",
        type=float,
        default=1.0,
        help="Scale pinhole render resolution while preserving its full FOV.",
    )
    parser.add_argument(
        "--no-backdrop",
        action="store_true",
        help="Disable the session-layer white-panel backdrop extension.",
    )
    parser.add_argument(
        "--backdrop-debug",
        action="store_true",
        help="Bind the backdrop to a bright magenta emissive debug material.",
    )
    parser.add_argument(
        "--light-profile",
        choices=LIGHT_PROFILES,
        default="real",
        help="Lighting color/temperature profile (default: real).",
    )
    parser.add_argument(
        "--light-scale",
        type=float,
        default=None,
        help="Base light intensity multiplier (default: 1.0 legacy, 1.35 real).",
    )
    parser.add_argument(
        "--no-floor",
        action="store_true",
        help="Disable the session-layer floor background.",
    )
    parser.add_argument(
        "--floor-albedo",
        type=parse_rgb_triplet,
        default=DEFAULT_FLOOR_ALBEDO,
        metavar="R,G,B",
        help="Linear RGB floor albedo (default: 0.36,0.31,0.22).",
    )
    parser.add_argument(
        "--floor-emissive",
        type=parse_nonnegative_rgb_triplet,
        default=DEFAULT_FLOOR_EMISSIVE,
        metavar="R,G,B",
        help="Non-negative RGB floor emission (default: 0.15,0.115,0.07).",
    )
    parser.add_argument(
        "--floor-brightness-jitter",
        type=float,
        default=0.25,
        metavar="FRACTION",
        help="Per-image floor brightness variation; zero disables (default: 0.25).",
    )
    return parser.parse_args()


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=SCRIPT_DIR.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def serialized_flags(args: argparse.Namespace) -> dict[str, object]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


def ensure_external_output(path: Path) -> None:
    resolved = path.resolve()
    forbidden = [Path("/job/in").resolve(), SCRIPT_DIR.parent.resolve()]
    for root in forbidden:
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        raise ValueError(f"Refusing to write generated data inside source tree: {path}")


def build_remap(
    width: int,
    height: int,
    focal: float,
    k1: float,
    k2: float,
    theta_max: float,
    width_scale: float,
):
    import numpy as np

    if width_scale <= 0.0:
        raise ValueError("--width-scale must be positive")
    cx, cy = PRINCIPAL_POINT
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    delta_x = grid_x.astype(np.float64) - cx
    delta_y = grid_y.astype(np.float64) - cy
    radius_px = np.sqrt(delta_x**2 + delta_y**2)
    theta_d = radius_px / focal
    theta = invert_theta(theta_d, k1, k2, upper=theta_max)
    pinhole_radius = np.tan(theta)
    radial_scale = np.zeros_like(radius_px)
    nonzero = radius_px > 1.0e-12
    radial_scale[nonzero] = pinhole_radius[nonzero] / radius_px[nonzero]
    ideal_x = delta_x * radial_scale
    ideal_y = delta_y * radial_scale

    # theta_max already includes a ten-percent angular margin. The additional
    # five-percent pixel margin prevents interpolation at the render boundary.
    full_extent = 2.0 * focal * math.tan(theta_max) * 1.05
    render_width = max(64, int(math.ceil(full_extent * width_scale)))
    render_height = render_width
    render_focal = focal * width_scale
    map_x = (ideal_x * render_focal + render_width / 2.0).astype(np.float32)
    map_y = (ideal_y * render_focal + render_height / 2.0).astype(np.float32)
    if (
        map_x.min() < -0.01
        or map_y.min() < -0.01
        or map_x.max() > render_width - 1 + 0.01
        or map_y.max() > render_height - 1 + 0.01
    ):
        raise RuntimeError("Computed pinhole render does not cover every final pixel")
    return render_width, render_height, render_focal, map_x, map_y


def main() -> None:
    kit = None
    try:
        args = parse_args()
        if args.n < 1:
            raise ValueError("--n must be at least 1")
        if args.defect_boost <= 0.0:
            raise ValueError("--defect-boost must be positive")
        if args.damage_weight < 0.0:
            raise ValueError("--damage-weight must be non-negative")
        if not 0 <= args.defects_per_fixture <= 30:
            raise ValueError("--defects-per-fixture must be from 0 to 30")
        if args.golden and (args.quota_mode or args.class_balance):
            print("--golden overrides --quota-mode and --class-balance", flush=True)
        elif args.quota_mode and args.class_balance:
            print("--quota-mode overrides --class-balance", flush=True)
        if args.light_scale is None:
            args.light_scale = default_light_scale(args.light_profile)
        if args.light_scale <= 0.0:
            raise ValueError("--light-scale must be positive")
        if not 0.0 <= args.floor_brightness_jitter <= 1.0:
            raise ValueError("--floor-brightness-jitter must be from 0 to 1")
        if Path(args.prefix).name != args.prefix or any(
            separator in args.prefix for separator in ("/", "\\")
        ):
            raise ValueError("--prefix must be a filename prefix, not a path")
        if args.camera_view != "top":
            raise NotImplementedError(
                f"Camera view {args.camera_view!r} is not implemented; use 'top'"
            )
        damage_sampler_types = (
            DAMAGE_KINDS
            if args.damage_types == QUOTA_DAMAGE_TYPES
            else tuple(args.damage_types)
        )
        damage_weight_by_type = dict(zip(DAMAGE_KINDS, DAMAGE_WEIGHTS, strict=True))
        damage_sampler_weights = tuple(
            damage_weight_by_type[damage_type]
            for damage_type in damage_sampler_types
        )
        ensure_external_output(args.out)

        import cv2
        import numpy as np

        camera = json.loads(CAMERA_PATH.read_text(encoding="utf-8"))
        width, height = (int(value) for value in camera["image_size"])
        if (width, height) != (1920, 1080):
            raise ValueError(f"Expected fitted image size 1920x1080, got {width}x{height}")
        focal = float(camera["focal_px"])
        k1 = float(camera["k1"])
        k2 = float(camera["k2"])
        theta_max = float(camera["theta_max"])
        if camera.get("model") != "opencv_fisheye":
            raise ValueError("camera_fit.json must use model='opencv_fisheye'")
        labeler_geometry = load_labeler_geometry(LABELER_CALIBRATION_PATH)
        rotation_world_to_cv = np.asarray(camera["R_world_to_cv"], dtype=np.float64)
        camera_position = np.asarray(camera["position_m"], dtype=np.float64)
        backdrop_coverage = compute_labeler_plane_coverage(
            camera, labeler_geometry, PANEL_SURFACE_Z
        )
        backdrop_overruns = panel_crop_overruns(backdrop_coverage)
        backdrop_spec = high_x_backdrop_spec(backdrop_coverage)
        image_left_side = (
            "+X"
            if backdrop_coverage["image_left_world"][0]
            > backdrop_coverage["image_right_world"][0]
            else "-X"
        )
        print(
            "panel coverage: "
            f"image-left={image_left_side}, "
            f"frame X=[{backdrop_coverage['frame_minimum'][0]:.6f}, "
            f"{backdrop_coverage['frame_maximum'][0]:.6f}], "
            f"crops X=[{backdrop_coverage['crops_minimum'][0]:.6f}, "
            f"{backdrop_coverage['crops_maximum'][0]:.6f}], "
            f"overruns={backdrop_overruns}",
            flush=True,
        )
        render_width, render_height, render_focal, map_x, map_y = build_remap(
            width, height, focal, k1, k2, theta_max, args.width_scale
        )

        from isaacsim import SimulationApp

        kit = SimulationApp(
            {
                "headless": True,
                "renderer": "PathTracing",
                "width": render_width,
                "height": render_height,
            }
        )

        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("omni.replicator.core")
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

        output_images = args.out / "images"
        output_crops = args.out / "crops"
        output_images.mkdir(parents=True, exist_ok=True)
        output_crops.mkdir(parents=True, exist_ok=True)

        context = omni.usd.get_context()
        if not context.open_stage(STAGE_PATH):
            raise RuntimeError(f"Could not open stage: {STAGE_PATH}")
        while context.get_stage_loading_status()[2] > 0:
            kit.update()
        for _ in range(10):
            kit.update()
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        time_code = Usd.TimeCode.Default()

        def translation(vector):
            return Gf.Matrix4d(1.0).SetTranslate(
                Gf.Vec3d(*(float(value) for value in vector))
            )

        def rotation(axis, degrees):
            return Gf.Matrix4d(1.0).SetRotate(
                Gf.Rotation(
                    Gf.Vec3d(*(float(value) for value in axis)), float(degrees)
                )
            )

        def scaling(x_scale, y_scale, z_scale):
            return Gf.Matrix4d(1.0).SetScale(
                Gf.Vec3d(float(x_scale), float(y_scale), float(z_scale))
            )

        def about(pivot, matrix):
            return translation([-value for value in pivot]) * matrix * translation(pivot)

        def set_visible(prim, visible):
            imageable = UsdGeom.Imageable(prim)
            imageable.MakeVisible() if visible else imageable.MakeInvisible()

        # USD cameras look down -Z with +Y up, while the fitted CV camera uses
        # +Z forward and +Y down.
        def usd_camera_matrix(cv_rotation, cv_position):
            x_axis = cv_rotation[0]
            y_axis = -cv_rotation[1]
            z_axis = -cv_rotation[2]
            return Gf.Matrix4d(
                *[
                    float(value)
                    for row in (x_axis, y_axis, z_axis)
                    for value in (*row, 0.0)
                ],
                float(cv_position[0]),
                float(cv_position[1]),
                float(cv_position[2]),
                1.0,
            )

        camera_matrix = usd_camera_matrix(rotation_world_to_cv, camera_position)
        camera_prim = UsdGeom.Camera.Define(stage, "/GenCamV2")
        camera_prim.GetHorizontalApertureAttr().Set(36.0)
        camera_prim.GetVerticalApertureAttr().Set(
            36.0 * render_height / render_width
        )
        camera_prim.GetFocalLengthAttr().Set(render_focal / render_width * 36.0)
        camera_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 20.0))
        camera_transform_op = UsdGeom.Xformable(camera_prim).AddTransformOp()
        camera_transform_op.Set(camera_matrix)

        render_product = rep.create.render_product(
            "/GenCamV2", (render_width, render_height)
        )
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([render_product])
        rep.orchestrator.set_capture_on_play(False)
        kit.update()

        bbox_cache = UsdGeom.BBoxCache(
            time_code, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
        )

        def create_backdrop_slab(spec, debug_material=False):
            """Create and material-bind one collision-checked session-layer slab."""
            if stage.GetEditTarget().GetLayer() != stage.GetSessionLayer():
                raise RuntimeError("Backdrop must be authored in the USD session layer")
            slab_minimum = np.asarray(spec["minimum"], dtype=np.float64)
            slab_maximum = np.asarray(spec["maximum"], dtype=np.float64)
            clearance = 0.00025
            slot_part = re.compile(r"^[GS]\d{2}_(?:cap|tube)$", re.IGNORECASE)
            collisions = []
            for prim in stage.Traverse():
                name = prim.GetName()
                lower_name = name.lower()
                if not (
                    slot_part.fullmatch(name)
                    or lower_name.startswith("tray_")
                    or lower_name.startswith("post")
                ):
                    continue
                try:
                    aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
                    obstacle_minimum = np.asarray(aligned.GetMin(), dtype=np.float64)
                    obstacle_maximum = np.asarray(aligned.GetMax(), dtype=np.float64)
                except Exception:
                    continue
                if not np.all(np.isfinite(obstacle_minimum)) or not np.all(
                    np.isfinite(obstacle_maximum)
                ):
                    continue
                overlaps = np.all(
                    slab_minimum - clearance < obstacle_maximum
                ) and np.all(slab_maximum + clearance > obstacle_minimum)
                if overlaps:
                    collisions.append(str(prim.GetPath()))
            if collisions:
                raise RuntimeError(
                    "Backdrop slab would touch vial/tray/post geometry: "
                    + ", ".join(collisions[:20])
                )

            material_path = spec["material_path"]
            if debug_material:
                material_path = (
                    "/TrayInspection/_materials/GeneratorBackdropDebugMagenta"
                )
                material = UsdShade.Material.Define(stage, material_path)
                shader = UsdShade.Shader.Define(
                    stage, f"{material_path}/PreviewSurface"
                )
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput(
                    "diffuseColor", Sdf.ValueTypeNames.Color3f
                ).Set(Gf.Vec3f(1.0, 0.0, 1.0))
                shader.CreateInput(
                    "emissiveColor", Sdf.ValueTypeNames.Color3f
                ).Set(Gf.Vec3f(4.0, 0.0, 4.0))
                shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.25)
                material.CreateSurfaceOutput().ConnectToSource(
                    shader.ConnectableAPI(), "surface"
                )
            else:
                material = UsdShade.Material.Get(stage, material_path)
            if not material or not material.GetPrim().IsValid():
                raise RuntimeError(
                    f"Backdrop material does not exist: {material_path}"
                )
            backdrop = UsdGeom.Cube.Define(stage, spec["prim_path"])
            backdrop.CreateSizeAttr(1.0)
            backdrop.CreatePurposeAttr(UsdGeom.Tokens.render)
            xform = UsdGeom.Xformable(backdrop.GetPrim())
            parent_world = UsdGeom.Xformable(
                backdrop.GetPrim().GetParent()
            ).ComputeLocalToWorldTransform(time_code)
            parent_world_values = np.asarray(
                [
                    [float(parent_world[row][column]) for column in range(4)]
                    for row in range(4)
                ],
                dtype=np.float64,
            )
            local_matrix_values = local_matrix_for_world_box(
                parent_world_values, spec["center"], spec["size"]
            )
            xform.ClearXformOpOrder()
            transform_op = xform.AddTransformOp()
            transform_op.Set(
                Gf.Matrix4d(
                    *[float(value) for value in local_matrix_values.reshape(-1)]
                )
            )
            binding_strength = (
                UsdShade.Tokens.strongerThanDescendants
                if debug_material
                else UsdShade.Tokens.weakerThanDescendants
            )
            UsdShade.MaterialBindingAPI.Apply(backdrop.GetPrim()).Bind(
                material, binding_strength
            )

            world_matrix = xform.ComputeLocalToWorldTransform(time_code)
            local_corners = [
                Gf.Vec3d(x_value, y_value, z_value)
                for x_value in (-0.5, 0.5)
                for y_value in (-0.5, 0.5)
                for z_value in (-0.5, 0.5)
            ]
            world_corners = np.asarray(
                [world_matrix.Transform(corner) for corner in local_corners],
                dtype=np.float64,
            )
            footprint = project_slab_corners_to_fisheye(camera, world_corners)
            verification_cache = UsdGeom.BBoxCache(
                time_code,
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                useExtentsHint=False,
            )
            actual_box = verification_cache.ComputeWorldBound(
                backdrop.GetPrim()
            ).ComputeAlignedBox()
            actual_minimum = np.asarray(actual_box.GetMin(), dtype=np.float64)
            actual_maximum = np.asarray(actual_box.GetMax(), dtype=np.float64)
            intended_minimum = np.asarray(spec["minimum"], dtype=np.float64)
            intended_maximum = np.asarray(spec["maximum"], dtype=np.float64)
            maximum_error = float(
                max(
                    np.max(np.abs(actual_minimum - intended_minimum)),
                    np.max(np.abs(actual_maximum - intended_maximum)),
                )
            )
            if maximum_error > 0.001:
                raise RuntimeError(
                    "Backdrop composed world bounds do not match the intended box: "
                    f"intendedMin={intended_minimum.tolist()} "
                    f"intendedMax={intended_maximum.tolist()} "
                    f"actualMin={actual_minimum.tolist()} "
                    f"actualMax={actual_maximum.tolist()} "
                    f"maxErrorM={maximum_error:.9f}"
                )
            runtime = {
                "world_minimum": actual_minimum.tolist(),
                "world_maximum": actual_maximum.tolist(),
                "world_bounds_max_error_m": maximum_error,
                "pixel_minimum": footprint["pixel_minimum"].tolist(),
                "pixel_maximum": footprint["pixel_maximum"].tolist(),
                "material_path": material_path,
                "binding_strength": str(binding_strength),
                "active": backdrop.GetPrim().IsActive(),
                "visibility": str(
                    UsdGeom.Imageable(backdrop.GetPrim()).ComputeVisibility(time_code)
                ),
                "purpose": str(backdrop.GetPurposeAttr().Get(time_code)),
            }
            print(
                f"created backdrop {spec['prim_path']} center={spec['center']} "
                f"size={spec['size']} material={material_path} "
                f"bindingStrength={binding_strength} active={runtime['active']} "
                f"visibility={runtime['visibility']} purpose={runtime['purpose']} "
                f"actualWorldMin={runtime['world_minimum']} "
                f"actualWorldMax={runtime['world_maximum']} "
                f"worldBoundsMaxErrorM={runtime['world_bounds_max_error_m']:.9f} "
                f"pixelBBox=[{runtime['pixel_minimum']}, {runtime['pixel_maximum']}]",
                flush=True,
            )
            return runtime

        backdrop_runtime = None
        if not args.no_backdrop:
            backdrop_runtime = create_backdrop_slab(
                backdrop_spec, debug_material=args.backdrop_debug
            )

        def create_floor_box(spec, albedo, emissive):
            """Create a collision-checked session-layer floor in world coordinates."""
            if stage.GetEditTarget().GetLayer() != stage.GetSessionLayer():
                raise RuntimeError("Floor must be authored in the USD session layer")
            floor_minimum = np.asarray(spec["minimum"], dtype=np.float64)
            floor_maximum = np.asarray(spec["maximum"], dtype=np.float64)
            collision_cache = UsdGeom.BBoxCache(
                time_code,
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                useExtentsHint=False,
            )
            slot_part = re.compile(r"^[GS]\d{2}_(?:cap|tube)$", re.IGNORECASE)
            collisions = []
            for prim in stage.Traverse():
                name = prim.GetName()
                lower_name = name.lower()
                if not (
                    slot_part.fullmatch(name)
                    or lower_name.startswith("tray_")
                    or lower_name.startswith("post")
                    or "steel" in lower_name
                ):
                    continue
                try:
                    aligned = collision_cache.ComputeWorldBound(
                        prim
                    ).ComputeAlignedBox()
                    obstacle_minimum = np.asarray(aligned.GetMin(), dtype=np.float64)
                    obstacle_maximum = np.asarray(aligned.GetMax(), dtype=np.float64)
                except Exception:
                    continue
                if not np.all(np.isfinite(obstacle_minimum)) or not np.all(
                    np.isfinite(obstacle_maximum)
                ):
                    continue
                overlaps = np.all(floor_minimum < obstacle_maximum) and np.all(
                    floor_maximum > obstacle_minimum
                )
                if overlaps:
                    collisions.append(str(prim.GetPath()))
            if collisions:
                raise RuntimeError(
                    "Floor would intersect vial/tray/post/steel geometry: "
                    + ", ".join(collisions[:20])
                )

            material = UsdShade.Material.Define(stage, spec["material_path"])
            shader = UsdShade.Shader.Define(
                stage, f"{spec['material_path']}/PreviewSurface"
            )
            shader.CreateIdAttr("UsdPreviewSurface")
            material_settings = floor_material_inputs(albedo, emissive)
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*material_settings["diffuseColor"])
            )
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
                material_settings["roughness"]
            )
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
                material_settings["metallic"]
            )
            if "emissiveColor" in material_settings:
                shader.CreateInput(
                    "emissiveColor", Sdf.ValueTypeNames.Color3f
                ).Set(Gf.Vec3f(*material_settings["emissiveColor"]))
            material.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface"
            )

            floor = UsdGeom.Cube.Define(stage, spec["prim_path"])
            floor.CreateSizeAttr(1.0)
            floor.CreatePurposeAttr(UsdGeom.Tokens.render)
            xform = UsdGeom.Xformable(floor.GetPrim())
            parent_world = UsdGeom.Xformable(
                floor.GetPrim().GetParent()
            ).ComputeLocalToWorldTransform(time_code)
            parent_world_values = np.asarray(
                [
                    [float(parent_world[row][column]) for column in range(4)]
                    for row in range(4)
                ],
                dtype=np.float64,
            )
            local_matrix_values = local_matrix_for_world_box(
                parent_world_values, spec["center"], spec["size"]
            )
            xform.ClearXformOpOrder()
            transform_op = xform.AddTransformOp()
            transform_op.Set(
                Gf.Matrix4d(
                    *[float(value) for value in local_matrix_values.reshape(-1)]
                )
            )
            UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(
                material, UsdShade.Tokens.strongerThanDescendants
            )

            world_matrix = xform.ComputeLocalToWorldTransform(time_code)
            local_corners = [
                Gf.Vec3d(x_value, y_value, z_value)
                for x_value in (-0.5, 0.5)
                for y_value in (-0.5, 0.5)
                for z_value in (-0.5, 0.5)
            ]
            world_corners = np.asarray(
                [world_matrix.Transform(corner) for corner in local_corners],
                dtype=np.float64,
            )
            footprint = project_slab_corners_to_fisheye(camera, world_corners)
            verification_cache = UsdGeom.BBoxCache(
                time_code,
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                useExtentsHint=False,
            )
            actual_box = verification_cache.ComputeWorldBound(
                floor.GetPrim()
            ).ComputeAlignedBox()
            actual_minimum = np.asarray(actual_box.GetMin(), dtype=np.float64)
            actual_maximum = np.asarray(actual_box.GetMax(), dtype=np.float64)
            intended_minimum = np.asarray(spec["minimum"], dtype=np.float64)
            intended_maximum = np.asarray(spec["maximum"], dtype=np.float64)
            maximum_error = float(
                max(
                    np.max(np.abs(actual_minimum - intended_minimum)),
                    np.max(np.abs(actual_maximum - intended_maximum)),
                )
            )
            if maximum_error > 0.001:
                raise RuntimeError(
                    "Floor composed world bounds do not match the intended box: "
                    f"intendedMin={intended_minimum.tolist()} "
                    f"intendedMax={intended_maximum.tolist()} "
                    f"actualMin={actual_minimum.tolist()} "
                    f"actualMax={actual_maximum.tolist()} "
                    f"maxErrorM={maximum_error:.9f}"
                )
            runtime = {
                "world_minimum": actual_minimum.tolist(),
                "world_maximum": actual_maximum.tolist(),
                "world_bounds_max_error_m": maximum_error,
                "pixel_minimum": footprint["pixel_minimum"].tolist(),
                "pixel_maximum": footprint["pixel_maximum"].tolist(),
                "albedo": [float(value) for value in albedo],
                "emissive": [float(value) for value in emissive],
            }
            print(
                f"created floor {spec['prim_path']} albedo={runtime['albedo']} "
                f"emissive={runtime['emissive']} "
                f"actualWorldMin={runtime['world_minimum']} "
                f"actualWorldMax={runtime['world_maximum']} "
                f"worldBoundsMaxErrorM={maximum_error:.9f} "
                f"pixelBBox=[{runtime['pixel_minimum']}, {runtime['pixel_maximum']}]",
                flush=True,
            )
            return runtime

        floor_specification = None
        floor_coverage = None
        floor_runtime = None
        approximate_uncovered = []

        def bounds_for(prim):
            bounds = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            return (
                np.asarray(bounds.GetMin(), dtype=np.float64),
                np.asarray(bounds.GetMax(), dtype=np.float64),
            )

        template_parts = {}
        for part in ("tube", "cap"):
            prim = stage.GetPrimAtPath(f"/TrayInspection/G01_{part}")
            if not prim.IsValid():
                raise RuntimeError(f"Missing S clone template: G01_{part}")
            minimum, maximum = bounds_for(prim)
            template_parts[part] = {
                "prim": prim,
                "world": UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                    time_code
                ),
                "minimum": minimum,
                "maximum": maximum,
            }
        template_cap_center = (
            template_parts["cap"]["minimum"] + template_parts["cap"]["maximum"]
        ) / 2.0
        fit_zmode = str(camera["zmode"])
        if fit_zmode == "cap_min":
            template_label_z = float(template_parts["cap"]["minimum"][2])
        elif fit_zmode == "tube_max":
            template_label_z = float(template_parts["tube"]["maximum"][2])
        elif fit_zmode == "cap_center":
            template_label_z = float(template_cap_center[2])
        elif fit_zmode == "tube_min":
            template_label_z = float(template_parts["tube"]["minimum"][2])
        else:
            raise ValueError(f"Unsupported camera zmode: {fit_zmode}")
        template_label_point = np.array(
            [template_cap_center[0], template_cap_center[1], template_label_z],
            dtype=np.float64,
        )

        def pixel_on_world_plane(pixel, world_z):
            normalized = fisheye_pixels_to_pinhole_normalized(
                np.asarray([pixel], dtype=np.float64), focal, k1, k2
            )[0]
            camera_ray = np.array(
                [normalized[0], normalized[1], 1.0], dtype=np.float64
            )
            world_ray = rotation_world_to_cv.T @ camera_ray
            if abs(float(world_ray[2])) < 1.0e-9:
                raise RuntimeError("S slot ray is parallel to the rack plane")
            distance = (world_z - camera_position[2]) / world_ray[2]
            if distance <= 0.0:
                raise RuntimeError("S slot plane lies behind the fitted camera")
            return camera_position + distance * world_ray

        for name in S_SLOT_NAMES:
            for part in ("tube", "cap"):
                authored = stage.GetPrimAtPath(f"/TrayInspection/{name}_{part}")
                if authored.IsValid():
                    set_visible(authored, False)

        synthetic_s_root = UsdGeom.Xform.Define(stage, "/SyntheticSFixtures")
        set_visible(synthetic_s_root.GetPrim(), True)
        s_prim_paths = {}
        for name in S_SLOT_NAMES:
            target_point = pixel_on_world_plane(
                labeler_geometry["fixtures"]["S"]["centers"][name],
                template_label_z,
            )
            delta = target_point - template_label_point
            s_prim_paths[name] = {}
            for part in ("tube", "cap"):
                path = f"/SyntheticSFixtures/{name}_{part}"
                clone = stage.DefinePrim(path, "Xform")
                clone.GetReferences().AddInternalReference(
                    f"/TrayInspection/G01_{part}"
                )
                clone.SetInstanceable(False)
                xform = UsdGeom.Xformable(clone)
                xform.ClearXformOpOrder()
                transform_op = xform.AddTransformOp()
                transform_op.Set(
                    template_parts[part]["world"] * translation(delta)
                )
                set_visible(clone, True)
                s_prim_paths[name][part] = path
        kit.update()
        bbox_cache.Clear()

        station_bounds_cache = UsdGeom.BBoxCache(
            time_code,
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=False,
        )
        station_root = stage.GetPrimAtPath("/TrayInspection")
        station_box = station_bounds_cache.ComputeWorldBound(
            station_root
        ).ComputeAlignedBox()
        station_minimum_z = float(station_box.GetMin()[2])
        floor_specification, floor_coverage = floor_box_spec(
            camera,
            labeler_geometry,
            station_minimum_z,
        )
        if not args.no_floor:
            floor_runtime = create_floor_box(
                floor_specification, args.floor_albedo, args.floor_emissive
            )

        authored_pixel_boxes = []
        authored_bounds_cache = UsdGeom.BBoxCache(
            time_code,
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=False,
        )
        excluded_paths = {BACKDROP_PRIM_PATH, FLOOR_PRIM_PATH}
        for prim in stage.Traverse():
            if str(prim.GetPath()) in excluded_paths or not prim.IsA(UsdGeom.Boundable):
                continue
            try:
                aligned = authored_bounds_cache.ComputeWorldBound(
                    prim
                ).ComputeAlignedBox()
                minimum = np.asarray(aligned.GetMin(), dtype=np.float64)
                maximum = np.asarray(aligned.GetMax(), dtype=np.float64)
                if not np.all(np.isfinite(minimum)) or not np.all(
                    np.isfinite(maximum)
                ):
                    continue
                corners = [
                    [x_value, y_value, z_value]
                    for x_value in (minimum[0], maximum[0])
                    for y_value in (minimum[1], maximum[1])
                    for z_value in (minimum[2], maximum[2])
                ]
                footprint = project_slab_corners_to_fisheye(camera, corners)
            except (ValueError, RuntimeError):
                continue
            authored_pixel_boxes.append(
                (footprint["pixel_minimum"], footprint["pixel_maximum"])
            )
        approximate_uncovered = approximate_uncovered_crop_corners(
            camera,
            labeler_geometry,
            PANEL_BOUNDS_XY,
            None if args.no_backdrop else backdrop_spec,
            None if args.no_floor else floor_specification,
            authored_pixel_boxes=tuple(authored_pixel_boxes),
        )
        print(
            "approximate crop background coverage (planar extents/projected AABBs; "
            "occlusion ignored): "
            f"authoredBounds={len(authored_pixel_boxes)} "
            f"notFullyCovered={approximate_uncovered}",
            flush=True,
        )

        slot_prim_paths = {
            **{
                name: {
                    part: f"/TrayInspection/{name}_{part}"
                    for part in ("tube", "cap")
                }
                for name in G_SLOT_NAMES
            },
            **s_prim_paths,
        }
        slot_info = {}
        for name in ALL_SLOT_NAMES:
            record = {}
            for part in ("tube", "cap"):
                prim_path = slot_prim_paths[name][part]
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    raise RuntimeError(f"Missing prim: {prim_path}")
                if part == "tube" and prim.IsInstance():
                    prim.SetInstanceable(False)
                    kit.update()
                world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time_code)
                parent_inverse = (
                    UsdGeom.Xformable(prim.GetParent())
                    .ComputeLocalToWorldTransform(time_code)
                    .GetInverse()
                )
                minimum, maximum = bounds_for(prim)
                record[part] = {
                    "prim": prim,
                    "world": world,
                    "parent_inverse": parent_inverse,
                    "minimum": minimum,
                    "maximum": maximum,
                }
            slot_info[name] = record

        # Cache the footprint used by direction-aware collision checks. A
        # fixed circular-cell limit can be zero even when a direction away
        # from occupied neighbors has ample room.
        slot_centers = {
            name: (record["tube"]["minimum"][:2] + record["tube"]["maximum"][:2])
            / 2.0
            for name, record in slot_info.items()
        }
        for name, record in slot_info.items():
            cap_size = (
                record["cap"]["maximum"][:2] - record["cap"]["minimum"][:2]
            )
            tube_size = (
                record["tube"]["maximum"][:2] - record["tube"]["minimum"][:2]
            )
            record["cap"]["radius"] = float(np.max(cap_size) / 2.0)
            record["tube"]["radius"] = float(np.max(tube_size) / 2.0)

        # Replace authored transform stacks with one session-layer transform,
        # matching the v1 generator while leaving source USD untouched.
        for record in slot_info.values():
            for part in ("tube", "cap"):
                item = record[part]
                xform = UsdGeom.Xformable(item["prim"])
                xform.ClearXformOpOrder()
                item["transform_op"] = xform.AddTransformOp()
                item["transform_op"].Set(
                    item["world"] * item["parent_inverse"]
                )

        # Author per-slot mesh point arrays. Overrides at distinct prim paths
        # keep one vial's damage from modifying any other vial.
        for name, record in slot_info.items():
            meshes = []
            for prim in Usd.PrimRange(record["tube"]["prim"]):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                mesh = UsdGeom.Mesh(prim)
                points = mesh.GetPointsAttr().Get(time_code)
                if points is None or len(points) == 0:
                    continue
                original = np.asarray(points, dtype=np.float64)
                mesh.GetPointsAttr().Set(
                    [Gf.Vec3f(*(float(value) for value in point)) for point in original]
                )
                meshes.append({"mesh": mesh, "points": original})
            record["tube"]["meshes"] = meshes

        fallback_root = UsdGeom.Xform.Define(stage, "/SyntheticDamageFallback")
        set_visible(fallback_root.GetPrim(), True)
        for name, record in slot_info.items():
            tube = record["tube"]
            tube_minimum = tube["minimum"]
            tube_maximum = tube["maximum"]
            center_xy = (tube_minimum[:2] + tube_maximum[:2]) / 2.0
            radius = float(min(tube_maximum[0] - tube_minimum[0], tube_maximum[1] - tube_minimum[1]) / 2.0)
            total_height = float(tube_maximum[2] - tube_minimum[2])
            segments = []
            if not tube["meshes"]:
                for segment_index in range(3):
                    cylinder = UsdGeom.Cylinder.Define(
                        stage,
                        f"/SyntheticDamageFallback/{name}_segment_{segment_index}",
                    )
                    cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
                    cylinder.CreateRadiusAttr(radius)
                    cylinder.CreateHeightAttr(total_height / 3.0)
                    cylinder.CreateDisplayColorAttr([Gf.Vec3f(0.72, 0.82, 0.90)])
                    transform_op = UsdGeom.Xformable(cylinder).AddTransformOp()
                    set_visible(cylinder.GetPrim(), False)
                    segments.append(
                        {
                            "prim": cylinder.GetPrim(),
                            "op": transform_op,
                            "index": segment_index,
                            "center_xy": center_xy,
                            "base_z": float(tube_minimum[2]),
                            "height": total_height,
                            "radius": radius,
                        }
                    )
            tube["fallback_segments"] = segments
            tube["deformation_backend"] = (
                "mesh_points" if tube["meshes"] else "stacked_segments"
            )

        backend_counts = {}
        for record in slot_info.values():
            backend = record["tube"]["deformation_backend"]
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
        print(f"deformation backends: {backend_counts}", flush=True)

        def project(world_point, cv_rotation, cv_position):
            return project_world(
                np.asarray([world_point], dtype=np.float64),
                cv_rotation,
                cv_position,
                focal,
                k1,
                k2,
            )[0].tolist()

        fit_zmode = str(camera["zmode"])
        label_world_points = {}
        for name, record in slot_info.items():
            cap = record["cap"]
            tube = record["tube"]
            cap_center = (cap["minimum"] + cap["maximum"]) / 2.0
            if fit_zmode == "cap_min":
                z_value = float(cap["minimum"][2])
            elif fit_zmode == "tube_max":
                z_value = float(tube["maximum"][2])
            elif fit_zmode == "cap_center":
                z_value = float(cap_center[2])
            elif fit_zmode == "tube_min":
                z_value = float(tube["minimum"][2])
            else:
                raise ValueError(f"Unsupported camera zmode: {fit_zmode}")
            label_world_points[name] = np.array(
                [cap_center[0], cap_center[1], z_value], dtype=np.float64
            )
        nominal_projected_centers = {
            name: project(point, rotation_world_to_cv, camera_position)
            for name, point in label_world_points.items()
        }
        label_world_array = np.vstack(list(label_world_points.values()))
        nominal_camera_points = (
            rotation_world_to_cv @ (label_world_array - camera_position).T
        ).T
        reference_depth = float(nominal_camera_points[:, 2].mean())

        def restore_meshes(tube):
            for mesh_record in tube["meshes"]:
                mesh_record["mesh"].GetPointsAttr().Set(
                    [
                        Gf.Vec3f(*(float(value) for value in point))
                        for point in mesh_record["points"]
                    ]
                )

        def hide_fallback(tube):
            for segment in tube["fallback_segments"]:
                set_visible(segment["prim"], False)

        def candidate_direction(name, labels, scene_rng, attempt):
            center = slot_centers[name]
            empty_slots = sorted(
                (
                    (float(np.linalg.norm(other_center - center)), other_center)
                    for other_name, other_center in slot_centers.items()
                    if other_name != name and labels[other_name] == "no_vial"
                ),
                key=lambda item: item[0],
            )
            if attempt < min(3, len(empty_slots)):
                vector = empty_slots[attempt][1] - center
            elif attempt == 3:
                vector = np.zeros(2, dtype=np.float64)
                for other_name, other_center in slot_centers.items():
                    if other_name == name or labels[other_name] == "no_vial":
                        continue
                    delta = other_center - center
                    distance = max(float(np.linalg.norm(delta)), 1.0e-6)
                    vector -= delta / distance**2
            else:
                angle = scene_rng.uniform(0.0, 2.0 * math.pi)
                vector = np.array([math.cos(angle), math.sin(angle)])
            norm = float(np.linalg.norm(vector))
            if norm < 1.0e-9:
                angle = scene_rng.uniform(0.0, 2.0 * math.pi)
                return np.array([math.cos(angle), math.sin(angle)])
            return vector / norm

        def maximum_safe_shift(name, labels, direction):
            center = slot_centers[name]
            moving_radius = slot_info[name]["cap"]["radius"]
            limit = float("inf")
            for other_name, other_center in slot_centers.items():
                if other_name == name or labels[other_name] == "no_vial":
                    continue
                other_part = (
                    "tube" if labels[other_name] == "missing_cap" else "cap"
                )
                collision_radius = (
                    moving_radius
                    + slot_info[other_name][other_part]["radius"]
                    + 2.0 * MAX_SEATED_CAP_OFFSET_M
                )
                delta = other_center - center
                projection = float(np.dot(delta, direction))
                if projection <= 0.0:
                    continue
                perpendicular_sq = max(
                    0.0, float(np.dot(delta, delta)) - projection**2
                )
                if perpendicular_sq >= collision_radius**2:
                    continue
                entry = projection - math.sqrt(
                    max(0.0, collision_radius**2 - perpendicular_sq)
                )
                limit = min(limit, max(0.0, entry))
            return limit

        def damage_crop_projection(name, plan, crop_context):
            tube = slot_info[name]["tube"]
            center = (tube["minimum"] + tube["maximum"]) / 2.0
            tip = np.array(
                [center[0], center[1], tube["maximum"][2]],
                dtype=np.float64,
            )
            direction = np.asarray(plan["direction"], dtype=np.float64)
            kind = plan["kind"]
            if kind in ("crush", "lean_crush"):
                shift = float(plan["band_shift_m"]) * direction
                center[:2] += shift
                tip[:2] += shift
            elif kind == "bend":
                center[:2] += 0.25 * float(plan["amplitude"]) * direction
                tip[:2] += float(plan["amplitude"]) * direction
            elif kind == "broken_top":
                tube_height = float(tube["maximum"][2] - tube["minimum"][2])
                tip[2] = tube["minimum"][2] + (
                    1.0 - float(plan["hidden_fraction"])
                ) * tube_height

            transform = crop_context["pre_damage_matrix"]
            if kind == "lean_crush":
                base_center = np.array(
                    [
                        (tube["minimum"][0] + tube["maximum"][0]) / 2.0,
                        (tube["minimum"][1] + tube["maximum"][1]) / 2.0,
                        tube["minimum"][2],
                    ]
                )
                lean_axis = (-direction[1], direction[0], 0.0)
                transform = transform * about(
                    base_center,
                    rotation(lean_axis, plan["tilt_degrees"]),
                )

            center_world = transform.Transform(Gf.Vec3d(*center))
            tip_world = transform.Transform(Gf.Vec3d(*tip))
            center_pixel = project(
                center_world,
                crop_context["cv_rotation"],
                crop_context["cv_position"],
            )
            tip_pixel = project(
                tip_world,
                crop_context["cv_rotation"],
                crop_context["cv_position"],
            )
            return projected_points_crop_check(
                center_pixel,
                tip_pixel,
                crop_context["crop_center"],
                crop_context["crop_offsets"],
            )

        def sample_damage_plan(
            name,
            labels,
            scene_rng,
            defect_boost,
            forced_kind=None,
            crop_context=None,
        ):
            if forced_kind is not None and forced_kind not in damage_sampler_types:
                raise ValueError(f"Unknown forced damage kind: {forced_kind!r}")
            tube = slot_info[name]["tube"]
            tube_height = max(
                float(tube["maximum"][2] - tube["minimum"][2]), 1.0e-6
            )
            for attempt in range(MAX_DAMAGE_ATTEMPTS):
                sampled_kind = scene_rng.choices(
                    damage_sampler_types, weights=damage_sampler_weights, k=1
                )[0]
                kind = forced_kind if forced_kind is not None else sampled_kind
                direction = candidate_direction(name, labels, scene_rng, attempt)
                safe_shift = maximum_safe_shift(name, labels, direction)
                plan = {
                    "kind": kind,
                    "boost": defect_boost,
                    "direction": direction,
                    "attempt": attempt + 1,
                }
                if kind in ("crush", "lean_crush"):
                    radius_scale = max(
                        0.15,
                        1.0
                        - (1.0 - scene_rng.uniform(0.35, 0.60))
                        * defect_boost,
                    )
                    desired_shift = (
                        scene_rng.uniform(0.004, 0.008) * defect_boost
                    )
                    shift_limit = safe_shift
                    if kind == "lean_crush":
                        minimum_lean_shift = tube_height * math.sin(
                            math.radians(MIN_LEAN_DEGREES)
                        )
                        shift_limit -= minimum_lean_shift
                    band_shift = min(desired_shift, shift_limit)
                    if band_shift < MIN_CRUSH_SHIFT_M:
                        continue
                    plan.update(
                        radius_scale=radius_scale,
                        squeeze_amount=1.0 - radius_scale,
                        band_shift_m=band_shift,
                        strength=1.0 - radius_scale,
                    )
                    if kind == "lean_crush":
                        remaining_shift = max(0.0, safe_shift - band_shift)
                        max_tilt = math.degrees(
                            math.asin(min(1.0, remaining_shift / tube_height))
                        )
                        tilt_degrees = min(
                            scene_rng.uniform(15.0, 25.0), max_tilt
                        )
                        if tilt_degrees < MIN_LEAN_DEGREES:
                            continue
                        plan.update(
                            tilt_degrees=tilt_degrees,
                            strength=tilt_degrees,
                            crush_strength=1.0 - radius_scale,
                        )
                        crop_check = damage_crop_projection(
                            name, plan, crop_context
                        )
                        if not crop_check["inside"]:
                            continue
                        plan["crop_check"] = crop_check
                    else:
                        plan["crop_check"] = damage_crop_projection(
                            name, plan, crop_context
                        )
                    return plan
                if kind == "bend":
                    amplitude = min(
                        scene_rng.uniform(0.009, 0.015) * defect_boost,
                        safe_shift,
                    )
                    if amplitude < MIN_BEND_SHIFT_M:
                        continue
                    plan.update(amplitude=amplitude, strength=amplitude)
                    plan["crop_check"] = damage_crop_projection(
                        name, plan, crop_context
                    )
                    return plan
                hidden_fraction = scene_rng.uniform(0.30, 0.50)
                plan.update(
                    hidden_fraction=hidden_fraction,
                    cap_tilt_degrees=scene_rng.uniform(10.0, 20.0),
                    strength=hidden_fraction,
                )
                plan["crop_check"] = damage_crop_projection(
                    name, plan, crop_context
                )
                return plan
            return None

        def deform_mesh_points(tube, plan):
            kind = plan["kind"]
            direction = plan["direction"]
            mouth_shift = np.zeros(2, dtype=np.float64)
            mouth_tilt = 0.0
            tube_height = max(
                float(tube["maximum"][2] - tube["minimum"][2]), 1.0e-6
            )
            if kind in ("crush", "lean_crush"):
                mouth_shift = plan["band_shift_m"] * direction
            elif kind == "bend":
                mouth_shift = plan["amplitude"] * direction
                mouth_tilt = math.degrees(
                    math.atan2(2.0 * plan["amplitude"], tube_height)
                )

            for mesh_record in tube["meshes"]:
                original = mesh_record["points"]
                changed = original.copy()
                z_min = float(original[:, 2].min())
                z_max = float(original[:, 2].max())
                height_local = max(z_max - z_min, 1.0e-6)
                normalized_height = np.clip(
                    (original[:, 2] - z_min) / height_local, 0.0, 1.0
                )
                center_xy = np.median(original[:, :2], axis=0)
                radial = original[:, :2] - center_xy
                if kind in ("crush", "lean_crush"):
                    upper_weight = np.clip(
                        (normalized_height - 0.40) / 0.10, 0.0, 1.0
                    )
                    upper_weight = upper_weight * upper_weight * (
                        3.0 - 2.0 * upper_weight
                    )
                    changed[:, :2] = (
                        center_xy
                        + radial
                        * (1.0 - plan["squeeze_amount"] * upper_weight)[:, None]
                        + (plan["band_shift_m"] * upper_weight)[:, None]
                        * direction
                    )
                elif kind == "bend":
                    curve = normalized_height * normalized_height
                    changed[:, :2] += (
                        curve[:, None] * plan["amplitude"] * direction
                    )
                elif kind == "broken_top":
                    cut_height = 1.0 - plan["hidden_fraction"]
                    above_cut = normalized_height > cut_height
                    changed[above_cut, 2] = z_min + cut_height * height_local
                else:
                    raise ValueError(f"Unknown damage kind: {kind}")
                mesh_record["mesh"].GetPointsAttr().Set(
                    [
                        Gf.Vec3f(*(float(value) for value in point))
                        for point in changed
                    ]
                )
            return mouth_shift, mouth_tilt

        def show_stacked_fallback(tube, plan):
            kind = plan["kind"]
            direction = plan["direction"]
            mouth_shift = np.zeros(2, dtype=np.float64)
            band_shift = np.zeros(2, dtype=np.float64)
            if kind in ("crush", "lean_crush"):
                band_shift = plan["band_shift_m"] * direction
                mouth_shift = band_shift.copy()
            elif kind == "bend":
                mouth_shift = plan["amplitude"] * direction

            for segment in tube["fallback_segments"]:
                fraction = (segment["index"] + 0.5) / 3.0
                upper = float(segment["index"] >= 1)
                if kind == "broken_top":
                    cut_height = 1.0 - plan["hidden_fraction"]
                    segment_start = segment["index"] / 3.0
                    visible_fraction = np.clip(
                        (cut_height - segment_start) * 3.0, 0.0, 1.0
                    )
                    if visible_fraction <= 0.0:
                        set_visible(segment["prim"], False)
                        continue
                    center_fraction = segment_start + visible_fraction / 6.0
                    center = [
                        float(segment["center_xy"][0]),
                        float(segment["center_xy"][1]),
                        float(
                            segment["base_z"]
                            + segment["height"] * center_fraction
                        ),
                    ]
                    segment["op"].Set(
                        scaling(1.0, 1.0, float(visible_fraction))
                        * translation(center)
                    )
                    set_visible(segment["prim"], True)
                    continue
                curved_shift = (
                    mouth_shift * fraction**2
                    if kind == "bend"
                    else np.zeros(2, dtype=np.float64)
                )
                center = [
                    float(
                        segment["center_xy"][0]
                        + curved_shift[0]
                        + band_shift[0] * upper
                    ),
                    float(
                        segment["center_xy"][1]
                        + curved_shift[1]
                        + band_shift[1] * upper
                    ),
                    float(segment["base_z"] + segment["height"] * fraction),
                ]
                bend_degrees = (
                    math.degrees(
                        math.atan2(
                            2.0 * np.linalg.norm(mouth_shift) * fraction,
                            segment["height"],
                        )
                    )
                    if kind == "bend"
                    else 0.0
                )
                bend_axis = (-direction[1], direction[0], 0.0)
                segment_scale = (
                    plan["radius_scale"]
                    if kind in ("crush", "lean_crush")
                    and segment["index"] >= 1
                    else 1.0
                )
                segment["op"].Set(
                    scaling(segment_scale, segment_scale, 1.0)
                    * rotation(bend_axis, bend_degrees)
                    * translation(center)
                )
                set_visible(segment["prim"], True)
            mouth_tilt = (
                math.degrees(
                    math.atan2(
                        2.0 * np.linalg.norm(mouth_shift),
                        tube["maximum"][2] - tube["minimum"][2],
                    )
                )
                if kind == "bend"
                else 0.0
            )
            return mouth_shift, mouth_tilt

        def apply_slot(
            name,
            labels,
            scene_rng,
            defect_boost,
            forced_damage_kind=None,
            crop_context=None,
        ):
            label = labels[name]
            record = slot_info[name]
            tube = record["tube"]
            cap = record["cap"]
            restore_meshes(tube)
            hide_fallback(tube)
            tube_world = tube["world"]
            cap_world = cap["world"]

            base_center = np.array(
                [
                    (tube["minimum"][0] + tube["maximum"][0]) / 2.0,
                    (tube["minimum"][1] + tube["maximum"][1]) / 2.0,
                    tube["minimum"][2],
                ]
            )
            mouth_center = np.array(
                [
                    (cap["minimum"][0] + cap["maximum"][0]) / 2.0,
                    (cap["minimum"][1] + cap["maximum"][1]) / 2.0,
                    cap["minimum"][2],
                ]
            )
            lean_azimuth = scene_rng.uniform(0.0, 2.0 * math.pi)
            lean_axis = (math.cos(lean_azimuth), math.sin(lean_azimuth), 0.0)
            lean_degrees = scene_rng.uniform(0.0, 6.0)
            lean_matrix = about(base_center, rotation(lean_axis, lean_degrees))
            tube_world = tube_world * lean_matrix
            cap_world = cap_world * lean_matrix

            cap_yaw = scene_rng.uniform(-180.0, 180.0)
            offset_radius = scene_rng.uniform(0.0, MAX_SEATED_CAP_OFFSET_M)
            offset_angle = scene_rng.uniform(0.0, 2.0 * math.pi)
            cap_offset = (
                offset_radius * math.cos(offset_angle),
                offset_radius * math.sin(offset_angle),
                0.0,
            )
            cap_world = (
                cap_world
                * about(mouth_center, rotation((0.0, 0.0, 1.0), cap_yaw))
                * translation(cap_offset)
            )
            tube_visible = True
            cap_visible = True
            variation = {
                "lean_deg": lean_degrees,
                "cap_yaw_deg": cap_yaw,
                "cap_offset_m": list(cap_offset[:2]),
            }

            if label == "no_vial":
                tube_visible = False
                cap_visible = False
            elif label == "missing_cap":
                cap_visible = False
            elif label == "crooked_cap":
                azimuth = scene_rng.uniform(0.0, 2.0 * math.pi)
                tilt_axis = (math.cos(azimuth), math.sin(azimuth), 0.0)
                rim_direction = (-math.sin(azimuth), math.cos(azimuth), 0.0)
                rim_radius = float(
                    min(
                        cap["maximum"][0] - cap["minimum"][0],
                        cap["maximum"][1] - cap["minimum"][1],
                    )
                    / 2.0
                )
                rim_pivot = mouth_center + rim_radius * np.asarray(rim_direction)
                crooked_degrees = scene_rng.uniform(12.0, 35.0)
                cap_world = cap_world * about(
                    rim_pivot, rotation(tilt_axis, crooked_degrees)
                )
                variation["crooked_tilt_deg"] = crooked_degrees
            elif label == "damaged_vial":
                damage_crop_context = dict(crop_context)
                damage_crop_context["pre_damage_matrix"] = lean_matrix
                plan = sample_damage_plan(
                    name,
                    labels,
                    scene_rng,
                    defect_boost,
                    forced_kind=forced_damage_kind,
                    crop_context=damage_crop_context,
                )
                if plan is None:
                    labels[name] = "pass"
                    variation["damage_relabel_reason"] = (
                        f"no valid deformation after {MAX_DAMAGE_ATTEMPTS} attempts"
                    )
                else:
                    damage_kind = plan["kind"]
                    damage_direction = plan["direction"]
                    if damage_kind == "lean_crush":
                        lean_axis = (
                            -damage_direction[1],
                            damage_direction[0],
                            0.0,
                        )
                        damage_lean = about(
                            base_center,
                            rotation(lean_axis, plan["tilt_degrees"]),
                        )
                        tube_world = tube_world * damage_lean
                        cap_world = cap_world * damage_lean
                    if tube["deformation_backend"] == "mesh_points":
                        mouth_shift, mouth_tilt = deform_mesh_points(tube, plan)
                    else:
                        tube_visible = False
                        mouth_shift, mouth_tilt = show_stacked_fallback(tube, plan)
                    if np.linalg.norm(mouth_shift) > 0.0:
                        bend_axis = (
                            -damage_direction[1],
                            damage_direction[0],
                            0.0,
                        )
                        shifted_mouth = mouth_center + np.array(
                            [mouth_shift[0], mouth_shift[1], 0.0]
                        )
                        cap_world = (
                            cap_world
                            * translation((mouth_shift[0], mouth_shift[1], 0.0))
                            * about(
                                shifted_mouth,
                                rotation(bend_axis, mouth_tilt),
                            )
                        )
                    if damage_kind == "broken_top":
                        cap_axis = (
                            -damage_direction[1],
                            damage_direction[0],
                            0.0,
                        )
                        rim_pivot = mouth_center + cap["radius"] * np.array(
                            [damage_direction[0], damage_direction[1], 0.0]
                        )
                        cap_world = cap_world * about(
                            rim_pivot,
                            rotation(cap_axis, plan["cap_tilt_degrees"]),
                        )
                    damage = {
                        key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in plan.items()
                    }
                    variation["damage"] = damage
                    variation["deformation_backend"] = tube[
                        "deformation_backend"
                    ]

            tube["transform_op"].Set(tube_world * tube["parent_inverse"])
            cap["transform_op"].Set(cap_world * cap["parent_inverse"])
            set_visible(tube["prim"], tube_visible)
            set_visible(cap["prim"], cap_visible)
            return variation

        key_light = stage.GetPrimAtPath(
            "/TrayInspection/key_overhead/key_overhead"
        )
        fill_light = stage.GetPrimAtPath("/TrayInspection/fill_side/fill_side")
        if not key_light.IsValid() or not fill_light.IsValid():
            raise RuntimeError("Expected key and fill light prims are missing")

        def prepare_light(prim):
            intensity = float(prim.GetAttribute("inputs:intensity").Get())
            xform = UsdGeom.Xformable(prim)
            world = xform.ComputeLocalToWorldTransform(time_code)
            parent_inverse = (
                UsdGeom.Xformable(prim.GetParent())
                .ComputeLocalToWorldTransform(time_code)
                .GetInverse()
            )
            xform.ClearXformOpOrder()
            transform_op = xform.AddTransformOp()
            transform_op.Set(world * parent_inverse)
            return {
                "prim": prim,
                "base_intensity": intensity,
                "world": world,
                "parent_inverse": parent_inverse,
                "transform_op": transform_op,
            }

        key_record = prepare_light(key_light)
        fill_record = prepare_light(fill_light)

        def randomize_light(light_record, intensity_range, scene_rng):
            sample = sample_light_settings(
                scene_rng,
                intensity_range,
                args.light_profile,
                args.light_scale,
            )
            prim = light_record["prim"]
            intensity = (
                light_record["base_intensity"] * sample["intensity_multiplier"]
            )
            prim.GetAttribute("inputs:intensity").Set(intensity)
            color_attribute = prim.GetAttribute("inputs:color")
            if color_attribute.IsValid():
                color_attribute.Set(Gf.Vec3f(*sample["color_rgb"]))
            elif args.light_profile == "real":
                raise RuntimeError(
                    f"Real light profile requires inputs:color on {prim.GetPath()}"
                )
            light_api = UsdLux.LightAPI(prim)
            light_api.CreateEnableColorTemperatureAttr(True).Set(True)
            light_api.CreateColorTemperatureAttr(sample["color_temperature_k"]).Set(
                sample["color_temperature_k"]
            )
            light_record["transform_op"].Set(
                light_record["world"]
                * translation(sample["position_jitter_m"])
                * light_record["parent_inverse"]
            )
            return {
                **sample,
                "base_intensity": light_record["base_intensity"],
                "intensity": intensity,
                "enable_color_temperature": True,
            }

        def set_golden_light(light_record):
            sample = {
                "profile": "golden",
                "intensity_factor": 1.0,
                "intensity_scale": float(args.light_scale),
                "intensity_multiplier": float(args.light_scale),
                "color_temperature_k": 6500.0,
                "color_rgb": [1.0, 1.0, 1.0],
                "position_jitter_m": [0.0, 0.0, 0.0],
            }
            prim = light_record["prim"]
            intensity = light_record["base_intensity"] * args.light_scale
            prim.GetAttribute("inputs:intensity").Set(intensity)
            color_attribute = prim.GetAttribute("inputs:color")
            if not color_attribute.IsValid():
                raise RuntimeError(
                    f"Golden mode requires inputs:color on {prim.GetPath()}"
                )
            color_attribute.Set(Gf.Vec3f(1.0, 1.0, 1.0))
            light_api = UsdLux.LightAPI(prim)
            light_api.CreateEnableColorTemperatureAttr(True).Set(True)
            light_api.CreateColorTemperatureAttr(6500.0).Set(6500.0)
            light_record["transform_op"].Set(
                light_record["world"] * light_record["parent_inverse"]
            )
            return {
                **sample,
                "base_intensity": light_record["base_intensity"],
                "intensity": intensity,
                "enable_color_temperature": True,
            }

        def write_crop(image, center, offsets, path):
            center_x = int(round(center[0]))
            center_y = int(round(center[1]))
            x0 = center_x + offsets[0]
            y0 = center_y + offsets[1]
            x1 = center_x + offsets[2]
            y1 = center_y + offsets[3]
            crop = np.zeros(
                (offsets[3] - offsets[1], offsets[2] - offsets[0], 3),
                dtype=image.dtype,
            )
            source_x0 = max(0, x0)
            source_y0 = max(0, y0)
            source_x1 = min(width, x1)
            source_y1 = min(height, y1)
            if source_x1 > source_x0 and source_y1 > source_y0:
                destination_x0 = source_x0 - x0
                destination_y0 = source_y0 - y0
                crop[
                    destination_y0 : destination_y0 + source_y1 - source_y0,
                    destination_x0 : destination_x0 + source_x1 - source_x0,
                ] = image[source_y0:source_y1, source_x0:source_x1]
            if not cv2.imwrite(str(path), crop):
                raise OSError(f"Could not write crop: {path}")

        master_rng = random.Random(args.seed)
        quota_active = args.quota_mode and not args.golden
        quota_schedule = (
            build_quota_schedule(
                args.n,
                args.defects_per_fixture,
                quota_classes=tuple(args.quota_classes),
                damage_types=tuple(args.damage_types),
            )
            if quota_active
            else None
        )
        quota_slot_rng = (
            random.Random(args.seed + QUOTA_SLOT_SEED_OFFSET)
            if quota_active
            else None
        )
        image_records = []
        labels_path = args.out / f"{args.prefix}labels.json"
        commit_hash = git_commit_hash()
        flags = serialized_flags(args)
        for image_index in range(args.n):
            image_name = f"{args.prefix}img_{image_index:04d}"
            image_seed = master_rng.randrange(0, 2**63)
            scene_rng = random.Random(image_seed)
            if args.golden:
                fixture_scenes = {
                    prefix: build_golden_fill_scene(prefix)
                    for prefix in ("G", "S")
                }
            else:
                fixture_scenes = {
                    prefix: sample_fill_scene(
                        scene_rng,
                        prefix=prefix,
                        class_balance=args.class_balance and not args.quota_mode,
                        damage_weight=args.damage_weight,
                    )
                    for prefix in ("G", "S")
                }
            baseline_labels = {
                name: label
                for prefix in ("G", "S")
                for name, label in fixture_scenes[prefix].labels.items()
            }
            quota_damage_kinds = {}
            if quota_active:
                for prefix in ("G", "S"):
                    fixture_scenes[prefix], fixture_damage_kinds = (
                        apply_quota_assignments(
                            fixture_scenes[prefix],
                            prefix=prefix,
                            assignments=quota_schedule[prefix][image_index],
                            rng=quota_slot_rng,
                        )
                    )
                    quota_damage_kinds.update(fixture_damage_kinds)
            labels = {
                name: label
                for prefix in ("G", "S")
                for name, label in fixture_scenes[prefix].labels.items()
            }
            if len(labels) != 60 or set(labels) != set(ALL_SLOT_NAMES):
                raise AssertionError("Each image must contain exactly 60 G/S labels")
            if args.golden:
                shared_offset_px = [0, 0]
                requested_shift_px = [0, 0]
                rotation_jitter_deg = [0.0, 0.0, 0.0]
                jittered_rotation = rotation_world_to_cv.copy()
                jittered_position = camera_position.copy()
                translation_world = np.zeros(3, dtype=np.float64)
            else:
                shared_offset_px = [
                    scene_rng.randint(-15, 15),
                    scene_rng.randint(-15, 15),
                ]
                requested_shift_px = list(shared_offset_px)
                rotation_jitter_deg = [
                    scene_rng.uniform(-0.5, 0.5),
                    scene_rng.uniform(-0.5, 0.5),
                    scene_rng.uniform(-0.5, 0.5),
                ]
                (
                    jittered_rotation,
                    jittered_position,
                    translation_world,
                ) = jitter_camera_pose(
                    rotation_world_to_cv,
                    camera_position,
                    focal,
                    reference_depth,
                    requested_shift_px,
                    rotation_jitter_deg,
                )
            camera_transform_op.Set(
                usd_camera_matrix(jittered_rotation, jittered_position)
            )
            projected_centers = {
                name: project(point, jittered_rotation, jittered_position)
                for name, point in label_world_points.items()
            }
            crop_centers = translated_crop_centers(
                labeler_geometry, shared_offset_px
            )
            damage_crop_contexts = {
                slot_name: {
                    "cv_rotation": jittered_rotation,
                    "cv_position": jittered_position,
                    "crop_center": crop_centers[slot_name],
                    "crop_offsets": labeler_geometry["fixtures"][
                        slot_name[0]
                    ]["crop_offsets"],
                }
                for slot_name in ALL_SLOT_NAMES
            }
            camera_jitter = {
                "requested_shift_px": requested_shift_px,
                "translation_world_m": translation_world.tolist(),
                "rotation_xyz_deg": rotation_jitter_deg,
                "position_m": jittered_position.tolist(),
                "R_world_to_cv": jittered_rotation.tolist(),
            }
            variation_rng = scene_rng
            if quota_active:
                variation_rng = clone_random_generator(scene_rng)
                shadow_labels = dict(baseline_labels)
                for slot_name in ALL_SLOT_NAMES:
                    apply_slot(
                        slot_name,
                        shadow_labels,
                        scene_rng,
                        args.defect_boost,
                        crop_context=damage_crop_contexts[slot_name],
                    )
            variations = {}
            for slot_name in ALL_SLOT_NAMES:
                variations[slot_name] = apply_slot(
                    slot_name,
                    labels,
                    variation_rng,
                    args.defect_boost,
                    forced_damage_kind=quota_damage_kinds.get(slot_name),
                    crop_context=damage_crop_contexts[slot_name],
                )
            if quota_active:
                for original_slot, damage_kind in list(
                    quota_damage_kinds.items()
                ):
                    if labels[original_slot] == "damaged_vial":
                        continue
                    prefix = original_slot[0]
                    filled_slots = list(
                        FIXTURE_SLOT_NAMES[prefix][
                            : fixture_scenes[prefix].expected_through
                        ]
                    )
                    candidates = [
                        slot for slot in filled_slots if labels[slot] == "pass"
                    ]
                    quota_slot_rng.shuffle(candidates)
                    replacement = None
                    for candidate in candidates:
                        labels[candidate] = "damaged_vial"
                        variations[candidate] = apply_slot(
                            candidate,
                            labels,
                            variation_rng,
                            args.defect_boost,
                            forced_damage_kind=damage_kind,
                            crop_context=damage_crop_contexts[candidate],
                        )
                        if labels[candidate] == "damaged_vial":
                            replacement = candidate
                            break
                    if replacement is None:
                        raise RuntimeError(
                            f"Could not realize quota damage {damage_kind!r} "
                            f"for fixture {prefix} in {image_name}"
                        )
                    del quota_damage_kinds[original_slot]
                    quota_damage_kinds[replacement] = damage_kind
                for prefix in ("G", "S"):
                    expected_classes = Counter(
                        assignment["class"]
                        for assignment in quota_schedule[prefix][image_index]
                    )
                    actual_classes = Counter(
                        labels[slot]
                        for slot in FIXTURE_SLOT_NAMES[prefix]
                        if labels[slot] in expected_classes
                    )
                    if actual_classes != expected_classes:
                        raise RuntimeError(
                            f"Quota class realization mismatch for {image_name} "
                            f"fixture {prefix}: expected={dict(expected_classes)} "
                            f"actual={dict(actual_classes)}"
                        )
                    expected_damage_types = Counter(
                        assignment["damage_type"]
                        for assignment in quota_schedule[prefix][image_index]
                        if assignment["class"] == "damaged_vial"
                    )
                    actual_damage_types = Counter(
                        variations[slot]["damage"]["kind"]
                        for slot in FIXTURE_SLOT_NAMES[prefix]
                        if labels[slot] == "damaged_vial"
                    )
                    if actual_damage_types != expected_damage_types:
                        raise RuntimeError(
                            f"Quota damage realization mismatch for {image_name} "
                            f"fixture {prefix}: expected={dict(expected_damage_types)} "
                            f"actual={dict(actual_damage_types)}"
                        )
            for slot_name, label in labels.items():
                if label != "damaged_vial":
                    continue
                damage = variations[slot_name].get("damage", {})
                assert float(damage.get("strength", 0.0)) > 0.0, (
                    f"{image_name} {slot_name} is damaged_vial without "
                    "positive recorded strength"
                )

            floor_sample = sample_floor_emissive(
                args.floor_emissive,
                0.0 if args.golden else args.floor_brightness_jitter,
                args.seed,
                image_index,
            )
            floor_sample["enabled"] = not args.no_floor
            floor_sample["applied"] = False
            if not args.no_floor:
                floor_shader = stage.GetPrimAtPath(
                    f"{FLOOR_MATERIAL_PATH}/PreviewSurface"
                )
                emissive_attribute = floor_shader.GetAttribute(
                    "inputs:emissiveColor"
                )
                if emissive_attribute.IsValid():
                    emissive_attribute.Set(Gf.Vec3f(*floor_sample["emissive"]))
                    floor_sample["applied"] = True
                elif any(value > 0.0 for value in floor_sample["emissive"]):
                    raise RuntimeError(
                        "Non-zero floor emission requested but the floor shader has "
                        "no inputs:emissiveColor"
                    )
            print(
                f"floor {image_name}: emissive={floor_sample['emissive']} "
                f"factor={floor_sample['brightness_factor']:.6f} "
                f"redFactor={floor_sample['red_factor']:.6f} "
                f"blueFactor={floor_sample['blue_factor']:.6f} "
                f"applied={floor_sample['applied']}",
                flush=True,
            )

            if args.golden:
                lights = {
                    "key": set_golden_light(key_record),
                    "fill": set_golden_light(fill_record),
                }
            else:
                lights = {
                    "key": randomize_light(key_record, (0.7, 1.8), scene_rng),
                    "fill": randomize_light(fill_record, (0.7, 2.5), scene_rng),
                }
            print(
                f"lighting {image_name}: profile={lights['key']['profile']} "
                f"scale={args.light_scale:.4f} "
                f"keyIntensity={lights['key']['intensity']:.3f} "
                f"keyTemperatureK={lights['key']['color_temperature_k']:.1f} "
                f"fillIntensity={lights['fill']['intensity']:.3f} "
                f"fillTemperatureK={lights['fill']['color_temperature_k']:.1f}",
                flush=True,
            )
            for _ in range(3):
                kit.update()
            for _ in range(2):
                rep.orchestrator.step(
                    rt_subframes=args.subframes, delta_time=0.0
                )
            render = np.asarray(annotator.get_data())
            if (
                render.ndim != 3
                or render.shape[:2] != (render_height, render_width)
                or render.shape[2] < 3
            ):
                raise RuntimeError(f"Unexpected render shape: {render.shape}")
            bgr = np.ascontiguousarray(render[..., :3][..., ::-1])
            final_image = cv2.remap(
                bgr,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            image_path = output_images / f"{image_name}.png"
            write_full_image = not args.crops_only or image_index < 3
            if write_full_image:
                if not cv2.imwrite(str(image_path), final_image):
                    raise OSError(f"Could not write image: {image_path}")
            for slot_name in ALL_SLOT_NAMES:
                fixture = slot_name[0]
                write_crop(
                    final_image,
                    crop_centers[slot_name],
                    labeler_geometry["fixtures"][fixture]["crop_offsets"],
                    output_crops / f"{image_name}_{slot_name}.png",
                )

            image_records.append(
                {
                    "image": image_name,
                    "full_image_path": (
                        f"images/{image_name}.png" if write_full_image else None
                    ),
                    "labels": labels,
                    "not_yet_filled": [
                        name
                        for prefix in ("G", "S")
                        for name in fixture_scenes[prefix].not_yet_filled
                    ],
                    "fixture_state": {
                        prefix: {
                            "expected_through": fixture_scenes[
                                prefix
                            ].expected_through,
                            "missed_position": fixture_scenes[
                                prefix
                            ].missed_position,
                            "not_yet_filled": fixture_scenes[
                                prefix
                            ].not_yet_filled,
                        }
                        for prefix in ("G", "S")
                    },
                    "lights": lights,
                    "floor": floor_sample,
                    "seed": image_seed,
                    "shared_offset_px": shared_offset_px,
                    "camera_jitter": camera_jitter,
                    "crop_centers": crop_centers,
                    "projected_centers": projected_centers,
                    "variations": variations,
                    "damaged_positions": {
                        slot_name: {
                            "type": variations[slot_name]["damage"]["kind"],
                            "strength": variations[slot_name]["damage"]["strength"],
                        }
                        for slot_name, label in labels.items()
                        if label == "damaged_vial"
                    },
                }
            )
            run_summary = build_run_summary(image_records)
            labels_document = {
                "image_size": [width, height],
                "generation": {
                    "git_commit": commit_hash,
                    "flags": flags,
                },
                "lighting": {
                    "profile": "golden" if args.golden else args.light_profile,
                    "scale": args.light_scale,
                    "key_base_intensity": key_record["base_intensity"],
                    "fill_base_intensity": fill_record["base_intensity"],
                    "key_factor_range": (
                        [1.0, 1.0] if args.golden else [0.7, 1.8]
                    ),
                    "fill_factor_range": (
                        [1.0, 1.0] if args.golden else [0.7, 2.5]
                    ),
                },
                "camera": {
                    "model": "opencv_fisheye",
                    "focal_px": focal,
                    "k1": k1,
                    "k2": k2,
                    "principal_point": list(PRINCIPAL_POINT),
                    "theta_max": theta_max,
                    "min_dtheta_d": float(camera["min_dtheta_d"]),
                    "render_size": [render_width, render_height],
                    "width_scale": args.width_scale,
                },
                "backdrop": {
                    "enabled": not args.no_backdrop,
                    "debug_material": args.backdrop_debug,
                    "panel_bounds_xy": list(PANEL_BOUNDS_XY),
                    "panel_surface_z": PANEL_SURFACE_Z,
                    "image_left_world_side": image_left_side,
                    "frame_minimum": backdrop_coverage["frame_minimum"].tolist(),
                    "frame_maximum": backdrop_coverage["frame_maximum"].tolist(),
                    "crops_minimum": backdrop_coverage["crops_minimum"].tolist(),
                    "crops_maximum": backdrop_coverage["crops_maximum"].tolist(),
                    "crop_overruns": backdrop_overruns,
                    "slab": backdrop_spec if not args.no_backdrop else None,
                    "runtime": backdrop_runtime,
                },
                "floor": {
                    "enabled": not args.no_floor,
                    "albedo": list(args.floor_albedo),
                    "emissive": list(args.floor_emissive),
                    "base_emissive": list(args.floor_emissive),
                    "brightness_jitter_fraction": (
                        0.0 if args.golden else args.floor_brightness_jitter
                    ),
                    "station_minimum_z": station_minimum_z,
                    "frame_minimum": floor_coverage["frame_minimum"].tolist(),
                    "frame_maximum": floor_coverage["frame_maximum"].tolist(),
                    "box": floor_specification if not args.no_floor else None,
                    "runtime": floor_runtime,
                    "approximate_uncovered_crops": approximate_uncovered,
                    "coverage_check_note": (
                        "Approximate planar/projected-AABB coverage; occlusion ignored."
                    ),
                },
                "crop_geometry": {
                    "source": "labeler_calibration",
                    "profile": labeler_geometry["profile"],
                    "fixtures": {
                        prefix: {
                            "grid": labeler_geometry["fixtures"][prefix]["grid"],
                            "crop_offsets": list(
                                labeler_geometry["fixtures"][prefix][
                                    "crop_offsets"
                                ]
                            ),
                            "crop_size": [
                                labeler_geometry["fixtures"][prefix][
                                    "crop_offsets"
                                ][2]
                                - labeler_geometry["fixtures"][prefix][
                                    "crop_offsets"
                                ][0],
                                labeler_geometry["fixtures"][prefix][
                                    "crop_offsets"
                                ][3]
                                - labeler_geometry["fixtures"][prefix][
                                    "crop_offsets"
                                ][1],
                            ],
                        }
                        for prefix in ("G", "S")
                    },
                },
                "positions": {
                    name: list(
                        labeler_geometry["fixtures"][name[0]]["centers"][name]
                    )
                    for name in ALL_SLOT_NAMES
                },
                "twin_projected_positions": nominal_projected_centers,
                "deformation_backends": {
                    name: slot_info[name]["tube"]["deformation_backend"]
                    for name in ALL_SLOT_NAMES
                },
                "run_summary": run_summary,
                "images": image_records,
            }
            if args.golden:
                labels_document["golden"] = True
            temporary_labels = labels_path.with_suffix(".json.tmp")
            temporary_labels.write_text(
                json.dumps(labels_document, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(temporary_labels, labels_path)
            print(f"done {image_name}", flush=True)
        print_run_summary(build_run_summary(image_records))
    except BaseException:
        traceback.print_exc()
        if kit is not None:
            kit.close(exit_code=1)
        raise
    else:
        if kit is not None:
            kit.close()


if __name__ == "__main__":
    main()
