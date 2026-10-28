"""Generate fill-order tray images in Isaac Sim 5.1.0 headless mode."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fill_logic import SLOT_NAMES, sample_fill_scene
from fisheye import (
    PRINCIPAL_POINT,
    invert_theta,
    jitter_camera_pose,
    project_world,
)


CAMERA_PATH = Path("/job/in/cameras/camera_fit.json")
STAGE_PATH = "/job/in/stations/tray_inspection/tray_inspection.usdc"
CROP_OFFSETS = (-38, -47, 42, 43)
CROP_SHIFT = (0, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("/job/out"))
    parser.add_argument("--subframes", type=int, default=24)
    parser.add_argument(
        "--width-scale",
        type=float,
        default=1.0,
        help="Scale pinhole render resolution while preserving its full FOV.",
    )
    return parser.parse_args()


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
        rotation_world_to_cv = np.asarray(camera["R_world_to_cv"], dtype=np.float64)
        camera_position = np.asarray(camera["position_m"], dtype=np.float64)
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
        from pxr import Gf, Usd, UsdGeom, UsdLux

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
        slot_info = {}
        for name in SLOT_NAMES:
            record = {}
            for part in ("tube", "cap"):
                prim = stage.GetPrimAtPath(f"/TrayInspection/{name}_{part}")
                if not prim.IsValid():
                    raise RuntimeError(f"Missing prim: {name}_{part}")
                if part == "tube" and prim.IsInstance():
                    prim.SetInstanceable(False)
                    kit.update()
                world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time_code)
                parent_inverse = (
                    UsdGeom.Xformable(prim.GetParent())
                    .ComputeLocalToWorldTransform(time_code)
                    .GetInverse()
                )
                bounds = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
                record[part] = {
                    "prim": prim,
                    "world": world,
                    "parent_inverse": parent_inverse,
                    "minimum": np.asarray(bounds.GetMin(), dtype=np.float64),
                    "maximum": np.asarray(bounds.GetMax(), dtype=np.float64),
                }
            slot_info[name] = record

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

        def deform_mesh_points(tube, damage_kind, scene_rng):
            angle = scene_rng.uniform(0.0, 2.0 * math.pi)
            direction = np.array([math.cos(angle), math.sin(angle)])
            parameters = {"kind": damage_kind}
            bend_shift = np.zeros(2, dtype=np.float64)
            bend_degrees = 0.0
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
                radius_local = max(
                    float(np.percentile(np.linalg.norm(radial, axis=1), 90)), 1.0e-6
                )

                if damage_kind == "crush":
                    strength = scene_rng.uniform(0.20, 0.42)
                    band_center = scene_rng.uniform(0.32, 0.72)
                    band_width = scene_rng.uniform(0.10, 0.20)
                    band = np.exp(
                        -0.5
                        * ((normalized_height - band_center) / band_width) ** 2
                    )
                    component = radial @ direction
                    changed[:, :2] -= (
                        strength * band * component
                    )[:, None] * direction
                    parameters.update(
                        strength=strength,
                        band_center=band_center,
                        band_width=band_width,
                    )
                elif damage_kind == "bend":
                    amplitude = scene_rng.uniform(0.003, 0.008)
                    curve = normalized_height * normalized_height
                    changed[:, :2] += curve[:, None] * amplitude * direction
                    bend_shift = amplitude * direction
                    bend_degrees = math.degrees(math.atan2(2.0 * amplitude, height_local))
                    parameters.update(
                        strength=amplitude,
                        mouth_shift_m=bend_shift.tolist(),
                        mouth_tilt_deg=bend_degrees,
                    )
                else:
                    strength = scene_rng.uniform(0.18, 0.38)
                    band_center = scene_rng.uniform(0.28, 0.76)
                    band_width = scene_rng.uniform(0.08, 0.17)
                    band = np.exp(
                        -0.5
                        * ((normalized_height - band_center) / band_width) ** 2
                    )
                    toward_dent = np.maximum(0.0, radial @ direction / radius_local)
                    indentation = strength * radius_local * band * toward_dent**2
                    changed[:, :2] -= indentation[:, None] * direction
                    parameters.update(
                        strength=strength,
                        band_center=band_center,
                        band_width=band_width,
                    )

                mesh_record["mesh"].GetPointsAttr().Set(
                    [
                        Gf.Vec3f(*(float(value) for value in point))
                        for point in changed
                    ]
                )
            return parameters, bend_shift, bend_degrees, direction

        def show_stacked_fallback(tube, scene_rng):
            direction_angle = scene_rng.uniform(0.0, 2.0 * math.pi)
            direction = np.array(
                [math.cos(direction_angle), math.sin(direction_angle)]
            )
            mouth_shift = scene_rng.uniform(0.003, 0.007) * direction
            squeeze_x = scene_rng.uniform(0.62, 0.88)
            squeeze_y = scene_rng.uniform(0.72, 0.95)
            for segment in tube["fallback_segments"]:
                fraction = (segment["index"] + 0.5) / 3.0
                center = [
                    float(segment["center_xy"][0] + mouth_shift[0] * fraction**2),
                    float(segment["center_xy"][1] + mouth_shift[1] * fraction**2),
                    float(segment["base_z"] + segment["height"] * fraction),
                ]
                bend_degrees = math.degrees(
                    math.atan2(2.0 * np.linalg.norm(mouth_shift) * fraction, segment["height"])
                )
                bend_axis = (-direction[1], direction[0], 0.0)
                segment["op"].Set(
                    scaling(squeeze_x, squeeze_y, 1.0)
                    * rotation(bend_axis, bend_degrees)
                    * translation(center)
                )
                set_visible(segment["prim"], True)
            parameters = {
                "kind": "stacked_bend",
                "strength": float(np.linalg.norm(mouth_shift)),
                "anisotropic_scale": [squeeze_x, squeeze_y],
            }
            tilt = math.degrees(
                math.atan2(2.0 * np.linalg.norm(mouth_shift), tube["maximum"][2] - tube["minimum"][2])
            )
            return parameters, mouth_shift, tilt, direction

        def apply_slot(name, label, scene_rng):
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
            offset_radius = scene_rng.uniform(0.0, 0.00149)
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
                damage_kind = scene_rng.choice(("crush", "bend", "dent"))
                if tube["deformation_backend"] == "mesh_points":
                    damage, mouth_shift, mouth_tilt, bend_direction = deform_mesh_points(
                        tube, damage_kind, scene_rng
                    )
                else:
                    tube_visible = False
                    damage, mouth_shift, mouth_tilt, bend_direction = show_stacked_fallback(
                        tube, scene_rng
                    )
                if np.linalg.norm(mouth_shift) > 0.0:
                    bend_axis = (-bend_direction[1], bend_direction[0], 0.0)
                    shifted_mouth = mouth_center + np.array(
                        [mouth_shift[0], mouth_shift[1], 0.0]
                    )
                    cap_world = (
                        cap_world
                        * translation((mouth_shift[0], mouth_shift[1], 0.0))
                        * about(shifted_mouth, rotation(bend_axis, mouth_tilt))
                    )
                variation["damage"] = damage
                variation["deformation_backend"] = tube["deformation_backend"]

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
            factor = scene_rng.uniform(*intensity_range)
            kelvin = scene_rng.uniform(3500.0, 6500.0)
            color = [
                1.0,
                scene_rng.uniform(0.9, 1.0),
                scene_rng.uniform(0.8, 1.0),
            ]
            jitter = [scene_rng.uniform(-0.045, 0.045) for _ in range(3)]
            prim = light_record["prim"]
            prim.GetAttribute("inputs:intensity").Set(
                light_record["base_intensity"] * factor
            )
            color_attribute = prim.GetAttribute("inputs:color")
            if color_attribute.IsValid():
                color_attribute.Set(Gf.Vec3f(*color))
            light_api = UsdLux.LightAPI(prim)
            light_api.CreateEnableColorTemperatureAttr(True).Set(True)
            light_api.CreateColorTemperatureAttr(kelvin).Set(kelvin)
            light_record["transform_op"].Set(
                light_record["world"]
                * translation(jitter)
                * light_record["parent_inverse"]
            )
            return {
                "intensity_factor": factor,
                "color_temperature_k": kelvin,
                "color_rgb": color,
                "position_jitter_m": jitter,
            }

        def write_crop(image, center, path):
            shifted_x = int(round(center[0] + CROP_SHIFT[0]))
            shifted_y = int(round(center[1] + CROP_SHIFT[1]))
            x0 = shifted_x + CROP_OFFSETS[0]
            y0 = shifted_y + CROP_OFFSETS[1]
            x1 = shifted_x + CROP_OFFSETS[2]
            y1 = shifted_y + CROP_OFFSETS[3]
            crop = np.zeros(
                (CROP_OFFSETS[3] - CROP_OFFSETS[1], CROP_OFFSETS[2] - CROP_OFFSETS[0], 3),
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
        image_records = []
        labels_path = args.out / "labels.json"
        for image_index in range(args.n):
            image_name = f"img_{image_index:04d}"
            image_seed = master_rng.randrange(0, 2**63)
            scene_rng = random.Random(image_seed)
            fill_scene = sample_fill_scene(scene_rng)
            requested_shift_px = [
                scene_rng.uniform(-20.0, 20.0),
                scene_rng.uniform(-20.0, 20.0),
            ]
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
            camera_jitter = {
                "requested_shift_px": requested_shift_px,
                "translation_world_m": translation_world.tolist(),
                "rotation_xyz_deg": rotation_jitter_deg,
                "position_m": jittered_position.tolist(),
                "R_world_to_cv": jittered_rotation.tolist(),
            }
            variations = {}
            for slot_name in SLOT_NAMES:
                variations[slot_name] = apply_slot(
                    slot_name, fill_scene.labels[slot_name], scene_rng
                )

            lights = {
                "key": randomize_light(key_record, (0.4, 1.8), scene_rng),
                "fill": randomize_light(fill_record, (0.3, 2.5), scene_rng),
            }
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
            if not cv2.imwrite(str(image_path), final_image):
                raise OSError(f"Could not write image: {image_path}")
            for slot_name in SLOT_NAMES:
                write_crop(
                    final_image,
                    projected_centers[slot_name],
                    output_crops / f"{image_name}_{slot_name}.png",
                )

            image_records.append(
                {
                    "image": image_name,
                    "labels": fill_scene.labels,
                    "not_yet_filled": fill_scene.not_yet_filled,
                    "expected_through": fill_scene.expected_through,
                    "missed_position": fill_scene.missed_position,
                    "lights": lights,
                    "seed": image_seed,
                    "camera_jitter": camera_jitter,
                    "projected_centers": projected_centers,
                    "variations": variations,
                }
            )
            labels_document = {
                "image_size": [width, height],
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
                "crop_offsets": list(CROP_OFFSETS),
                "crop_shift": list(CROP_SHIFT),
                "positions": nominal_projected_centers,
                "deformation_backends": {
                    name: slot_info[name]["tube"]["deformation_backend"]
                    for name in SLOT_NAMES
                },
                "images": image_records,
            }
            temporary_labels = labels_path.with_suffix(".json.tmp")
            temporary_labels.write_text(
                json.dumps(labels_document, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(temporary_labels, labels_path)
            print(f"done {image_name}", flush=True)
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
