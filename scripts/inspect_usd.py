"""Inspect the tray-inspection USD stage without authoring or saving it."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path


STAGE_PATH = Path("/job/in/stations/tray_inspection/tray_inspection.usdc")
REPORT_PATH = Path("/job/out/usd_report.txt")
ROOT_PATH = "/TrayInspection"
MAX_REPORT_LINES = 400
MATCH_RE = re.compile(
    r"cap|panel|plate|fixture|backdrop|table|G01|S01|G30|S30",
    re.IGNORECASE,
)
SLOT_CAP_RE = re.compile(r"^([GS])(\d{2}).*cap", re.IGNORECASE)
BOUNDARY_SLOT_RE = re.compile(r"(?:^|_)(?:G01|S01|G30|S30)(?:_|$)", re.IGNORECASE)
MATERIAL_INPUT_RE = re.compile(
    r"color|diffuse|emissive|opacity|roughness|metallic", re.IGNORECASE
)


class Report:
    """Collect a bounded plain-text report."""

    def __init__(self, maximum_lines: int = MAX_REPORT_LINES) -> None:
        self.maximum_lines = maximum_lines
        self.lines: list[str] = []
        self.omitted = 0

    def add(self, line: object = "") -> None:
        text = str(line).replace("\r", "")
        for part in text.split("\n"):
            if len(self.lines) < self.maximum_lines - 1:
                self.lines.append(part)
            else:
                self.omitted += 1

    def render(self) -> str:
        lines = list(self.lines)
        if self.omitted:
            lines.append(f"[output capped: {self.omitted} additional lines omitted]")
        return "\n".join(lines) + "\n"


def compact_value(value: object, limit: int = 220) -> str:
    text = repr(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def run_inspection() -> None:
    # SimulationApp is constructed in main() before this function imports pxr.
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdShade

    report = Report()
    report.add("READ-ONLY USD INSPECTION")
    report.add(f"stage: {STAGE_PATH}")
    report.add("The stage is opened and traversed only; this script never saves or exports it.")

    stage = Usd.Stage.Open(str(STAGE_PATH), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {STAGE_PATH}")

    time_code = Usd.TimeCode.Default()
    bbox_cache = UsdGeom.BBoxCache(
        time_code,
        [UsdGeom.Tokens.default_],
        useExtentsHint=True,
    )
    root = stage.GetPrimAtPath(ROOT_PATH)
    if not root.IsValid():
        raise RuntimeError(f"Required root prim does not exist: {ROOT_PATH}")

    prims = list(stage.Traverse())

    def bound_material_path(prim) -> str:
        try:
            result = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            material = result[0] if isinstance(result, tuple) else result
            if material and material.GetPrim().IsValid():
                return str(material.GetPath())
        except Exception as exc:  # Report malformed or unsupported bindings, then continue.
            return f"<material-error:{type(exc).__name__}>"
        return "<none>"

    def world_bounds_values(prim):
        try:
            aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            minimum = aligned.GetMin()
            maximum = aligned.GetMax()
            values = [*minimum, *maximum]
            if any(value == float("inf") or value == float("-inf") for value in values):
                return None
            return minimum, maximum
        except Exception:
            return None

    def world_bounds(prim) -> str:
        try:
            values = world_bounds_values(prim)
            if values is None:
                return "<empty>"
            minimum, maximum = values
            return (
                "min=(%.4f, %.4f, %.4f) max=(%.4f, %.4f, %.4f)"
                % (*minimum, *maximum)
            )
        except Exception as exc:
            return f"<bounds-error:{type(exc).__name__}>"

    def world_position(prim) -> str:
        try:
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time_code)
            position = matrix.ExtractTranslation()
            return "(%.4f, %.4f, %.4f)" % tuple(position)
        except Exception as exc:
            return f"<position-error:{type(exc).__name__}>"

    def describe_prim(prim) -> str:
        return (
            f"{prim.GetPath()} type={prim.GetTypeName() or '<untyped>'} "
            f"bounds=[{world_bounds(prim)}] material={bound_material_path(prim)}"
        )

    report.add()
    report.add("1. PRIM HIERARCHY: TOP 3 LEVELS UNDER /TrayInspection")
    root_depth = len(root.GetPath().pathString.strip("/").split("/"))
    report.add(f"{root.GetPath()} type={root.GetTypeName() or '<untyped>'}")
    for prim in Usd.PrimRange(root):
        if prim == root:
            continue
        depth = len(prim.GetPath().pathString.strip("/").split("/")) - root_depth
        if depth <= 3:
            indent = "  " * depth
            report.add(f"{indent}{prim.GetName()} type={prim.GetTypeName() or '<untyped>'}")

    report.add()
    report.add("2. MATCHED GEOMETRY AND SCENE PRIMS")
    matched = [prim for prim in prims if MATCH_RE.search(prim.GetName())]
    report.add(f"matched prim count: {len(matched)}")

    individual = []
    repeated_groups = defaultdict(list)
    for prim in matched:
        name = prim.GetName()
        slot_match = re.match(r"^([GS])(\d{2})(.*)$", name, re.IGNORECASE)
        is_boundary = bool(BOUNDARY_SLOT_RE.search(name))
        if slot_match and "cap" in name.lower() and not is_boundary:
            fixture, _, suffix = slot_match.groups()
            key = (
                fixture.upper(),
                suffix.lower(),
                prim.GetTypeName() or "<untyped>",
                bound_material_path(prim),
            )
            repeated_groups[key].append(prim)
        else:
            individual.append(prim)

    for prim in sorted(individual, key=lambda item: str(item.GetPath())):
        report.add(describe_prim(prim))
    for key, members in sorted(repeated_groups.items(), key=lambda item: item[0]):
        fixture, suffix, type_name, material_path = key
        members.sort(key=lambda item: str(item.GetPath()))
        report.add(
            f"summary pattern={fixture}##{suffix} count={len(members)} "
            f"type={type_name} material={material_path}; first 2 follow"
        )
        for prim in members[:2]:
            report.add(describe_prim(prim))

    report.add()
    report.add("2b. BOUNDABLE PRIMS NEAR THE LEFT/RIGHT SCENE EDGES")
    root_bounds = world_bounds_values(root)
    if root_bounds is None:
        report.add("Could not compute /TrayInspection bounds; edge scan unavailable.")
    else:
        root_minimum, root_maximum = root_bounds
        x_span = float(root_maximum[0] - root_minimum[0])
        edge_band = max(0.0, x_span * 0.10)
        left_limit = float(root_minimum[0] + edge_band)
        right_limit = float(root_maximum[0] - edge_band)
        report.add(
            "edge definition: outer 10% of /TrayInspection world-X span; "
            f"rootX=[{root_minimum[0]:.4f}, {root_maximum[0]:.4f}] "
            f"left<= {left_limit:.4f}, right>= {right_limit:.4f}"
        )
        left_prims = []
        right_prims = []
        for prim in prims:
            if not prim.IsA(UsdGeom.Boundable):
                continue
            values = world_bounds_values(prim)
            if values is None:
                continue
            minimum, maximum = values
            if float(minimum[0]) <= left_limit:
                left_prims.append(prim)
            if float(maximum[0]) >= right_limit:
                right_prims.append(prim)
        for label, edge_prims in (("left", left_prims), ("right", right_prims)):
            edge_prims.sort(key=lambda item: str(item.GetPath()))
            report.add(f"{label} edge boundable prim count: {len(edge_prims)}")
            for prim in edge_prims[:30]:
                report.add(describe_prim(prim))
            if len(edge_prims) > 30:
                report.add(
                    f"{label} edge summary: {len(edge_prims) - 30} additional "
                    "boundable prims omitted"
                )

    report.add()
    report.add("3. CAP MATERIALS AND SELECTED SHADER INPUTS")
    cap_roots = [prim for prim in prims if SLOT_CAP_RE.match(prim.GetName())]
    cap_materials_by_fixture: dict[str, set[str]] = {"G": set(), "S": set()}
    caps_without_material = {"G": 0, "S": 0}
    all_material_paths: set[str] = set()
    for cap_root in cap_roots:
        fixture = SLOT_CAP_RE.match(cap_root.GetName()).group(1).upper()
        paths = set()
        for candidate in Usd.PrimRange(cap_root):
            material_path = bound_material_path(candidate)
            if material_path != "<none>" and not material_path.startswith("<material-error"):
                paths.add(material_path)
        if not paths:
            caps_without_material[fixture] += 1
        cap_materials_by_fixture[fixture].update(paths)
        all_material_paths.update(paths)

    report.add(
        f"cap roots discovered: G={sum(1 for p in cap_roots if p.GetName().upper().startswith('G'))} "
        f"S={sum(1 for p in cap_roots if p.GetName().upper().startswith('S'))}"
    )
    report.add(f"G cap material paths: {sorted(cap_materials_by_fixture['G'])}")
    report.add(f"S cap material paths: {sorted(cap_materials_by_fixture['S'])}")
    report.add(f"cap roots without a discoverable material: {caps_without_material}")
    share_one = (
        len(all_material_paths) == 1
        and bool(cap_materials_by_fixture["G"])
        and cap_materials_by_fixture["G"] == cap_materials_by_fixture["S"]
    )
    report.add(f"all discovered G and S caps share ONE material: {share_one}")

    for material_path in sorted(all_material_paths):
        material = UsdShade.Material.Get(stage, material_path)
        report.add(f"material {material_path}")
        source = material.ComputeSurfaceSource()
        shader = source[0] if isinstance(source, tuple) else source
        if not shader or not shader.GetPrim().IsValid():
            report.add("  surface shader: <none>")
            continue
        shader = UsdShade.Shader(shader.GetPrim())
        report.add(
            f"  surface shader: {shader.GetPath()} id={compact_value(shader.GetIdAttr().Get())}"
        )
        selected_inputs = [
            shader_input
            for shader_input in shader.GetInputs()
            if MATERIAL_INPUT_RE.search(shader_input.GetFullName())
        ]
        if not selected_inputs:
            report.add("  selected shader inputs: <none>")
        for shader_input in selected_inputs:
            report.add(
                f"  {shader_input.GetFullName()} = "
                f"{compact_value(shader_input.GetAttr().Get())}"
            )

    report.add()
    report.add("4. CAMERAS")
    cameras = [prim for prim in prims if prim.IsA(UsdGeom.Camera)]
    report.add(f"camera count: {len(cameras)}")
    for prim in cameras:
        camera = UsdGeom.Camera(prim)
        report.add(
            f"{prim.GetPath()} focalLength={compact_value(camera.GetFocalLengthAttr().Get())} "
            f"horizontalAperture={compact_value(camera.GetHorizontalApertureAttr().Get())} "
            f"verticalAperture={compact_value(camera.GetVerticalApertureAttr().Get())} "
            f"clippingRange={compact_value(camera.GetClippingRangeAttr().Get())} "
            f"worldPosition={world_position(prim)}"
        )

    report.add()
    report.add("5. LIGHTS")
    light_schema_names = (
        "DistantLight",
        "SphereLight",
        "RectLight",
        "DiskLight",
        "CylinderLight",
        "DomeLight",
        "GeometryLight",
        "PortalLight",
    )
    light_schemas = tuple(
        schema
        for schema in (getattr(UsdLux, name, None) for name in light_schema_names)
        if schema is not None
    )
    lights = [
        prim
        for prim in prims
        if any(prim.IsA(schema) for schema in light_schemas)
        or (
            prim.GetTypeName().endswith("Light")
            and prim.GetAttribute("inputs:intensity").IsValid()
        )
    ]
    report.add(f"light count: {len(lights)}")
    light_attributes = (
        "inputs:intensity",
        "inputs:exposure",
        "inputs:color",
        "inputs:colorTemperature",
        "inputs:enableColorTemperature",
    )
    for prim in lights:
        values = []
        for attribute_name in light_attributes:
            attribute = prim.GetAttribute(attribute_name)
            value = attribute.Get(time_code) if attribute.IsValid() else None
            values.append(f"{attribute_name.removeprefix('inputs:')}={compact_value(value)}")
        report.add(
            f"{prim.GetPath()} type={prim.GetTypeName()} {' '.join(values)} "
            f"worldPosition={world_position(prim)}"
        )

    text = report.render()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(text, end="", flush=True)


def main() -> int:
    # Isaac Sim initializes the USD/PXR runtime. Keep this before any pxr import.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        run_inspection()
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
