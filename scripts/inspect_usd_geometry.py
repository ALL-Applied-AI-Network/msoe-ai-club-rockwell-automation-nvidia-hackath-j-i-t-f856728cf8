"""Inspect tray-station geometry without authoring or saving the USD stage."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from pathlib import Path


STAGE_PATH = Path("/job/in/stations/tray_inspection/tray_inspection.usdc")
REPORT_PATH = Path("/job/out/geometry_report.txt")
ROOT_PATH = "/TrayInspection"
MESH_PATH = "/TrayInspection/AU_001436920200716/AU_001436920200716"
TARGET_SUBSETS = (
    "tw_table_paint",
    "tw_black_plate",
    "tw_steel",
    "tw_knob_red",
    "tw_robot_yellow",
)
DIRECT_CHILD_PREFIXES = ("tray_", "note_", "post", "label")
MAX_REPORT_LINES = 150
HORIZONTAL_NORMAL_Z = 0.95

CAMERA_FOCAL = 0.3673
CAMERA_H_APERTURE = 0.432
CAMERA_V_APERTURE = 0.324
CAMERA_POSITION = (-0.3156, -0.7200, 2.3734)
CAP_TOP_Z = 1.29


class Report:
    """Collect a bounded report while reserving one line for truncation notice."""

    def __init__(self, maximum_lines: int = MAX_REPORT_LINES) -> None:
        self.maximum_lines = maximum_lines
        self.lines: list[str] = []
        self.omitted = 0

    def add(self, value: object = "") -> None:
        for line in str(value).replace("\r", "").split("\n"):
            if len(self.lines) < self.maximum_lines - 1:
                self.lines.append(line)
            else:
                self.omitted += 1

    def render(self) -> str:
        lines = list(self.lines)
        if self.omitted:
            lines.append(f"[output capped: {self.omitted} additional lines omitted]")
        return "\n".join(lines) + "\n"


def run_inspection() -> None:
    # SimulationApp is started in main() before this function imports pxr.
    from pxr import Gf, Usd, UsdGeom

    report = Report()
    report.add("READ-ONLY USD GEOMETRY INSPECTION")
    report.add(f"stage: {STAGE_PATH}")
    report.add("The stage is traversed only; this script never saves or exports it.")

    stage = Usd.Stage.Open(str(STAGE_PATH), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not open stage: {STAGE_PATH}")
    root = stage.GetPrimAtPath(ROOT_PATH)
    if not root.IsValid():
        raise RuntimeError(f"Missing root prim: {ROOT_PATH}")

    time_code = Usd.TimeCode.Default()
    bbox_cache = UsdGeom.BBoxCache(
        time_code,
        [UsdGeom.Tokens.default_],
        useExtentsHint=True,
    )

    def vector_text(vector) -> str:
        return "(%.6f, %.6f, %.6f)" % tuple(float(value) for value in vector)

    def bounds_text(prim) -> str:
        try:
            aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            minimum = aligned.GetMin()
            maximum = aligned.GetMax()
            values = [*minimum, *maximum]
            if not all(math.isfinite(float(value)) for value in values):
                return "<empty>"
            return f"min={vector_text(minimum)} max={vector_text(maximum)}"
        except Exception as exc:
            return f"<bounds-error:{type(exc).__name__}>"

    report.add()
    report.add("1. SELECTED DIRECT CHILDREN OF /TrayInspection")
    selected_children = [
        prim
        for prim in root.GetChildren()
        if prim.GetName().lower().startswith(DIRECT_CHILD_PREFIXES)
    ]
    report.add(f"matching direct-child count: {len(selected_children)}")
    for prim in sorted(selected_children, key=lambda item: item.GetName().lower()):
        report.add(
            f"{prim.GetPath()} type={prim.GetTypeName() or '<untyped>'} "
            f"worldBBox=[{bounds_text(prim)}]"
        )

    report.add()
    report.add("2. COMBINED STATION MESH AND MATERIAL SUBSETS")
    mesh_prim = stage.GetPrimAtPath(MESH_PATH)
    if not mesh_prim.IsValid() or not mesh_prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"Expected mesh does not exist: {MESH_PATH}")
    mesh = UsdGeom.Mesh(mesh_prim)
    local_to_world = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(time_code)
    report.add(f"mesh: {MESH_PATH}")
    report.add("local-to-world transform (rows):")
    for row in range(4):
        report.add(
            "  [" + ", ".join(f"{float(local_to_world[row][column]):.9f}" for column in range(4)) + "]"
        )

    points = mesh.GetPointsAttr().Get(time_code) or []
    face_counts = [int(value) for value in (mesh.GetFaceVertexCountsAttr().Get(time_code) or [])]
    face_indices = [int(value) for value in (mesh.GetFaceVertexIndicesAttr().Get(time_code) or [])]
    world_points = [local_to_world.Transform(Gf.Vec3d(point)) for point in points]
    face_offsets = []
    running_offset = 0
    for count in face_counts:
        face_offsets.append(running_offset)
        running_offset += count
    if running_offset != len(face_indices):
        raise RuntimeError(
            "faceVertexCounts do not sum to the faceVertexIndices length: "
            f"{running_offset} != {len(face_indices)}"
        )

    def face_vertex_ids(face_index: int) -> list[int]:
        if not 0 <= face_index < len(face_counts):
            raise IndexError(f"face index outside mesh: {face_index}")
        start = face_offsets[face_index]
        return face_indices[start : start + face_counts[face_index]]

    def polygon_normal_and_area(vertex_ids: list[int]):
        vertices = [world_points[index] for index in vertex_ids]
        newell = Gf.Vec3d(0.0, 0.0, 0.0)
        for index, current in enumerate(vertices):
            following = vertices[(index + 1) % len(vertices)]
            newell[0] += (current[1] - following[1]) * (current[2] + following[2])
            newell[1] += (current[2] - following[2]) * (current[0] + following[0])
            newell[2] += (current[0] - following[0]) * (current[1] + following[1])
        magnitude = float(newell.GetLength())
        if magnitude <= 1.0e-15:
            return Gf.Vec3d(0.0, 0.0, 0.0), 0.0
        return newell / magnitude, magnitude / 2.0

    def subset_metrics(face_ids: list[int]):
        used_vertices: set[int] = set()
        weighted_normal = Gf.Vec3d(0.0, 0.0, 0.0)
        face_records = {}
        for face_id in face_ids:
            vertex_ids = face_vertex_ids(face_id)
            if len(vertex_ids) < 3:
                continue
            normal, area = polygon_normal_and_area(vertex_ids)
            used_vertices.update(vertex_ids)
            weighted_normal += normal * area
            face_records[face_id] = {
                "vertices": vertex_ids,
                "normal": normal,
                "area": area,
            }
        if not used_vertices:
            return None
        vertices = [world_points[index] for index in used_vertices]
        minimum = Gf.Vec3d(*(min(float(point[axis]) for point in vertices) for axis in range(3)))
        maximum = Gf.Vec3d(*(max(float(point[axis]) for point in vertices) for axis in range(3)))
        normal_length = float(weighted_normal.GetLength())
        average_normal = (
            weighted_normal / normal_length
            if normal_length > 1.0e-15
            else Gf.Vec3d(0.0, 0.0, 0.0)
        )
        return minimum, maximum, average_normal, face_records

    subsets = {subset.GetPrim().GetName(): subset for subset in UsdGeom.Subset.GetAllGeomSubsets(mesh)}
    table_face_records = {}
    for subset_name in TARGET_SUBSETS:
        subset = subsets.get(subset_name)
        if subset is None:
            report.add(f"subset {subset_name}: NOT FOUND")
            continue
        element_type = subset.GetElementTypeAttr().Get(time_code)
        family_name = subset.GetFamilyNameAttr().Get(time_code)
        member_indices = [int(value) for value in (subset.GetIndicesAttr().Get(time_code) or [])]
        report.add(
            f"subset {subset.GetPath()} elementType={element_type} "
            f"familyName={family_name!r} memberCount={len(member_indices)}"
        )
        if element_type != UsdGeom.Tokens.face:
            report.add("  face metrics: unavailable because this is not a face-element subset")
            continue
        metrics = subset_metrics(member_indices)
        if metrics is None:
            report.add("  faceCount=0 worldBBox=<empty> averageWorldFaceNormal=<undefined>")
            continue
        minimum, maximum, average_normal, records = metrics
        report.add(
            f"  faceCount={len(member_indices)} worldBBox=[min={vector_text(minimum)} "
            f"max={vector_text(maximum)}] averageWorldFaceNormal={vector_text(average_normal)} "
            f"worldZ=[{minimum[2]:.6f}, {maximum[2]:.6f}]"
        )
        if subset_name == "tw_table_paint":
            table_face_records = records

    report.add()
    report.add("3. LARGEST UPWARD-FACING HORIZONTAL tw_table_paint REGION")
    table_region_bounds = None
    horizontal_faces = {
        face_id: record
        for face_id, record in table_face_records.items()
        if float(record["normal"][2]) >= HORIZONTAL_NORMAL_Z and record["area"] > 0.0
    }
    if not horizontal_faces:
        report.add(
            f"No tw_table_paint faces found with world normal Z >= {HORIZONTAL_NORMAL_Z:.2f}."
        )
    else:
        edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
        for face_id, record in horizontal_faces.items():
            vertex_ids = record["vertices"]
            for index, first in enumerate(vertex_ids):
                second = vertex_ids[(index + 1) % len(vertex_ids)]
                edge_to_faces[tuple(sorted((first, second)))].append(face_id)
        neighbors: dict[int, set[int]] = defaultdict(set)
        for shared_faces in edge_to_faces.values():
            for face_id in shared_faces:
                neighbors[face_id].update(other for other in shared_faces if other != face_id)

        components = []
        unseen = set(horizontal_faces)
        while unseen:
            first = unseen.pop()
            component = {first}
            queue = deque([first])
            while queue:
                current = queue.popleft()
                new_faces = neighbors[current] & unseen
                unseen.difference_update(new_faces)
                component.update(new_faces)
                queue.extend(new_faces)
            area = sum(horizontal_faces[face_id]["area"] for face_id in component)
            components.append((area, component))
        area, largest = max(components, key=lambda item: item[0])
        vertex_ids = {
            vertex_id
            for face_id in largest
            for vertex_id in horizontal_faces[face_id]["vertices"]
        }
        vertices = [world_points[index] for index in vertex_ids]
        minimum = Gf.Vec3d(*(min(float(point[axis]) for point in vertices) for axis in range(3)))
        maximum = Gf.Vec3d(*(max(float(point[axis]) for point in vertices) for axis in range(3)))
        table_region_bounds = minimum, maximum
        report.add(
            f"threshold: normal.z >= {HORIZONTAL_NORMAL_Z:.2f}; connected components={len(components)}"
        )
        report.add(f"faceCount={len(largest)} worldArea={area:.6f}")
        report.add(
            f"worldX=[{minimum[0]:.6f}, {maximum[0]:.6f}] "
            f"worldY=[{minimum[1]:.6f}, {maximum[1]:.6f}] "
            f"worldZ=[{minimum[2]:.6f}, {maximum[2]:.6f}]"
        )

    report.add()
    report.add("4. STRAIGHT-DOWN CAMERA FOOTPRINT")
    report.add(
        "Assumption: optical axis is -world Z and camera image horizontal/vertical axes "
        "align with world X/Y. The rectangle uses the supplied pinhole values."
    )
    matching_cameras = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Camera):
            continue
        camera = UsdGeom.Camera(prim)
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time_code)
        position = matrix.ExtractTranslation()
        optical_axis = matrix.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0)).GetNormalized()
        position_error = math.sqrt(
            sum((float(position[index]) - CAMERA_POSITION[index]) ** 2 for index in range(3))
        )
        matching_cameras.append((position_error, prim, camera, position, optical_axis))
    if matching_cameras:
        _, prim, camera, position, optical_axis = min(matching_cameras, key=lambda item: item[0])
        straight_down = float(optical_axis[2]) <= -0.999
        report.add(
            f"stage verification candidate: {prim.GetPath()} position={vector_text(position)} "
            f"opticalAxis={vector_text(optical_axis)} straightDown={straight_down} "
            f"focalLength={camera.GetFocalLengthAttr().Get()} "
            f"apertures=({camera.GetHorizontalApertureAttr().Get()}, "
            f"{camera.GetVerticalApertureAttr().Get()})"
        )
        if not straight_down:
            report.add("WARNING: supplied straight-down orientation was not verified from the stage.")
    else:
        report.add("Camera orientation could not be verified: no UsdGeom.Camera prim was found.")

    def report_footprint(label: str, plane_z: float) -> None:
        distance = CAMERA_POSITION[2] - plane_z
        if distance <= 0.0:
            report.add(f"{label}: plane is not below the supplied camera; footprint unavailable")
            return
        half_x = distance * CAMERA_H_APERTURE / (2.0 * CAMERA_FOCAL)
        half_y = distance * CAMERA_V_APERTURE / (2.0 * CAMERA_FOCAL)
        report.add(
            f"{label}: Z={plane_z:.6f} X=[{CAMERA_POSITION[0] - half_x:.6f}, "
            f"{CAMERA_POSITION[0] + half_x:.6f}] Y=[{CAMERA_POSITION[1] - half_y:.6f}, "
            f"{CAMERA_POSITION[1] + half_y:.6f}]"
        )

    report_footprint("G cap-top plane", CAP_TOP_Z)
    if table_region_bounds is None:
        report.add("table-paint top plane: unavailable because no candidate region was found")
    else:
        table_top_z = float(table_region_bounds[1][2])
        report_footprint("table-paint top plane", table_top_z)

    text = report.render()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(text, end="", flush=True)


def main() -> int:
    # Isaac Sim initializes the USD/PXR runtime; do not move pxr imports above this.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        run_inspection()
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
