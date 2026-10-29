"""Pure fill-order scene sampling for the tray generator."""

from __future__ import annotations

import random
from dataclasses import dataclass


G_SLOT_NAMES = tuple(f"G{index:02d}" for index in range(1, 31))
S_SLOT_NAMES = tuple(f"S{index:02d}" for index in range(1, 31))
ALL_SLOT_NAMES = G_SLOT_NAMES + S_SLOT_NAMES
# Backward-compatible name for callers that sample the original G fixture.
SLOT_NAMES = G_SLOT_NAMES
OCCUPIED_CLASSES = ("pass", "missing_cap", "crooked_cap", "damaged_vial")
OCCUPIED_WEIGHTS = (0.80, 0.07, 0.07, 0.06)
QUOTA_DEFECT_CLASSES = ("missing_cap", "crooked_cap", "damaged_vial")
QUOTA_DAMAGE_TYPES = ("bend", "crush", "lean_crush", "broken_top")


@dataclass(frozen=True)
class FillScene:
    labels: dict[str, str]
    not_yet_filled: list[str]
    expected_through: int
    missed_position: str | None


def build_golden_fill_scene(prefix: str) -> FillScene:
    """Return a completely filled, all-pass fixture for reference renders."""
    if prefix not in ("G", "S"):
        raise ValueError("prefix must be 'G' or 'S'")
    slot_names = G_SLOT_NAMES if prefix == "G" else S_SLOT_NAMES
    return FillScene(
        labels={slot: "pass" for slot in slot_names},
        not_yet_filled=[],
        expected_through=30,
        missed_position=None,
    )


def build_quota_schedule(
    image_count: int,
    defects_per_fixture: int,
) -> dict[str, list[list[dict[str, str | None]]]]:
    """Build balanced per-fixture defect and damage schedules for a run."""
    if image_count < 1:
        raise ValueError("image_count must be at least 1")
    if not 0 <= defects_per_fixture <= 30:
        raise ValueError("defects_per_fixture must be from 0 to 30")

    schedules: dict[str, list[list[dict[str, str | None]]]] = {}
    for prefix in ("G", "S"):
        class_index = 0
        damage_index = 0
        fixture_schedule = []
        for _ in range(image_count):
            image_assignments = []
            for _ in range(defects_per_fixture):
                label = QUOTA_DEFECT_CLASSES[class_index % len(QUOTA_DEFECT_CLASSES)]
                class_index += 1
                damage_type = None
                if label == "damaged_vial":
                    damage_type = QUOTA_DAMAGE_TYPES[
                        damage_index % len(QUOTA_DAMAGE_TYPES)
                    ]
                    damage_index += 1
                image_assignments.append(
                    {"class": label, "damage_type": damage_type}
                )
            fixture_schedule.append(image_assignments)
        schedules[prefix] = fixture_schedule
    return schedules


def apply_quota_assignments(
    scene: FillScene,
    *,
    prefix: str,
    assignments: list[dict[str, str | None]],
    rng: random.Random,
) -> tuple[FillScene, dict[str, str]]:
    """Apply exact scheduled defects to distinct slots in a filled prefix."""
    if prefix not in ("G", "S"):
        raise ValueError("prefix must be 'G' or 'S'")
    if len(assignments) > 30:
        raise ValueError("at most 30 defects may be assigned per fixture")
    slot_names = G_SLOT_NAMES if prefix == "G" else S_SLOT_NAMES
    filled_count = max(scene.expected_through, len(assignments))
    filled_slots = list(slot_names[:filled_count])
    selected_slots = rng.sample(filled_slots, len(assignments))
    labels = {
        slot: "pass" if index < filled_count else "no_vial"
        for index, slot in enumerate(slot_names)
    }
    damage_types = {}
    for slot, assignment in zip(selected_slots, assignments, strict=True):
        label = assignment["class"]
        if label not in QUOTA_DEFECT_CLASSES:
            raise ValueError(f"Unsupported quota class: {label!r}")
        labels[slot] = label
        damage_type = assignment.get("damage_type")
        if label == "damaged_vial":
            if damage_type not in QUOTA_DAMAGE_TYPES:
                raise ValueError(
                    f"Unsupported quota damage type: {damage_type!r}"
                )
            damage_types[slot] = str(damage_type)
        elif damage_type is not None:
            raise ValueError("Only damaged_vial assignments may have a damage type")

    return (
        FillScene(
            labels=labels,
            not_yet_filled=list(slot_names[filled_count:]),
            expected_through=filled_count,
            missed_position=None,
        ),
        damage_types,
    )


def fill_count_weights() -> tuple[float, ...]:
    """Favor later fill stages while retaining explicit empty/full coverage."""
    weights = []
    for count in range(31):
        if count == 0:
            weights.append(2.0)
        elif count < 10:
            weights.append(0.45)
        elif count < 30:
            weights.append(2.0 + 0.06 * (count - 10))
        else:
            weights.append(4.0)
    return tuple(weights)


def sample_fill_scene(
    rng: random.Random,
    *,
    prefix: str = "G",
    class_balance: bool = False,
    damage_weight: float = 1.0,
) -> FillScene:
    """Sample one fill-prefix scene without importing Isaac Sim."""
    if prefix not in ("G", "S"):
        raise ValueError("prefix must be 'G' or 'S'")
    slot_names = G_SLOT_NAMES if prefix == "G" else S_SLOT_NAMES
    if damage_weight < 0.0:
        raise ValueError("damage_weight must be non-negative")
    expected_through = rng.choices(range(31), weights=fill_count_weights(), k=1)[0]
    if class_balance:
        required_defects = 3 if damage_weight > 0.0 else 2
        expected_through = max(required_defects, expected_through)

    missed_position = None
    if not class_balance and expected_through >= 2 and rng.random() < 0.03:
        missed_index = rng.randint(1, expected_through - 1)
        missed_position = f"{prefix}{missed_index:02d}"

    labels: dict[str, str] = {}
    not_yet_filled = list(slot_names[expected_through:])
    for index, slot in enumerate(slot_names, start=1):
        if index > expected_through or slot == missed_position:
            labels[slot] = "no_vial"
        else:
            occupied_weights = (*OCCUPIED_WEIGHTS[:3], OCCUPIED_WEIGHTS[3] * damage_weight)
            labels[slot] = rng.choices(
                OCCUPIED_CLASSES, weights=occupied_weights, k=1
            )[0]

    if class_balance:
        forced_classes = ["crooked_cap", "missing_cap"]
        if damage_weight > 0.0:
            forced_classes.append("damaged_vial")
        forced_slots = rng.sample(
            list(slot_names[:expected_through]), len(forced_classes)
        )
        rng.shuffle(forced_classes)
        for slot, label in zip(forced_slots, forced_classes, strict=True):
            labels[slot] = label

    return FillScene(
        labels=labels,
        not_yet_filled=not_yet_filled,
        expected_through=expected_through,
        missed_position=missed_position,
    )
