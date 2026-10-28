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


@dataclass(frozen=True)
class FillScene:
    labels: dict[str, str]
    not_yet_filled: list[str]
    expected_through: int
    missed_position: str | None


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
