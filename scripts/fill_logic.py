"""Pure fill-order scene sampling for the tray generator."""

from __future__ import annotations

import random
from dataclasses import dataclass


SLOT_NAMES = tuple(f"G{index:02d}" for index in range(1, 31))
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
    rng: random.Random, *, class_balance: bool = False
) -> FillScene:
    """Sample one fill-prefix scene without importing Isaac Sim."""
    expected_through = rng.choices(range(31), weights=fill_count_weights(), k=1)[0]
    if class_balance:
        expected_through = max(3, expected_through)

    missed_position = None
    if not class_balance and expected_through >= 2 and rng.random() < 0.03:
        missed_index = rng.randint(1, expected_through - 1)
        missed_position = f"G{missed_index:02d}"

    labels: dict[str, str] = {}
    not_yet_filled = list(SLOT_NAMES[expected_through:])
    for index, slot in enumerate(SLOT_NAMES, start=1):
        if index > expected_through or slot == missed_position:
            labels[slot] = "no_vial"
        else:
            labels[slot] = rng.choices(
                OCCUPIED_CLASSES, weights=OCCUPIED_WEIGHTS, k=1
            )[0]

    if class_balance:
        forced_slots = rng.sample(list(SLOT_NAMES[:expected_through]), 3)
        forced_classes = ["damaged_vial", "crooked_cap", "missing_cap"]
        rng.shuffle(forced_classes)
        for slot, label in zip(forced_slots, forced_classes, strict=True):
            labels[slot] = label

    return FillScene(
        labels=labels,
        not_yet_filled=not_yet_filled,
        expected_through=expected_through,
        missed_position=missed_position,
    )
