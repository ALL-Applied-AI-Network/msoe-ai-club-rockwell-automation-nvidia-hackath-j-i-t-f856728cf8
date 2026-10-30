"""Pure manifest and decision-rule helpers for crop classification."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Sequence


CLASS_NAMES = ("pass", "missing_cap", "crooked_cap", "damaged_vial", "no_vial")
DEFECT_INDICES = (1, 2, 3)
PASS_INDEX = 0
NO_VIAL_INDEX = 4


def load_manifest(
    data_root: Path,
    split: str,
    manifest_path: Path | None = None,
) -> list[dict[str, str]]:
    """Load one split and resolve its crop paths relative to ``data_root``."""
    if manifest_path is None:
        manifest_path = data_root / "manifest.csv"
    required = {"path", "label", "split", "job", "image_id", "slot", "damage_type"}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or ()))
            raise ValueError(f"Manifest is missing columns: {missing}")
        manifest_rows = [dict(row) for row in reader]
    missing_paths = [
        data_root / Path(row["path"])
        for row in manifest_rows
        if not (data_root / Path(row["path"])).is_file()
    ]
    if missing_paths:
        details = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            f"Manifest references {len(missing_paths)} missing file(s):\n{details}"
        )
    rows = [row for row in manifest_rows if row["split"] == split]
    if not rows:
        raise ValueError(f"Manifest has no rows for split {split!r}")
    for row in rows:
        if row["label"] not in CLASS_NAMES:
            raise ValueError(f"Unknown label {row['label']!r}")
        path = data_root / Path(row["path"])
        row["absolute_path"] = str(path)
    return rows


def _as_rows(probabilities) -> list[list[float]]:
    if hasattr(probabilities, "tolist"):
        probabilities = probabilities.tolist()
    return [[float(value) for value in row] for row in probabilities]


def _as_targets(targets) -> list[int]:
    if hasattr(targets, "tolist"):
        targets = targets.tolist()
    return [int(value) for value in targets]


def separate_no_vial_predictions(probabilities) -> list[int]:
    """Use no_vial only when it is the confident argmax; otherwise classify 0..3."""
    predictions = []
    for row in _as_rows(probabilities):
        raw = max(range(len(row)), key=row.__getitem__)
        occupied = max(range(NO_VIAL_INDEX), key=row.__getitem__)
        predictions.append(
            NO_VIAL_INDEX
            if raw == NO_VIAL_INDEX and row[NO_VIAL_INDEX] > 0.5
            else occupied
        )
    return predictions


def write_probability_csv(
    path: Path,
    rows: Sequence[dict[str, str]],
    probabilities,
    targets,
    indices,
) -> None:
    """Write per-sample class probabilities in dataset-inference order."""
    probability_rows = _as_rows(probabilities)
    target_rows = _as_targets(targets)
    index_rows = _as_targets(indices)
    predictions = separate_no_vial_predictions(probability_rows)
    probability_fields = [f"p_{name}" for name in CLASS_NAMES]
    fieldnames = [
        "path",
        "label",
        "job",
        "image_id",
        "slot",
        "damage_type",
        "target_class",
        "predicted_class",
        *probability_fields,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for probability_row, target, index, prediction in zip(
            probability_rows,
            target_rows,
            index_rows,
            predictions,
            strict=True,
        ):
            manifest_row = rows[index]
            output_row = {
                field: manifest_row.get(field, "")
                for field in (
                    "path",
                    "label",
                    "job",
                    "image_id",
                    "slot",
                    "damage_type",
                )
            }
            output_row["target_class"] = CLASS_NAMES[target]
            output_row["predicted_class"] = CLASS_NAMES[prediction]
            output_row.update(
                {
                    field: f"{probability:.10f}"
                    for field, probability in zip(
                        probability_fields, probability_row, strict=True
                    )
                }
            )
            writer.writerow(output_row)


def binary_defect_predictions(
    probabilities, tau: float
) -> list[bool]:
    rows = _as_rows(probabilities)
    class_predictions = separate_no_vial_predictions(probabilities)
    return [
        prediction in DEFECT_INDICES or row[PASS_INDEX] < tau
        for row, prediction in zip(rows, class_predictions, strict=True)
    ]


def binary_numbers(
    probabilities, targets, tau: float
) -> dict[str, float | int]:
    relevant = [
        (row, target)
        for row, target in zip(
            _as_rows(probabilities), _as_targets(targets), strict=True
        )
        if target != NO_VIAL_INDEX
    ]
    rows = [row for row, _target in relevant]
    actual = [target in DEFECT_INDICES for _row, target in relevant]
    predicted = binary_defect_predictions(rows, tau)
    true_positive = sum(a and p for a, p in zip(actual, predicted, strict=True))
    false_negative = sum(a and not p for a, p in zip(actual, predicted, strict=True))
    false_positive = sum(not a and p for a, p in zip(actual, predicted, strict=True))
    true_negative = sum(not a and not p for a, p in zip(actual, predicted, strict=True))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1.0e-12, precision + recall)
    false_alarm_rate = false_positive / max(1, false_positive + true_negative)
    return {
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": false_alarm_rate,
    }


def select_tau(
    probabilities,
    targets,
    minimum_recall: float = 0.99,
) -> tuple[float, dict[str, float | int]]:
    """Choose the lowest-FAR validation threshold meeting defect recall."""
    relevant = [
        (row, target)
        for row, target in zip(
            _as_rows(probabilities), _as_targets(targets), strict=True
        )
        if target != NO_VIAL_INDEX
    ]
    probabilities = [row for row, _target in relevant]
    targets = [target for _row, target in relevant]
    pass_probabilities = [row[PASS_INDEX] for row in probabilities]
    candidates = {0.0, 1.0}
    candidates.update(
        math.nextafter(value, 1.0) for value in pass_probabilities
    )
    evaluated = [
        (tau, binary_numbers(probabilities, targets, tau))
        for tau in sorted(candidates)
    ]
    reachable = [item for item in evaluated if item[1]["recall"] >= minimum_recall]
    if reachable:
        tau, numbers = min(
            reachable,
            key=lambda item: (
                item[1]["false_alarm_rate"],
                -item[1]["recall"],
                item[0],
            ),
        )
    else:
        tau, numbers = min(
            evaluated,
            key=lambda item: (
                -item[1]["recall"],
                item[1]["false_alarm_rate"],
                item[0],
            ),
        )
    return float(tau), numbers
