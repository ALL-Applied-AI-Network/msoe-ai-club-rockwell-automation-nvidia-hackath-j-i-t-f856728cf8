"""Evaluate a TorchScript classifier on labeled, test-only real crops."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

try:
    from classifier_utils import CLASS_NAMES
    from predict import DEFAULT_THRESHOLD, TorchScriptClassifier
except ModuleNotFoundError:
    from scripts.classifier_utils import CLASS_NAMES
    from scripts.predict import DEFAULT_THRESHOLD, TorchScriptClassifier


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--crops-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def ensure_external_output(path: Path) -> None:
    try:
        path.resolve().relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        return
    raise ValueError(f"Refusing to write real-image evaluation inside repository: {path}")


def load_labeled_crops(
    labels_path: Path, crops_dir: Path
) -> tuple[list[dict[str, str]], list[Image.Image], list[int]]:
    with labels_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"tray", "slot", "label", "model_pred", "p_pass"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"labels CSV must contain columns {sorted(required)}")
    if len(rows) != 60:
        raise ValueError(f"Expected 60 labeled real crops, got {len(rows)}")
    identities = [(row["tray"], row["slot"]) for row in rows]
    if len(set(identities)) != len(rows):
        raise ValueError("labels CSV contains duplicate tray/slot rows")
    images = []
    targets = []
    for row in rows:
        if row["label"] not in CLASS_NAMES:
            raise ValueError(f"Unknown or blank label for {row['tray']} {row['slot']}")
        crop_path = crops_dir / f"{row['tray']}_{row['slot']}.png"
        with Image.open(crop_path) as image:
            images.append(image.convert("RGB"))
        targets.append(CLASS_NAMES.index(row["label"]))
    return rows, images, targets


def confusion_matrix(
    targets: Sequence[int], predictions: Sequence[int]
) -> list[list[int]]:
    matrix = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    for target, prediction in zip(targets, predictions, strict=True):
        matrix[int(target)][int(prediction)] += 1
    return matrix


def per_class_recall(matrix: Sequence[Sequence[int]]) -> dict[str, float | None]:
    recalls = {}
    for index, name in enumerate(CLASS_NAMES):
        support = sum(int(value) for value in matrix[index])
        recalls[name] = int(matrix[index][index]) / support if support else None
    return recalls


def binary_numbers(
    targets: Sequence[int], probabilities: np.ndarray, threshold: float
) -> dict[str, int | float]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(targets), len(CLASS_NAMES)):
        raise ValueError(
            f"Expected probabilities shaped ({len(targets)}, {len(CLASS_NAMES)}), "
            f"got {values.shape}"
        )
    actual_defect = [int(target) != 0 for target in targets]
    predicted_defect = [float(row[0]) < threshold for row in values]
    defect_count = sum(actual_defect)
    pass_count = len(actual_defect) - defect_count
    defects_missed = sum(
        actual and not predicted
        for actual, predicted in zip(actual_defect, predicted_defect, strict=True)
    )
    false_alarms = sum(
        not actual and predicted
        for actual, predicted in zip(actual_defect, predicted_defect, strict=True)
    )
    return {
        "threshold": threshold,
        "defect_count": defect_count,
        "pass_count": pass_count,
        "defects_missed": defects_missed,
        "defect_miss_rate": defects_missed / max(1, defect_count),
        "false_alarms_on_pass": false_alarms,
        "false_alarm_rate": false_alarms / max(1, pass_count),
    }


def evaluate_probabilities(
    targets: Sequence[int], probabilities: np.ndarray, threshold: float
) -> dict[str, object]:
    values = np.asarray(probabilities, dtype=np.float64)
    predictions = np.argmax(values, axis=1).astype(int).tolist()
    matrix = confusion_matrix(targets, predictions)
    sweep = [binary_numbers(targets, values, step / 10.0) for step in range(1, 10)]
    return {
        "class_order": list(CLASS_NAMES),
        "sample_count": len(targets),
        "confusion_matrix": matrix,
        "per_class_recall": per_class_recall(matrix),
        "binary": binary_numbers(targets, values, threshold),
        "threshold_sweep": sweep,
    }


def print_report(report: dict[str, object]) -> None:
    print("5-class confusion matrix (rows=true, columns=predicted)")
    print("true\\pred," + ",".join(CLASS_NAMES))
    matrix = report["confusion_matrix"]
    for name, row in zip(CLASS_NAMES, matrix, strict=True):
        print(name + "," + ",".join(str(value) for value in row))
    print("per-class recall")
    for name, recall in report["per_class_recall"].items():
        value = "n/a" if recall is None else f"{recall:.4f}"
        print(f"{name}={value}")
    numbers = report["binary"]
    print(
        f"binary threshold={numbers['threshold']:.2f} "
        f"defects_missed={numbers['defects_missed']}/{numbers['defect_count']} "
        f"false_alarms_on_pass={numbers['false_alarms_on_pass']}/{numbers['pass_count']}"
    )
    print("threshold sweep")
    for row in report["threshold_sweep"]:
        print(
            f"threshold={row['threshold']:.1f} "
            f"defects_missed={row['defects_missed']}/{row['defect_count']} "
            f"false_alarms_on_pass={row['false_alarms_on_pass']}/{row['pass_count']}"
        )


def write_report(report: dict[str, object], out_dir: Path) -> None:
    ensure_external_output(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "real_evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (out_dir / "threshold_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "threshold",
            "defect_count",
            "pass_count",
            "defects_missed",
            "defect_miss_rate",
            "false_alarms_on_pass",
            "false_alarm_rate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["threshold_sweep"])


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    rows, images, targets = load_labeled_crops(args.labels, args.crops_dir)
    classifier = TorchScriptClassifier(args.weights)
    probabilities = classifier(images)
    report = evaluate_probabilities(targets, probabilities, args.threshold)
    report["labels"] = str(args.labels)
    report["crops_dir"] = str(args.crops_dir)
    report["weights"] = str(args.weights)
    print_report(report)
    write_report(report, args.out_dir)


if __name__ == "__main__":
    main()
