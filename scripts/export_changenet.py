"""Export generated vial crops to the Visual ChangeNet classify layout."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


FIXTURES = ("G", "S")
CLASSES = ("pass", "missing_cap", "crooked_cap", "damaged_vial", "no_vial")
EXPECTED_SLOTS = tuple(
    f"{fixture}{position:02d}"
    for fixture in FIXTURES
    for position in range(1, 31)
)
CSV_COLUMNS = ("input_path", "object_name", "golden_path", "label")
META_COLUMNS = (
    "split",
    "input_path",
    "object_name",
    "golden_path",
    "label",
    "orig_class",
    "slot",
    "image_id",
    "fixture",
    "golden_source_image_id",
    "golden_source_slot",
    "golden_fallback",
)


@dataclass(frozen=True)
class ExportSummary:
    """Counts produced by one export."""

    split_counts: Counter[tuple[str, str, str]]
    golden_counts: Counter[tuple[str, str]]
    fallback_counts: Counter[tuple[str, str]]
    row_counts: Counter[tuple[str, str]]


@dataclass(frozen=True)
class GoldenSource:
    """A train-split PASS crop eligible to supply golden content."""

    image_id: str
    slot: str
    crop: Path


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _load_images(labels_path: Path) -> list[dict[str, object]]:
    try:
        document = json.loads(labels_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Generator labels file not found: {labels_path}") from exc

    images = document.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError(f"Expected a non-empty 'images' list in {labels_path}")
    return images


def export_dataset(
    generator_out: Path,
    output_dir: Path,
    *,
    valid_images: Iterable[str] = ("img_0003",),
    seed: int = 0,
) -> ExportSummary:
    """Copy crops and write ChangeNet CSV files.

    All images not named in ``valid_images`` are assigned to the training split.
    Golden crops are PASS examples from training images only.
    """

    generator_out = generator_out.resolve()
    output_dir = output_dir.resolve()
    labels_path = generator_out / "labels.json"
    crops_dir = generator_out / "crops"
    if not crops_dir.is_dir():
        raise FileNotFoundError(f"Generator crops directory not found: {crops_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    valid_image_set = set(valid_images)
    images = _load_images(labels_path)
    seen_image_ids: set[str] = set()
    split_rows: dict[str, list[dict[str, str]]] = {"train": [], "valid": []}
    pending_rows: list[tuple[str, str, str, Path, dict[str, str]]] = []
    split_counts: Counter[tuple[str, str, str]] = Counter()
    golden_counts: Counter[tuple[str, str]] = Counter()
    fallback_counts: Counter[tuple[str, str]] = Counter()
    row_counts: Counter[tuple[str, str]] = Counter()
    pass_by_slot: dict[str, list[GoldenSource]] = {
        slot: [] for slot in EXPECTED_SLOTS
    }
    pass_by_fixture: dict[str, list[GoldenSource]] = {
        fixture: [] for fixture in FIXTURES
    }

    for image_record in images:
        image_id = image_record.get("image")
        labels = image_record.get("labels")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("Every image record must have a non-empty string 'image'")
        if image_id in seen_image_ids:
            raise ValueError(f"Duplicate image id in labels.json: {image_id}")
        seen_image_ids.add(image_id)
        if not isinstance(labels, dict):
            raise ValueError(f"Image {image_id} has no labels mapping")
        if set(labels) != set(EXPECTED_SLOTS):
            missing = sorted(set(EXPECTED_SLOTS) - set(labels))
            extra = sorted(set(labels) - set(EXPECTED_SLOTS))
            raise ValueError(
                f"Image {image_id} must contain exactly G01-G30 and S01-S30; "
                f"missing={missing}, extra={extra}"
            )

        split = "valid" if image_id in valid_image_set else "train"
        for slot in EXPECTED_SLOTS:
            orig_class = labels[slot]
            if orig_class not in CLASSES:
                raise ValueError(f"Unknown class for {image_id}/{slot}: {orig_class!r}")
            fixture = slot[0]
            crop_name = f"{image_id}_{slot}.png"
            source_crop = crops_dir / crop_name
            if not source_crop.is_file():
                raise FileNotFoundError(f"Crop listed by labels.json is missing: {source_crop}")

            relative_input = Path(fixture) / image_id
            destination_dir = output_dir / "images" / relative_input
            destination_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_crop, destination_dir / crop_name)

            golden_path = Path("golden") / fixture
            label = "PASS" if orig_class == "pass" else orig_class
            row = {
                "input_path": relative_input.as_posix(),
                "object_name": source_crop.stem,
                "golden_path": golden_path.as_posix(),
                "label": label,
            }
            split_rows[split].append(row)
            pending_rows.append((split, image_id, slot, source_crop, row))
            split_counts[(split, fixture, orig_class)] += 1
            row_counts[(split, fixture)] += 1
            if split == "train" and orig_class == "pass":
                source = GoldenSource(image_id=image_id, slot=slot, crop=source_crop)
                pass_by_slot[slot].append(source)
                pass_by_fixture[fixture].append(source)

    missing_valid = valid_image_set - seen_image_ids
    if missing_valid:
        raise ValueError(f"Validation image ids not present in labels.json: {sorted(missing_valid)}")

    for fixture in FIXTURES:
        if not pass_by_fixture[fixture]:
            raise ValueError(
                f"Fixture {fixture} has no train PASS crops available for golden images"
            )

    rng = random.Random(seed)
    meta_rows: list[dict[str, str]] = []
    for split, image_id, slot, source_crop, row in pending_rows:
        fixture = slot[0]
        candidates = pass_by_slot[slot]
        fallback = not candidates
        if fallback:
            candidates = pass_by_fixture[fixture]
        different_image = [
            candidate for candidate in candidates if candidate.image_id != image_id
        ]
        selection_pool = different_image or candidates
        golden_source = rng.choice(selection_pool)
        golden_dir = output_dir / "images" / row["golden_path"]
        golden_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(golden_source.crop, golden_dir / f"{source_crop.stem}.png")
        golden_counts[(split, fixture)] += 1
        if fallback:
            fallback_counts[(split, fixture)] += 1
        meta_rows.append(
            {
                "split": split,
                **row,
                "orig_class": row["label"] if row["label"] != "PASS" else "pass",
                "slot": slot,
                "image_id": image_id,
                "fixture": fixture,
                "golden_source_image_id": golden_source.image_id,
                "golden_source_slot": golden_source.slot,
                "golden_fallback": str(fallback).lower(),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "train.csv", CSV_COLUMNS, split_rows["train"])
    _write_csv(output_dir / "valid.csv", CSV_COLUMNS, split_rows["valid"])
    _write_csv(output_dir / "meta.csv", META_COLUMNS, meta_rows)
    return ExportSummary(
        split_counts=split_counts,
        golden_counts=golden_counts,
        fallback_counts=fallback_counts,
        row_counts=row_counts,
    )


def _print_summary(summary: ExportSummary, output_dir: Path) -> None:
    for split in ("train", "valid"):
        for fixture in FIXTURES:
            counts = ", ".join(
                f"{class_name}={summary.split_counts[(split, fixture, class_name)]}"
                for class_name in CLASSES
            )
            print(
                f"{split} {fixture}: rows={summary.row_counts[(split, fixture)]}, "
                f"golden_files={summary.golden_counts[(split, fixture)]}, "
                f"fallbacks={summary.fallback_counts[(split, fixture)]}; {counts}"
            )
    print(f"wrote: {output_dir.resolve()}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generator_out", type=Path, help="Folder containing labels.json and crops/")
    parser.add_argument("output_dir", type=Path, help="Destination ChangeNet dataset folder")
    parser.add_argument(
        "--valid-images",
        nargs="+",
        default=["img_0003"],
        metavar="IMAGE_ID",
        help="Image ids assigned wholly to validation (default: img_0003)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic golden-source selection (default: 0)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = export_dataset(
        args.generator_out,
        args.output_dir,
        valid_images=args.valid_images,
        seed=args.seed,
    )
    _print_summary(summary, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
