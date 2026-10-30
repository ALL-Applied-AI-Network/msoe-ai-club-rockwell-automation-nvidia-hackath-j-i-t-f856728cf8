"""Run the five-class vial classifier on all 30 G-fixture slots."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
from PIL import Image

try:
    from classifier_utils import CLASS_NAMES, PASS_INDEX
    from model_preprocessing import IMAGE_SIZE, build_evaluation_transform
except ModuleNotFoundError:
    from scripts.classifier_utils import CLASS_NAMES, PASS_INDEX
    from scripts.model_preprocessing import IMAGE_SIZE, build_evaluation_transform


EXPECTED_SLOTS = tuple(f"G{index:02d}" for index in range(1, 31))
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
DEFAULT_THRESHOLD = 0.90


@dataclass(frozen=True)
class SlotROI:
    slot: str
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class Prediction:
    slot: str
    predicted_class: str
    p_pass: float
    flag: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--watch-dir", type=Path)
    source.add_argument("--camera-index", type=int)
    parser.add_argument("--slots", type=Path, default=Path("slots_G.json"))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, default=Path("results/factory"))
    return parser.parse_args()


def load_slots(path: Path) -> list[SlotROI]:
    document = json.loads(path.read_text(encoding="utf-8"))
    records = document.get("slots")
    if not isinstance(records, list):
        raise ValueError("Slot document must contain a 'slots' list")
    slots = [
        SlotROI(
            slot=str(record["slot"]),
            x=int(record["x"]),
            y=int(record["y"]),
            w=int(record["w"]),
            h=int(record["h"]),
        )
        for record in records
    ]
    names = [slot.slot for slot in slots]
    if len(slots) != len(EXPECTED_SLOTS) or set(names) != set(EXPECTED_SLOTS):
        missing = sorted(set(EXPECTED_SLOTS) - set(names))
        extra = sorted(set(names) - set(EXPECTED_SLOTS))
        raise ValueError(
            f"Expected exactly G01..G30; missing={missing}, extra={extra}, "
            f"records={len(slots)}"
        )
    if len(names) != len(set(names)):
        raise ValueError("Slot document contains duplicate slot IDs")
    if any(slot.w <= 0 or slot.h <= 0 for slot in slots):
        raise ValueError("Every slot ROI must have positive width and height")
    by_name = {slot.slot: slot for slot in slots}
    return [by_name[name] for name in EXPECTED_SLOTS]


def extract_crops(image_bgr: np.ndarray, slots: Sequence[SlotROI]) -> list[Image.Image]:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Input image must have three BGR color channels")
    height, width = image_bgr.shape[:2]
    crops = []
    for slot in slots:
        x1, y1 = slot.x + slot.w, slot.y + slot.h
        if slot.x < 0 or slot.y < 0 or x1 > width or y1 > height:
            raise ValueError(
                f"ROI {slot.slot} ({slot.x}, {slot.y}, {slot.w}, {slot.h}) "
                f"is outside image size {width}x{height}"
            )
        crop_bgr = image_bgr[slot.y:y1, slot.x:x1]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crops.append(Image.fromarray(crop_rgb))
    return crops


def probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            f"Expected logits shaped (N, {len(CLASS_NAMES)}), got {values.shape}"
        )
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def predictions_from_probabilities(
    slots: Sequence[SlotROI], probabilities: np.ndarray, threshold: float
) -> list[Prediction]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(slots), len(CLASS_NAMES)):
        raise ValueError(
            f"Expected probabilities shaped ({len(slots)}, {len(CLASS_NAMES)}), "
            f"got {values.shape}"
        )
    predictions = []
    for slot, row in zip(slots, values, strict=True):
        class_index = int(np.argmax(row))
        p_pass = float(row[PASS_INDEX])
        predictions.append(
            Prediction(
                slot=slot.slot,
                predicted_class=CLASS_NAMES[class_index],
                p_pass=p_pass,
                flag="DEFECT" if p_pass < threshold else "PASS",
            )
        )
    return predictions


class TorchScriptClassifier:
    """Offline, CPU-only wrapper around the TorchScript export from training."""

    def __init__(self, weights: Path) -> None:
        import torch

        self._torch = torch
        self._transform = build_evaluation_transform(IMAGE_SIZE)
        self._model = torch.jit.load(str(weights), map_location="cpu")
        self._model.eval()

    def __call__(self, crops: Sequence[Image.Image]) -> np.ndarray:
        batch = self._torch.stack([self._transform(crop) for crop in crops])
        with self._torch.inference_mode():
            output = self._model(batch)
        if isinstance(output, (tuple, list)):
            output = output[0]
        logits = output.detach().cpu().numpy()
        return probabilities_from_logits(logits)


def draw_overlay(
    image_bgr: np.ndarray,
    slots: Sequence[SlotROI],
    predictions: Sequence[Prediction],
) -> np.ndarray:
    overlay = image_bgr.copy()
    for slot, prediction in zip(slots, predictions, strict=True):
        color = (0, 190, 0) if prediction.flag == "PASS" else (0, 0, 255)
        top_left = (slot.x, slot.y)
        bottom_right = (slot.x + slot.w, slot.y + slot.h)
        cv2.rectangle(overlay, top_left, bottom_right, color, 2, cv2.LINE_AA)
        text_origin = (slot.x, max(14, slot.y - 5))
        cv2.putText(
            overlay,
            slot.slot,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay


def write_outputs(
    source_name: str,
    image_bgr: np.ndarray,
    slots: Sequence[SlotROI],
    predictions: Sequence[Prediction],
    threshold: float,
    out_dir: Path,
) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(source_name).stem
    csv_path = out_dir / f"{stem}_predictions.csv"
    json_path = out_dir / f"{stem}_predictions.json"
    overlay_path = out_dir / f"{stem}_overlay.png"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["slot", "predicted_class", "p_pass", "flag"]
        )
        writer.writeheader()
        for prediction in predictions:
            writer.writerow(
                {
                    "slot": prediction.slot,
                    "predicted_class": prediction.predicted_class,
                    "p_pass": f"{prediction.p_pass:.8f}",
                    "flag": prediction.flag,
                }
            )
    json_path.write_text(
        json.dumps(
            {
                "image": source_name,
                "threshold": threshold,
                "predictions": [
                    {
                        "slot": prediction.slot,
                        "predicted_class": prediction.predicted_class,
                        "p_pass": prediction.p_pass,
                        "flag": prediction.flag,
                    }
                    for prediction in predictions
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    overlay = draw_overlay(image_bgr, slots, predictions)
    if not cv2.imwrite(str(overlay_path), overlay):
        raise OSError(f"Could not write overlay: {overlay_path}")
    return csv_path, json_path, overlay_path


def process_frame(
    source_name: str,
    image_bgr: np.ndarray,
    slots: Sequence[SlotROI],
    classifier: Callable[[Sequence[Image.Image]], np.ndarray],
    threshold: float,
    out_dir: Path,
) -> list[Prediction]:
    crops = extract_crops(image_bgr, slots)
    probabilities = classifier(crops)
    predictions = predictions_from_probabilities(slots, probabilities, threshold)
    write_outputs(source_name, image_bgr, slots, predictions, threshold, out_dir)
    defects = sum(prediction.flag == "DEFECT" for prediction in predictions)
    print(
        f"image={source_name} slots={len(predictions)} pass={len(predictions) - defects} "
        f"defects={defects} threshold={threshold:.3f}",
        flush=True,
    )
    return predictions


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def watch_directory(
    directory: Path,
    process_path: Callable[[Path], None],
    poll_seconds: float,
) -> None:
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    seen: set[Path] = set()
    while True:
        candidates = sorted(
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
            and not path.stem.endswith("_overlay")
        )
        for path in candidates:
            resolved = path.resolve()
            if resolved not in seen:
                process_path(path)
                seen.add(resolved)
        time.sleep(poll_seconds)


def watch_camera(
    camera_index: int,
    process_camera_frame: Callable[[str, np.ndarray], None],
    poll_seconds: float,
) -> None:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read camera index {camera_index}")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            process_camera_frame(f"camera{camera_index}_{timestamp}.png", frame)
            time.sleep(poll_seconds)
    finally:
        capture.release()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.poll_seconds <= 0.0:
        raise ValueError("--poll-seconds must be positive")
    slots = load_slots(args.slots)
    classifier = TorchScriptClassifier(args.weights)

    def process_path(path: Path) -> None:
        process_frame(
            path.name,
            read_image(path),
            slots,
            classifier,
            args.threshold,
            args.out_dir,
        )

    def process_camera_frame(source_name: str, frame: np.ndarray) -> None:
        process_frame(
            source_name,
            frame,
            slots,
            classifier,
            args.threshold,
            args.out_dir,
        )

    try:
        if args.image is not None:
            process_path(args.image)
        elif args.watch_dir is not None:
            watch_directory(args.watch_dir, process_path, args.poll_seconds)
        else:
            watch_camera(
                args.camera_index,
                process_camera_frame,
                args.poll_seconds,
            )
    except KeyboardInterrupt:
        print("stopped", flush=True)


if __name__ == "__main__":
    main()
