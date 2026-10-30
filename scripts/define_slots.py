"""Define the 30 G-slot crop centers interactively with OpenCV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


SLOT_NAMES = tuple(f"G{index:02d}" for index in range(1, 31))
DEFAULT_CROP_OFFSETS = (-38, -47, 42, 43)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("slots_G.json"))
    parser.add_argument(
        "--crop-offsets",
        type=int,
        nargs=4,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        default=DEFAULT_CROP_OFFSETS,
        help="Offsets from each clicked center; defaults to the 80x90 G training crop",
    )
    return parser.parse_args()


def slot_document(
    image_width: int,
    image_height: int,
    centers: list[tuple[int, int]],
    crop_offsets: tuple[int, int, int, int],
) -> dict[str, object]:
    if len(centers) != len(SLOT_NAMES):
        raise ValueError(f"Expected 30 centers, got {len(centers)}")
    left, top, right, bottom = crop_offsets
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0:
        raise ValueError("Crop offsets must define a positive rectangle")
    slots = []
    for slot, (center_x, center_y) in zip(SLOT_NAMES, centers, strict=True):
        x, y = center_x + left, center_y + top
        if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
            raise ValueError(f"Crop for {slot} extends outside the image")
        slots.append({"slot": slot, "x": x, "y": y, "w": width, "h": height})
    return {
        "image_size": [image_width, image_height],
        "crop_size": [width, height],
        "crop_offsets": list(crop_offsets),
        "slots": slots,
    }


def draw_state(
    image,
    centers: list[tuple[int, int]],
    crop_offsets: tuple[int, int, int, int],
):
    canvas = image.copy()
    left, top, right, bottom = crop_offsets
    for index, (center_x, center_y) in enumerate(centers):
        slot = SLOT_NAMES[index]
        cv2.rectangle(
            canvas,
            (center_x + left, center_y + top),
            (center_x + right, center_y + bottom),
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(canvas, (center_x, center_y), 3, (0, 0, 255), -1)
        cv2.putText(
            canvas,
            slot,
            (center_x + left, max(14, center_y + top - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )
    next_slot = SLOT_NAMES[len(centers)] if len(centers) < len(SLOT_NAMES) else "done"
    cv2.putText(
        canvas,
        f"click {next_slot} | right-click/U undo | R reset | Enter save | Esc cancel",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def main() -> None:
    args = parse_args()
    crop_offsets = tuple(args.crop_offsets)
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    centers: list[tuple[int, int]] = []
    window = "Define G01..G30"

    def mouse_callback(event, x, y, _flags, _parameter) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(centers) < len(SLOT_NAMES):
            centers.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and centers:
            centers.pop()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse_callback)
    while True:
        cv2.imshow(window, draw_state(image, centers, crop_offsets))
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            cv2.destroyAllWindows()
            print("cancelled")
            return
        if key in (8, ord("u")) and centers:
            centers.pop()
        elif key == ord("r"):
            centers.clear()
        elif key in (10, 13) and len(centers) == len(SLOT_NAMES):
            break
    cv2.destroyAllWindows()
    height, width = image.shape[:2]
    document = slot_document(width, height, centers, crop_offsets)
    args.out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} with {len(centers)} slots")


if __name__ == "__main__":
    main()
