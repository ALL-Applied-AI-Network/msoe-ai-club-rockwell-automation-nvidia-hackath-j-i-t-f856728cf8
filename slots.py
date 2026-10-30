"""Tray slot grid: 4 corner slot centres per tray (trays.json) -> a crop box for every slot.

python slots.py frame.jpg               # writes frame_grid.png so you can check the calibration
python slots.py frame.jpg --crops DIR   # also saves one crop per slot, ready to sort into data/<class>/
"""
import argparse
import json
from glob import glob
from pathlib import Path

from PIL import Image, ImageDraw

CONFIG = Path(__file__).with_name("trays.json")


def load_config(path=CONFIG):
    return json.loads(Path(path).read_text())


def slot_boxes(trays):
    """Yield (slot_id, (x0, y0, x1, y1)) for every slot of every tray."""
    for name, t in trays.items():
        h = t["crop"] / 2
        dx, dy = t.get("offset", (0, 0))
        for r in range(t["rows"]):
            for c in range(t["cols"]):
                u, v = c / max(t["cols"] - 1, 1), r / max(t["rows"] - 1, 1)
                # ponytail: bilinear between corner centres, fine for a near top-down camera;
                # switch to cv2.getPerspectiveTransform if the view gets steep
                x, y = ((1-u)*(1-v)*a + u*(1-v)*b + (1-u)*v*c_ + u*v*d
                        for a, b, c_, d in zip(t["tl"], t["tr"], t["bl"], t["br"]))
                yield f"{name}_r{r+1}c{c+1}", (round(x + dx - h), round(y + dy - h),
                                               round(x + dx + h), round(y + dy + h))


def crop_slots(img, trays):
    return [(sid, box, img.crop(box)) for sid, box in slot_boxes(trays)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", nargs="+")
    ap.add_argument("--crops", help="folder to save per-slot crops into")
    args = ap.parse_args()
    trays = load_config()["top"]
    for f in (Path(p) for a in args.frames for p in glob(a) or [a]):  # PowerShell doesn't expand *.jpg
        img = Image.open(f).convert("RGB")
        grid = img.copy()
        draw = ImageDraw.Draw(grid)
        for sid, box, crop in crop_slots(img, trays):
            draw.rectangle(box, outline=(0, 255, 255), width=2)
            draw.text((box[0] + 3, box[1] + 2), sid.split("_")[-1], fill=(255, 0, 255))
            if args.crops:
                Path(args.crops).mkdir(parents=True, exist_ok=True)
                crop.save(Path(args.crops) / f"{f.stem}_{sid}.png")
        grid.save(f.with_name(f"{f.stem}_grid.png"))
        print(f"{f}: grid -> {f.stem}_grid.png" + (f", crops -> {args.crops}" if args.crops else ""))


if __name__ == "__main__":
    main()
