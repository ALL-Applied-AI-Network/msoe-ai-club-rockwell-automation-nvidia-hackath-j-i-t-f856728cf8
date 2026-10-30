"""Turn the Isaac Sim fill-sequence dataset into what the checker needs.

python prep.py dataset_v1_partial/dataset_v1_partial
  -> slots_sim.json                          per-slot boxes (top, diag), column ranges (side), skyline + upright refs
  -> ref_side_empty.png                      empty-tray side frame that each episode's skyline is measured against
  -> data_sim/{train,val}/<class>/*.png      top crops of slot k after step k (its class) and before it (empty)

Vials always go in fill order G01..G30, so step k = slot k and every box comes straight from the labels.
Split is by episode (last VAL_EPISODES held out, used for nothing here), so val really is unseen trays.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from upright import cap_shape

TOP_CROP = 32      # px square around the slot centre in the 640x480 top image (~1.2 slot pitches)
SIDE_PAD = 12      # px added either side of the vial's column range in the side image
DIAG_PAD = 20      # px added around the vial's box in the 45-degree image
VAL_EPISODES = 6


def load_episodes(root):
    return [json.loads(p.read_text()) for p in sorted(Path(root).glob("labels/*.json"))]


def med(vals):
    return float(np.median(vals))


def slot_table(eps):
    """Median per-slot geometry over all episodes, so per-episode camera jitter averages out."""
    top, side, diag, offsets = {}, {}, {}, {}
    # per-episode skyline = median top row of the good caps in the side view
    sky = {e["episode_id"]: med([s["cameras"]["side"]["bbox_xyxy"][1] for s in e["steps"] if s["defect"] == "good"])
           for e in eps}
    for k in range(1, 31):
        steps = [(e["episode_id"], e["steps"][k - 1]) for e in eps]
        b = np.array([s["cameras"]["top"]["bbox_projected_xyxy"] for _, s in steps if s["cameras"]["top"]["bbox_projected_xyxy"]])
        cx, cy, h = med((b[:, 0] + b[:, 2]) / 2), med((b[:, 1] + b[:, 3]) / 2), TOP_CROP / 2
        top[k] = [round(cx - h), round(cy - h), round(cx + h), round(cy + h)]
        sb = np.array([s["cameras"]["side"]["bbox_xyxy"] for _, s in steps if s["cameras"]["side"]["bbox_xyxy"]])
        side[k] = [round(med(sb[:, 0])) - SIDE_PAD, round(med(sb[:, 2])) + SIDE_PAD]
        db = np.median([s["cameras"]["diag"]["bbox_xyxy"] for _, s in steps if s["cameras"]["diag"]["bbox_xyxy"]], axis=0)
        diag[k] = [round(db[0]) - DIAG_PAD, round(db[1]) - DIAG_PAD, round(db[2]) + DIAG_PAD, round(db[3]) + DIAG_PAD]
        # where this slot's good cap sits relative to its episode's skyline (depth / lens distortion)
        offsets[k] = round(med([s["cameras"]["side"]["bbox_xyxy"][1] - sky[e] for e, s in steps if s["defect"] == "good"]), 2)
    ref = eps[0]["episode_id"]
    return {"top": top, "side_cols": side, "diag": diag, "side_slot_offset": offsets, "side_ref_skyline": sky[ref],
            "ref_episode": ref}


def upright_refs(root, eps, diag):
    """Per-slot outline (pixel covariance) of an upright cap in the 45-degree view, from the 'pass' steps."""
    refs = {}
    for k in range(1, 31):
        covs = [cap_shape(cv2.imread(str(root / c["before"])), cv2.imread(str(root / c["after"])), diag[k])
                for c in (e["steps"][k - 1]["cameras"]["diag"] for e in eps if e["steps"][k - 1]["cls"] == "pass")]
        refs[k] = np.median([c for c in covs if c is not None], axis=0).round(3).tolist()
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="data_sim")
    args = ap.parse_args()
    root = Path(args.root)
    eps = load_episodes(root)
    train_eps = eps[:-VAL_EPISODES]  # geometry and references come from training episodes only
    table = slot_table(train_eps)
    table["diag_ref_cov"] = upright_refs(root, train_eps, table["diag"])
    Path("slots_sim.json").write_text(json.dumps(table, indent=1))
    shutil.copy(root / f"frames/side/{table['ref_episode']}_s00.png", "ref_side_empty.png")

    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    for i, e in enumerate(eps):
        split = "val" if i >= len(eps) - VAL_EPISODES else "train"
        for s in e["steps"]:
            k, cam = s["k"], s["cameras"]["top"]
            x0, y0, x1, y1 = table["top"][k]
            for cls, frame in ((s["cls"], cam["after"]), ("empty", cam["before"])):
                d = out / split / cls
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{e['episode_id']}_k{k:02d}.png"), cv2.imread(str(root / frame))[y0:y1, x0:x1])
    for split in ("train", "val"):
        print(split, {d.name: len(list(d.iterdir())) for d in sorted((out / split).iterdir())})


if __name__ == "__main__":
    main()
