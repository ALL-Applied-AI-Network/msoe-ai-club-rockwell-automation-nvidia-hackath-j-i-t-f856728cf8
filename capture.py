"""live.py for USB cameras: grab one frame from each camera after every placement and check it on the spot,
with the same dashboard and per-step lines. Frames stay in memory (only the previous state is kept); nothing is saved.

python capture.py --list                                 # which index is which camera (saves cam_<i>.png)
python capture.py --top 0 --side 1 --diag 2 --settings    # lock exposure / focus / white balance, then check
python capture.py --frames dataset_v1_partial/dataset_v1_partial/frames --tray train_0031   # no cameras: a recording

A placement = something moves over the tray in the top view (the arm), then the tray stays still for --settle s.
That also fires when the arm comes and goes without leaving a vial, so state k always lines up with slot k.
Space captures now, N starts a new tray (its next capture is the empty tray), Esc quits.
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from infer import load_models
from live import WIN, check
from skyline import episode_skyline

SIZES = {"top": (640, 480), "side": (1920, 1080), "diag": (1920, 1080)}  # what models / references were built for
FONT = cv2.FONT_HERSHEY_SIMPLEX


class PlacementTrigger:
    """Fires once the tray has stayed still for `settle` s after something moved over it.
    Starts armed, so the first still moment captures the empty tray."""
    # ponytail: motion-based; if the arm ever pauses over the tray longer than `settle`, swap this for the
    # robot's own "placed" signal (capture when it arrives)

    def __init__(self, roi, settle=1.0, motion_px=40):
        self.roi, self.settle, self.motion_px = roi, settle, motion_px
        self.prev, self.last_motion, self.armed, self.moving = None, time.monotonic(), True, False

    def update(self, top, now=None):
        """Feed every top-camera frame; True when a placement has just finished."""
        now = time.monotonic() if now is None else now
        x0, y0, x1, y1 = self.roi
        g = cv2.GaussianBlur(cv2.cvtColor(top[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.int16)
        self.moving = self.prev is not None and int((np.abs(g - self.prev) > 20).sum()) > self.motion_px
        self.prev = g
        if self.moving:
            self.last_motion, self.armed = now, True
        if self.armed and now - self.last_motion >= self.settle:
            self.armed = False
            return True
        return False


def open_cam(index, size, settings):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"camera {index} didn't open (python capture.py --list shows what's there)")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # raw 1080p x2 + 480p won't fit USB 2 bandwidth
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    # BEFORE / AFTER get compared pixel by pixel, so exposure, white balance and focus must not drift between them.
    # ponytail: best effort, some drivers ignore these; --settings opens the driver dialog to lock them by hand
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # DirectShow: 0.25 = manual
    if settings:
        cap.set(cv2.CAP_PROP_SETTINGS, 1)
    return cap


class FolderCam:
    """Stands in for a camera (--frames): plays a recorded tray <frames>/<cam>/<tray>_sSS.png, one state every
    `pace` s from t0, then holds the last one. Each new state opens with one black frame, the arm passing over the
    tray, so the placement trigger fires even for a slot left empty (where the picture doesn't change)."""

    def __init__(self, frames, cam, tray, pace, t0):
        self.paths = sorted((Path(frames) / cam).glob(f"{tray}_s??.png"))
        if not self.paths:
            raise SystemExit(f"no {tray}_sSS.png under {Path(frames) / cam}")
        self.pace, self.t0, self.i, self.arm = pace, t0, -1, False
        self.img = cv2.imread(str(self.paths[0]))

    def grab(self):
        i = min(int((time.monotonic() - self.t0) / self.pace), len(self.paths) - 1)
        if i != self.i:
            self.i, self.img, self.arm = i, cv2.imread(str(self.paths[i])), True
        return True

    def retrieve(self):
        img, self.arm = (np.zeros_like(self.img) if self.arm else self.img), False
        return True, img

    def get(self, prop):
        return self.img.shape[1] if prop == cv2.CAP_PROP_FRAME_WIDTH else self.img.shape[0]

    def release(self):
        pass


def read_all(caps):
    """Latest frame from every camera, grabbed back to back so the three line up in time."""
    for cap in caps.values():
        cap.grab()
    frames = {}
    for cam, cap in caps.items():
        ok, img = cap.retrieve()
        if not ok:
            raise SystemExit(f"{cam} camera stopped delivering frames")
        frames[cam] = img if (img.shape[1], img.shape[0]) == SIZES[cam] else cv2.resize(img, SIZES[cam])
    return frames


def preview(frames, roi, tray, s, moving):
    top = frames["top"].copy()
    cv2.rectangle(top, roi[:2], roi[2:], (0, 255, 255) if moving else (0, 200, 0), 2)
    view = cv2.hconcat([cv2.resize(top, (320, 240)), cv2.resize(frames["side"], (427, 240)),
                        cv2.resize(frames["diag"], (427, 240))])
    nxt = "empty tray" if s == 0 else f"after vial {s}"
    cv2.putText(view, f"{tray}  next: s{s:02d} ({nxt})  {'MOVING' if moving else 'still'}", (8, 22), FONT, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return view


def list_cams(n=6):
    for i in range(n):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        ok, img = cap.read() if cap.isOpened() else (False, None)
        if ok:
            cv2.imwrite(f"cam_{i}.png", img)
            print(f"camera {i}: {img.shape[1]}x{img.shape[0]} -> cam_{i}.png")
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="snapshot every camera index and exit")
    ap.add_argument("--top", type=int)
    ap.add_argument("--side", type=int)
    ap.add_argument("--diag", type=int)
    ap.add_argument("--settle", type=float, default=1.0, help="seconds of stillness that end a placement")
    ap.add_argument("--motion-px", type=int, default=40, help="changed pixels in the tray area that count as motion")
    ap.add_argument("--settings", action="store_true", help="open each camera's driver dialog to lock exposure etc.")
    ap.add_argument("--models", default="models")
    ap.add_argument("--frames", help="no cameras: play a recorded tray from this folder (top/ side/ diag/) instead")
    ap.add_argument("--tray", default="train_0031", help="with --frames: which recorded tray")
    ap.add_argument("--pace", type=float, default=2.0, help="with --frames: seconds per state, more than --settle")
    args = ap.parse_args()
    if args.list:
        return list_cams()
    if args.frames and args.pace <= args.settle:
        raise SystemExit("--pace must be more than --settle, or no state stays still long enough to be captured")
    if not args.frames and None in (args.top, args.side, args.diag):
        raise SystemExit("give --top, --side and --diag camera indices (python capture.py --list), or --frames")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models, table = load_models(args.models, dev), json.loads(Path("slots_sim.json").read_text())
    ref_empty = cv2.imread("ref_side_empty.png")
    if args.frames:
        t0 = time.monotonic()  # one clock, so the three streams switch states together
        caps = {cam: FolderCam(args.frames, cam, args.tray, args.pace, t0) for cam in SIZES}
    else:
        caps = {cam: open_cam(i, SIZES[cam], args.settings) for cam, i in (("top", args.top), ("side", args.side), ("diag", args.diag))}
    for cam, cap in caps.items():
        w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if (w, h) != SIZES[cam]:
            print(f"warning: {cam} gives {w:.0f}x{h:.0f}, resizing to {SIZES[cam]}; slot positions may be off")
    boxes = np.array(list(table["top"].values()))
    roi = (int(boxes[:, 0].min()) - 10, int(boxes[:, 1].min()) - 10, int(boxes[:, 2].max()) + 10, int(boxes[:, 3].max()) + 10)
    trigger = PlacementTrigger(roi, args.settle, args.motion_px)
    n, s, verdicts = 1, 0, {}  # tray number, next state (0 = empty tray), this tray's verdicts
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    print("checking tray_0001 ... Space = capture now, N = new tray, Esc = quit")

    while True:
        frames = read_all(caps)
        fire = trigger.update(frames["top"])
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if fire or key == ord(" "):
            if s == 0:  # the empty tray: what step 1 is compared against, and where this tray's skyline sits
                print(f"--- tray_{n:04d}  ({time.strftime('%H:%M:%S')}, empty tray)")
                sky = episode_skyline(frames["side"], ref_empty, table["side_ref_skyline"])
            else:
                cv2.imshow(WIN, check(s, before, frames, sky, verdicts, models, table, dev))
            before = frames  # the only frames kept: the next step is compared against them
            s += 1
        if s > 0 and (key in (ord("n"), ord("N")) or s > 30):  # N, or tray full: next capture is a new empty tray
            if verdicts:
                print(f"tray_{n:04d} done:", dict(Counter(verdicts.values())))
            n, s, verdicts = n + 1, 0, {}
            print(f"new tray tray_{n:04d}: next capture is its empty tray")
        cv2.imshow("capture", preview(frames, roi, f"tray_{n:04d}", s, trigger.moving))
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
