"""Follow a tray as it fills: check each new fill state the moment its three camera frames land, with a live view.

Frames arrive as <frames>/<cam>/<tray>_sSS.png for cam in top / side / diag (the dataset / Isaac Sim writer layout):
state 00 is the empty tray, state k has vials 1..k (vial k went into slot k, fill order G01..G30).

python live.py dataset_v1_partial/dataset_v1_partial/frames train_0031 --pace 1   # replay a recorded tray, 1 s a step
python live.py live_frames                    # replay / follow the frames capture.py saved: newest tray, then new ones
Esc closes the window. With no display (the Spark over ssh) there is no window: each tray's dashboard is saved as
live_videos/<tray>.mp4 instead (--videos), one frame per step.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from infer import check_step, load_models
from skyline import episode_skyline

CAMS = ("top", "side", "diag")
WIN = "vial check"
GREEN, GREY, ORANGE, RED, CYAN = (0, 200, 0), (150, 150, 150), (0, 165, 255), (0, 0, 255), (255, 255, 0)  # BGR
HEADLESS = sys.platform.startswith("linux") and not os.environ.get("DISPLAY")  # cv2.imshow would abort here


def pause(ms):
    """cv2.waitKey keeps the window responsive and returns the key; with no display there's no window, so just sleep."""
    if HEADLESS:
        time.sleep(ms / 1000)
        return -1
    return cv2.waitKey(ms)


def colour(verdict):
    return {"pass": GREEN, "no_vial": GREY, "check": ORANGE}.get(verdict, RED)  # every defect is red


def wait_for_state(frames, tray, s, timeout):
    """{cam: image} for fill state s once all three frames exist and decode (a writer may be mid-file); None on timeout."""
    paths = {cam: frames / cam / f"{tray}_s{s:02d}.png" for cam in CAMS}
    end = time.time() + timeout
    while time.time() < end:
        if all(p.exists() for p in paths.values()):
            imgs = {cam: cv2.imread(str(p)) for cam, p in paths.items()}
            if all(im is not None for im in imgs.values()):
                return imgs
        pause(200)
    return None


def dashboard(k, after, verdicts, p, side, tilt, sky, table, ms):
    """One frame of the live view: top view with every slot's verdict so far, side + 45-deg views of slot k, readout."""
    s, c = str(k), colour(verdicts[k])
    top = after["top"].copy()
    for j, v in verdicts.items():
        x0, y0, x1, y1 = table["top"][str(j)]
        cv2.rectangle(top, (x0, y0), (x1, y1), colour(v), 2 if j == k else 1)
    sd = after["side"].copy()
    cv2.line(sd, (0, round(sky)), (sd.shape[1], round(sky)), CYAN, 2)  # the skyline: seated caps' tops sit on it
    x0, x1 = table["side_cols"][s]
    cv2.rectangle(sd, (x0, round(sky) - 120), (x1, round(sky) + 260), c, 6)
    dg = after["diag"].copy()
    x0, y0, x1, y1 = table["diag"][s]
    cv2.rectangle(dg, (x0, y0), (x1, y1), c, 6)

    panel = np.zeros((360, 480, 3), np.uint8)
    counts = Counter(verdicts.values())
    lines = [
        (f"step {k}/30   slot G{k:02d}", (255, 255, 255), 0.7),
        (verdicts[k].upper().replace("_", " "), c, 1.3),
        (f"top    cap {p['cap']:.2f}  vial {p['vial']:.2f}  empty {p['empty']:.2f}", (220, 220, 220), 0.55),
        ("side   nothing new" if side is None else f"side   rise {side[0]:+.1f} px  dome {side[1]:.2f}", (220, 220, 220), 0.55),
        ("45deg  no cap seen" if tilt is None else f"45deg  tilt {tilt:.3f}", (220, 220, 220), 0.55),
        (f"{ms:.0f} ms", (150, 150, 150), 0.5),
        (f"pass {counts['pass']}   bad {sum(n for v, n in counts.items() if v not in ('pass', 'no_vial', 'check'))}"
         f"   no vial {counts['no_vial']}   check {counts['check']}", (255, 255, 255), 0.6),
    ]
    y = 40
    for text, col, scale in lines:
        cv2.putText(panel, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2 if scale > 1 else 1, cv2.LINE_AA)
        y += 70 if scale > 1 else 42

    size = lambda im, w, h: cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.vconcat([cv2.hconcat([size(top, 480, 360), size(sd, 640, 360)]),
                        cv2.hconcat([panel, size(dg, 640, 360)])])


def check(k, before, after, sky, verdicts, models, table, dev):
    """Check fill state k into verdicts, print its line and return the dashboard frame (live.py and capture.py)."""
    t0 = time.perf_counter()
    verdicts[k], p, side, tilt = check_step(k, before, after, sky, models, table, dev)
    ms = (time.perf_counter() - t0) * 1000
    print(f"G{k:02d} {verdicts[k]:<12} cap {p['cap']:.2f} vial {p['vial']:.2f} empty {p['empty']:.2f}  "
          + ("side -" if side is None else f"rise {side[0]:+.1f} dome {side[1]:.2f}")
          + ("  45deg -" if tilt is None else f"  tilt {tilt:.3f}") + f"  ({ms:.0f} ms)")
    return dashboard(k, after, verdicts, p, side, tilt, sky, table, ms)


def follow(frames):
    """Tray ids in arrival order: the newest tray already started, then each new one as its empty-tray frame lands."""
    last = None
    while True:
        trays = sorted(p.name[:-len("_s00.png")] for p in (frames / "top").glob("*_s00.png"))
        new = [t for t in trays if last is None or t > last]
        if new:
            last = new[-1] if last is None else new[0]
            yield last
        else:
            pause(500)


def run_tray(frames, tray, models, table, ref_empty, dev, pace, timeout, videos=None):
    """Check one tray state by state as its frames land. False if Esc was pressed.
    videos: folder to save the dashboard to as <tray>.mp4 instead of showing it (no display)."""
    before = wait_for_state(frames, tray, 0, timeout)
    if before is None:
        print(f"no empty-tray frames {tray}_s00.png under {frames}/{{top,side,diag}}")
        return True
    print(f"--- {tray}")
    sky = episode_skyline(before["side"], ref_empty, table["side_ref_skyline"])
    verdicts, video = {}, None
    try:
        for k in range(1, 31):
            after = wait_for_state(frames, tray, k, timeout)
            if after is None:
                print(f"no state {k} within {timeout:.0f} s, stopping this tray")
                break
            view = check(k, before, after, sky, verdicts, models, table, dev)
            if videos:
                if video is None:  # one frame per step, played back at one step per --pace seconds
                    video = cv2.VideoWriter(str(videos / f"{tray}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                            1 / pace if pace else 1, view.shape[1::-1])
                video.write(view)
            else:
                cv2.imshow(WIN, view)
                if cv2.waitKey(max(1, int(pace * 1000))) == 27:
                    return False
            before = after
    finally:
        if video is not None:
            video.release()  # also on Ctrl+C: an unfinished .mp4 won't play
            print(f"dashboard video -> {videos / tray}.mp4")
    print(f"{tray} done:", dict(Counter(verdicts.values())))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", help="folder with top/ side/ diag/ subfolders")
    ap.add_argument("tray", nargs="?", help="tray id in the file names, e.g. train_0031 (default: follow new trays)")
    ap.add_argument("--pace", type=float, default=0, help="minimum seconds per step (slow down a replay)")
    ap.add_argument("--timeout", type=float, default=60, help="seconds to wait for the next state before giving up")
    ap.add_argument("--models", default="models")
    ap.add_argument("--videos", default="live_videos", help="with no display: folder for the <tray>.mp4 dashboards")
    args = ap.parse_args()
    frames = Path(args.frames)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models, table = load_models(args.models, dev), json.loads(Path("slots_sim.json").read_text())
    ref_empty = cv2.imread("ref_side_empty.png")
    videos = Path(args.videos) if HEADLESS else None
    if videos:
        videos.mkdir(parents=True, exist_ok=True)
        print(f"no display: saving each tray's dashboard to {videos}/<tray>.mp4 instead of opening a window")
    else:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    for tray in [args.tray] if args.tray else follow(frames):
        if not run_tray(frames, tray, models, table, ref_empty, dev, args.pace, args.timeout, videos):
            break
    else:  # single tray finished: keep the last view up until the window is closed or Esc
        while not videos and cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) >= 1 and cv2.waitKey(100) != 27:
            pass
    if not videos:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
