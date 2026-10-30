"""python test_vials.py -- checks the slot grid maths, the side skyline and 45-deg upright checks, the verdicts,
the capture trigger, and live.py finding trays in a frames folder."""
import tempfile
from pathlib import Path

import cv2
import numpy as np

from capture import PlacementTrigger
from infer import decide
from live import follow
from skyline import measure
from slots import slot_boxes
from upright import cap_shape, tilt


def test_grid():
    tray = {"t": {"tl": [0, 0], "tr": [100, 0], "bl": [0, 50], "br": [100, 50], "rows": 2, "cols": 3, "crop": 10}}
    boxes = dict(slot_boxes(tray))
    assert len(boxes) == 6
    assert boxes["t_r1c1"] == (-5, -5, 5, 5)
    assert boxes["t_r1c2"] == (45, -5, 55, 5)
    assert boxes["t_r2c3"] == (95, 45, 105, 55)


def test_skyline():
    before, cols, sky = np.zeros((200, 100, 3), np.uint8), (0, 100), 100.0
    assert measure(before, before, cols, sky) is None  # nothing placed

    raised = before.copy()
    raised[90:150, 20:80] = 255                        # flat top 10 px above the skyline
    rise, dome = measure(before, raised, cols, sky)
    assert 8 <= rise <= 14 and dome < 0.25, (rise, dome)

    tilted = before.copy()
    cv2.fillPoly(tilted, [np.array([[20, 50], [80, 95], [80, 150], [20, 150]])], (255, 255, 255))  # wedge top
    rise, dome = measure(before, tilted, cols, sky)
    assert rise > 40 and dome > 0.25, (rise, dome)

    open_tube = before.copy()
    open_tube[130:180, 30:70] = 255                    # top well below the skyline
    assert measure(before, open_tube, cols, sky)[0] < -20


def test_upright():
    before, box = np.zeros((200, 200, 3), np.uint8), (0, 0, 200, 200)
    yellow = tuple(int(v) for v in cv2.cvtColor(np.uint8([[[26, 200, 200]]]), cv2.COLOR_HSV2BGR)[0, 0])
    assert cap_shape(before, before, box) is None  # no new cap

    def cap(centre, axes, angle=0):
        im = before.copy()
        cv2.ellipse(im, centre, axes, angle, 0, 360, yellow, -1)
        return cap_shape(before, im, box)

    ref = cap((100, 100), (30, 26))
    assert tilt(cap((100, 80), (30, 26)), ref) < 0.01      # raised: same outline, just higher
    assert tilt(cap((100, 100), (30, 20), 30), ref) > 0.08  # tilted: squashed and rotated outline


def test_decide():
    good = {"cap": 0.99, "vial": 0.99, "empty": 0.01}
    assert decide(good, (1.0, 0.1), 0.01) == "pass"
    assert decide(good, (10.0, 0.1), 0.01) == "raised_cap"
    assert decide(good, (10.0, 0.4), 0.05) == "crooked_cap"   # both cameras see a tilt
    assert decide(good, (10.0, 0.1), 0.05) == "raised_cap"    # 45 deg unsure, side top is flat
    assert decide(good, (10.0, 0.1), 0.12) == "crooked_cap"   # 45 deg sure on its own
    assert decide(good, (1.0, 0.1), 0.12) == "crooked_cap"    # right height but clearly tilting
    assert decide(good, (10.0, 0.4), None) == "crooked_cap"   # 45 deg can't see it: side decides
    assert decide({"cap": 0.01, "vial": 0.99, "empty": 0.01}, (-20.0, 0.0), None) == "missing_cap"
    assert decide({"cap": 0.01, "vial": 0.01, "empty": 0.99}, None, None) == "no_vial"
    assert decide(good, None, None) == "check"  # top sees a vial, side saw nothing arrive


def test_trigger():
    still = np.full((100, 100, 3), 120, np.uint8)
    arm = still.copy()
    arm[20:60, 20:60] = 255
    t = PlacementTrigger((0, 0, 100, 100), settle=1.0, motion_px=40)
    t.last_motion = 0.0
    assert [t.update(still, now) for now in (0.0, 0.5, 1.1, 2.0)] == [False, False, True, False]  # empty tray, once
    frames = [(arm, 3.0), (still, 3.1), (still, 3.6), (still, 4.2), (still, 5.0)]
    assert [t.update(f, now) for f, now in frames] == [False, False, False, True, False]  # arm came and went: one placement


def test_follow():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "top").mkdir()
        for tray in ("tray_0001", "tray_0002"):
            cv2.imwrite(str(Path(d) / "top" / f"{tray}_s00.png"), np.zeros((4, 4, 3), np.uint8))
        assert next(follow(Path(d))) == "tray_0002"  # starts at the newest tray already begun


if __name__ == "__main__":
    test_grid()
    test_skyline()
    test_upright()
    test_decide()
    test_trigger()
    test_follow()
    print("ok")
