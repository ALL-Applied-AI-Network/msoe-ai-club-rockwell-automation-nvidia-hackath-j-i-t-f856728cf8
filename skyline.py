"""Side "skyline" camera: its lens sits at cap-top height, so every seated cap's top lands on the same image row.
A new vial whose top rises above that row has a raised or tilted cap; one well below it has no cap.

Fill order is fixed, so step k only needs BEFORE / AFTER and slot k's columns: whatever changed there is vial k.
"""
import cv2
import numpy as np

DIFF_THR = 25  # grey-level change that counts as "vial k is here"; path-tracer noise stays well below it


def episode_skyline(empty_frame, ref_empty, ref_skyline):
    """Skyline row for this episode: the reference row plus however far this camera shifted (it jitters ~0.2 deg)."""
    a, b = (cv2.cvtColor(i, cv2.COLOR_BGR2GRAY).astype(np.float32) for i in (ref_empty, empty_frame))
    (_, dy), _ = cv2.phaseCorrelate(a, b, cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F))
    return ref_skyline + dy


def measure(before, after, cols, sky_row):
    """None if nothing appeared in columns cols, else (rise, dome) of the new vial:
    rise = px its top sits above the skyline (negative = below, i.e. no cap)
    dome = how far its outer fifths drop below its centre, as a share of the height above the skyline:
           ~0.1 for a flat raised cap, ~0.3+ for a tilted one (domed if toward / away, a wedge if left / right)
    """
    x0, x1 = cols
    blur = lambda im: cv2.GaussianBlur(im, (5, 5), 0)
    d = cv2.absdiff(blur(after), blur(before)).max(2)[:, x0:x1] > DIFF_THR
    hit = np.flatnonzero(d.sum(0) > 5)  # columns the new vial covers
    if len(hit) < 10:
        return None
    rise = sky_row - d[:, hit].argmax(0).min()
    row = int(round(sky_row)) - 3
    up = d[:row]
    up_cols = np.flatnonzero(up.any(0))
    if len(up_cols) < 5:
        return float(rise), 0.0
    top = up[:, up_cols].argmax(0).astype(float)
    q = len(top) // 5
    dome = (np.mean(np.r_[top[:q], top[-q:]]) - top[q:-q].min()) / (row - top.min())
    return float(rise), float(dome)
