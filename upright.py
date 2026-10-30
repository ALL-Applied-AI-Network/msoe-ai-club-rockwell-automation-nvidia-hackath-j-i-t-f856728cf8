"""45-degree camera: is the new vial's cap upright? Seen from 35 deg above, a seated or merely raised cap keeps the
same outline as an upright one (just shifted), while a tilted cap's outline stretches or squashes.

Fill order is fixed, so step k only needs BEFORE / AFTER and slot k's window: the new cap is cap-yellow in AFTER
that changed since BEFORE.
"""
import cv2
import numpy as np

CAP_HSV = (np.array([23, 120, 45]), np.array([30, 255, 255]))  # cap yellow; the robot arm sits near hue 20
DIFF_THR = 25


def cap_shape(before, after, box):
    """2x2 covariance of the new cap's pixels (the spread of its outline), or None if no new cap is visible."""
    x0, y0, x1, y1 = box
    a, b = after[y0:y1, x0:x1], before[y0:y1, x0:x1]
    yellow = cv2.inRange(cv2.cvtColor(a, cv2.COLOR_BGR2HSV), *CAP_HSV) > 0
    changed = cv2.absdiff(cv2.GaussianBlur(a, (5, 5), 0), cv2.GaussianBlur(b, (5, 5), 0)).max(2) > DIFF_THR
    n, lab, stats, _ = cv2.connectedComponentsWithStats((yellow & changed).astype(np.uint8))
    if n < 2 or stats[1:, cv2.CC_STAT_AREA].max() < 50:
        return None
    ys, xs = np.nonzero(lab == 1 + stats[1:, cv2.CC_STAT_AREA].argmax())  # biggest blob = the new cap
    return np.cov(np.vstack([xs, ys]))


def tilt(cov, ref_cov):
    """How far the cap's outline is from an upright cap in the same slot (0 = identical, tilted caps ~0.03+)."""
    return float(np.linalg.norm(np.asarray(cov) - ref_cov) / np.linalg.norm(ref_cov))
