"""Check every fill step. Vials go in fill order, so step k = slot k:
  top camera  -> 3 CNNs give cap / vial / empty confidences for slot k (AFTER image)
  side camera -> skyline check on what changed between BEFORE and AFTER (is the cap at the right height?)
  45-deg cam  -> upright check on the new cap's outline (is it tilting?)

python infer.py dataset_v1_partial/dataset_v1_partial                          # held-out episodes, confusion matrix
python infer.py dataset_v1_partial/dataset_v1_partial --episodes train_0031 -v   # one episode, every step
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import torch
from PIL import Image

from prep import VAL_EPISODES, load_episodes
from skyline import episode_skyline, measure
from train import EVAL_TF, TARGETS, build_model
from upright import cap_shape, tilt

CLASSES = ["pass", "no_vial", "missing_cap", "raised_cap", "crooked_cap"]
# thresholds picked on the 26 training episodes
RISE_HIGH = 6.5     # px above the skyline: good caps stay within +-6, a 3 mm raise starts at +7
RISE_LOW = -10.0    # px below it: good caps >= -5, an open tube (missing cap) <= -15
TILT = 0.032        # 45-deg outline change: upright (good / raised) caps mostly under 0.03, tilted ones from 0.032
TILT_BAD = 0.08     # no good cap gets near this, so the 45-deg view alone is sure (bad even at the right height)
DOME_TILT = 0.25    # side silhouette: a raised cap keeps a flat top (~0.1), a tilted one a dome or wedge (~0.3+)


def decide(p, side, tilt):
    """p: top-CNN confidences {cap, vial, empty}; side: (rise, dome), or None if nothing new in the side view;
    tilt: 45-deg outline change vs an upright cap, or None if no new cap is visible there."""
    vial = p["vial"] > 0.5 and p["empty"] < 0.5
    if vial != (side is not None):
        return "check"  # top and side disagree on whether a vial arrived
    if not vial:
        return "no_vial"
    rise, dome = side
    if p["cap"] < 0.5:
        return "missing_cap" if rise < RISE_HIGH else "check"
    if rise < RISE_LOW:
        return "check"  # top sees a cap, side sees an open tube
    if tilt is None:  # 45-deg camera couldn't see the cap: side silhouette alone
        tilting = dome > DOME_TILT
    else:             # both cameras agree, or the 45-deg view is sure on its own
        tilting = tilt > TILT_BAD or (tilt > TILT and dome > DOME_TILT)
    if rise > RISE_HIGH:
        return "crooked_cap" if tilting else "raised_cap"
    if tilt is not None and tilt > TILT_BAD:
        return "crooked_cap"  # right height, but clearly tilting from 45 deg
    return "pass"


def load_models(model_dir="models", dev="cpu"):
    models = {}
    for t in TARGETS:
        m = build_model(pretrained=False)
        m.load_state_dict(torch.load(Path(model_dir) / f"{t}.pt", map_location=dev))
        models[t] = m.to(dev).eval()
    return models


@torch.no_grad()
def check_step(k, before, after, sky, models, table, dev="cpu"):
    """Verdict for fill step k (vial k went into slot k). before / after: {"top" | "side" | "diag": BGR image};
    sky: this tray's skyline row (episode_skyline on its empty-tray side frame).
    Returns (verdict, top confidences, side (rise, dome) or None, 45-deg tilt or None)."""
    s = str(k)
    crop = Image.fromarray(cv2.cvtColor(after["top"], cv2.COLOR_BGR2RGB)).crop(table["top"][s])
    x = EVAL_TF(crop).unsqueeze(0).to(dev)
    p = {t: torch.sigmoid(m(x)).item() for t, m in models.items()}
    side = measure(before["side"], after["side"], table["side_cols"][s], sky + table["side_slot_offset"][s])
    cov = cap_shape(before["diag"], after["diag"], table["diag"][s])
    t = None if cov is None else tilt(cov, table["diag_ref_cov"][s])
    return decide(p, side, t), p, side, t


def run_episode(root, ep, models, table, ref_empty, dev="cpu"):
    """[(k, verdict, top confidences, side measurement, 45-deg tilt)] for every step of one recorded episode."""
    load = lambda st, when: {cam: cv2.imread(str(root / st["cameras"][cam][when])) for cam in ("top", "side", "diag")}
    before = load(ep["steps"][0], "before")  # state 0: empty tray
    sky = episode_skyline(before["side"], ref_empty, table["side_ref_skyline"])
    out = []
    for st in ep["steps"]:
        after = load(st, "after")
        out.append((st["k"], *check_step(st["k"], before, after, sky, models, table, dev)))
        before = after  # BEFORE of step k+1 is AFTER of step k
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--episodes", nargs="*", help="episode ids (default: the held-out val episodes)")
    ap.add_argument("--models", default="models")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every step, not just the misses")
    args = ap.parse_args()
    root = Path(args.root)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models, table = load_models(args.models, dev), json.loads(Path("slots_sim.json").read_text())
    ref_empty = cv2.imread("ref_side_empty.png")
    eps = load_episodes(root)
    eps = [e for e in eps if e["episode_id"] in args.episodes] if args.episodes else eps[-VAL_EPISODES:]

    conf = Counter()
    for ep in eps:
        for k, verdict, p, side, t in run_episode(root, ep, models, table, ref_empty, dev):
            gt = ep["steps"][k - 1]["cls"]
            conf[gt, verdict] += 1
            if args.verbose or verdict != gt:
                s = "nothing new" if side is None else "rise %+.1f dome %.2f" % side
                d = "no cap" if t is None else f"tilt {t:.3f}"
                print(f"{ep['episode_id']} k{k:02d}  {gt:<12} -> {verdict:<12} cap {p['cap']:.2f} vial {p['vial']:.2f} "
                      f"empty {p['empty']:.2f} | side {s} | 45deg {d}")

    cols = CLASSES + ["check"]
    print("\n" + "truth / verdict".ljust(16) + "".join(f"{c:>12}" for c in cols))
    for g in CLASSES:
        print(f"{g:<16}" + "".join(f"{conf[g, c]:>12}" for c in cols))
    n = sum(conf.values())
    exact = sum(conf[g, g] for g in CLASSES)
    good_bad = sum(v for (g, c), v in conf.items() if (g == "pass") == (c == "pass"))
    print(f"\n{len(eps)} episodes, {n} steps: class accuracy {exact / n:.3f}, good-vs-bad accuracy {good_bad / n:.3f}")


if __name__ == "__main__":
    main()
