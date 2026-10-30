# Runbook: vial cap checker (ResNet + camera geometry)

For an AI agent (or a person) operating this repo. It covers how to run, check, retrain and troubleshoot the working model on branch `Ash's-run-it-through-ResNets`. `README.md` has the background; this file has the procedures.

---

## 0. Ground rules

1. **Run everything from the repo root** (`hackathon/`). Every script loads `slots_sim.json`, `ref_side_empty.png` and `models/` by relative path.
2. **Check before you overwrite.** These commands write over existing outputs without asking:

   | Command | Overwrites / deletes |
   |---|---|
   | `prep.py` | `slots_sim.json` (**tracked in git**), `ref_side_empty.png`, and **deletes all of** `--out` (default `data_sim/`) |
   | `train.py <t>` | `models/<t>.pt` (~45 MB; git-ignored, so there's no backup) |
   | `capture.py --list` | `cam_<i>.png` |

   Copy `models/` and `slots_sim.json` somewhere safe before retraining or re-prepping.
3. **Never tune on the held-out episodes.** The last 6 episodes (in sorted `labels/*.json` order, `prep.VAL_EPISODES = 6`) are the test set. Pick thresholds and geometry from the 26 training episodes only.
4. **Keep `slots_sim.json` and `ref_side_empty.png` together.** `side_ref_skyline` in the JSON was measured on that exact PNG. Regenerate both with `prep.py`, never just one.
5. **Run `test_vials.py` after any code change.** It needs no data, models or GPU, and prints `ok`.

---

## 1. What the model is

Per fill step k (vial k goes into slot G`k`, always in G01 → G30 order), `infer.check_step()` compares the frames from **before** and **after** that placement:

| Stage | Code | Output |
|---|---|---|
| Top camera, 32 px crop of slot k → 3 ResNet-18 binary classifiers | `train.build_model`, `models/{cap,vial,empty}.pt` | `p = {cap, vial, empty}` sigmoid scores |
| Side camera, pixel diff in slot k's columns vs the tray's skyline row | `skyline.measure` | `(rise_px, dome)`, or `None` if nothing changed |
| 45° camera, new cap-yellow blob covariance vs an upright reference | `upright.cap_shape`, `upright.tilt` | `tilt`, or `None` if no new cap is visible |
| Rules | `infer.decide` | `pass` · `no_vial` · `missing_cap` · `raised_cap` · `crooked_cap` · `check` |

`check` means the cameras disagree and a person should look. It always counts as a miss in scoring.

**Thresholds** (constants at the top of `infer.py`):

| Name | Value | Meaning |
|---|---|---|
| `RISE_HIGH` | 6.5 px | above this, the cap is raised or crooked (good caps sit within ±6 px, so the margin is tight) |
| `RISE_LOW` | −10 px | below this with a cap seen from the top gives `check` (an open tube is ≤ −15) |
| `TILT` | 0.032 | 45° tilt, counted only when the side dome agrees |
| `TILT_BAD` | 0.08 | 45° tilt that's enough on its own |
| `DOME_TILT` | 0.25 | side-view top shape: flat ≈ 0.1, tilted ≈ 0.3+ |

**Known-good baseline:** 180/180 on the 6 held-out episodes, 778/780 on the 26 training episodes, about 56 ms per step on an RTX 5060.

---

## 2. Required files

| Path | In git? | Made by | Needed by |
|---|---|---|---|
| `dataset_v1_partial/dataset_v1_partial/{frames,masks,labels,cameras.json}` | no | Isaac Sim twin (supplied) | `prep.py`, `infer.py`, replay |
| `slots_sim.json` | **yes** | `prep.py` | everything at runtime |
| `ref_side_empty.png` | no | `prep.py` (copy of the reference episode's side `_s00`) | `infer.py`, `live.py` |
| `data_sim/{train,val}/<class>/*.png` | no | `prep.py` | `train.py` |
| `models/{cap,vial,empty}.pt` | no | `train.py` | `infer.py`, `live.py` |

Quick preflight:

```bash
ls slots_sim.json ref_side_empty.png models/cap.pt models/vial.pt models/empty.pt
ls dataset_v1_partial/dataset_v1_partial/labels | head
```

> **Current state of this checkout (2026-10-30):** `models/*.pt` and `slots_sim.json` are present, but `ref_side_empty.png` and the dataset are **not**. So `infer.py` and `live.py` will fail until the dataset is restored and `ref_side_empty.png` is recovered. Either copy `frames/side/<ref_episode>_s00.png` to `ref_side_empty.png`, where `ref_episode` is the value in `slots_sim.json`, or re-run `prep.py`. Re-running `prep.py` also rewrites `slots_sim.json`, so diff it afterwards. `prep.py` also has an uncommitted change (`np.atleast_2d` / `.get()` for missing bboxes); check `git diff prep.py` before committing anything.

---

## 3. Environment

The README targets Windows + Python 3.11 + CUDA 12.8 (`torch==2.11.0`, `torchvision==0.26.0`, `opencv-python==5.0.0.93`, `pillow==12.3.0`, `numpy==2.4.6`).

What's on this host (Linux aarch64, NVIDIA GB10, Python 3.12, torch 2.5.1, OpenCV 4.6) is enough for `test_vials.py`, `prep.py`, `train.py` and `infer.py`. Two things differ from the README setup:
- `capture.py` opens cameras with `cv2.CAP_DSHOW`, which is **Windows only**. Don't run live capture here.
- `live.py` calls `cv2.imshow` and needs a display. On a headless host, use `infer.py` instead.

GPU check: `python -c "import torch; print(torch.cuda.is_available())"`. The scripts fall back to CPU on their own, just more slowly.

---

## 4. Procedures

### 4.1 Health check (safe, read-only, ~seconds)

```bash
python test_vials.py            # expect: ok
```

This covers the slot-grid maths, skyline, upright, the `decide()` truth table, the placement trigger and the capture → live file hand-off. It does **not** cover the models or real data.

### 4.2 Score the held-out episodes (read-only)

```bash
python infer.py dataset_v1_partial/dataset_v1_partial
```

Expected: a confusion matrix with everything on the diagonal, ending in
`6 episodes, 180 steps: class accuracy 1.000, good-vs-bad accuracy 1.000`.
Only misses are printed above the matrix. Any line there is a regression.

Other forms:

```bash
python infer.py <root> --episodes train_0031 -v          # every step of one episode
python infer.py <root> --episodes <id1> <id2> ...        # chosen episodes (e.g. training ones, for threshold work)
python infer.py <root> --models other_models_dir         # compare a candidate model set
```

How to read a step line: `<ep> kNN <truth> -> <verdict> cap .. vial .. empty .. | side rise ±x dome y | 45deg tilt z`.

### 4.3 Rebuild crops and references

```bash
cp slots_sim.json slots_sim.json.bak                     # it is tracked; keep the old one to diff
python prep.py dataset_v1_partial/dataset_v1_partial     # optional: --out <dir> instead of data_sim
git diff --stat slots_sim.json
```

The last lines print per-class crop counts for `train` and `val`. For the baseline, `missing_cap` should be about 63 train / 17 val. If a class is missing, the dataset is incomplete.

### 4.4 Retrain the top-view models

```bash
mkdir -p models_backup && cp models/*.pt models_backup/
python train.py cap
python train.py vial
python train.py empty
python infer.py dataset_v1_partial/dataset_v1_partial    # must still be 180/180
```

- The first run downloads ImageNet ResNet-18 weights, so it needs network access.
- Each run saves the epoch with the lowest validation loss (look for `<- saved`). Balanced accuracy should reach about 1.000.
- Options: `--epochs 20`, `--lr 3e-4`, `--data`, `--val` (`--val ''` makes a random 20% per-class split of `--data`), `--out models`.
- The seed is **not** fixed, so results vary from run to run. If a retrain scores worse, restore `models_backup/`. Don't loosen thresholds to cover for it.
- To write to a separate folder instead: `--out models_new`, then `infer.py ... --models models_new`.

### 4.5 Change a threshold

1. Edit the constant in `infer.py` (not inside `decide()`).
2. Evaluate on **training** episodes: pass every non-held-out id to `--episodes` with `-v`, and look at the margins (rise, tilt, dome) near the new cut.
3. Then run the held-out score once (4.2) to confirm.
4. If `decide()` behaviour changed, update the asserts in `test_vials.py::test_decide`, then run `python test_vials.py`.

### 4.6 Replay a tray in the dashboard (needs a display)

```bash
python live.py dataset_v1_partial/dataset_v1_partial/frames train_0031 --pace 1
```

The panels show the top view with verdict boxes (green pass, red defect, grey no vial, orange check), the side view with the cyan skyline, the 45° view, and the readout. Esc closes it.

### 4.7 Live USB cameras (Windows, needs a display)

```bash
python capture.py --list                                   # identify indices via cam_<i>.png
python capture.py --top 0 --side 1 --diag 2 --settings     # lock exposure / WB / focus in the dialogs
python live.py live_frames                                 # second terminal
```

- Sizes must be top 640×480 and side / diag 1920×1080. If capture prints a `warning: ... resizing`, slot positions may be off.
- Capture keys: Space captures now, N starts a new tray, Esc quits. A tray rolls over by itself after s30.
- Placement trigger options: `--settle` (default 1.0 s) and `--motion-px` (default 40).

**Frame contract** for any other source (Isaac Sim, a robot signal):

```
<frames>/{top,side,diag}/<tray>_s00.png    empty tray, written first (it sets the skyline)
<frames>/{top,side,diag}/<tray>_sNN.png    after vial NN, NN = 01..30
```

Write each file atomically (temp file + rename, as `capture.save` does). `live.py` waits `--timeout` (default 60 s) per state.

---

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `cv2.error` in `phaseCorrelate`, or `NoneType` has no `cvtColor` | `ref_side_empty.png` is missing, or not run from the repo root | `cd` to the repo root; restore the file (§2) |
| `FileNotFoundError: slots_sim.json` / `models/x.pt` | wrong working directory, or untrained | `cd` to the repo root; run §4.4 |
| `train.py`: `unknown class folders` | stray folder under `data_sim/<split>/` | allowed classes: `pass raised_cap crooked_cap missing_cap no_vial empty` |
| `train.py`: `needs both positive and negative crops` | `prep.py` didn't run fully, or the data is wrong | re-run §4.3 and check the counts |
| Many `check` verdicts across a whole tray | side camera moved more than ±6 px, or `ref_side_empty.png` doesn't match `slots_sim.json` | regenerate both with `prep.py`; look at the cyan skyline in `live.py` |
| `raised_cap` false alarms on good caps | rise is close to the 6.5 px limit (good caps reach about +6) | check the rise values with `-v`; recalibrate the camera before touching `RISE_HIGH` |
| `crooked_cap` ↔ `raised_cap` confusions | 45° view missed the cap (`45deg no cap`), or lighting moved the yellow hue | check `upright.CAP_HSV` (hue 23–30; the arm is near 20) |
| Slot boxes off-target on real cameras | geometry is fit to the twin | real cameras need their own calibration and labelled crops (see the limits below) |
| `camera N didn't open` | wrong index, or not on Windows | `capture.py --list`; run on Windows |
| `live.py` hangs on "waiting" | `<tray>_s00.png` is missing from one of the camera folders | check that all three `top/side/diag` files exist |
| Placements missed or doubled | arm paused longer than `--settle`, or the scene is noisy | adjust `--settle` / `--motion-px`, or use Space by hand |

---

## 6. Limits (don't claim more than this)

- Validated **only on the digital twin**. On one real photo, the vial/empty models called 50 of 53 empty holes "vial".
- `damaged_vial` is not detected. Only one fixture (30 slots, fixed fill order) is handled.
- Rare-class counts are small, and training isn't seeded.
- `slots.py` / `trays.json` is an older grid tool and is **not** part of this pipeline.

## 7. Before reporting "done"

- [ ] `python test_vials.py` prints `ok`
- [ ] `python infer.py <root>` gives 180/180 (or the change is explained step by step)
- [ ] `git status` shows only the intended changes; `slots_sim.json` was diffed if `prep.py` ran
- [ ] model backups were kept if `models/` was overwritten
