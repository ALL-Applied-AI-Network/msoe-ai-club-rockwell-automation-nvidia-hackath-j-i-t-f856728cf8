# Vial cap checker: ResNets + camera geometry

Checks every vial the robot places on the tray and labels it `pass`, `no_vial`, `missing_cap`, `raised_cap` or `crooked_cap`, using the top, side and 45° cameras.

Vials always go in the same fill order (G01 → G30), so step k always means "vial k just went into slot k". Each step compares the frames from just **before** and just **after** that placement.

On the 6 held-out digital-twin episodes (180 steps) it gets **180/180** right, both good vs bad and the exact defect class. On the 26 training episodes it gets 778/780. A check takes about 56 ms per step on an RTX 5060.

---

## 1. Setup

Python 3.11 on Windows, with an NVIDIA GPU. RTX 50-series cards need the CUDA 12.8 PyTorch build.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # PowerShell   (Git Bash: source .venv/Scripts/activate)
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install opencv-python==5.0.0.93 pillow==12.3.0 numpy==2.4.6
```

Run every command below from the repo root with the venv active.

### Data (not in git)

Put the Isaac Sim fill-sequence dataset at `dataset_v1_partial/dataset_v1_partial/`, with its `frames/`, `masks/`, `labels/` and `cameras.json`. Datasets, crops, captures and model weights are all git-ignored. The next two steps regenerate them.

---

## 2. Build the training data and references

```bash
python prep.py dataset_v1_partial/dataset_v1_partial
```

This creates:

| Output | What it is |
|---|---|
| `slots_sim.json` | Where slot k appears in each camera, plus the side-view skyline and the per-slot upright-cap references for the 45° view |
| `ref_side_empty.png` | Empty-tray side frame; each tray's camera drift is measured against it |
| `data_sim/{train,val}/<class>/` | 32x32 top-view crops of slot k after step k (labelled with its class) and before it (`empty`) |

The last 6 episodes are held out, and geometry and references come from the training episodes only.

## 3. Train the three top-view models

```bash
python train.py cap
python train.py vial
python train.py empty
```

Each writes `models/<name>.pt`, saving the epoch with the lowest validation loss. Options: `--epochs` (default 20), `--lr` (default 3e-4), and `--data` / `--val` to point at other crop folders.

## 4. Check the results

Score the held-out episodes and print a confusion matrix:

```bash
python infer.py dataset_v1_partial/dataset_v1_partial
```

Show one episode step by step:

```bash
python infer.py dataset_v1_partial/dataset_v1_partial --episodes train_0031 -v
```

Run the self-check (synthetic tests of the geometry, the decision rules and the capture hand-off):

```bash
python test_vials.py
```

## 5. Watch it live

**Replay a recorded tray** at 1 s per step:

```bash
python live.py dataset_v1_partial/dataset_v1_partial/frames train_0031 --pace 1
```

The dashboard has four panels:
- **Top-left:** the top view, with every slot checked so far boxed (green = pass, red = defect, grey = no vial).
- **Top-right:** the side view, with the skyline in cyan and the current slot boxed.
- **Bottom-right:** the 45° view, with the current slot boxed.
- **Bottom-left:** the verdict, the measurements behind it, and running counts.

Esc closes it.

**Run with USB cameras.** You need two terminals.

1. Find which camera index is which (saves `cam_<i>.png`; index 0 is often the laptop webcam):
   ```bash
   python capture.py --list
   ```
2. Capture. `--settings` opens each driver's dialog, where you should turn off auto exposure, auto white balance and autofocus:
   ```bash
   python capture.py --top 0 --side 1 --diag 2 --settings
   ```
3. In the second terminal, check each tray as it fills:
   ```bash
   python live.py live_frames
   ```

`capture.py` saves one frame per camera after every placement. It counts a placement when something moves over the tray in the top view and the tray then stays still for `--settle` seconds (default 1). The first still moment is saved as the empty tray. Keys: Space captures now, N starts a new tray, Esc quits.

Expected sizes are top 640×480 and side / 45° 1920×1080. `live.py` follows any folder laid out like this, so another frame source (Isaac Sim, a robot "placed" signal) only has to write the same files:

```
<frames>/{top,side,diag}/<tray>_s00.png   empty tray (first: it sets that tray's skyline)
<frames>/{top,side,diag}/<tray>_sNN.png   after vial NN was placed, NN = 01..30
```

---

## Architecture

```
BEFORE (state k-1) ─┐
AFTER  (state k)   ─┼─► top  : crop slot k ──► 3 × ResNet-18 ─────────────► cap / vial / empty confidence
                    ├─► side : what changed in slot k ──► rise above skyline, top-edge dome
                    └─► 45°  : new cap-yellow in slot k ──► outline vs upright cap ──► tilt
                                            ▼
             decide() ─► pass | no_vial | missing_cap | raised_cap | crooked_cap | check
```

| Camera | Method | Answers |
|---|---|---|
| Top (640×480) | Three ImageNet-pretrained ResNet-18 binary classifiers on a 32 px crop of slot k | Is there a vial? Does it have a cap? |
| Side (skyline) | Plain OpenCV. The lens sits at cap-top height, so every seated cap's top lands on one image row. | Is the cap at the right height (raised or missing), and is its top flat or tilted? |
| 45° | Plain OpenCV. Compares the new cap's outline (pixel covariance) with an upright cap in the same slot. | Is the cap upright? |

**Decision rules** (`infer.decide`; thresholds were picked on training episodes only):
- **`no_vial`:** the side view saw nothing change and the top models say empty.
- **`missing_cap`:** the top cap model says no cap.
- **`raised_cap` or `crooked_cap`:** the cap's top is more than 6.5 px above the skyline (good caps stay within ±6 px).
  - It counts as **crooked** if both cameras see a tilt (45° tilt > 0.032 and side dome > 0.25), or if the 45° view alone is sure (tilt > 0.08).
  - Otherwise it's **raised**.
- **`check`:** the cameras disagree, for example the top view sees a vial but the side view saw nothing arrive. It flags the step for a person.

| File | Role |
|---|---|
| `prep.py` | Dataset → `slots_sim.json`, `ref_side_empty.png`, `data_sim/` |
| `train.py` | Trains one top-view ResNet-18 (`cap`, `vial` or `empty`) |
| `skyline.py` | Side-view measurement (rise and dome) and per-tray skyline estimate |
| `upright.py` | 45° cap-outline measurement and tilt score |
| `infer.py` | `check_step()` combines all three; the CLI scores recorded episodes |
| `live.py` | Follows a frames folder and shows the live dashboard |
| `capture.py` | USB camera capture with the placement trigger |
| `test_vials.py` | Self-check |
| `slots.py`, `trays.json` | Older grid-based crop tool for real photos (not used by the pipeline above) |

---

## How it was built

- **Labels straight from the twin:** every step in the dataset is labelled, and the fill order fixes which slot changed. Slot positions in each camera are medians of the label bounding boxes, so nothing is hand-calibrated.
- **Top models:** each step gives two crops of the same slot on the same tray and lighting: one after placement (its class) and one before (empty). The only difference between them is the slot's contents, so the models can't learn lighting or neighbouring vials instead.
  - **What each model learns:** the cap model's negatives include both empty slots and vials with no cap, so it learns "yellow cap present" rather than "vial present". The vial model's positives include capless vials.
  - **Training details:** crops are resized to 128 px. Augmentation is flips, small crop and scale changes, and colour jitter. Loss is weighted BCE for class balance, with AdamW. Episodes are split for validation, never individual crops.
  - **Checks:** pasting an empty slot into the centre of a crop drops the cap score from 1.000 to 0.000. Scrambling the neighbouring slots changes no decision.
- **Why the side view needs no model:** the side camera sits at cap-top height, so a raised or tilted cap pokes above one fixed image row, even behind other vials.
  - **The numbers:** within a tray, good caps sit within ±4 px of that row. A 3 mm raise is +7 px or more, and a missing cap is −14.7 px or less.
  - **Camera drift:** the camera shifts up to about ±6 px between trays. That shift is measured by phase-correlating each tray's empty frame against `ref_side_empty.png`.
- **Why the 45° view is needed:** from the side, a tilt toward or away from the camera looks like a raise. From 45°, a tilted cap's outline squashes or rotates while a raised one keeps its shape, which separates them.

## Known limitations

- **Simulator only so far.** Every model and threshold is fit to the digital twin. On crops from one real photo, cap detection worked, but the vial/empty models called 50 of 53 empty holes "vial". Expect to need real labelled crops before trusting it on real cameras.
- **Tight 3 mm margin.** Good caps reach about +6 px against a 6.5 px threshold. That's the first thing to watch when cameras or lighting differ.
- **Partial coverage of the challenge.** `damaged_vial` isn't detected (the dataset has no examples), `raised_cap` isn't one of the challenge's classes, and only one fixture is handled.
- **Small rare-class counts.** Only 63 capless vials in training and 17 in validation, and the training seed isn't fixed.
- **Motion trigger.** `capture.py` relies on motion; a robot "placed" signal would be more robust.
