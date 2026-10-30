# Factory Inference Runbook

This runbook targets a Linux factory computer connected to a fixed 1920x1080 top-view camera. The computer may be CPU-only or GPU-equipped; the current predictor deliberately loads its TorchScript model on the CPU on either type of host.

## What it does

`scripts/predict.py` reads one tray image, a watched folder, or camera frames. It crops the 30 calibrated G-fixture positions, resizes each crop to 112x112, runs the five-class classifier, and writes machine-readable results plus an annotated image. A position is flagged `DEFECT` from its pass probability, not merely from its highest-probability class.

The runtime is offline after the Python packages and `weights/model.pt` are installed. It does not require Isaac Sim or TAO.

## One-time setup

Clone the repository, select the deployment branch, and create a virtual environment:

```bash
git clone REPOSITORY_URL vial-inspection
cd vial-inspection
git checkout gen-v4
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Replace `REPOSITORY_URL` with the repository's HTTPS or SSH clone URL before running the command.

For a CPU-only PyTorch installation, install the pinned CPU wheels first, then install the remaining requirements. The second command will recognize that the matching Torch packages are already installed:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.0 torchvision==0.24.0
python -m pip install -r requirements.txt
```

On a GPU-equipped computer, use the normal requirements command and the PyTorch build appropriate for its NVIDIA driver. The current `predict.py` still performs inference on CPU.

Create the ignored weights directory and copy the training export into it:

```bash
mkdir -p weights
cp /path/to/exported/model.pt weights/model.pt
```

Do not commit `model.pt`, factory images, prediction results, or credentials. Confirm the command-line interfaces after installation:

```bash
python scripts/predict.py --help
python scripts/define_slots.py --help
```

## Camera check

Mount the camera rigidly above the tray with the same orientation used for slot definition. Disable autofocus, automatic zoom, and any image rotation in the camera or capture software when possible. Verify camera index 0, capture one frame, and require 1920x1080 resolution:

```bash
mkdir -p ../factory-runtime/checks
python -c 'import cv2; cap=cv2.VideoCapture(0); ok,frame=cap.read(); cap.release(); assert ok, "camera read failed"; assert frame.shape[:2] == (1080,1920), f"expected 1920x1080, got {frame.shape[1]}x{frame.shape[0]}"; assert cv2.imwrite("../factory-runtime/checks/camera_check.png", frame)'
```

Inspect `../factory-runtime/checks/camera_check.png`. The full G fixture must be visible, upright, sharply focused, and free of severe glare or obstruction. If index 0 is incorrect, repeat the check with the correct integer camera index.

## Define slots

The checked-in `slots_G.json` is the default for the calibrated 1920x1080 `view_001` top view. Use it unchanged only when its overlay boxes align with the physical G fixture.

If the camera, lens, zoom, tray, or fixture moves, redefine the slots from a current reference image:

```bash
python scripts/define_slots.py --image ../factory-runtime/checks/camera_check.png --out slots_G.json
```

Click the centers in exact `G01` through `G30` order. Left-click adds a center, right-click or `U` undoes, `R` resets, Enter saves after all 30 centers are present, and Esc cancels. The default crop offsets produce the same 80x90 G crops used to build the classifier dataset.

Run one known image after redefining slots and inspect its overlay before operating continuously.

## Run

All examples below use the real flags exposed by `python scripts/predict.py --help`. The program accepts exactly one of `--image`, `--watch-dir`, or `--camera-index`.

Process one image:

```bash
python scripts/predict.py --weights weights/model.pt --image ../factory-runtime/incoming/tray_000001.png --slots slots_G.json --threshold 0.90 --out-dir ../factory-runtime/predictions
```

Poll a folder every two seconds. Existing supported images are processed once when the program starts, and new images are processed as they appear. Stop with Ctrl+C:

```bash
python scripts/predict.py --weights weights/model.pt --watch-dir ../factory-runtime/incoming --slots slots_G.json --threshold 0.90 --poll-seconds 2 --out-dir ../factory-runtime/predictions
```

Read camera index 0 every two seconds. Stop with Ctrl+C:

```bash
python scripts/predict.py --weights weights/model.pt --camera-index 0 --slots slots_G.json --threshold 0.90 --poll-seconds 2 --out-dir ../factory-runtime/predictions
```

The `0.90` value is the software default and is suitable for a wiring smoke test, not a finalized production setting. Record and deploy the threshold selected from validation results.

## Reading the output

For an input named `tray_000001.png`, the output directory receives:

- `tray_000001_predictions.csv`: 30 rows with `slot`, `predicted_class`, `p_pass`, and `flag`.
- `tray_000001_predictions.json`: the same predictions plus the input name and threshold.
- `tray_000001_overlay.png`: all 30 slot boxes and IDs drawn over the source image.

Green means `PASS`; red means `DEFECT`. Always investigate red boxes, and also inspect the predicted class to distinguish `missing_cap`, `crooked_cap`, `damaged_vial`, and `no_vial` behavior.

The console prints one line per tray in this form:

```text
image=tray_000001.png slots=30 pass=28 defects=2 threshold=0.900
```

Archive CSV and JSON outputs with the corresponding source image when traceability is required.

## Collecting and labeling real test data

Real factory images are test-only. Never use them for training, validation, augmentation, threshold selection, or model selection.

Store real images outside the repository, for example:

```text
../factory-runtime/test/2026-10-30/images/
../factory-runtime/test/2026-10-30/labels.csv
```

For every captured tray, preserve the original image and label all 30 positions. A practical label file contains `image`, `slot`, `label`, `reviewer`, and `notes`, with labels restricted to:

```text
pass
missing_cap
crooked_cap
damaged_vial
no_vial
```

Use a second reviewer for ambiguous defects. Freeze the labeled real test set before running comparisons, record the model identifier and threshold with every evaluation, and report per-class recall, false negatives, and false alarms. Do not copy test images or labels into the Git repository.

## Troubleshooting

### Misaligned boxes

- Confirm the camera still outputs 1920x1080 without rotation or scaling.
- Confirm the correct `--slots` file is being used.
- Re-run `define_slots.py` whenever the camera, lens, zoom, tray, or fixture moves.
- Inspect a single-image overlay before restarting watch or camera mode.

### Too many false alarms

- Review `p_pass` in CSV or JSON instead of relying only on the predicted class.
- A lower threshold flags fewer positions as defects.
- Check focus, glare, exposure, and ROI alignment first.
- Select the production threshold only from validation data; do not tune it on the real test set.

### Missed defects

- A higher threshold flags more positions as defects and favors recall over false-alarm rate.
- Confirm the defect is inside the slot crop and that the overlay is aligned.
- Review per-class recall and the validation probability table before changing the threshold.
- Preserve missed examples in the frozen real test set; do not feed them back into training.

### Python or package errors

- Activate the environment with `source .venv/bin/activate`.
- Re-run `python -m pip install -r requirements.txt`.
- On CPU-only systems, use the CPU-wheel commands from one-time setup.
- Confirm `weights/model.pt`, `slots_G.json`, and the input path exist.
- Run `python scripts/predict.py --help` to detect an incomplete installation before factory operation.

## Model card

- Artifact: TorchScript `weights/model.pt`
- Input tensor: `1x3x112x112`
- Source crop: 80x90 RGB G-slot crop, resized to 112x112
- Normalization: ImageNet mean and standard deviation
- Class order:
  1. `pass`
  2. `missing_cap`
  3. `crooked_cap`
  4. `damaged_vial`
  5. `no_vial`
- Defect rule: flag `DEFECT` when `P(pass) < threshold`, regardless of the top class
- Final production threshold: **TBD from validation**

The asymmetric inspection objective favors defect recall: a missed defect is more costly than a false alarm. Record the final validation-derived threshold, model commit, training dataset version, and validation metrics before deployment.
