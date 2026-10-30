# Final Vial Inspection

Rockwell Automation/NVIDIA Hackathon Challenge 1 project for classifying every vial-fixture position as one of five classes: `pass`, `missing_cap`, `crooked_cap`, `damaged_vial`, or `no_vial`.

## Data rule

Training uses digital-twin images only. Real images are reserved strictly for testing and are never used for training, validation, augmentation, or model selection.

## Repository layout

- `scripts/gen_tray_v1.py`: versioned digital-twin tray generation script.
- `scripts/draw_overlay.py`: laptop-side visualization and overlay script.
- `docs/`: project notes and documentation.
- `AGENTS.md`: repository rules and development workflow.

Datasets, images, generated outputs, model weights, and secrets are excluded from version control.

## Running the scripts

Run digital-twin generation on the DGX Spark through YAAD jobs. Copy scripts from the laptop to the temporary Spark workspace as needed. Run analysis and visualization scripts on the laptop.

## Factory run

Factory inference runs locally on CPU from the exported TorchScript `model.pt`. It does not require Isaac Sim, TAO, a network connection, or pretrained-weight downloads at runtime.

For complete Linux setup, camera calibration, operation, output interpretation, test-data collection, and troubleshooting instructions, see the [Factory Inference Runbook](docs/RUNBOOK.md).

Create and activate a virtual environment, then install the pinned runtime packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Keep model files in the ignored `weights/` directory, for example `weights/model.pt`. Do not commit weights or factory images.

The checked-in `slots_G.json` contains the 30 calibrated G-fixture ROIs for 1920x1080 `view_001` images. For a different camera pose, click the slot centers in `G01` through `G30` order. Left-click adds a center, right-click or `U` undoes, `R` resets, Enter saves after all 30 clicks, and Esc cancels:

```powershell
python scripts/define_slots.py --image C:\factory\reference.png --out slots_G.json
```

Run one image with the default `P(pass) < 0.90` defect threshold:

```powershell
python scripts/predict.py --weights weights/model.pt --image C:\factory\tray.png --slots slots_G.json --out-dir C:\factory\predictions
```

Override the threshold with `--threshold`. Monitor a folder every two seconds, or sample camera index 0 at the same interval:

```powershell
python scripts/predict.py --weights weights/model.pt --watch-dir C:\factory\incoming --slots slots_G.json --poll-seconds 2 --out-dir C:\factory\predictions
python scripts/predict.py --weights weights/model.pt --camera-index 0 --slots slots_G.json --poll-seconds 2 --out-dir C:\factory\predictions
```

Each processed image produces `<name>_predictions.csv`, `<name>_predictions.json`, and `<name>_overlay.png`. CSV and JSON contain the slot ID, five-class argmax prediction, `P(pass)`, and `PASS` or `DEFECT` flag. The overlay uses green boxes for pass and red boxes for defect. A crop is flagged whenever `P(pass)` is below the threshold, even when `pass` is still its highest-probability class. The console prints a one-line tray summary.
