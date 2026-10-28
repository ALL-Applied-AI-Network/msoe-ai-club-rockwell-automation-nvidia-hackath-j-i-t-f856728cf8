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
