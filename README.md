# Final Vial Inspection

Challenge 1 submission for the Rockwell Automation/NVIDIA hackathon.

## Problem

Classify each of the 30 positions in fixtures G and S as `pass`, `missing_cap`, `crooked_cap`, `damaged_vial`, or `no_vial`. The evaluation penalizes missed defects more heavily than false alarms.

## Approach

Train on digital-twin images only and reserve all real images for testing. Document the synthetic-data generation, preprocessing, model, and cost-sensitive decision strategy here.

## Results

Add validation and real-image test results here. Report the dataset version, random seed, per-class metrics, confusion matrix, missed-defect rate, and false-alarm rate.

## How to run

Document environment setup, synthetic-data generation, training, evaluation, and inference commands here. Keep datasets, renders, checkpoints, exported models, and secrets outside version control.

## Credits

List team members and credit the hackathon organizers, datasets, models, libraries, code, and other tools used by the project.
