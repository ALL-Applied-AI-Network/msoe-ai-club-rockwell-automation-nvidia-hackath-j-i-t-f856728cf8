# Runbook: vial cap checker (ResNet + camera geometry)

This file is for a person or an AI agent operating the checker on branch `Ash's-run-it-through-ResNets`. It says how to run, check, retrain and troubleshoot it. `README.md` explains how the checker works.

**Conventions**
- **Where to run.** Run every command from the repo root, the folder that contains `infer.py`. The scripts read and write `slots_sim.json`, `ref_side_empty.png` and `models/` by relative path.
- **Shell.** Commands are written for bash: Git Bash on Windows, or the normal shell on the Spark. Every `python ...` line also works as is in PowerShell.
- **venv.** Activate it in every new terminal (step 1).

---

## How to run

Do the steps in order. Each one says what you should see. If you see something else, stop and look it up in [Troubleshooting](#troubleshooting).

| Goal | Steps |
|---|---|
| Check that the checker works | 1, 2, 3, 4 |
| Watch a recorded tray | 1, 2, 5 |
| Run on the USB cameras (Windows only), or try the camera path on a recording | 1, 2, 6 |
| No `models/` yet, or the dataset changed | 1, 2, [R1](#r1-rebuild-crops-and-references-preppy), [R2](#r2-retrain-the-three-top-view-models-trainpy), 3, 4 |

### Step 1. Install (once per machine)

Windows laptop with Python 3.11 and an NVIDIA GPU. RTX 50-series cards need the CUDA 12.8 build:

```bash
python -m venv .venv
source .venv/Scripts/activate        # PowerShell: .venv\Scripts\Activate.ps1
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install opencv-python==5.0.0.93 pillow==12.3.0 numpy==2.4.6
python -c "import torch; print(torch.cuda.is_available())"   # True. False still works, only slower (CPU).
```

**On the DGX Spark** (Linux aarch64, GB10, Python 3.12):
- **Checkout.** It is at `/home/yt-jit3783/workspace/hackathon`; connect with `ssh yt-jit3783@172.30.102.95` over the CSI VPN. That checkout has no GitHub login, so `git pull` there fails; copy files over from the laptop instead.
- **venv.** It uses the same pinned versions as the laptop. The cu128 torch build ships sm_120 GPU kernels, which run on the GB10. The venv is already set up there; redo the first four lines below only if the workspace was wiped.

```bash
cd /home/yt-jit3783/workspace/hackathon
python3 -m venv .venv
source .venv/bin/activate             # every new terminal
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install opencv-python==5.0.0.93 pillow==12.3.0 numpy==2.4.6
python -c "import torch; print(torch.cuda.get_device_name(0))"   # NVIDIA GB10
```

- **Running.** With the venv active, every command in this file works as written. On the Spark GPU:
  - step 4 takes about 15 s;
  - the training trays take about 80 s;
  - R2 takes about 20 s per epoch, about 20 min for all three models.
- **Don't run the scripts with the system `python3`, outside the venv.** Its torch 2.5.1 has no kernels for the GB10 (sm_121) but still reports CUDA as available, so the scripts crash with `CUDA error: no kernel image is available for execution on the device`.
- **Steps 5–6.** Step 5 works on the Spark, but it saves the dashboard as a video because the Spark has no display. Do step 6 on the laptop: `capture.py` opens cameras through DirectShow, which only exists on Windows.

### Step 2. Put in place the files git doesn't have

| Path | Where it comes from |
|---|---|
| `dataset_v1_partial/dataset_v1_partial/` with `frames/`, `labels/`, `masks/`, `cameras.json` | The Isaac Sim fill-sequence dataset. Copy it from a teammate. |
| `models/cap.pt`, `models/vial.pt`, `models/empty.pt` | Copy them from a teammate's checkout (about 45 MB each), or train them with [R2](#r2-retrain-the-three-top-view-models-trainpy). |
| `ref_side_empty.png` | Copy it with the command below, or re-run [R1](#r1-rebuild-crops-and-references-preppy). It is the side frame `s00` of `ref_episode` in `slots_sim.json`, which is `train_0000`. |
| `slots_sim.json` | Already in git. Don't regenerate it just to run. |

```bash
cp dataset_v1_partial/dataset_v1_partial/frames/side/train_0000_s00.png ref_side_empty.png
```

Preflight check. The first line prints nothing when every file is there; otherwise it prints `ls: cannot access ...` for each missing one:

```bash
ls slots_sim.json ref_side_empty.png models/{cap,vial,empty}.pt dataset_v1_partial/dataset_v1_partial/cameras.json > /dev/null
ls dataset_v1_partial/dataset_v1_partial/labels/*.json | wc -l      # 32
```

### Step 3. Self-check (seconds, needs no data, models or GPU)

```bash
python test_vials.py        # prints: ok
```

This tests:
- the slot-grid maths (`slots.py`);
- the side skyline check and the 45° check;
- every `decide()` rule;
- the placement trigger;
- how `live.py` finds trays in a frames folder.

It doesn't load the models or the dataset; step 4 does.

### Step 4. Score the held-out trays

```bash
python infer.py dataset_v1_partial/dataset_v1_partial
```

This scores the last 6 episodes in sorted order, `train_0026` to `train_0031` (180 steps), and takes about 45 s on the laptop. **Every episode is named `train_NNNN`, the held-out ones too**, so `train_0031` in the examples is a held-out tray. A correct run prints only this:

```
truth / verdict         pass     no_vial missing_cap  raised_cap crooked_cap       check
pass                      93           0           0           0           0           0
no_vial                    0           6           0           0           0           0
missing_cap                0           0          17           0           0           0
raised_cap                 0           0           0          29           0           0
crooked_cap                0           0           0           0          35           0

6 episodes, 180 steps: class accuracy 1.000, good-vs-bad accuracy 1.000
```

Each wrong step prints one line above the matrix:

```
<episode> k<NN>  <truth> -> <verdict> cap <p> vial <p> empty <p> | side rise <±px> dome <d> | 45deg tilt <t>
```

`side nothing new` means the side camera saw no change. `45deg no cap` means no new cap-yellow showed up in the 45° view. Any such line is a regression: don't commit until it's explained.

Other forms:

```bash
python infer.py dataset_v1_partial/dataset_v1_partial --episodes train_0031 -v      # every step of one tray
python infer.py dataset_v1_partial/dataset_v1_partial --models models_new           # score another model folder
python infer.py dataset_v1_partial/dataset_v1_partial --episodes $(printf 'train_%04d ' $(seq 0 25))
                                                       # the 26 training trays, about 3 min
```

On the training trays, a correct run gives 778/780 (class and good-vs-bad accuracy 0.997). The two misses are known: in `train_0008 k28` and `train_0017 k28`, a good cap at rise +7.4 px is called `raised_cap`. That is the tight margin in [Thresholds](#thresholds).

### Step 5. Replay a recorded tray

```bash
python live.py dataset_v1_partial/dataset_v1_partial/frames train_0031 --pace 1
```

A window steps through the tray at one placement per second. Without `--pace`, it goes as fast as it can. The panels are:
- **Top-left:** the top view, with every slot so far boxed: green = pass, red = defect, grey = no vial, orange = check.
- **Top-right:** the side view, with the skyline in cyan.
- **Bottom-right:** the 45° view.
- **Bottom-left:** the readout.

The terminal prints `--- train_0031`, then one line per step and a summary:

```
G<NN> <verdict> cap <p> vial <p> empty <p>  rise <±px> dome <d>  tilt <t>  (<ms> ms)
train_0031 done: {'raised_cap': 5, 'pass': 15, 'no_vial': 2, 'crooked_cap': 5, 'missing_cap': 3}
```

`side -` or `45deg -` in place of the numbers means that camera saw nothing new. The last view stays up until you press Esc or close the window.

**On the Spark, or any Linux machine with no display,** the same command opens no window and saves the dashboard as a video:
- **What it prints:** first `no display: saving each tray's dashboard to live_videos/<tray>.mp4 ...`, then the same step lines, then `dashboard video -> live_videos/train_0031.mp4`.
- **Speed:** it runs at full speed; on the Spark GPU that is about 35 ms per step.
- **The video:** one frame per step (1120×720), played back at one step per `--pace` seconds, or 1 s without `--pace`. `--videos <dir>` saves to another folder.

To watch it, copy it to the laptop by running this on the laptop:

```bash
scp yt-jit3783@172.30.102.95:/home/yt-jit3783/workspace/hackathon/live_videos/train_0031.mp4 .
```

### Step 6. Live USB cameras (Windows only, needs a display)

1. **Find the camera indices.** This saves `cam_<i>.png` for each index 0–5 that opens, overwriting old ones. Index 0 is often the laptop webcam.
   ```bash
   python capture.py --list
   ```
2. **Capture and check** with your indices (venv active). `--settings` opens each camera's driver dialog, where you turn off auto exposure, auto white balance and autofocus. Start with the empty tray in view, because the first still moment is taken as the empty tray.
   ```bash
   python capture.py --top 0 --side 1 --diag 2 --settings
   ```
   It prints `checking tray_0001 ...`, then `--- tray_0001  (<time>, empty tray)`. After each placement it checks that slot on the spot, the same way as step 5: one `GNN <verdict> ...` line, and the `vial check` window shows the dashboard. After `G30` it prints `tray_0001 done: {...}` and `new tray tray_0002 ...`. A second window, `capture`, shows the three cameras live, with the tray box yellow while something moves.

   Nothing is written to disk. Only the previous state's frames are kept, for comparison with the next one, so a tray can't be replayed afterwards.

- **Frame sizes.** The top camera must give 640×480, and the side and 45° cameras 1920×1080. `warning: <cam> gives WxH, resizing ...` means the slot boxes may be off.
- **Keys in the capture window.**
  - Space captures now.
  - N starts a new tray; its next capture is the empty tray.
  - Esc quits.
  - After `G30`, the next capture starts a new tray by itself.
- **What counts as a placement.** Motion over the tray in the top view, then `--settle` seconds of stillness (default 1.0). Motion means more than `--motion-px` changed pixels (default 40).
- **Restarting.** Tray numbers start again at `tray_0001`.
- **Without cameras.** `--frames` feeds a recorded tray in place of the three cameras, so you can try this step with no hardware. It shows one state every `--pace` seconds (default 2, must be more than `--settle`). A black frame comes before each state, standing in for the arm passing, so the trigger also fires for empty slots. After the last state the stream holds its last frame, and Esc quits.
  ```bash
  python capture.py --frames dataset_v1_partial/dataset_v1_partial/frames --tray train_0031
  ```
  The 30 `GNN` lines match step 5's replay of the same tray, apart from the ms times. A run takes about 62 s.

---

## Rebuild and retrain

Only needed when the dataset changes, `models/` is missing, or you're trying a training change.

### R1. Rebuild crops and references (`prep.py`)

`prep.py` overwrites `slots_sim.json` (tracked in git) and `ref_side_empty.png`. It also **deletes the whole `--out` folder** (default `data_sim/`) before writing it, so never point `--out` at a folder that holds anything else.

```bash
python prep.py dataset_v1_partial/dataset_v1_partial
git diff --stat slots_sim.json          # same dataset: no diff
```

It ends with the crop count per class. For `dataset_v1_partial` the counts are:

```
train {'crooked_cap': 141, 'empty': 780, 'missing_cap': 63, 'no_vial': 59, 'pass': 383, 'raised_cap': 134}
val {'crooked_cap': 35, 'empty': 180, 'missing_cap': 17, 'no_vial': 6, 'pass': 93, 'raised_cap': 29}
```

A missing class means the dataset is incomplete.

### R2. Retrain the three top-view models (`train.py`)

Each run overwrites `models/<name>.pt` without a backup. Training isn't seeded, so a retrain can come out worse. Back up first:

```bash
mkdir -p models_backup && cp models/*.pt models_backup/
python train.py cap
python train.py vial
python train.py empty
python infer.py dataset_v1_partial/dataset_v1_partial      # must still be 180/180 (step 4)
```

- **Inputs.** It needs `data_sim/` from R1.
- **Internet.** The first run downloads ImageNet ResNet-18 weights (`resnet18-f37072fd.pth` into torch's hub cache), so it needs internet once.
- **What it prints.** First `<name>: N train / M val crops, P positive in train`. Then one line per epoch, `epoch E: val loss L  balanced acc A`. The epoch with the lowest validation loss so far is marked `<- saved`, and that is the one kept.
- **If it scores below 180/180,** restore the backup with `cp models_backup/*.pt models/`. Don't loosen thresholds to make up for a weaker model.
- **To try new models without touching the current ones,** add `--out models_new` to all three runs, then run `python infer.py dataset_v1_partial/dataset_v1_partial --models models_new`.
- **Options and defaults:**
  - `--epochs`: 20
  - `--lr`: 3e-4
  - `--data`: `data_sim/train`
  - `--val`: `data_sim/val`. `--val ''` uses a random 20% per-class split of `--data` instead.
  - `--out`: `models`

### R3. Change a threshold

1. Edit the constant at the top of `infer.py` (see [Thresholds](#thresholds)). Don't edit the numbers inside `decide()`.
2. Judge the change on the training trays only. Look at the rise, dome and tilt values of the steps near the new cut-off:
   ```bash
   python infer.py dataset_v1_partial/dataset_v1_partial --episodes $(printf 'train_%04d ' $(seq 0 25)) -v
   ```
3. Run step 4 once to confirm. Don't go back and forth between the held-out score and the threshold.
4. If a `decide()` outcome changed, update the asserts in `test_vials.py::test_decide` and re-run step 3.

---

## Reference

### What runs at each step

Step k means vial k went into slot G`k`; the fill order is always G01 → G30. `infer.check_step()` takes the three camera frames from just before and just after that placement:

| Camera | Code | What it does | Output |
|---|---|---|---|
| Top, 640×480 | `train.build_model`, `models/{cap,vial,empty}.pt` | Crops 32 px around slot k from the AFTER frame, resizes it to 128 px, and runs three ResNet-18 binary classifiers | `cap`, `vial`, `empty` sigmoid scores |
| Side, 1920×1080 | `skyline.measure` | Diffs BEFORE and AFTER in slot k's columns and compares the result with this tray's skyline row | `(rise px, dome)`, or `None` if nothing changed |
| 45°, 1920×1080 | `upright.cap_shape`, `upright.tilt` | Compares the outline of the new cap-yellow pixels in slot k with an upright cap in that slot | `tilt`, or `None` if no new cap is seen |
| — | `infer.decide` | Applies the rules to the three outputs | `pass`, `no_vial`, `missing_cap`, `raised_cap`, `crooked_cap` or `check` |

`check` means the cameras disagree and a person should look. Scoring always counts it as wrong.

### Thresholds

These are the constants at the top of `infer.py`, picked on the 26 training trays:

| Name | Value | Meaning |
|---|---|---|
| `RISE_HIGH` | 6.5 px | Above this, the cap is raised or crooked. Good caps reach about +6 px, so the margin is tight. |
| `RISE_LOW` | −10 px | Below this while the top camera sees a cap: `check`. An open tube is at −15 px or lower. |
| `TILT` | 0.032 | 45° tilt that counts only when the side dome agrees |
| `TILT_BAD` | 0.08 | 45° tilt that is enough on its own |
| `DOME_TILT` | 0.25 | Side-view top shape: flat ≈ 0.1, tilted ≈ 0.3 or more |

### Files

| Path | In git | Made by | Used by |
|---|---|---|---|
| `slots_sim.json` | yes | `prep.py` | `infer.py`, `live.py`, `capture.py` |
| `ref_side_empty.png` | no | `prep.py` (copy of `train_0000`'s side `s00`) | `infer.py`, `live.py`, `capture.py` |
| `data_sim/{train,val}/<class>/` | no | `prep.py` | `train.py` |
| `models/{cap,vial,empty}.pt` | no | `train.py` | `infer.py`, `live.py`, `capture.py` |
| `live_videos/<tray>.mp4` | no | `live.py` on a machine with no display | you, to watch the dashboard |
| `cam_<i>.png` | no | `capture.py --list` | you, to identify the cameras |

- **Keep `slots_sim.json` and `ref_side_empty.png` together.** `side_ref_skyline` in the JSON was measured on that exact PNG, so regenerate both or neither.
- **`slots.py` and `trays.json`** are an older grid tool for real photos. Only `test_grid` uses them.

### Frame layout `live.py` expects

Any frame source can feed `live.py`, including Isaac Sim or a robot's "placed" signal, as long as it writes these files:

```
<frames>/{top,side,diag}/<tray>_s00.png    empty tray, written first (it sets that tray's skyline)
<frames>/{top,side,diag}/<tray>_sNN.png    after vial NN went in, NN = 01..30
```

Write each file atomically: write a temp file, then rename it (`os.replace`), so `live.py` never reads a half-written PNG. `live.py` waits `--timeout` seconds (default 60) for each state.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` (or `cv2`) | The venv isn't active | `source .venv/Scripts/activate` (on the Spark: `source .venv/bin/activate`) |
| `RuntimeError: CUDA error: no kernel image is available for execution on the device` | On the Spark: run with the system `python3`, whose torch 2.5.1 can't use the GB10 | `source .venv/bin/activate` (step 1) |
| `ImportError: ... libwebpmux.so.3: undefined symbol: WebPMalloc` | On the Spark, outside the venv: the system `torchvision` was imported before `cv2` | `source .venv/bin/activate`; the venv's packages don't have this problem |
| `FileNotFoundError` for `models\cap.pt` or `slots_sim.json` | Not run from the repo root, or there are no models yet | `cd` to the repo root; step 2 or R2 |
| `cv2.error: ... !_src.empty() in function 'cvtColor'` in `skyline.episode_skyline` | `ref_side_empty.png` is missing from the current folder | Step 2 |
| `train.py`: `unknown class folders {...}` | Extra folder under `data_sim/train/` or `data_sim/val/` | Re-run R1. Allowed folders: `pass raised_cap crooked_cap missing_cap no_vial empty`. |
| `train.py`: `'<name>' needs both positive and negative crops` | `data_sim/` is incomplete | Re-run R1 and compare the counts |
| Many `check` verdicts across one tray | The side camera moved more than about ±6 px, or `ref_side_empty.png` doesn't match `slots_sim.json` | Regenerate both with R1. In step 5, check that the cyan skyline sits on the cap tops. |
| `raised_cap` on good caps | Rise is near the 6.5 px limit; good caps reach about +6 | Read the rise values with `-v`. Fix the camera before touching `RISE_HIGH`. |
| `crooked_cap` and `raised_cap` mixed up | The 45° view missed the cap (`45deg no cap`), or the lighting moved the cap's hue | Check `upright.CAP_HSV`: hue 23–30; the robot arm is near 20 |
| `qt.qpa.xcb: could not connect to display` then `Aborted (core dumped)` from `live.py` | `DISPLAY` is set, but no X server answers (for example `ssh -X` without an X server on the laptop) | `unset DISPLAY` and rerun; `live.py` then saves a video (step 5) |
| `camera N didn't open` | Wrong index, or not on Windows | `python capture.py --list` on Windows |
| `live.py` prints nothing, then `no empty-tray frames <tray>_s00.png under ...` after 60 s | That tray's `s00` is missing from one of `top/`, `side/`, `diag/` | Make sure all three files exist; `--timeout` waits longer |
| `python live.py <frames>` (no tray) shows nothing | No `*_s00.png` in `<frames>/top/` yet; it waits without printing | Start the frame writer first |
| `no state N within 60 s, stopping this tray` | The frame writer stopped before `s30`, or one camera's file is missing | Expected for an unfinished tray; otherwise check all three folders |
| `capture.py` verdicts look shifted by one slot | It missed a placement, so its step count is one behind the vials on the tray | Press Space before the next placement to capture the missed state; see the next row |
| Placements missed or doubled | The arm paused over the tray longer than `--settle`, or the scene is noisy | Tune `--settle` and `--motion-px`, or capture with Space |
| Slot boxes off target on real cameras | The geometry is fit to the digital twin | Real cameras need their own calibration (see Limits) |

---

## Limits (don't claim more than this)

- **Digital twin only.** On crops from one real photo, the vial and empty models called 50 of 53 empty holes "vial".
- **Coverage.** `damaged_vial` isn't detected. Only fixture G is handled (30 slots in a fixed fill order), not fixture S.
- **Held-out trays aren't fully untouched.** `train.py` picks the saved epoch by loss on `data_sim/val`, which holds the same 6 trays that step 4 scores. The thresholds came from training trays only. So 180/180 is slightly optimistic; 778/780 on the training trays is the other data point.
- **Small numbers.** Rare classes are small: 63 train and 17 val `missing_cap`, and 6 val `no_vial`. Training isn't seeded.
- **Speed.** In the step 5 replay on an RTX 5060 Laptop GPU, a check takes 90–125 ms per step, including drawing the dashboard. The README's 56 ms is the check alone. A placement comes about every 6 s.

## Before reporting "done"

- [ ] `python test_vials.py` prints `ok`
- [ ] `python infer.py dataset_v1_partial/dataset_v1_partial` gives 180/180, or every miss is explained
- [ ] `git status` shows only the intended changes; `slots_sim.json` was diffed if `prep.py` ran
- [ ] `models_backup/` exists if `models/` was overwritten
