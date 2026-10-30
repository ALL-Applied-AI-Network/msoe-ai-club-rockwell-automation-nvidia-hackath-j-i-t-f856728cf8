"""Train one of the three top-view slot classifiers (ResNet-18, ImageNet-pretrained) on labelled crops.

Crops are sorted into <data>/<class>/ where <class> is a key of LABELS (prep.py builds data_sim/). One run per model:
    python train.py cap
    python train.py vial
    python train.py empty
"""
import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms as T

# folder -> (cap, vial, empty). "cap" = a cap is on the vial. The top camera can't see a 3 mm raise
# or a 10 deg tilt, so raised / crooked caps count as present here; the side skyline check judges seating.
LABELS = {
    "pass":        (1, 1, 0),
    "raised_cap":  (1, 1, 0),
    "crooked_cap": (1, 1, 0),
    "missing_cap": (0, 1, 0),
    "no_vial":     (0, 0, 1),
    "empty":       (0, 0, 1),
}
TARGETS = ("cap", "vial", "empty")
SIZE = 128
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

EVAL_TF = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])
TRAIN_TF = T.Compose([
    T.RandomResizedCrop(SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),  # slack for calibration drift
    T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
    T.ColorJitter(0.3, 0.3, 0.3, 0.03),  # factory lighting changes
    T.ToTensor(), T.Normalize(MEAN, STD),
])


def build_model(pretrained=True):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, 1)  # one logit; sigmoid gives the confidence
    return m


def split(samples, n_classes, frac=0.2, seed=0):
    """Per-class split so rare classes still show up in val. Same split for all three targets."""
    # ponytail: splits by crop; split by capture session instead if val scores look too good
    train, val = [], []
    for c in range(n_classes):
        ids = [i for i, (_, ci) in enumerate(samples) if ci == c]
        random.Random(seed).shuffle(ids)
        n = round(len(ids) * frac) if len(ids) > 1 else 0
        val += ids[:n]
        train += ids[n:]
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=TARGETS)
    ap.add_argument("--data", default="data_sim/train")
    ap.add_argument("--val", default="data_sim/val", help="held-out folder; '' = random 20%% split of --data")
    ap.add_argument("--out", default="models")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    train_ds = datasets.ImageFolder(args.data, TRAIN_TF)
    val_ds = datasets.ImageFolder(args.val or args.data, EVAL_TF)
    unknown = (set(train_ds.classes) | set(val_ds.classes)) - LABELS.keys()
    if unknown:
        raise SystemExit(f"unknown class folders {unknown}; expected some of {list(LABELS)}")
    k = TARGETS.index(args.target)
    lut = torch.tensor([LABELS[c][k] for c in train_ds.classes], dtype=torch.float32)  # class idx -> 0/1
    val_lut = torch.tensor([LABELS[c][k] for c in val_ds.classes], dtype=torch.float32)

    if args.val:
        train_idx, val_idx = list(range(len(train_ds))), list(range(len(val_ds)))
    else:
        train_idx, val_idx = split(train_ds.samples, len(train_ds.classes))
    y = lut[[train_ds.samples[i][1] for i in train_idx]]
    pos, neg = y.sum().item(), len(y) - y.sum().item()
    if not pos or not neg:
        raise SystemExit(f"'{args.target}' needs both positive and negative crops (got {pos:.0f} pos, {neg:.0f} neg)")
    print(f"{args.target}: {len(train_idx)} train / {len(val_idx)} val crops, {pos:.0f} positive in train")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, device=dev))  # empties dominate
    train_dl = DataLoader(Subset(train_ds, train_idx), batch_size=32, shuffle=True)
    val_dl = DataLoader(Subset(val_ds, val_idx), batch_size=64)
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, c in train_dl:
            loss = loss_fn(model(x.to(dev)).squeeze(1), lut[c].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()

        if not val_idx:  # too little data for a val split: just keep the last epoch
            torch.save(model.state_dict(), out / f"{args.target}.pt")
            print(f"epoch {epoch}: train loss {loss.item():.4f}")
            continue
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(x.to(dev)).squeeze(1) for x, _ in val_dl])
            t = torch.cat([val_lut[c] for _, c in val_dl]).to(dev)
        val_loss = loss_fn(logits, t).item()
        pred = logits > 0
        tpr = (pred & (t == 1)).sum() / (t == 1).sum().clamp(min=1)
        tnr = (~pred & (t == 0)).sum() / (t == 0).sum().clamp(min=1)
        mark = ""
        if val_loss < best:
            best, mark = val_loss, "  <- saved"
            torch.save(model.state_dict(), out / f"{args.target}.pt")
        print(f"epoch {epoch}: val loss {val_loss:.4f}  balanced acc {(tpr + tnr).item() / 2:.3f}{mark}")
    print(f"best model -> {out / args.target}.pt")


if __name__ == "__main__":
    main()
