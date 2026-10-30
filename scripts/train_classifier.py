"""Train and evaluate a five-class vial crop classifier."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import functional as transform_functional

try:
    from classifier_utils import (
        CLASS_NAMES,
        DEFECT_INDICES,
        NO_VIAL_INDEX,
        PASS_INDEX,
        binary_defect_predictions,
        binary_numbers,
        load_manifest,
        select_tau,
        separate_no_vial_predictions,
    )
except ModuleNotFoundError:
    from scripts.classifier_utils import (
        CLASS_NAMES,
        DEFECT_INDICES,
        NO_VIAL_INDEX,
        PASS_INDEX,
        binary_defect_predictions,
        binary_numbers,
        load_manifest,
        select_tau,
        separate_no_vial_predictions,
    )

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=112)
    parser.add_argument("--batch", type=int, default=128)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RandomGamma:
    def __init__(self, minimum: float = 0.75, maximum: float = 1.35) -> None:
        self.minimum = minimum
        self.maximum = maximum

    def __call__(self, image: Image.Image) -> Image.Image:
        gamma = random.uniform(self.minimum, self.maximum)
        return transform_functional.adjust_gamma(image, gamma)


class GaussianNoise:
    def __init__(self, standard_deviation: float = 0.025) -> None:
        self.standard_deviation = standard_deviation

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            tensor + torch.randn_like(tensor) * self.standard_deviation,
            0.0,
            1.0,
        )


class CropDataset(Dataset):
    def __init__(self, rows: Sequence[dict[str, str]], transform) -> None:
        self.rows = list(rows)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        row = self.rows[index]
        with Image.open(row["absolute_path"]) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, CLASS_NAMES.index(row["label"]), index


def build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.05,
            ),
            RandomGamma(),
            transforms.RandomAffine(
                degrees=2.0,
                translate=(0.06, 0.06),
                scale=(0.95, 1.05),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
                p=0.3,
            ),
            transforms.ToTensor(),
            GaussianNoise(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, evaluation_transform


def confusion_matrix(
    targets: np.ndarray, predictions: np.ndarray, class_count: int
) -> np.ndarray:
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for target, prediction in zip(targets, predictions, strict=True):
        matrix[int(target), int(prediction)] += 1
    return matrix


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, object]:
    per_class = {}
    f1_values = []
    for index, name in enumerate(CLASS_NAMES):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index, :].sum() - true_positive)
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2.0 * precision * recall / max(1.0e-12, precision + recall)
        f1_values.append(f1)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(matrix[index, :].sum()),
        }
    return {"per_class": per_class, "macro_f1": float(np.mean(f1_values))}


def infer(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    probability_batches = []
    target_batches = []
    index_batches = []
    with torch.inference_mode():
        for images, targets, indices in loader:
            logits = model(images.to(device, non_blocking=True))
            probability_batches.append(torch.softmax(logits, dim=1).cpu().numpy())
            target_batches.append(targets.numpy())
            index_batches.append(indices.numpy())
    return (
        np.concatenate(probability_batches),
        np.concatenate(target_batches),
        np.concatenate(index_batches),
    )


def draw_confusion(matrix: np.ndarray, path: Path) -> None:
    cell = 130
    margin_left = 170
    margin_top = 100
    canvas = Image.new(
        "RGB",
        (margin_left + cell * len(CLASS_NAMES), margin_top + cell * len(CLASS_NAMES)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    maximum = max(1, int(matrix.max()))
    for index, name in enumerate(CLASS_NAMES):
        draw.text((margin_left + index * cell + 5, 30), name, fill="black", font=font)
        draw.text((5, margin_top + index * cell + 5), name, fill="black", font=font)
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            value = int(matrix[row, column])
            intensity = int(255 - 180 * value / maximum)
            color = (intensity, intensity, 255)
            box = (
                margin_left + column * cell,
                margin_top + row * cell,
                margin_left + (column + 1) * cell,
                margin_top + (row + 1) * cell,
            )
            draw.rectangle(box, fill=color, outline="black")
            draw.text((box[0] + 50, box[1] + 55), str(value), fill="black", font=font)
    canvas.save(path)


def draw_false_negative_sheet(
    rows: Sequence[dict[str, str]],
    probabilities: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    tau: float,
    path: Path,
) -> None:
    missed = []
    defect_predictions = binary_defect_predictions(probabilities, tau)
    for position, (target, predicted) in enumerate(
        zip(targets, defect_predictions, strict=True)
    ):
        if int(target) in DEFECT_INDICES and not bool(predicted):
            missed.append((float(probabilities[position, PASS_INDEX]), position))
    missed.sort(reverse=True)
    selected = missed[:24]
    cell_w, cell_h, caption_h = 150, 150, 34
    canvas = Image.new("RGB", (cell_w * 6, (cell_h + caption_h) * 4), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for sheet_index, (pass_probability, position) in enumerate(selected):
        row = rows[int(indices[position])]
        with Image.open(row["absolute_path"]) as image:
            image = image.convert("RGB").resize(
                (cell_w, cell_h), Image.Resampling.NEAREST
            )
        x = (sheet_index % 6) * cell_w
        y = (sheet_index // 6) * (cell_h + caption_h)
        canvas.paste(image, (x, y))
        draw.text(
            (x + 3, y + cell_h + 2),
            f"{row['label']} {row['slot']} p={pass_probability:.3f}",
            fill="black",
            font=font,
        )
    canvas.save(path)


def class_weights(rows: Iterable[dict[str, str]]) -> torch.Tensor:
    counts = Counter(row["label"] for row in rows)
    weights = np.asarray(
        [1.0 / math.sqrt(counts[name]) for name in CLASS_NAMES], dtype=np.float32
    )
    weights /= weights.mean()
    return torch.from_numpy(weights)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.img_size < 1 or args.batch < 1:
        raise ValueError("--epochs, --img-size, and --batch must be positive")
    seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    train_rows = load_manifest(args.data, "train")
    validation_rows = load_manifest(args.data, "val")
    test_rows = load_manifest(args.data, "test")
    train_transform, evaluation_transform = build_transforms(args.img_size)
    train_dataset = CropDataset(train_rows, train_transform)
    validation_dataset = CropDataset(validation_rows, evaluation_transform)
    test_dataset = CropDataset(test_rows, evaluation_transform)
    generator = torch.Generator().manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    loader_options = {
        "batch_size": args.batch,
        "num_workers": 4,
        "pin_memory": use_cuda,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    device = torch.device("cuda" if use_cuda else "cpu")
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.to(device)
    loss_function = nn.CrossEntropyLoss(
        weight=class_weights(train_rows).to(device),
        label_smoothing=0.05,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    best_validation_f1 = -1.0
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, targets, _indices in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_cuda,
            ):
                logits = model(images)
                loss = loss_function(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * images.size(0)
            seen += images.size(0)
        scheduler.step()
        validation_probabilities, validation_targets, _ = infer(
            model, validation_loader, device
        )
        validation_predictions = separate_no_vial_predictions(
            validation_probabilities
        )
        validation_matrix = confusion_matrix(
            validation_targets, validation_predictions, len(CLASS_NAMES)
        )
        validation_metrics = metrics_from_confusion(validation_matrix)
        validation_f1 = float(validation_metrics["macro_f1"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(1, seen),
                "validation_macro_f1": validation_f1,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"epoch={epoch:02d} loss={history[-1]['train_loss']:.6f} "
            f"val_macro_f1={validation_f1:.6f}",
            flush=True,
        )
        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a model state")
    model.load_state_dict(best_state)
    model.to(device)
    validation_probabilities, validation_targets, _ = infer(
        model, validation_loader, device
    )
    tau, validation_binary = select_tau(
        validation_probabilities, validation_targets
    )

    test_probabilities, test_targets, test_indices = infer(model, test_loader, device)
    test_predictions = separate_no_vial_predictions(test_probabilities)
    test_matrix = confusion_matrix(
        test_targets, test_predictions, len(CLASS_NAMES)
    )
    test_metrics = metrics_from_confusion(test_matrix)
    test_binary = binary_numbers(test_probabilities, test_targets, tau)
    false_negatives = {}
    test_defect_predictions = binary_defect_predictions(test_probabilities, tau)
    for index in DEFECT_INDICES:
        false_negatives[CLASS_NAMES[index]] = sum(
            int(target) == index and not predicted
            for target, predicted in zip(
                test_targets, test_defect_predictions, strict=True
            )
        )

    metrics_document = {
        "split": "test",
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_validation_f1,
        "tau": tau,
        **test_metrics,
        "confusion_matrix": test_matrix.tolist(),
        "false_negatives_per_defect_class": false_negatives,
        "false_alarm_rate_at_tau": test_binary["false_alarm_rate"],
        "binary_defect_vs_pass": test_binary,
        "history": history,
    }
    (args.out / "metrics.json").write_text(
        json.dumps(metrics_document, indent=2) + "\n", encoding="utf-8"
    )
    model_card = {
        "architecture": "torchvision_resnet18",
        "class_order": list(CLASS_NAMES),
        "image_size": [args.img_size, args.img_size],
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "tau": tau,
        "no_vial_threshold": 0.5,
        "validation": {
            "best_epoch": best_epoch,
            "macro_f1": best_validation_f1,
            "binary_defect_vs_pass": validation_binary,
        },
        "decision_rule": (
            "Treat no_vial as separate only when it is the argmax with P(no_vial) "
            "> 0.5; otherwise flag defect when P(pass) < tau or the occupied-class "
            "argmax is missing_cap, crooked_cap, or damaged_vial."
        ),
    }
    (args.out / "model_card.json").write_text(
        json.dumps(model_card, indent=2) + "\n", encoding="utf-8"
    )
    draw_confusion(test_matrix, args.out / "confusion.png")
    draw_false_negative_sheet(
        test_rows,
        test_probabilities,
        test_targets,
        test_indices,
        tau,
        args.out / "fn_sheet.png",
    )

    model.to("cpu").eval()
    example = torch.zeros(1, 3, args.img_size, args.img_size)
    scripted = torch.jit.trace(model, example)
    scripted.save(str(args.out / "model.pt"))
    print(
        f"best_epoch={best_epoch} val_macro_f1={best_validation_f1:.6f} "
        f"tau={tau:.8f} test_macro_f1={test_metrics['macro_f1']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
