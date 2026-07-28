"""Train and evaluate the supervised full-player goalkeeper classifier."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
import cv2
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from handball_annotator.runtime import get_device, seed_everything

from .config import project_path
from .goalkeeper_classifier import (
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_goalkeeper_model,
    save_checkpoint,
)
from .goalkeeper_dataset import assign_group_folds


@dataclass(frozen=True)
class TrainGoalkeeperConfig:
    candidates_manifest: Path
    training_manifest: Path
    folds: int
    validation_fold: int
    test_fold: int
    seed: int
    pretrained: bool
    input_size: int
    device: str
    batch_size: int
    epochs: int
    freeze_backbone_epochs: int
    head_learning_rate: float
    backbone_learning_rate: float
    weight_decay: float
    patience: int
    num_workers: int
    target_class_precision: float
    checkpoint: Path
    history: Path
    metrics: Path
    predictions: Path
    mistakes: Path


def load_config(path: str | Path) -> TrainGoalkeeperConfig:
    config_path = project_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Goalkeeper training config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    try:
        data, model, training, paths = (
            raw["data"],
            raw["model"],
            raw["training"],
            raw["paths"],
        )
        if model.get("architecture") != "mobilenet_v3_small":
            raise ValueError("Only mobilenet_v3_small is supported")
        config = TrainGoalkeeperConfig(
            candidates_manifest=project_path(data["candidates_manifest"]),
            training_manifest=project_path(data["training_manifest"]),
            folds=int(data.get("folds", 5)),
            validation_fold=int(data.get("validation_fold", 0)),
            test_fold=int(data.get("test_fold", 1)),
            seed=int(data.get("seed", 42)),
            pretrained=bool(model.get("pretrained", True)),
            input_size=int(model.get("input_size", 224)),
            device=str(training.get("device", "auto")),
            batch_size=int(training.get("batch_size", 16)),
            epochs=int(training.get("epochs", 35)),
            freeze_backbone_epochs=int(
                training.get("freeze_backbone_epochs", 4)
            ),
            head_learning_rate=float(
                training.get("head_learning_rate", 0.001)
            ),
            backbone_learning_rate=float(
                training.get("backbone_learning_rate", 0.0001)
            ),
            weight_decay=float(training.get("weight_decay", 0.0001)),
            patience=int(training.get("patience", 7)),
            num_workers=int(training.get("num_workers", 0)),
            target_class_precision=float(
                training.get("target_class_precision", 0.85)
            ),
            checkpoint=project_path(paths["checkpoint"]),
            history=project_path(paths["history"]),
            metrics=project_path(paths["metrics"]),
            predictions=project_path(paths["predictions"]),
            mistakes=project_path(paths["mistakes"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid goalkeeper training config: {exc}") from exc
    if config.validation_fold == config.test_fold:
        raise ValueError("Validation and test folds must be different")
    if not 0 <= config.validation_fold < config.folds:
        raise ValueError("validation_fold is outside the configured folds")
    if not 0 <= config.test_fold < config.folds:
        raise ValueError("test_fold is outside the configured folds")
    return config


class CropDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: Any):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.labels = self.frame["review_label"].map(
            {"not_goalkeeper": 0, "goalkeeper": 1}
        ).astype(int).tolist()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(str(row["crop_path"])) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.labels[index], index


def build_transforms(input_size: int) -> tuple[Any, Any]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                input_size, scale=(0.75, 1.0), ratio=(0.65, 1.35)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.20,
                contrast=0.20,
                saturation=0.12,
                hue=0.03,
            ),
            transforms.RandomRotation(7),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3)], p=0.15
            ),
            transforms.ToTensor(),
            transforms.RandomErasing(
                p=0.12, scale=(0.02, 0.10), ratio=(0.4, 2.5)
            ),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(int(input_size * 1.14)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, evaluation_transform


def prepare_manifest(config: TrainGoalkeeperConfig) -> pd.DataFrame:
    if not config.candidates_manifest.is_file():
        raise FileNotFoundError(
            f"Crop candidate manifest not found: {config.candidates_manifest}"
        )
    candidates = pd.read_csv(
        config.candidates_manifest, keep_default_na=False
    )
    manifest = assign_group_folds(
        candidates, folds=config.folds, seed=config.seed
    )
    missing_files = [
        path
        for path in manifest["crop_path"].map(Path)
        if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} reviewed crops are missing; first: "
            f"{missing_files[0]}"
        )
    config.training_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(config.training_manifest, index=False)
    return manifest


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "examples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "confusion_matrix": matrix.tolist(),
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
    }


def select_abstaining_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    target_precision: float,
) -> tuple[float, float]:
    def threshold_for_positive(
        targets: np.ndarray, scores: np.ndarray
    ) -> float:
        candidates: list[tuple[float, float, float]] = []
        for threshold in np.linspace(0.50, 0.95, 46):
            predicted = scores >= threshold
            if not predicted.any():
                continue
            precision = float(np.mean(targets[predicted] == 1))
            coverage = float(np.mean(predicted))
            if precision >= target_precision:
                candidates.append((coverage, precision, float(threshold)))
        if candidates:
            return max(candidates)[2]
        return 0.75

    high = threshold_for_positive(labels, probabilities)
    field_high = threshold_for_positive(1 - labels, 1 - probabilities)
    low = 1.0 - field_high
    if low >= high:
        return 0.25, 0.75
    return float(low), float(high)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for features, target, index in loader:
        logits = model(features.to(device))
        probabilities.append(
            torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        )
        labels.append(target.numpy())
        indices.append(index.numpy())
    return (
        np.concatenate(labels).astype(int),
        np.concatenate(probabilities).astype(float),
        np.concatenate(indices).astype(int),
    )


def _save_gradcam(
    model: nn.Module,
    image_path: Path,
    transform: Any,
    device: str,
    destination: Path,
) -> None:
    activations: list[torch.Tensor] = []

    def capture(_module, _inputs, output):
        activations.append(output)
        output.retain_grad()

    handle = model.features.register_forward_hook(capture)
    try:
        with Image.open(image_path) as image:
            tensor = transform(image.convert("RGB"))[None].to(device)
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        predicted_class = int(logits.argmax(dim=1)[0])
        logits[0, predicted_class].backward()
        activation = activations[-1]
        if activation.grad is None:
            raise RuntimeError("Grad-CAM activation gradients are unavailable")
        weights = activation.grad.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * activation).sum(dim=1))[0]
        maximum = float(heatmap.max().detach().cpu())
        if maximum > 0:
            heatmap = heatmap / maximum
        heatmap_array = heatmap.detach().cpu().numpy()
        original = cv2.imread(str(image_path))
        if original is None:
            raise RuntimeError(f"Could not read Grad-CAM source: {image_path}")
        resized = cv2.resize(
            heatmap_array,
            (original.shape[1], original.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        color = cv2.applyColorMap(
            np.uint8(np.clip(resized, 0, 1) * 255), cv2.COLORMAP_JET
        )
        overlay = cv2.addWeighted(original, 0.55, color, 0.45, 0)
        if not cv2.imwrite(str(destination), overlay):
            raise OSError(f"Could not write Grad-CAM image: {destination}")
    finally:
        handle.remove()


def _save_mistakes(
    predictions: pd.DataFrame,
    destination: Path,
    model: nn.Module,
    transform: Any,
    device: str,
) -> None:
    # This directory represents one training run. Remove stale review images
    # from an earlier run so its contents always match test_predictions.csv.
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for _, row in predictions[predictions["correct"] == False].iterrows():  # noqa: E712
        source = Path(row["crop_path"])
        name = (
            f"true-{row['review_label']}_p-{row['goalkeeper_probability']:.3f}_"
            f"{source.name}"
        )
        shutil.copy2(source, destination / name)
        _save_gradcam(
            model,
            source,
            transform,
            device,
            destination / f"gradcam_{name}",
        )


def train(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    seed_everything(config.seed)
    random.seed(config.seed)
    manifest = prepare_manifest(config)
    train_frame = manifest[
        ~manifest["fold"].isin([config.validation_fold, config.test_fold])
    ]
    validation_frame = manifest[manifest["fold"] == config.validation_fold]
    test_frame = manifest[manifest["fold"] == config.test_fold]
    if min(len(train_frame), len(validation_frame), len(test_frame)) == 0:
        raise ValueError("Train, validation, and test partitions must be non-empty")
    train_transform, eval_transform = build_transforms(config.input_size)
    train_dataset = CropDataset(train_frame, train_transform)
    validation_dataset = CropDataset(validation_frame, eval_transform)
    test_dataset = CropDataset(test_frame, eval_transform)
    counts = np.bincount(train_dataset.labels, minlength=2)
    sample_weights = [
        1.0 / max(int(counts[label]), 1) for label in train_dataset.labels
    ]
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    device = get_device(config.device)
    model = build_goalkeeper_model(pretrained=config.pretrained).to(device)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.features.parameters(),
                "lr": config.backbone_learning_rate,
            },
            {
                "params": model.classifier.parameters(),
                "lr": config.head_learning_rate,
            },
        ],
        weight_decay=config.weight_decay,
    )
    class_weights = torch.tensor(
        [
            len(train_dataset) / max(2 * int(counts[0]), 1),
            len(train_dataset) / max(2 * int(counts[1]), 1),
        ],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    history: list[dict[str, Any]] = []
    best_f1, stale = -1.0, 0
    best_thresholds = (0.25, 0.75)
    for epoch in range(config.epochs):
        if epoch == config.freeze_backbone_epochs:
            for parameter in model.features.parameters():
                parameter.requires_grad = True
        model.train()
        if epoch < config.freeze_backbone_epochs:
            # Keep pretrained BatchNorm statistics fixed during head-only
            # warmup as well as freezing the feature weights.
            model.features.eval()
        losses: list[float] = []
        for features, labels, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(
                model(features.to(device)), labels.to(device)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        labels, probabilities, _ = predict(
            model, validation_loader, device
        )
        metrics = binary_metrics(labels, probabilities)
        thresholds = select_abstaining_thresholds(
            labels,
            probabilities,
            target_precision=config.target_class_precision,
        )
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            **metrics,
            "not_goalkeeper_threshold": thresholds[0],
            "goalkeeper_threshold": thresholds[1],
        }
        history.append(record)
        print(
            f"epoch={epoch:02d} loss={record['loss']:.4f} "
            f"precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f}",
            flush=True,
        )
        if metrics["f1"] > best_f1:
            best_f1, stale, best_thresholds = metrics["f1"], 0, thresholds
            save_checkpoint(
                config.checkpoint,
                model,
                input_size=(config.input_size, config.input_size),
                decision_thresholds=thresholds,
                training_metadata={
                    "epoch": epoch,
                    "validation_metrics": metrics,
                    "validation_fold": config.validation_fold,
                    "test_fold": config.test_fold,
                    "seed": config.seed,
                },
            )
        else:
            stale += 1
        if stale >= config.patience:
            break
    config.history.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(config.history, index=False)

    payload = torch.load(
        config.checkpoint, map_location=device, weights_only=True
    )
    model.load_state_dict(payload["state_dict"])
    labels, probabilities, indices = predict(model, test_loader, device)
    metrics = binary_metrics(labels, probabilities)
    prediction_rows = test_dataset.frame.iloc[indices].copy()
    prediction_rows["label"] = labels
    prediction_rows["goalkeeper_probability"] = probabilities
    prediction_rows["prediction"] = (probabilities >= 0.5).astype(int)
    prediction_rows["correct"] = (
        prediction_rows["label"] == prediction_rows["prediction"]
    )
    config.predictions.parent.mkdir(parents=True, exist_ok=True)
    prediction_rows.to_csv(config.predictions, index=False)
    _save_mistakes(
        prediction_rows,
        config.mistakes,
        model,
        eval_transform,
        device,
    )
    result = {
        "test_metrics": metrics,
        "decision_thresholds": list(best_thresholds),
        "dataset": {
            "train": len(train_frame),
            "validation": len(validation_frame),
            "test": len(test_frame),
            "source_groups": int(manifest["source_group"].nunique()),
        },
        "checkpoint": str(config.checkpoint),
        "predictions": str(config.predictions),
        "mistakes": str(config.mistakes),
    }
    config.metrics.parent.mkdir(parents=True, exist_ok=True)
    config.metrics.write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the supervised goalkeeper image classifier."
    )
    parser.add_argument(
        "--config", default="configs/goalkeeper_classifier.yaml"
    )
    args = parser.parse_args()
    print(json.dumps(train(args.config), indent=2))


if __name__ == "__main__":
    main()
