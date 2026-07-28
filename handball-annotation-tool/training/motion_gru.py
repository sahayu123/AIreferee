"""Train the original temporal GRU on trajectory/optical-flow augmentation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import yaml

from handball_annotator.runtime import get_device, seed_everything

from .config import project_path
from .features import feature_path
from .gru import TemporalGRU
from .logging_utils import configure_logging
from .metrics import binary_metrics, save_metrics
from .motion_features import (
    MOTION_GRU_FEATURE_NAMES,
    motion_feature_path,
)


@dataclass(frozen=True)
class MotionGRUConfig:
    manifest: Path
    motion_features_dir: Path
    checkpoints_dir: Path
    reports_dir: Path
    logs_dir: Path
    baseline_reports_dir: Path
    device: str
    seed: int
    folds: int
    fold: int
    epochs: int
    batch_size: int
    hidden_size: int
    layers: int
    dropout: float
    learning_rate: float
    weight_decay: float
    patience: int


@dataclass(frozen=True)
class MotionView:
    example_id: str
    view_id: str
    label: int
    domain: str
    fold: int
    features: np.ndarray


def load_motion_gru_config(path: str | Path) -> MotionGRUConfig:
    config_path = project_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        training = raw["training"]
        paths = raw["paths"]
        config = MotionGRUConfig(
            manifest=project_path(paths["manifest"]),
            motion_features_dir=project_path(paths["motion_features"]),
            checkpoints_dir=project_path(paths["motion_checkpoints"]),
            reports_dir=project_path(paths["motion_reports"]),
            logs_dir=project_path(paths["motion_logs"]),
            baseline_reports_dir=project_path(
                paths.get("baseline_reports", "artifacts/reports")
            ),
            device=str(training.get("device", "auto")),
            seed=int(training.get("seed", 42)),
            folds=int(training.get("folds", 5)),
            fold=int(training.get("fold", 0)),
            epochs=int(training.get("epochs", 60)),
            batch_size=int(training.get("batch_size", 16)),
            hidden_size=int(training.get("hidden_size", 64)),
            layers=int(training.get("layers", 2)),
            dropout=float(training.get("dropout", 0.25)),
            learning_rate=float(training.get("learning_rate", 1e-3)),
            weight_decay=float(training.get("weight_decay", 1e-4)),
            patience=int(training.get("patience", 10)),
        )
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid motion GRU configuration: {exc}") from exc
    positive = {
        "folds": config.folds,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "hidden_size": config.hidden_size,
        "layers": config.layers,
        "patience": config.patience,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError(
            "Motion GRU values must be positive: " + ", ".join(invalid)
        )
    if not 0 <= config.dropout < 1:
        raise ValueError("training.dropout must be in [0, 1)")
    return config


def _load_motion_artifact(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Motion feature artifact not found: {path}")
    try:
        with np.load(path, allow_pickle=False) as loaded:
            features = loaded["features"].astype(np.float32)
            metadata = json.loads(str(loaded["metadata"]))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid motion feature artifact {path}: {exc}") from exc
    if metadata.get("schema") != "ai_referee.motion_features":
        raise ValueError(f"Unexpected motion feature schema in {path}")
    if metadata.get("schema_version") != 2:
        raise ValueError(f"Unexpected motion feature schema version in {path}")
    if metadata.get("feature_names") != MOTION_GRU_FEATURE_NAMES:
        raise ValueError(f"Motion feature names do not match in {path}")
    expected = (len(metadata.get("selected_frame_indices", [])), len(MOTION_GRU_FEATURE_NAMES))
    if features.ndim != 2 or features.shape != expected:
        raise ValueError(
            f"Unexpected motion feature shape in {path}: "
            f"{features.shape}, expected {expected}"
        )
    if not np.isfinite(features).all():
        raise ValueError(f"Motion features contain non-finite values in {path}")
    return features


def load_motion_views(config: MotionGRUConfig) -> list[MotionView]:
    if not config.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {config.manifest}")
    manifest = pd.read_csv(config.manifest)
    views: list[MotionView] = []
    missing: list[Path] = []
    for _, row in manifest.iterrows():
        path = motion_feature_path(config.motion_features_dir, row)
        if not path.is_file():
            missing.append(path)
            continue
        views.append(
            MotionView(
                example_id=str(row["example_id"]),
                view_id=str(row["view_id"]),
                label=int(row["label"]),
                domain=str(row["domain"]),
                fold=int(row["fold"]),
                features=_load_motion_artifact(path),
            )
        )
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} motion feature artifacts are missing. "
            "Run `python -m training.motion_features` first. "
            f"First missing file: {missing[0]}"
        )
    if not views:
        raise ValueError("No motion feature views were loaded")
    return views


def motion_normalization(
    views: list[MotionView],
) -> tuple[np.ndarray, np.ndarray]:
    if not views:
        raise ValueError("Cannot normalize an empty training set")
    matrix = np.concatenate([view.features for view in views], axis=0)
    return matrix.mean(axis=0), matrix.std(axis=0)


class MotionRandomViewDataset(Dataset):
    def __init__(
        self,
        views: list[MotionView],
        mean: np.ndarray,
        std: np.ndarray,
        *,
        random_view: bool,
    ):
        self.by_example: dict[str, list[MotionView]] = {}
        for view in views:
            self.by_example.setdefault(view.example_id, []).append(view)
        self.example_ids = sorted(self.by_example)
        self.labels = [
            self.by_example[example_id][0].label
            for example_id in self.example_ids
        ]
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)
        self.random_view = random_view

    def __len__(self) -> int:
        return len(self.example_ids)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        example_id = self.example_ids[index]
        candidates = self.by_example[example_id]
        view = (
            random.choice(candidates) if self.random_view else candidates[0]
        )
        normalized = (view.features - self.mean) / self.std
        return (
            torch.from_numpy(normalized),
            torch.tensor(view.label, dtype=torch.float32),
            example_id,
        )


def _example_predictions(view_predictions: pd.DataFrame) -> pd.DataFrame:
    return view_predictions.groupby("example_id", as_index=False).agg(
        label=("label", "first"),
        probability=("probability", "mean"),
        domain=("domain", "first"),
        fold=("fold", "first"),
        views=("view_id", "count"),
    )


@torch.no_grad()
def predict_motion_views(
    model: nn.Module,
    views: list[MotionView],
    mean: np.ndarray,
    std: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    model.eval()
    safe_std = np.maximum(std, 1e-6)
    rows: list[dict[str, object]] = []
    for start in range(0, len(views), batch_size):
        batch = views[start:start + batch_size]
        matrix = np.stack(
            [(view.features - mean) / safe_std for view in batch]
        ).astype(np.float32)
        probabilities = (
            torch.sigmoid(model(torch.from_numpy(matrix).to(device)))
            .cpu()
            .numpy()
        )
        for view, probability in zip(batch, probabilities):
            rows.append(
                {
                    "example_id": view.example_id,
                    "view_id": view.view_id,
                    "label": view.label,
                    "domain": view.domain,
                    "fold": view.fold,
                    "probability": float(probability),
                }
            )
    return pd.DataFrame(rows)


def _metrics_with_accuracy(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, object]:
    labels = labels.astype(int)
    predictions = (probabilities >= 0.5).astype(int)
    metrics = binary_metrics(labels, probabilities)
    metrics["accuracy"] = float((predictions == labels).mean())
    metrics["correct"] = int((predictions == labels).sum())
    metrics["incorrect"] = int((predictions != labels).sum())
    return metrics


def train_motion_gru(
    config_path: str | Path,
    *,
    fold: int | None = None,
) -> dict[str, object]:
    config = load_motion_gru_config(config_path)
    fold = config.fold if fold is None else fold
    if not 0 <= fold < config.folds:
        raise ValueError(f"fold must be between 0 and {config.folds - 1}")
    seed_everything(config.seed)
    device = get_device(config.device)
    views = load_motion_views(config)
    train_views = [view for view in views if view.fold != fold]
    validation_views = [view for view in views if view.fold == fold]
    if not train_views or not validation_views:
        raise ValueError(f"Fold {fold} has an empty train or validation split")
    mean, std = motion_normalization(train_views)
    dataset = MotionRandomViewDataset(
        train_views, mean, std, random_view=True
    )
    class_counts = np.bincount(dataset.labels, minlength=2)
    sample_weights = [
        1.0 / max(class_counts[label], 1) for label in dataset.labels
    ]
    generator = torch.Generator().manual_seed(config.seed)
    sampler = WeightedRandomSampler(
        sample_weights,
        len(dataset),
        replacement=True,
        generator=generator,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=0,
    )
    model = TemporalGRU(
        len(MOTION_GRU_FEATURE_NAMES),
        config.hidden_size,
        config.layers,
        config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(config.logs_dir / f"motion_gru_fold{fold}.log")
    best_path = config.checkpoints_dir / f"motion_gru_fold{fold}_best.pt"
    best_pr_auc = -1.0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        losses: list[float] = []
        for features, labels, _ in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        view_predictions = predict_motion_views(
            model,
            validation_views,
            mean,
            std,
            device=device,
            batch_size=config.batch_size,
        )
        examples = _example_predictions(view_predictions)
        metrics = _metrics_with_accuracy(
            examples["label"].to_numpy(),
            examples["probability"].to_numpy(),
        )
        epoch_result = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "validation_precision": float(metrics["precision"]),
            "validation_recall": float(metrics["recall"]),
            "validation_f1": float(metrics["f1"]),
            "validation_pr_auc": float(metrics["pr_auc"]),
            "validation_accuracy": float(metrics["accuracy"]),
        }
        history.append(epoch_result)
        logger.info(
            "epoch=%d loss=%.4f precision=%.3f recall=%.3f "
            "f1=%.3f pr_auc=%.3f accuracy=%.3f",
            epoch,
            epoch_result["train_loss"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["pr_auc"],
            metrics["accuracy"],
        )
        if float(metrics["pr_auc"]) > best_pr_auc:
            best_pr_auc = float(metrics["pr_auc"])
            stale_epochs = 0
            checkpoint = {
                "schema": "ai_referee.motion_gru",
                "schema_version": 2,
                "model": model.state_dict(),
                "epoch": epoch,
                "best_pr_auc": best_pr_auc,
                "mean": mean,
                "std": std,
                "feature_names": MOTION_GRU_FEATURE_NAMES,
                "fold": fold,
                "model_config": {
                    "input_size": len(MOTION_GRU_FEATURE_NAMES),
                    "hidden_size": config.hidden_size,
                    "layers": config.layers,
                    "dropout": config.dropout,
                },
            }
            temporary = best_path.with_suffix(".temporary.pt")
            torch.save(checkpoint, temporary)
            temporary.replace(best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                logger.info(
                    "early_stopping epoch=%d stale=%d",
                    epoch,
                    stale_epochs,
                )
                break
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    view_predictions = predict_motion_views(
        model,
        validation_views,
        mean,
        std,
        device=device,
        batch_size=config.batch_size,
    )
    examples = _example_predictions(view_predictions)
    final_metrics = _metrics_with_accuracy(
        examples["label"].to_numpy(),
        examples["probability"].to_numpy(),
    )
    final_metrics.update(
        {
            "model": "motion_gru",
            "fold": fold,
            "best_epoch": int(best["epoch"]),
            "device": device,
            "input_features": len(MOTION_GRU_FEATURE_NAMES),
        }
    )
    pd.DataFrame(history).to_csv(
        config.reports_dir / f"motion_gru_fold{fold}_history.csv",
        index=False,
    )
    view_predictions.to_csv(
        config.reports_dir / f"motion_gru_fold{fold}_view_predictions.csv",
        index=False,
    )
    examples.to_csv(
        config.reports_dir / f"motion_gru_fold{fold}_predictions.csv",
        index=False,
    )
    save_metrics(
        final_metrics,
        config.reports_dir / f"motion_gru_fold{fold}_metrics.json",
    )
    logger.info("final=%s", json.dumps(final_metrics, indent=2))
    return final_metrics


def _load_baseline_oof(config: MotionGRUConfig) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    for fold in range(config.folds):
        path = config.baseline_reports_dir / f"gru_fold{fold}_predictions.csv"
        if not path.is_file():
            return None
        frame = pd.read_csv(path)
        if {"example_id", "label", "probability"} - set(frame.columns):
            return None
        frame = frame.copy()
        frame["fold"] = fold
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined["example_id"].astype(str).duplicated().any():
        raise ValueError("Baseline OOF predictions contain duplicate examples")
    return combined


def train_all_motion_folds(config_path: str | Path) -> dict[str, object]:
    config = load_motion_gru_config(config_path)
    fold_metrics = [
        train_motion_gru(config_path, fold=fold)
        for fold in range(config.folds)
    ]
    predictions = pd.concat(
        [
            pd.read_csv(
                config.reports_dir
                / f"motion_gru_fold{fold}_predictions.csv"
            )
            for fold in range(config.folds)
        ],
        ignore_index=True,
    )
    if predictions["example_id"].astype(str).duplicated().any():
        raise ValueError("Motion GRU OOF predictions contain duplicates")
    labels = predictions["label"].to_numpy()
    probabilities = predictions["probability"].to_numpy()
    motion_metrics = _metrics_with_accuracy(labels, probabilities)
    baseline_frame = _load_baseline_oof(config)
    baseline_metrics: dict[str, object] | None = None
    deltas: dict[str, float] | None = None
    if baseline_frame is not None:
        baseline_frame = baseline_frame.set_index("example_id").loc[
            predictions["example_id"].astype(str)
        ]
        if not np.array_equal(
            baseline_frame["label"].to_numpy().astype(int),
            labels.astype(int),
        ):
            raise ValueError("Baseline and motion GRU labels do not match")
        baseline_metrics = _metrics_with_accuracy(
            baseline_frame["label"].to_numpy(),
            baseline_frame["probability"].to_numpy(),
        )
        deltas = {
            name: float(motion_metrics[name]) - float(baseline_metrics[name])
            for name in ("accuracy", "precision", "recall", "f1", "pr_auc")
        }
        predictions["baseline_probability"] = (
            baseline_frame["probability"].to_numpy()
        )
    predictions.to_csv(
        config.reports_dir / "motion_gru_oof_predictions.csv",
        index=False,
    )
    report: dict[str, object] = {
        "model": "motion_gru",
        "evaluation": "five_fold_out_of_fold",
        "input_features": len(MOTION_GRU_FEATURE_NAMES),
        "motion": motion_metrics,
        "baseline": baseline_metrics,
        "delta": deltas,
        "folds": fold_metrics,
        "limitations": (
            "Each fold checkpoint is selected on that same outer fold, "
            "matching the existing GRU comparison but remaining mildly "
            "optimistic."
        ),
    }
    save_metrics(
        report,
        config.reports_dir / "motion_gru_oof_metrics.json",
    )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the original GRU with trajectory and flow features."
    )
    parser.add_argument("--config", default="configs/motion_gru.yaml")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--all-folds", action="store_true")
    args = parser.parse_args()
    if args.all_folds:
        train_all_motion_folds(args.config)
    else:
        train_motion_gru(args.config, fold=args.fold)


if __name__ == "__main__":
    main()
