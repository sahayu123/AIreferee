"""Fuse frozen visual embeddings with the existing temporal handball GRU."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import yaml

from handball_annotator.runtime import get_device, seed_everything

from .config import project_path
from .features import FEATURE_NAMES, feature_path
from .logging_utils import configure_logging
from .metrics import binary_metrics, save_metrics
from .visual_features import (
    VISUAL_BACKBONE,
    VISUAL_EMBEDDING_SIZE,
    visual_feature_path,
)


@dataclass(frozen=True)
class VisualFusionConfig:
    manifest: Path
    base_features_dir: Path
    visual_features_dir: Path
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
    visual_projection_size: int
    dropout: float
    learning_rate: float
    weight_decay: float
    patience: int


@dataclass(frozen=True)
class FusionView:
    example_id: str
    view_id: str
    label: int
    domain: str
    fold: int
    numerical: np.ndarray
    visual: np.ndarray


def load_visual_fusion_config(path: str | Path) -> VisualFusionConfig:
    config_path = project_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        training = raw["training"]
        paths = raw["paths"]
        config = VisualFusionConfig(
            manifest=project_path(paths["manifest"]),
            base_features_dir=project_path(paths["base_features"]),
            visual_features_dir=project_path(paths["visual_features"]),
            checkpoints_dir=project_path(paths["visual_checkpoints"]),
            reports_dir=project_path(paths["visual_reports"]),
            logs_dir=project_path(paths["visual_logs"]),
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
            visual_projection_size=int(
                training.get("visual_projection_size", 32)
            ),
            dropout=float(training.get("dropout", 0.35)),
            learning_rate=float(training.get("learning_rate", 1e-3)),
            weight_decay=float(training.get("weight_decay", 1e-3)),
            patience=int(training.get("patience", 10)),
        )
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid visual fusion configuration: {exc}") from exc
    positive_values = {
        "folds": config.folds,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "hidden_size": config.hidden_size,
        "layers": config.layers,
        "visual_projection_size": config.visual_projection_size,
        "patience": config.patience,
    }
    invalid = [name for name, value in positive_values.items() if value < 1]
    if invalid:
        raise ValueError(
            "Visual fusion values must be positive: " + ", ".join(invalid)
        )
    if not 0 <= config.dropout < 1:
        raise ValueError("training.dropout must be in [0, 1)")
    return config


def _load_npz_matrix(
    path: Path,
    key: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Feature artifact not found: {path}")
    try:
        with np.load(path, allow_pickle=False) as loaded:
            matrix = loaded[key].astype(np.float32)
            metadata = json.loads(str(loaded["metadata"]))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid feature artifact {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Feature metadata must be an object in {path}")
    return matrix, metadata


def load_fusion_views(config: VisualFusionConfig) -> list[FusionView]:
    if not config.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {config.manifest}")
    manifest = pd.read_csv(config.manifest)
    views: list[FusionView] = []
    missing: list[Path] = []
    for _, row in manifest.iterrows():
        numerical_path = feature_path(config.base_features_dir, row)
        visual_path = visual_feature_path(config.visual_features_dir, row)
        if not numerical_path.is_file():
            missing.append(numerical_path)
            continue
        if not visual_path.is_file():
            missing.append(visual_path)
            continue
        numerical, numerical_metadata = _load_npz_matrix(
            numerical_path, "features"
        )
        visual, visual_metadata = _load_npz_matrix(
            visual_path, "embeddings"
        )
        if numerical.ndim != 2 or numerical.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"Unexpected numerical shape in {numerical_path}: "
                f"{numerical.shape}"
            )
        if (
            visual.ndim != 2
            or visual.shape[1] != VISUAL_EMBEDDING_SIZE
        ):
            raise ValueError(
                f"Unexpected visual shape in {visual_path}: {visual.shape}"
            )
        numerical_selected = [
            int(value)
            for value in numerical_metadata.get(
                "selected_frame_indices", []
            )
        ]
        visual_selected = [
            int(value)
            for value in visual_metadata.get("selected_frame_indices", [])
        ]
        if (
            len(numerical) != len(visual)
            or numerical_selected != visual_selected
            or len(numerical_selected) != len(numerical)
        ):
            raise ValueError(
                f"Numerical/visual temporal alignment mismatch for "
                f"{row['example_id']} {row['view_id']}"
            )
        if visual_metadata.get("backbone") != VISUAL_BACKBONE:
            raise ValueError(
                f"Unexpected visual backbone in {visual_path}: "
                f"{visual_metadata.get('backbone')!r}"
            )
        views.append(
            FusionView(
                example_id=str(row["example_id"]),
                view_id=str(row["view_id"]),
                label=int(row["label"]),
                domain=str(row["domain"]),
                fold=int(row["fold"]),
                numerical=numerical,
                visual=visual,
            )
        )
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} fusion feature artifacts are missing. "
            "Run `python -m training.visual_features` first. "
            f"First missing file: {missing[0]}"
        )
    if not views:
        raise ValueError("No fusion views were loaded")
    return views


def fusion_normalization(
    views: list[FusionView],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not views:
        raise ValueError("Cannot normalize an empty training set")
    numerical = np.concatenate([view.numerical for view in views], axis=0)
    visual = np.concatenate([view.visual for view in views], axis=0)
    return (
        numerical.mean(axis=0),
        numerical.std(axis=0),
        visual.mean(axis=0),
        visual.std(axis=0),
    )


class FusionRandomViewDataset(Dataset):
    def __init__(
        self,
        views: list[FusionView],
        numerical_mean: np.ndarray,
        numerical_std: np.ndarray,
        visual_mean: np.ndarray,
        visual_std: np.ndarray,
        *,
        random_view: bool,
    ):
        self.by_example: dict[str, list[FusionView]] = {}
        for view in views:
            self.by_example.setdefault(view.example_id, []).append(view)
        self.example_ids = sorted(self.by_example)
        self.labels = [
            self.by_example[example_id][0].label
            for example_id in self.example_ids
        ]
        self.numerical_mean = numerical_mean.astype(np.float32)
        self.numerical_std = np.maximum(
            numerical_std.astype(np.float32), 1e-6
        )
        self.visual_mean = visual_mean.astype(np.float32)
        self.visual_std = np.maximum(visual_std.astype(np.float32), 1e-6)
        self.random_view = random_view

    def __len__(self) -> int:
        return len(self.example_ids)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        example_id = self.example_ids[index]
        candidates = self.by_example[example_id]
        view = (
            random.choice(candidates) if self.random_view else candidates[0]
        )
        numerical = (
            view.numerical - self.numerical_mean
        ) / self.numerical_std
        visual = (view.visual - self.visual_mean) / self.visual_std
        return (
            torch.from_numpy(numerical),
            torch.from_numpy(visual),
            torch.tensor(view.label, dtype=torch.float32),
            example_id,
        )


class VisualFusionGRU(nn.Module):
    """Existing temporal GRU plus a compact pooled visual branch."""

    def __init__(
        self,
        *,
        numerical_size: int,
        visual_size: int,
        visual_projection_size: int,
        hidden_size: int,
        layers: int,
        dropout: float,
    ):
        super().__init__()
        self.numerical_gru = nn.GRU(
            input_size=numerical_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_size, visual_projection_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        fused_size = hidden_size * 2 + visual_projection_size * 2
        self.head = nn.Sequential(
            nn.Linear(fused_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        numerical: torch.Tensor,
        visual: torch.Tensor,
    ) -> torch.Tensor:
        sequence, hidden = self.numerical_gru(numerical)
        numerical_pooled = torch.cat(
            [hidden[-1], sequence.max(dim=1).values], dim=1
        )
        visual_sequence = self.visual_projection(visual)
        visual_pooled = torch.cat(
            [
                visual_sequence.mean(dim=1),
                visual_sequence.max(dim=1).values,
            ],
            dim=1,
        )
        return self.head(
            torch.cat([numerical_pooled, visual_pooled], dim=1)
        ).squeeze(1)


def _example_predictions(view_predictions: pd.DataFrame) -> pd.DataFrame:
    return view_predictions.groupby("example_id", as_index=False).agg(
        label=("label", "first"),
        probability=("probability", "mean"),
        domain=("domain", "first"),
        fold=("fold", "first"),
        views=("view_id", "count"),
    )


@torch.no_grad()
def predict_fusion_views(
    model: nn.Module,
    views: list[FusionView],
    numerical_mean: np.ndarray,
    numerical_std: np.ndarray,
    visual_mean: np.ndarray,
    visual_std: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, object]] = []
    safe_numerical_std = np.maximum(numerical_std, 1e-6)
    safe_visual_std = np.maximum(visual_std, 1e-6)
    for start in range(0, len(views), batch_size):
        batch = views[start:start + batch_size]
        numerical = np.stack(
            [
                (view.numerical - numerical_mean) / safe_numerical_std
                for view in batch
            ]
        ).astype(np.float32)
        visual = np.stack(
            [
                (view.visual - visual_mean) / safe_visual_std
                for view in batch
            ]
        ).astype(np.float32)
        probabilities = (
            torch.sigmoid(
                model(
                    torch.from_numpy(numerical).to(device),
                    torch.from_numpy(visual).to(device),
                )
            )
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
    metrics = binary_metrics(labels, probabilities)
    predictions = (probabilities >= 0.5).astype(int)
    metrics["accuracy"] = float((predictions == labels.astype(int)).mean())
    metrics["correct"] = int((predictions == labels.astype(int)).sum())
    metrics["incorrect"] = int((predictions != labels.astype(int)).sum())
    return metrics


def train_visual_fusion(
    config_path: str | Path,
    *,
    fold: int | None = None,
) -> dict[str, object]:
    config = load_visual_fusion_config(config_path)
    fold = config.fold if fold is None else fold
    if not 0 <= fold < config.folds:
        raise ValueError(f"fold must be between 0 and {config.folds - 1}")
    seed_everything(config.seed)
    device = get_device(config.device)
    views = load_fusion_views(config)
    train_views = [view for view in views if view.fold != fold]
    validation_views = [view for view in views if view.fold == fold]
    if not train_views or not validation_views:
        raise ValueError(f"Fold {fold} has an empty train or validation split")
    normalization = fusion_normalization(train_views)
    numerical_mean, numerical_std, visual_mean, visual_std = normalization
    dataset = FusionRandomViewDataset(
        train_views,
        *normalization,
        random_view=True,
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
    model = VisualFusionGRU(
        numerical_size=len(FEATURE_NAMES),
        visual_size=VISUAL_EMBEDDING_SIZE,
        visual_projection_size=config.visual_projection_size,
        hidden_size=config.hidden_size,
        layers=config.layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(
        config.logs_dir / f"visual_fusion_fold{fold}.log"
    )
    best_path = (
        config.checkpoints_dir / f"visual_fusion_fold{fold}_best.pt"
    )
    history: list[dict[str, float]] = []
    best_pr_auc = -1.0
    stale_epochs = 0
    for epoch in range(config.epochs):
        model.train()
        losses: list[float] = []
        for numerical, visual, labels, _ in loader:
            numerical = numerical.to(device)
            visual = visual.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(numerical, visual), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        view_predictions = predict_fusion_views(
            model,
            validation_views,
            numerical_mean,
            numerical_std,
            visual_mean,
            visual_std,
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
                "schema": "ai_referee.visual_fusion_gru",
                "schema_version": 1,
                "model": model.state_dict(),
                "epoch": epoch,
                "best_pr_auc": best_pr_auc,
                "numerical_mean": numerical_mean,
                "numerical_std": numerical_std,
                "visual_mean": visual_mean,
                "visual_std": visual_std,
                "feature_names": FEATURE_NAMES,
                "visual_backbone": VISUAL_BACKBONE,
                "visual_embedding_size": VISUAL_EMBEDDING_SIZE,
                "fold": fold,
                "model_config": {
                    "numerical_size": len(FEATURE_NAMES),
                    "visual_size": VISUAL_EMBEDDING_SIZE,
                    "visual_projection_size": (
                        config.visual_projection_size
                    ),
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
    view_predictions = predict_fusion_views(
        model,
        validation_views,
        numerical_mean,
        numerical_std,
        visual_mean,
        visual_std,
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
            "model": "visual_fusion_gru",
            "fold": fold,
            "best_epoch": int(best["epoch"]),
            "device": device,
            "visual_backbone": VISUAL_BACKBONE,
        }
    )
    pd.DataFrame(history).to_csv(
        config.reports_dir / f"visual_fusion_fold{fold}_history.csv",
        index=False,
    )
    view_predictions.to_csv(
        config.reports_dir
        / f"visual_fusion_fold{fold}_view_predictions.csv",
        index=False,
    )
    examples.to_csv(
        config.reports_dir / f"visual_fusion_fold{fold}_predictions.csv",
        index=False,
    )
    save_metrics(
        final_metrics,
        config.reports_dir / f"visual_fusion_fold{fold}_metrics.json",
    )
    logger.info("final=%s", json.dumps(final_metrics, indent=2))
    return final_metrics


def _load_baseline_oof(config: VisualFusionConfig) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    for fold in range(config.folds):
        path = config.baseline_reports_dir / f"gru_fold{fold}_predictions.csv"
        if not path.is_file():
            return None
        frame = pd.read_csv(path)
        required = {"example_id", "label", "probability"}
        if required - set(frame.columns):
            return None
        frame = frame.copy()
        frame["fold"] = fold
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined["example_id"].astype(str).duplicated().any():
        raise ValueError("Baseline OOF predictions contain duplicate examples")
    return combined


def train_all_visual_folds(config_path: str | Path) -> dict[str, object]:
    config = load_visual_fusion_config(config_path)
    fold_metrics = [
        train_visual_fusion(config_path, fold=fold)
        for fold in range(config.folds)
    ]
    prediction_frames = [
        pd.read_csv(
            config.reports_dir
            / f"visual_fusion_fold{fold}_predictions.csv"
        )
        for fold in range(config.folds)
    ]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    if predictions["example_id"].astype(str).duplicated().any():
        raise ValueError("Visual fusion OOF predictions contain duplicates")
    labels = predictions["label"].to_numpy()
    probabilities = predictions["probability"].to_numpy()
    fusion_metrics = _metrics_with_accuracy(labels, probabilities)
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
            raise ValueError("Baseline and visual fusion labels do not match")
        baseline_metrics = _metrics_with_accuracy(
            baseline_frame["label"].to_numpy(),
            baseline_frame["probability"].to_numpy(),
        )
        deltas = {
            name: float(fusion_metrics[name]) - float(baseline_metrics[name])
            for name in ("accuracy", "precision", "recall", "f1", "pr_auc")
        }
        predictions["baseline_probability"] = (
            baseline_frame["probability"].to_numpy()
        )
    predictions.to_csv(
        config.reports_dir / "visual_fusion_oof_predictions.csv",
        index=False,
    )
    report: dict[str, object] = {
        "model": "visual_fusion_gru",
        "evaluation": "five_fold_out_of_fold",
        "visual_backbone": VISUAL_BACKBONE,
        "fusion": fusion_metrics,
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
        config.reports_dir / "visual_fusion_oof_metrics.json",
    )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the temporal handball GRU with visual fusion."
    )
    parser.add_argument("--config", default="configs/visual_fusion.yaml")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--all-folds", action="store_true")
    args = parser.parse_args()
    if args.all_folds:
        train_all_visual_folds(args.config)
    else:
        train_visual_fusion(args.config, fold=args.fold)


if __name__ == "__main__":
    main()
