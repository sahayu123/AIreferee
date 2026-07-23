from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from handball_annotator.runtime import get_device, seed_everything

from .config import load_train_config
from .data import FeatureView, RandomViewDataset, load_views, normalization
from .features import FEATURE_NAMES
from .logging_utils import configure_logging
from .metrics import binary_metrics, save_metrics


class TemporalGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        sequence, hidden = self.gru(features)
        pooled = torch.cat([hidden[-1], sequence.max(dim=1).values], dim=1)
        return self.head(pooled).squeeze(1)


@torch.no_grad()
def predict_views(
    model: nn.Module,
    views: list[FeatureView],
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, object]] = []
    safe_std = np.maximum(std, 1e-6)
    for start in range(0, len(views), batch_size):
        batch = views[start:start + batch_size]
        matrix = np.stack([(view.features - mean) / safe_std for view in batch]).astype(np.float32)
        probabilities = torch.sigmoid(model(torch.from_numpy(matrix).to(device))).cpu().numpy()
        for view, probability in zip(batch, probabilities):
            rows.append({
                "example_id": view.example_id, "view_id": view.view_id,
                "label": view.label, "domain": view.domain, "probability": float(probability),
            })
    return pd.DataFrame(rows)


def _example_predictions(view_predictions: pd.DataFrame) -> pd.DataFrame:
    return view_predictions.groupby("example_id", as_index=False).agg(
        label=("label", "first"), probability=("probability", "mean"),
        domain=("domain", "first"), views=("view_id", "count"),
    )


def train_gru(config_path: str | Path, fold: int | None = None, resume: Path | None = None) -> dict[str, object]:
    config = load_train_config(config_path)
    fold = config.fold if fold is None else fold
    seed_everything(config.seed)
    device = get_device(config.device)
    views = load_views(config.manifest, config.features_dir)
    train_views = [view for view in views if view.fold != fold]
    validation_views = [view for view in views if view.fold == fold]
    mean, std = normalization(train_views)
    train_dataset = RandomViewDataset(train_views, mean, std, random_view=True)
    class_counts = np.bincount(train_dataset.labels, minlength=2)
    sample_weights = [1.0 / max(class_counts[label], 1) for label in train_dataset.labels]
    generator = torch.Generator().manual_seed(config.seed)
    sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True, generator=generator)
    loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=sampler, num_workers=0)

    model = TemporalGRU(len(FEATURE_NAMES), config.hidden_size, config.layers, config.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    start_epoch, best_pr_auc, stale_epochs = 0, -1.0, 0
    logger = configure_logging(config.logs_dir / f"gru_fold{fold}.log")
    config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    best_path = config.checkpoints_dir / f"gru_fold{fold}_best.pt"
    last_path = config.checkpoints_dir / f"gru_fold{fold}_last.pt"

    if resume is not None:
        if not resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        best_pr_auc = float(state.get("best_pr_auc", -1.0))
        logger.info("Resuming at epoch %d from %s", start_epoch, resume)

    history: list[dict[str, float]] = []
    for epoch in range(start_epoch, config.epochs):
        model.train()
        losses = []
        for features, labels, _ in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        view_predictions = predict_views(model, validation_views, mean, std, device, config.batch_size)
        examples = _example_predictions(view_predictions)
        metrics = binary_metrics(examples["label"].to_numpy(), examples["probability"].to_numpy())
        epoch_result = {
            "epoch": float(epoch), "train_loss": float(np.mean(losses)),
            "validation_precision": float(metrics["precision"]),
            "validation_recall": float(metrics["recall"]),
            "validation_f1": float(metrics["f1"]),
            "validation_pr_auc": float(metrics["pr_auc"]),
        }
        history.append(epoch_result)
        logger.info(
            "epoch=%d loss=%.4f precision=%.3f recall=%.3f f1=%.3f pr_auc=%.3f",
            epoch, epoch_result["train_loss"], metrics["precision"], metrics["recall"],
            metrics["f1"], metrics["pr_auc"],
        )
        checkpoint = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "best_pr_auc": max(best_pr_auc, float(metrics["pr_auc"])),
            "mean": mean, "std": std, "feature_names": FEATURE_NAMES,
            "fold": fold, "model_config": {
                "input_size": len(FEATURE_NAMES), "hidden_size": config.hidden_size,
                "layers": config.layers, "dropout": config.dropout,
            },
        }
        torch.save(checkpoint, last_path)
        if float(metrics["pr_auc"]) > best_pr_auc:
            best_pr_auc = float(metrics["pr_auc"])
            stale_epochs = 0
            torch.save(checkpoint, best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                logger.info("Early stopping after %d stale epochs", stale_epochs)
                break

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    view_predictions = predict_views(model, validation_views, mean, std, device, config.batch_size)
    examples = _example_predictions(view_predictions)
    final_metrics = binary_metrics(examples["label"].to_numpy(), examples["probability"].to_numpy())
    final_metrics.update({"model": "gru", "fold": fold, "best_epoch": int(best["epoch"]), "device": device})
    pd.DataFrame(history).to_csv(config.reports_dir / f"gru_fold{fold}_history.csv", index=False)
    view_predictions.to_csv(config.reports_dir / f"gru_fold{fold}_view_predictions.csv", index=False)
    examples.to_csv(config.reports_dir / f"gru_fold{fold}_predictions.csv", index=False)
    save_metrics(final_metrics, config.reports_dir / f"gru_fold{fold}_metrics.json")
    logger.info("final=%s", json.dumps(final_metrics, indent=2))
    return final_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the temporal MediaPipe/ball-track GRU.")
    parser.add_argument("--config", default="configs/temporal_classifier.yaml")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    train_gru(args.config, args.fold, args.resume)


if __name__ == "__main__":
    main()

