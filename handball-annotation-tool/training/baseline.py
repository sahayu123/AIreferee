from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import load_train_config
from .data import aggregate_sequence, load_views
from .features import FEATURE_NAMES
from .logging_utils import configure_logging
from .metrics import binary_metrics, save_metrics


def _evaluate_views(views, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    table = pd.DataFrame({
        "example_id": [view.example_id for view in views],
        "view_id": [view.view_id for view in views],
        "label": [view.label for view in views],
        "domain": [view.domain for view in views],
        "probability": probabilities,
    })
    grouped = table.groupby("example_id", as_index=False).agg(
        label=("label", "first"), probability=("probability", "mean"),
        domain=("domain", "first"), views=("view_id", "count"),
    )
    return grouped["label"].to_numpy(), grouped["probability"].to_numpy(), grouped


def train_baseline(config_path: str | Path, model_name: str, fold: int | None = None) -> dict[str, object]:
    config = load_train_config(config_path)
    fold = config.fold if fold is None else fold
    views = load_views(config.manifest, config.features_dir)
    train = [view for view in views if view.fold != fold]
    validation = [view for view in views if view.fold == fold]
    logger = configure_logging(config.logs_dir / f"baseline_{model_name}_fold{fold}.log")
    train_x = np.stack([aggregate_sequence(view.features) for view in train])
    train_y = np.array([view.label for view in train])
    validation_x = np.stack([aggregate_sequence(view.features) for view in validation])
    if model_name == "logistic":
        model = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.seed)),
        ])
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=config.seed, n_jobs=-1,
        )
    else:
        raise ValueError("model must be logistic or random_forest")
    # Each independent incident receives total training weight one regardless of view count.
    view_counts = pd.Series([view.example_id for view in train]).value_counts()
    weights = np.array([1.0 / view_counts[view.example_id] for view in train])
    if isinstance(model, Pipeline):
        model.fit(train_x, train_y, model__sample_weight=weights)
    else:
        model.fit(train_x, train_y, sample_weight=weights)
    probabilities = model.predict_proba(validation_x)[:, 1]
    labels, example_probabilities, predictions = _evaluate_views(validation, probabilities)
    metrics = binary_metrics(labels, example_probabilities)
    metrics.update({"model": model_name, "fold": fold, "feature_count": int(train_x.shape[1])})
    config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = config.checkpoints_dir / f"{model_name}_fold{fold}.joblib"
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES, "fold": fold}, checkpoint)
    predictions.to_csv(config.reports_dir / f"{model_name}_fold{fold}_predictions.csv", index=False)
    save_metrics(metrics, config.reports_dir / f"{model_name}_fold{fold}_metrics.json")
    logger.info("%s", json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an interpretable trajectory/pose baseline.")
    parser.add_argument("--config", default="configs/temporal_classifier.yaml")
    parser.add_argument("--model", choices=["logistic", "random_forest"], default="logistic")
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    train_baseline(args.config, args.model, args.fold)


if __name__ == "__main__":
    main()

