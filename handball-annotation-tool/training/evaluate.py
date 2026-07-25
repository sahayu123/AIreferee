from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from handball_annotator.runtime import get_device

from .config import load_train_config
from .data import load_views
from .gru import TemporalGRU, _example_predictions, predict_views
from .metrics import binary_metrics, save_metrics


def evaluate_checkpoint(config_path: str | Path, checkpoint_path: Path, threshold: float = 0.5) -> dict[str, object]:
    config = load_train_config(config_path)
    device = get_device(config.device)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TemporalGRU(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    fold = int(checkpoint["fold"])
    feature_names = tuple(str(name) for name in checkpoint["feature_names"])
    if checkpoint["model_config"]["input_size"] != len(feature_names):
        raise ValueError("Checkpoint input size does not match its feature schema")
    views = [
        view for view in load_views(
            config.manifest, config.features_dir, target_feature_names=feature_names
        ) if view.fold == fold
    ]
    view_predictions = predict_views(
        model, views, checkpoint["mean"], checkpoint["std"], device, config.batch_size
    )
    examples = _example_predictions(view_predictions)
    metrics = binary_metrics(
        examples["label"].to_numpy(), examples["probability"].to_numpy(), threshold
    )
    metrics.update({"checkpoint": str(checkpoint_path), "fold": fold, "device": device})
    output = config.reports_dir / f"{checkpoint_path.stem}_evaluation.json"
    save_metrics(metrics, output)
    print(json.dumps(metrics, indent=2))
    return metrics


def summarize_folds(reports_dir: Path, pattern: str = "gru_fold*_metrics.json") -> dict[str, object]:
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(reports_dir.glob(pattern))]
    if not records:
        raise FileNotFoundError(f"No fold reports matching {pattern} in {reports_dir}")
    metrics = ["precision", "recall", "f1", "pr_auc", "roc_auc"]
    summary = {
        name: {"mean": float(np.mean([record[name] for record in records if record[name] is not None])),
               "std": float(np.std([record[name] for record in records if record[name] is not None]))}
        for name in metrics
    }
    result = {"folds": len(records), "metrics": summary}
    save_metrics(result, reports_dir / "cross_validation_summary.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a GRU checkpoint or summarize cross-validation.")
    parser.add_argument("--config", default="configs/temporal_classifier.yaml")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    config = load_train_config(args.config)
    if args.summarize:
        print(json.dumps(summarize_folds(config.reports_dir), indent=2))
    elif args.checkpoint:
        evaluate_checkpoint(args.config, args.checkpoint, args.threshold)
    else:
        parser.error("Provide --checkpoint or --summarize")


if __name__ == "__main__":
    main()
