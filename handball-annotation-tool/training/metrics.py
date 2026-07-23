from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, object]:
    labels = labels.astype(int)
    predictions = (probabilities >= threshold).astype(int)
    result: dict[str, object] = {
        "examples": int(len(labels)),
        "positives": int(labels.sum()),
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }
    result["roc_auc"] = float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None
    return result


def save_metrics(metrics: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

