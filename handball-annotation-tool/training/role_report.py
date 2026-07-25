from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import project_path
from .data import feature_metadata
from .features import feature_path


def build_report(manifest_path: Path, features_dir: Path, output: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    rows: list[dict[str, object]] = []
    for _, item in manifest.iterrows():
        path = feature_path(features_dir, item)
        if not path.is_file():
            continue
        role = feature_metadata(path).get("player_role")
        if not isinstance(role, dict):
            continue
        scores = role.get("aggregate_scores", {})
        rows.append({
            "example_id": str(item["example_id"]),
            "label": int(item["label"]),
            "goalkeeper_score": float(scores.get("goalkeeper", 0.0)),
            "outfield_score": float(scores.get("outfield", 0.0)),
            "referee_score": float(scores.get("referee", 0.0)),
            "goalkeeper_margin": float(role.get("goalkeeper_margin", 0.0)),
            "uncertain": bool(role.get("uncertain", True)),
            "valid_crop_count": int(role.get("valid_crop_count", 0)),
            "feature_path": str(path),
        })
    report = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize zero-shot goalkeeper scores.")
    parser.add_argument("--manifest", default="artifacts/manifests/dataset.csv")
    parser.add_argument("--features", default="artifacts/features_goalkeeper")
    parser.add_argument("--output", default="artifacts/reports_goalkeeper/player_roles.csv")
    args = parser.parse_args()
    report = build_report(
        project_path(args.manifest), project_path(args.features), project_path(args.output)
    )
    summary = {
        "examples": len(report),
        "mean_goalkeeper_score": float(report["goalkeeper_score"].mean()) if len(report) else None,
        "uncertain": int(report["uncertain"].sum()) if len(report) else 0,
        "high_confidence_goalkeepers": int(
            ((report["goalkeeper_score"] >= 0.70) & (report["goalkeeper_margin"] >= 0.20)).sum()
        ) if len(report) else 0,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
