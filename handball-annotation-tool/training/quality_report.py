from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import project_path
from .features import FEATURE_NAMES, feature_path


def build_report(manifest_path: Path, features_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(manifest_path)
    index = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    rows: list[dict[str, object]] = []
    for _, item in manifest.iterrows():
        path = feature_path(features_dir, item)
        if not path.is_file():
            continue
        loaded = np.load(path)
        matrix = loaded["features"]
        metadata = json.loads(str(loaded["metadata"]))
        rows.append({
            "example_id": item["example_id"],
            "view_id": item["view_id"],
            "label": int(item["label"]),
            "domain": item["domain"],
            "ball_detection_rate": float(matrix[:, index["ball_valid"]].mean()),
            "player_detection_rate": float(matrix[:, index["player_valid"]].mean()),
            "pose_valid_rate": float(matrix[:, index["pose_valid_fraction"]].mean()),
            "complete_left_arm_rate": float(np.minimum.reduce([
                matrix[:, index["left_shoulder_valid"]],
                matrix[:, index["left_elbow_valid"]],
                matrix[:, index["left_wrist_valid"]],
            ]).mean()),
            "complete_right_arm_rate": float(np.minimum.reduce([
                matrix[:, index["right_shoulder_valid"]],
                matrix[:, index["right_elbow_valid"]],
                matrix[:, index["right_wrist_valid"]],
            ]).mean()),
            "mean_ball_confidence": float(matrix[:, index["ball_conf"]].mean()),
            "mean_player_confidence": float(matrix[:, index["player_conf"]].mean()),
            "mean_ball_width": float(matrix[:, index["ball_w"]].mean()),
            "mean_player_height": float(matrix[:, index["player_h"]].mean()),
            "selected_frames": json.dumps(metadata["selected_frame_indices"]),
        })
    if not rows:
        raise ValueError("No extracted feature files were found. Run training.features first.")
    per_view = pd.DataFrame(rows)
    metrics = [
        "ball_detection_rate", "player_detection_rate", "pose_valid_rate",
        "complete_left_arm_rate", "complete_right_arm_rate",
        "mean_ball_confidence", "mean_player_confidence",
        "mean_ball_width", "mean_player_height",
    ]
    summary = per_view.groupby(["domain", "label"])[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in summary.columns
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    per_view.to_csv(output_dir / "detection_quality_per_view.csv", index=False)
    summary.to_csv(output_dir / "detection_quality_summary.csv", index=False)

    plot_metrics = metrics[:5]
    means = per_view.groupby(["domain", "label"])[plot_metrics].mean()
    axis = means.plot(kind="bar", figsize=(12, 6), ylim=(0, 1), title="Detection quality by domain and label")
    axis.set_ylabel("Mean valid-frame rate")
    axis.set_xlabel("Domain, label")
    plt.tight_layout()
    plt.savefig(output_dir / "detection_quality.png", dpi=160)
    plt.close()
    return per_view, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Report YOLO/MediaPipe detection quality by class and domain.")
    parser.add_argument("--manifest", default="artifacts/manifests/dataset.csv")
    parser.add_argument("--features", default="artifacts/features")
    parser.add_argument("--output", default="artifacts/reports/quality")
    args = parser.parse_args()
    per_view, summary = build_report(
        project_path(args.manifest), project_path(args.features), project_path(args.output)
    )
    print(f"Reported {len(per_view)} extracted views.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
