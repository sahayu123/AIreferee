from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import PROJECT_ROOT, project_path


def frame_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        raise ValueError(f"Frame filename does not end in a number: {path.name}")
    return int(match.group(1))


def sorted_frames(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.jpg"), key=frame_number)


def _manifest_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _native_rows(dataset: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for class_name, label in (("handball", 1), ("not_handball", 0)):
        for candidate in sorted((dataset / class_name).iterdir()):
            if not candidate.is_dir():
                continue
            metadata_path, frames_dir = candidate / "metadata.json", candidate / "frames"
            if not metadata_path.is_file() or not frames_dir.is_dir():
                raise ValueError(f"Incomplete labeled candidate: {candidate}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            frames = sorted_frames(frames_dir)
            if not frames:
                raise ValueError(f"No frames found in {frames_dir}")
            source_name = str(metadata.get("source_name") or candidate.name)
            rows.append({
                "example_id": candidate.name,
                "view_id": "primary",
                "label": label,
                "domain": "native",
                "source_name": source_name,
                "source_group": f"native::{source_name}",
                "action_id": "",
                "frames_dir": _manifest_path(frames_dir),
                "frame_count": len(frames),
                "fps": metadata.get("fps", ""),
                "center_frame": metadata.get("center_frame", ""),
                "auxiliary_label": "",
            })
    return rows


def _imported_rows(imported: Path, allowed_label: str) -> list[dict[str, object]]:
    csv_path = imported / "test" / "labels.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Imported labels file not found: {csv_path}")
    labels = pd.read_csv(csv_path, dtype={"label": str, "action_id": str, "clip": str})
    labels = labels[labels["label"] == str(allowed_label)]
    rows: list[dict[str, object]] = []
    for (action_id, clip), group in labels.groupby(["action_id", "clip"], sort=False):
        frames_dir = imported / "test" / f"action_{action_id}" / str(clip)
        frames = sorted_frames(frames_dir)
        if len(frames) != len(group):
            raise ValueError(
                f"CSV/files disagree for action {action_id} {clip}: "
                f"{len(group)} CSV rows versus {len(frames)} images"
            )
        example_id = f"imported_action_{int(action_id):04d}"
        rows.append({
            "example_id": example_id,
            "view_id": str(clip),
            "label": 0,
            "domain": "imported",
            "source_name": "processed_frames_no_handball",
            "source_group": f"imported::action_{int(action_id):04d}",
            "action_id": int(action_id),
            "frames_dir": _manifest_path(frames_dir),
            "frame_count": len(frames),
            "fps": "",
            "center_frame": "",
            "auxiliary_label": str(allowed_label),
        })
    return rows


def assign_folds(manifest: pd.DataFrame, folds: int, seed: int) -> pd.DataFrame:
    examples = manifest.drop_duplicates("example_id").reset_index(drop=True)
    if examples["label"].value_counts().min() < folds:
        raise ValueError(f"Each class needs at least {folds} independent examples")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_by_example: dict[str, int] = {}
    for fold, (_, validation) in enumerate(
        splitter.split(examples, examples["label"], groups=examples["source_group"])
    ):
        for example_id in examples.iloc[validation]["example_id"]:
            fold_by_example[str(example_id)] = fold
    result = manifest.copy()
    result["fold"] = result["example_id"].map(fold_by_example)
    if result["fold"].isna().any():
        raise RuntimeError("Some examples were not assigned to a fold")
    result["fold"] = result["fold"].astype(int)
    return result


def build_manifest(
    dataset: Path,
    output: Path,
    folds: int = 5,
    seed: int = 42,
    imported: Path | None = None,
    imported_label: str = "1",
) -> pd.DataFrame:
    rows = _native_rows(dataset)
    if imported is not None:
        rows += _imported_rows(imported, imported_label)
    manifest = assign_folds(pd.DataFrame(rows), folds, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a grouped handball training manifest.")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument(
        "--include-imported",
        action="store_true",
        help="Include processed_frames_no_handball data. Disabled by default.",
    )
    parser.add_argument("--imported", default="dataset/processed_frames_no_handball")
    parser.add_argument("--output", default="artifacts/manifests/dataset.csv")
    parser.add_argument("--imported-label", default="1", help="Only this imported CSV label is used as negative.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build_manifest(
        project_path(args.dataset),
        project_path(args.output),
        args.folds,
        args.seed,
        project_path(args.imported) if args.include_imported else None,
        args.imported_label,
    )
    unique = manifest.drop_duplicates("example_id")
    print(f"Wrote {len(manifest)} views representing {len(unique)} independent examples to {project_path(args.output)}")
    print(unique.groupby(["domain", "label"]).size().to_string())


if __name__ == "__main__":
    main()
