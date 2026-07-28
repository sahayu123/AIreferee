"""Streamlit UI for auditing combined handball/goalkeeper clip decisions."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

from training.features import FEATURE_NAMES, feature_path
from training.manifest import sorted_frames

REQUIRED_PREDICTION_COLUMNS = {
    "example_id",
    "view_id",
    "label",
    "domain",
    "frames_dir",
    "frame_count",
    "fps",
    "fold",
    "raw_handball_probability",
    "raw_prediction",
    "final_prediction",
    "raw_predicted_label",
    "final_predicted_label",
    "raw_correct",
    "final_correct",
    "goalkeeper_status",
    "goalkeeper_evidence_score",
    "goalkeeper_reason",
    "goalkeeper_analysis_invoked",
    "role_cache_path",
    "combined_event_label",
    "goalkeeper_veto",
    "actor_track_id",
    "association_score",
    "final_decision_rule",
}
AUDIT_COLUMNS = (
    "example_id",
    "view_id",
    "reviewed",
    "actual_actor_role",
    "actor_track_correct",
    "root_cause",
    "notes",
    "updated_at_utc",
)
CATEGORY_OPTIONS = (
    "Combined mistakes",
    "Final false positives",
    "Final false negatives",
    "Base-model misses",
    "Surviving false alarms",
    "Veto introduced errors",
    "Veto fixed errors",
    "All goalkeeper vetoes",
    "Unchanged combined mistakes",
    "Raw false positives",
    "Raw false negatives",
    "All goalkeeper checks",
    "Unknown goalkeeper checks",
    "All clips",
)
ROOT_CAUSES = (
    "",
    "Handball model error",
    "Wrong actor track",
    "Goalkeeper classifier error",
    "Insufficient visual evidence",
    "Dataset label questionable",
    "Correct goalkeeper veto",
    "Other",
)
ACTOR_ROLES = ("", "Goalkeeper", "Outfield", "Not visible", "Uncertain")
TRACK_CHOICES = ("", "Correct", "Wrong", "Uncertain")


def bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    rendered = str(value).strip().lower()
    if rendered in ("true", "1", "yes"):
        return True
    if rendered in ("false", "0", "no", ""):
        return False
    raise ValueError(f"Cannot interpret boolean value: {value!r}")


def validate_predictions(frame: pd.DataFrame) -> None:
    missing = REQUIRED_PREDICTION_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            "Combined predictions are missing columns: "
            + ", ".join(sorted(missing))
        )
    keys = frame[["example_id", "view_id"]].astype(str)
    if keys.duplicated().any():
        raise ValueError("Combined predictions contain duplicate clip/view keys")
    if not frame["label"].isin((0, 1)).all():
        raise ValueError("Clip labels must be binary")


def prepare_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    validate_predictions(frame)
    prepared = frame.copy()
    for column in (
        "raw_correct",
        "final_correct",
        "goalkeeper_veto",
        "goalkeeper_analysis_invoked",
    ):
        if column in prepared:
            prepared[column] = prepared[column].map(bool_value)
    raw_correct = prepared["raw_correct"]
    final_correct = prepared["final_correct"]
    prepared["review_category"] = "correct_unchanged"
    prepared.loc[
        raw_correct & ~final_correct, "review_category"
    ] = "veto_introduced_error"
    prepared.loc[
        ~raw_correct & final_correct, "review_category"
    ] = "veto_fixed_error"
    prepared.loc[
        ~raw_correct & ~final_correct, "review_category"
    ] = "unchanged_combined_error"
    prepared["audit_key"] = [
        audit_key(example_id, view_id)
        for example_id, view_id in zip(
            prepared["example_id"], prepared["view_id"]
        )
    ]
    return prepared


def audit_key(example_id: object, view_id: object) -> str:
    return hashlib.sha256(
        f"{example_id}::{view_id}".encode("utf-8")
    ).hexdigest()[:20]


def load_audit(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=AUDIT_COLUMNS)
    audit = pd.read_csv(path, keep_default_na=False)
    missing = set(AUDIT_COLUMNS) - set(audit.columns)
    if missing:
        raise ValueError(
            "Audit file is missing columns: " + ", ".join(sorted(missing))
        )
    if audit[["example_id", "view_id"]].astype(str).duplicated().any():
        raise ValueError("Audit file contains duplicate clip/view keys")
    audit["reviewed"] = audit["reviewed"].map(bool_value)
    return audit[list(AUDIT_COLUMNS)].copy()


def save_audit_record(
    audit: pd.DataFrame,
    path: Path,
    record: Mapping[str, object],
) -> pd.DataFrame:
    missing = set(AUDIT_COLUMNS) - set(record)
    if missing:
        raise ValueError(
            "Audit record is missing fields: " + ", ".join(sorted(missing))
        )
    updated = audit.copy()
    match = (
        updated["example_id"].astype(str)
        == str(record["example_id"])
    ) & (
        updated["view_id"].astype(str) == str(record["view_id"])
    )
    serialized = {column: record[column] for column in AUDIT_COLUMNS}
    if match.any():
        row_index = updated.index[match][0]
        for column, value in serialized.items():
            updated.at[row_index, column] = value
    else:
        updated = pd.concat(
            [updated, pd.DataFrame([serialized])], ignore_index=True
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = path.with_suffix(path.suffix + ".before_current_session")
    if path.is_file() and not backup.exists():
        shutil.copy2(path, backup)
    temporary = path.with_suffix(path.suffix + ".temporary")
    updated.to_csv(temporary, index=False)
    temporary.replace(path)
    return updated


def audit_lookup(audit: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["example_id"]), str(row["view_id"])): row.to_dict()
        for _, row in audit.iterrows()
    }


def filter_predictions(
    frame: pd.DataFrame,
    *,
    category: str,
    audit_status: str = "Unaudited",
    truth: str = "Both",
    goalkeeper_status: str = "Any",
    search: str = "",
    audit: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if category not in CATEGORY_OPTIONS:
        raise ValueError(f"Unsupported review category: {category}")
    selected = frame.copy()
    if category == "Combined mistakes":
        selected = selected[~selected["final_correct"]]
    elif category == "Final false positives":
        selected = selected[
            (selected["label"] == 0) & (selected["final_prediction"] == 1)
        ]
    elif category == "Final false negatives":
        selected = selected[
            (selected["label"] == 1) & (selected["final_prediction"] == 0)
        ]
    elif category == "Base-model misses":
        selected = selected[
            (selected["label"] == 1) & (selected["raw_prediction"] == 0)
        ]
    elif category == "Surviving false alarms":
        selected = selected[
            (selected["label"] == 0) & (selected["final_prediction"] == 1)
        ]
    elif category == "Veto introduced errors":
        selected = selected[
            selected["review_category"] == "veto_introduced_error"
        ]
    elif category == "Veto fixed errors":
        selected = selected[
            selected["review_category"] == "veto_fixed_error"
        ]
    elif category == "All goalkeeper vetoes":
        selected = selected[selected["goalkeeper_veto"]]
    elif category == "Unchanged combined mistakes":
        selected = selected[
            selected["review_category"] == "unchanged_combined_error"
        ]
    elif category == "Raw false positives":
        selected = selected[
            (selected["label"] == 0) & (selected["raw_prediction"] == 1)
        ]
    elif category == "Raw false negatives":
        selected = selected[
            (selected["label"] == 1) & (selected["raw_prediction"] == 0)
        ]
    elif category == "All goalkeeper checks":
        selected = selected[
            selected["goalkeeper_analysis_invoked"].astype(bool)
        ]
    elif category == "Unknown goalkeeper checks":
        selected = selected[selected["goalkeeper_status"] == "unknown"]

    if truth == "Handball":
        selected = selected[selected["label"] == 1]
    elif truth == "No handball":
        selected = selected[selected["label"] == 0]
    if goalkeeper_status != "Any":
        selected = selected[
            selected["goalkeeper_status"] == goalkeeper_status
        ]
    query = search.strip().casefold()
    if query:
        source_names = (
            selected["source_name"].astype(str)
            if "source_name" in selected
            else pd.Series("", index=selected.index, dtype=str)
        )
        searchable = (
            selected["example_id"].astype(str)
            + " "
            + source_names
        ).str.casefold()
        selected = selected[searchable.str.contains(query, regex=False)]

    reviewed_keys: set[tuple[str, str]] = set()
    if audit is not None and not audit.empty:
        reviewed_keys = {
            (str(row["example_id"]), str(row["view_id"]))
            for _, row in audit[audit["reviewed"].map(bool_value)].iterrows()
        }
    reviewed_mask = pd.Series(
        [
            (str(row["example_id"]), str(row["view_id"])) in reviewed_keys
            for _, row in selected.iterrows()
        ],
        index=selected.index,
    )
    if audit_status == "Unaudited":
        selected = selected[~reviewed_mask]
    elif audit_status == "Audited":
        selected = selected[reviewed_mask]
    return selected.reset_index(drop=True)


def resolve_artifact_path(value: object, project_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (project_root / path).resolve()


def load_role_payload(path_value: object, project_root: Path) -> dict[str, Any]:
    rendered = str(path_value).strip()
    if not rendered:
        return {}
    path = resolve_artifact_path(rendered, project_root)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def observation_map(payload: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    observations: dict[int, dict[str, Any]] = {}
    for item in payload.get("actor_observations", []):
        if isinstance(item, dict) and "frame_index" in item:
            observations[int(item["frame_index"])] = item
    return observations


def crop_map(payload: Mapping[str, Any]) -> dict[int, list[dict[str, Any]]]:
    crops: dict[int, list[dict[str, Any]]] = {}
    for item in payload.get("crop_metadata", []):
        if isinstance(item, dict) and "frame_index" in item:
            crops.setdefault(int(item["frame_index"]), []).append(item)
    return crops


def parse_box(value: object) -> tuple[int, int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("Bounding box must be a four-item sequence")
    parts = [int(round(float(item))) for item in value]
    if len(parts) != 4:
        raise ValueError("Bounding box must contain four values")
    return parts[0], parts[1], parts[2], parts[3]


def normalized_feature_box(
    feature_row: np.ndarray,
    feature_indices: Mapping[str, int],
    *,
    prefix: str,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    if float(feature_row[feature_indices[f"{prefix}_valid"]]) <= 0:
        return None
    center_x = float(feature_row[feature_indices[f"{prefix}_x"]]) * frame_width
    center_y = float(feature_row[feature_indices[f"{prefix}_y"]]) * frame_height
    width = float(feature_row[feature_indices[f"{prefix}_w"]]) * frame_width
    height = float(feature_row[feature_indices[f"{prefix}_h"]]) * frame_height
    if width <= 0 or height <= 0:
        return None
    return (
        max(0, round(center_x - width / 2)),
        max(0, round(center_y - height / 2)),
        min(frame_width, round(center_x + width / 2)),
        min(frame_height, round(center_y + height / 2)),
    )


@st.cache_data(show_spinner=False, max_entries=96)
def load_feature_evidence(
    features_dir: str,
    domain: str,
    example_id: str,
    view_id: str,
    frame_path_values: tuple[str, ...],
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], str]:
    row = pd.Series(
        {
            "domain": domain,
            "example_id": example_id,
            "view_id": view_id,
        }
    )
    artifact = feature_path(Path(features_dir), row)
    if not artifact.is_file():
        return {}, {}, str(artifact)
    try:
        with np.load(artifact, allow_pickle=False) as loaded:
            features = loaded["features"].astype(np.float32)
            metadata = json.loads(str(loaded["metadata"]))
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return {}, {}, str(artifact)
    names = [str(name) for name in metadata.get("feature_names", [])]
    selected = [
        int(index)
        for index in metadata.get("selected_frame_indices", [])
    ]
    if (
        names != FEATURE_NAMES
        or features.ndim != 2
        or len(features) != len(selected)
    ):
        return {}, {}, str(artifact)
    indices = {name: index for index, name in enumerate(names)}
    players: dict[int, dict[str, Any]] = {}
    balls: dict[int, dict[str, Any]] = {}
    for feature_row, frame_index in zip(features, selected):
        if not 0 <= frame_index < len(frame_path_values):
            continue
        try:
            with Image.open(frame_path_values[frame_index]) as opened:
                frame_width, frame_height = opened.size
        except OSError:
            continue
        player_box = normalized_feature_box(
            feature_row,
            indices,
            prefix="player",
            frame_width=frame_width,
            frame_height=frame_height,
        )
        if player_box is not None:
            players[frame_index] = {
                "bbox": list(player_box),
                "overlay_label": "feature-selected player",
            }
        ball_box = normalized_feature_box(
            feature_row,
            indices,
            prefix="ball",
            frame_width=frame_width,
            frame_height=frame_height,
        )
        if ball_box is not None:
            balls[frame_index] = {
                "bbox": list(ball_box),
                "overlay_label": "detected ball",
            }
    return players, balls, str(artifact)


def render_frame(
    frame_path: Path,
    *,
    observation: Mapping[str, Any] | None = None,
    detected_ball: Mapping[str, Any] | None = None,
    sampled_crops: Sequence[Mapping[str, Any]] = (),
    maximum_width: int = 960,
) -> Image.Image:
    with Image.open(frame_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(3, round(min(image.size) / 250))
    if observation and observation.get("bbox") is not None:
        box = parse_box(observation["bbox"])
        actor_label = str(
            observation.get("overlay_label", "associated actor")
        )
        draw.rectangle(box, outline="#21c55d", width=line_width)
        draw.text(
            (box[0] + 4, max(0, box[1] - 14)),
            actor_label,
            fill="#21c55d",
            stroke_width=2,
            stroke_fill="black",
        )
    if detected_ball and detected_ball.get("bbox") is not None:
        box = parse_box(detected_ball["bbox"])
        draw.rectangle(box, outline="#22d3ee", width=line_width)
        draw.text(
            (box[0] + 4, max(0, box[1] - 14)),
            str(detected_ball.get("overlay_label", "detected ball")),
            fill="#22d3ee",
            stroke_width=2,
            stroke_fill="black",
        )
    for crop in sampled_crops:
        if crop.get("bbox") is None:
            continue
        box = parse_box(crop["bbox"])
        probability = crop.get("goalkeeper_probability")
        caption = (
            f"sample p(GK)={float(probability):.2f}"
            if probability is not None
            else "sampled crop"
        )
        draw.rectangle(box, outline="#f4b942", width=line_width)
        draw.text(
            (box[0] + 4, box[1] + 4),
            caption,
            fill="#f4b942",
            stroke_width=2,
            stroke_fill="black",
        )
    if image.width > maximum_width:
        scale = maximum_width / image.width
        image = image.resize(
            (maximum_width, max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return image


@st.cache_data(show_spinner=False, max_entries=48)
def animated_preview(
    frame_paths: tuple[str, ...],
    observations_json: str,
    balls_json: str,
    crops_json: str,
    maximum_width: int = 640,
) -> bytes:
    observations = {
        int(key): value for key, value in json.loads(observations_json).items()
    }
    crops = {
        int(key): value for key, value in json.loads(crops_json).items()
    }
    balls = {
        int(key): value for key, value in json.loads(balls_json).items()
    }
    images = [
        render_frame(
            Path(path),
            observation=observations.get(index),
            detected_ball=balls.get(index),
            sampled_crops=crops.get(index, []),
            maximum_width=maximum_width,
        ).quantize(colors=128)
        for index, path in enumerate(frame_paths)
    ]
    if not images:
        raise ValueError("Cannot animate an empty clip")
    stream = io.BytesIO()
    images[0].save(
        stream,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0,
        disposal=2,
    )
    return stream.getvalue()


def actor_crops(
    frame_paths: Sequence[Path],
    payload: Mapping[str, Any],
) -> list[tuple[Image.Image, dict[str, Any]]]:
    results: list[tuple[Image.Image, dict[str, Any]]] = []
    for item in payload.get("crop_metadata", []):
        if not isinstance(item, dict):
            continue
        frame_index = int(item.get("frame_index", -1))
        if not 0 <= frame_index < len(frame_paths):
            continue
        try:
            box = parse_box(item.get("bbox"))
            with Image.open(frame_paths[frame_index]) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            left = max(0, min(box[0], image.width))
            top = max(0, min(box[1], image.height))
            right = max(left + 1, min(box[2], image.width))
            bottom = max(top + 1, min(box[3], image.height))
            results.append((image.crop((left, top, right, bottom)), item))
        except (OSError, TypeError, ValueError):
            continue
    return results


def category_explanation(row: Mapping[str, Any]) -> str:
    category = str(row["review_category"])
    if category == "veto_introduced_error":
        return (
            "The raw handball prediction was correct, but the goalkeeper "
            "veto changed it into an incorrect no-handball result."
        )
    if category == "veto_fixed_error":
        return (
            "The raw model produced a false handball, and the goalkeeper "
            "veto corrected it."
        )
    if bool_value(row["final_correct"]):
        return "The final combined decision is correct."
    if int(row["label"]) == 1 and int(row["raw_prediction"]) == 0:
        return (
            "The handball model missed the event, so goalkeeper analysis "
            "was correctly skipped."
        )
    return (
        "The raw handball error was not changed by goalkeeper analysis."
    )


def queue_navigation(target: str | None) -> None:
    if target is not None:
        st.session_state.combined_pending_key = target


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "artifacts/reports/"
            "combined_supervised_gated_predictions.csv"
        ),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path(
            "artifacts/reports/"
            "combined_supervised_gated_audit.csv"
        ),
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("artifacts/features"),
    )
    args, _ = parser.parse_known_args()
    project_root = Path.cwd().resolve()
    predictions_path = resolve_artifact_path(args.predictions, project_root)
    audit_path = resolve_artifact_path(args.audit, project_root)
    features_dir = resolve_artifact_path(args.features_dir, project_root)

    st.set_page_config(
        page_title="Combined Pipeline Mistake Review",
        page_icon="🔎",
        layout="wide",
    )
    st.title("Combined handball + goalkeeper mistake review")
    st.caption(
        "Green boxes show the goalkeeper-associated actor when that stage "
        "ran, otherwise the player selected by base feature extraction. "
        "Cyan is the detected ball; amber is the exact goalkeeper crop."
    )
    if not predictions_path.is_file():
        st.error(f"Predictions file not found: {predictions_path}")
        st.stop()
    try:
        predictions = prepare_predictions(
            pd.read_csv(predictions_path, keep_default_na=False)
        )
        audit = load_audit(audit_path)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.sidebar.header("Review queue")
    category = st.sidebar.selectbox("Category", CATEGORY_OPTIONS)
    audit_status = st.sidebar.selectbox(
        "Audit status", ("Unaudited", "Audited", "All")
    )
    truth = st.sidebar.selectbox(
        "Ground truth", ("Both", "Handball", "No handball")
    )
    statuses = ["Any"] + sorted(
        predictions["goalkeeper_status"].astype(str).unique().tolist()
    )
    goalkeeper_status = st.sidebar.selectbox(
        "Goalkeeper result", statuses
    )
    search = st.sidebar.text_input("Search clip", placeholder="Optional")
    queue = filter_predictions(
        predictions,
        category=category,
        audit_status=audit_status,
        truth=truth,
        goalkeeper_status=goalkeeper_status,
        search=search,
        audit=audit,
    )

    total_errors = int((~predictions["final_correct"]).sum())
    introduced = int(
        (
            predictions["review_category"] == "veto_introduced_error"
        ).sum()
    )
    fixed = int(
        (predictions["review_category"] == "veto_fixed_error").sum()
    )
    audited = (
        int(audit["reviewed"].map(bool_value).sum())
        if not audit.empty
        else 0
    )
    st.sidebar.metric("Combined mistakes", total_errors)
    st.sidebar.metric("Veto introduced / fixed", f"{introduced} / {fixed}")
    st.sidebar.metric("Audited clips", audited)
    st.sidebar.caption(
        f"Notes save separately to {audit_path.name}; predictions are never "
        "modified."
    )

    if queue.empty:
        st.success("No clips match the current filters.")
        st.stop()
    queue_keys = queue["audit_key"].astype(str).tolist()
    pending = st.session_state.pop("combined_pending_key", None)
    if pending in queue_keys:
        st.session_state.combined_clip_picker = pending
    if st.session_state.get("combined_clip_picker") not in queue_keys:
        st.session_state.combined_clip_picker = queue_keys[0]
    labels = {
        str(row["audit_key"]): (
            f"{str(row['review_category']).replace('_', ' ')} — "
            f"{str(row['example_id'])[:80]}"
        )
        for _, row in queue.iterrows()
    }
    selected_key = st.sidebar.selectbox(
        "Jump to clip",
        queue_keys,
        key="combined_clip_picker",
        format_func=lambda value: labels[value],
    )
    position = queue_keys.index(selected_key)
    row = queue[queue["audit_key"] == selected_key].iloc[0]
    previous_key = queue_keys[position - 1] if position > 0 else None
    next_key = (
        queue_keys[position + 1]
        if position + 1 < len(queue_keys)
        else None
    )

    previous, counter, next_column = st.columns([1, 2, 1])
    previous.button(
        "← Previous",
        disabled=previous_key is None,
        width="stretch",
        on_click=queue_navigation,
        args=(previous_key,),
    )
    counter.markdown(
        f"<div style='text-align:center'>Clip <strong>{position + 1}</strong> "
        f"of <strong>{len(queue)}</strong> in this queue</div>",
        unsafe_allow_html=True,
    )
    next_column.button(
        "Next →",
        disabled=next_key is None,
        width="stretch",
        on_click=queue_navigation,
        args=(next_key,),
    )

    truth_label = "handball" if int(row["label"]) else "not handball"
    raw_label = str(row["raw_predicted_label"])
    final_label = str(row["final_predicted_label"])
    metric_columns = st.columns(5)
    metric_columns[0].metric("Ground truth", truth_label)
    metric_columns[1].metric(
        "Raw prediction",
        raw_label,
        f"p={float(row['raw_handball_probability']):.3f}",
    )
    metric_columns[2].metric(
        "Goalkeeper stage", str(row["goalkeeper_status"])
    )
    metric_columns[3].metric("Final prediction", final_label)
    metric_columns[4].metric(
        "Final result",
        "Correct" if bool_value(row["final_correct"]) else "WRONG",
    )
    if bool_value(row["final_correct"]):
        st.success(category_explanation(row))
    else:
        st.error(category_explanation(row))
    st.write(
        {
            "clip": row["example_id"],
            "source": row.get("source_name") or None,
            "fold": int(row["fold"]),
            "combined_event": row["combined_event_label"],
            "goalkeeper_probability": (
                float(row["goalkeeper_evidence_score"])
                if str(row["goalkeeper_evidence_score"]).strip()
                else None
            ),
            "goalkeeper_reason": row["goalkeeper_reason"] or None,
            "actor_track_id": row["actor_track_id"] or None,
            "association_score": row["association_score"] or None,
            "final_rule": row["final_decision_rule"],
        }
    )

    frames_dir = resolve_artifact_path(row["frames_dir"], project_root)
    frame_paths = sorted_frames(frames_dir) if frames_dir.is_dir() else []
    payload = load_role_payload(row["role_cache_path"], project_root)
    observations = observation_map(payload)
    sampled = crop_map(payload)
    feature_players: dict[int, dict[str, Any]] = {}
    feature_balls: dict[int, dict[str, Any]] = {}
    feature_artifact = ""
    if frame_paths:
        feature_players, feature_balls, feature_artifact = (
            load_feature_evidence(
                str(features_dir),
                str(row["domain"]),
                str(row["example_id"]),
                str(row["view_id"]),
                tuple(str(path) for path in frame_paths),
            )
        )
    display_observations = dict(feature_players)
    display_observations.update(observations)
    if not frame_paths:
        st.error(f"Clip frames are unavailable: {frames_dir}")
    else:
        animation_column, frame_column = st.columns(2)
        with animation_column:
            st.subheader("Animated clip")
            if st.checkbox(
                "Render animated preview",
                value=True,
                key=f"animation_{selected_key}",
            ):
                with st.spinner("Rendering actor overlay…"):
                    try:
                        gif = animated_preview(
                            tuple(str(path) for path in frame_paths),
                            json.dumps(display_observations, sort_keys=True),
                            json.dumps(feature_balls, sort_keys=True),
                            json.dumps(sampled, sort_keys=True),
                        )
                        st.image(gif, width="stretch")
                    except (OSError, TypeError, ValueError) as exc:
                        st.error(f"Could not render animation: {exc}")
        with frame_column:
            st.subheader("Exact frame inspector")
            default_frame = min(len(frame_paths) - 1, len(frame_paths) // 2)
            frame_index = st.slider(
                "Frame",
                min_value=0,
                max_value=len(frame_paths) - 1,
                value=default_frame,
                key=f"frame_{selected_key}",
            )
            try:
                st.image(
                    render_frame(
                        frame_paths[frame_index],
                        observation=display_observations.get(frame_index),
                        detected_ball=feature_balls.get(frame_index),
                        sampled_crops=sampled.get(frame_index, []),
                    ),
                    width="stretch",
                    caption=(
                        f"{frame_paths[frame_index].name} "
                        f"({frame_index + 1}/{len(frame_paths)})"
                    ),
                )
            except OSError as exc:
                st.error(f"Could not open frame: {exc}")

    crops = actor_crops(frame_paths, payload)
    st.subheader("Goalkeeper-classifier crops")
    if not bool_value(row.get("goalkeeper_analysis_invoked", False)):
        st.info(
            "The raw handball score was below threshold, so the goalkeeper "
            "stage correctly did not run."
        )
    elif not crops:
        st.warning(
            "No crops reached the classifier. "
            f"Reason: {row['goalkeeper_reason'] or 'unknown'}"
        )
    else:
        crop_columns = st.columns(min(4, len(crops)))
        for number, (image, metadata) in enumerate(crops, start=1):
            probability = metadata.get("goalkeeper_probability")
            with crop_columns[(number - 1) % len(crop_columns)]:
                st.image(image, width="stretch")
                st.caption(
                    f"Frame {int(metadata['frame_index'])}; "
                    f"p(goalkeeper)={float(probability):.3f}"
                )

    association = payload.get("association", {})
    with st.expander("Actor-association and cache details"):
        st.json(
            {
                "association": association,
                "tracked_people": payload.get("tracked_people"),
                "valid_crops": payload.get("valid_crops"),
                "frame_probabilities": payload.get(
                    "frame_probabilities", []
                ),
                "cache_path": row["role_cache_path"] or None,
                "feature_artifact": feature_artifact or None,
                "frames_dir": str(frames_dir),
            }
        )

    lookup = audit_lookup(audit)
    existing = lookup.get(
        (str(row["example_id"]), str(row["view_id"])), {}
    )
    st.subheader("Your audit")
    st.caption(
        "These notes are independent ground truth for diagnosing whether the "
        "handball model, actor association, or goalkeeper classifier failed."
    )
    widget_suffix = selected_key
    with st.form(f"audit_form_{widget_suffix}"):
        audit_columns = st.columns(3)
        actual_actor_role = audit_columns[0].selectbox(
            "Actual actor role",
            ACTOR_ROLES,
            index=ACTOR_ROLES.index(
                str(existing.get("actual_actor_role", ""))
                if str(existing.get("actual_actor_role", "")) in ACTOR_ROLES
                else ""
            ),
        )
        actor_track_correct = audit_columns[1].selectbox(
            "Is the green actor track correct?",
            TRACK_CHOICES,
            index=TRACK_CHOICES.index(
                str(existing.get("actor_track_correct", ""))
                if str(existing.get("actor_track_correct", ""))
                in TRACK_CHOICES
                else ""
            ),
        )
        root_cause = audit_columns[2].selectbox(
            "Primary root cause",
            ROOT_CAUSES,
            index=ROOT_CAUSES.index(
                str(existing.get("root_cause", ""))
                if str(existing.get("root_cause", "")) in ROOT_CAUSES
                else ""
            ),
        )
        notes = st.text_area(
            "Notes",
            value=str(existing.get("notes", "")),
            placeholder="What do you see in the clip?",
        )
        save, save_next = st.columns(2)
        save_clicked = save.form_submit_button(
            "Save audit", type="primary", width="stretch"
        )
        save_next_clicked = save_next.form_submit_button(
            "Save and next", type="primary", width="stretch"
        )
    if save_clicked or save_next_clicked:
        record = {
            "example_id": str(row["example_id"]),
            "view_id": str(row["view_id"]),
            "reviewed": True,
            "actual_actor_role": actual_actor_role,
            "actor_track_correct": actor_track_correct,
            "root_cause": root_cause,
            "notes": notes,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        save_audit_record(audit, audit_path, record)
        if save_next_clicked:
            queue_navigation(next_key)
            st.rerun()
        st.success("Audit saved.")


if __name__ == "__main__":
    main()
