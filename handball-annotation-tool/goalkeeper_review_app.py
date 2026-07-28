"""Source-centric Streamlit UI for labeling YOLO person crops."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Mapping

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

REQUIRED_COLUMNS = {
    "source_image",
    "source_label",
    "source_group",
    "crop_path",
    "crop_id",
    "detection_index",
    "detection_count",
    "bbox",
    "status",
    "review_label",
}
REVIEW_LABELS = ("", "goalkeeper", "not_goalkeeper", "uncertain")
LABEL_CAPTIONS = {
    "": "Unreviewed",
    "goalkeeper": "Goalkeeper",
    "not_goalkeeper": "Outfield",
    "uncertain": "Reject / uncertain",
}
BOX_COLORS = {
    "": "#ff4b4b",
    "goalkeeper": "#21c55d",
    "not_goalkeeper": "#3291ff",
    "uncertain": "#f4b942",
}


def validate_manifest(frame: pd.DataFrame) -> None:
    """Raise a useful error before the UI can overwrite a malformed file."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            "Candidate manifest is missing columns: "
            + ", ".join(sorted(missing))
        )
    invalid = set(frame["review_label"].astype(str).str.strip()) - set(
        REVIEW_LABELS
    )
    if invalid:
        raise ValueError(
            "Candidate manifest contains unsupported review labels: "
            + ", ".join(sorted(invalid))
        )


def save_manifest(frame: pd.DataFrame, path: Path) -> None:
    """Create a first-save backup, then atomically replace the CSV."""
    backup = path.with_suffix(path.suffix + ".before_manual_review")
    if not backup.exists():
        shutil.copy2(path, backup)
    temporary = path.with_suffix(path.suffix + ".temporary")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def reviewable_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only rows backed by a generated person crop."""
    return frame[frame["crop_path"].astype(str).str.strip() != ""].copy()


def source_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Build one progress row per source photograph."""
    crops = reviewable_rows(frame)
    if crops.empty:
        return pd.DataFrame(
            columns=[
                "source_image",
                "source_label",
                "source_group",
                "crop_count",
                "remaining",
                "completed",
            ]
        )
    crops["is_unreviewed"] = (
        crops["review_label"].astype(str).str.strip() == ""
    )
    summary = (
        crops.groupby("source_image", sort=False)
        .agg(
            source_label=("source_label", "first"),
            source_group=("source_group", "first"),
            crop_count=("crop_path", "size"),
            remaining=("is_unreviewed", "sum"),
        )
        .reset_index()
    )
    summary["completed"] = summary["remaining"] == 0
    return summary


def apply_source_labels(
    frame: pd.DataFrame,
    source_image: str,
    labels: Mapping[int, str],
) -> pd.DataFrame:
    """Apply reviewed labels to one source without touching other sources."""
    updated = frame.copy()
    source_indices = set(
        updated.index[updated["source_image"].astype(str) == source_image]
    )
    unexpected = set(labels) - source_indices
    if unexpected:
        raise ValueError(
            f"Label update includes rows outside this source: {unexpected}"
        )
    for row_index, raw_label in labels.items():
        label = str(raw_label).strip()
        if label not in REVIEW_LABELS:
            raise ValueError(f"Unsupported review label: {label}")
        updated.at[row_index, "review_label"] = label
        updated.at[row_index, "status"] = (
            "reviewed" if label else "needs_review"
        )
    return updated


def parse_bbox(value: object) -> tuple[int, int, int, int]:
    parts = [int(round(float(part))) for part in str(value).split(",")]
    if len(parts) != 4:
        raise ValueError(f"Invalid bounding box: {value}")
    return parts[0], parts[1], parts[2], parts[3]


def annotated_source(source_rows: pd.DataFrame) -> Image.Image:
    """Draw crop numbers and current saved labels over the source image."""
    source_path = Path(str(source_rows.iloc[0]["source_image"]))
    with Image.open(source_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(3, round(min(image.size) / 250))
    for crop_number, (_, row) in enumerate(source_rows.iterrows(), start=1):
        try:
            box = parse_bbox(row["bbox"])
        except (TypeError, ValueError):
            continue
        label = str(row["review_label"]).strip()
        color = BOX_COLORS.get(label, BOX_COLORS[""])
        draw.rectangle(box, outline=color, width=line_width)
        caption = f" {crop_number} "
        text_box = draw.textbbox((box[0], box[1]), caption)
        draw.rectangle(text_box, fill=color)
        draw.text((box[0], box[1]), caption, fill="black")
    return image


def choose_next_source(
    sources: list[str], current: str, *, forward: bool = True
) -> str | None:
    if not sources:
        return None
    try:
        position = sources.index(current)
    except ValueError:
        return sources[0]
    offset = 1 if forward else -1
    target = position + offset
    if 0 <= target < len(sources):
        return sources[target]
    return None


def queue_navigation(target: str | None) -> None:
    if target is not None:
        st.session_state.gk_pending_source = target


def clear_crop_widgets(source_rows: pd.DataFrame) -> None:
    for _, row in source_rows.iterrows():
        st.session_state.pop(f"gk_label_{row['crop_id']}", None)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--manifest",
        default="workspace/goalkeeper_classifier/candidates.csv",
        type=Path,
    )
    args, _ = parser.parse_known_args()
    path = args.manifest.expanduser().resolve()

    st.set_page_config(
        page_title="Goalkeeper Dataset Labeler",
        page_icon="🥅",
        layout="wide",
    )
    st.title("Goalkeeper dataset labeler")
    st.caption(
        "Review one source photo at a time. Red boxes are still unreviewed; "
        "green is goalkeeper, blue is outfield, and amber is rejected."
    )
    if not path.is_file():
        st.error(f"Candidate manifest not found: {path}")
        st.stop()

    frame = pd.read_csv(path, keep_default_na=False)
    try:
        validate_manifest(frame)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    frame["review_label"] = frame["review_label"].astype(str).str.strip()
    summary = source_summary(frame)
    if summary.empty:
        st.warning("There are no generated player crops to review.")
        st.stop()

    st.sidebar.header("Review queue")
    status_filter = st.sidebar.selectbox(
        "Status", ("Needs review", "Completed", "All")
    )
    class_filter = st.sidebar.selectbox(
        "Folder class", ("Both", "Goalkeeper", "Outfield")
    )
    search = st.sidebar.text_input(
        "Search filename or group", placeholder="Optional"
    ).strip().casefold()

    filtered = summary.copy()
    if status_filter == "Needs review":
        filtered = filtered[~filtered["completed"]]
    elif status_filter == "Completed":
        filtered = filtered[filtered["completed"]]
    if class_filter == "Goalkeeper":
        filtered = filtered[filtered["source_label"] == "goalkeeper"]
    elif class_filter == "Outfield":
        filtered = filtered[filtered["source_label"] == "not_goalkeeper"]
    if search:
        searchable = (
            filtered["source_image"].map(lambda value: Path(value).name)
            + " "
            + filtered["source_group"].astype(str)
        ).str.casefold()
        filtered = filtered[searchable.str.contains(search, regex=False)]

    all_crops = reviewable_rows(frame)
    label_counts = (
        all_crops["review_label"]
        .replace("", "unreviewed")
        .value_counts()
        .reindex(
            ["goalkeeper", "not_goalkeeper", "uncertain", "unreviewed"],
            fill_value=0,
        )
    )
    completed_sources = int(summary["completed"].sum())
    st.sidebar.metric(
        "Completed photos", f"{completed_sources} / {len(summary)}"
    )
    st.sidebar.progress(completed_sources / len(summary))
    st.sidebar.dataframe(
        label_counts.rename(
            index={
                "not_goalkeeper": "outfield",
                "uncertain": "rejected",
            }
        ).rename("crops"),
        width="stretch",
    )
    st.sidebar.caption(
        "Training uses only goalkeeper and outfield labels. Unreviewed and "
        "rejected crops are ignored."
    )

    sources = filtered["source_image"].astype(str).tolist()
    if not sources:
        st.success("No photos match this queue. This section is complete.")
        st.stop()

    pending = st.session_state.pop("gk_pending_source", None)
    if pending in sources:
        st.session_state.gk_source_picker = pending
    if st.session_state.get("gk_source_picker") not in sources:
        st.session_state.gk_source_picker = sources[0]

    display_names = {
        row.source_image: (
            f"{Path(row.source_image).name} "
            f"({int(row.remaining)} left, {int(row.crop_count)} crops)"
        )
        for row in filtered.itertuples()
    }
    source_image = st.sidebar.selectbox(
        "Jump to photo",
        sources,
        key="gk_source_picker",
        format_func=lambda value: display_names[value],
    )
    source_rows = all_crops[
        all_crops["source_image"].astype(str) == source_image
    ].sort_values("detection_index")
    source_position = sources.index(source_image)

    previous_source = choose_next_source(
        sources, source_image, forward=False
    )
    next_source = choose_next_source(sources, source_image, forward=True)
    previous, position, next_column = st.columns([1, 2, 1])
    previous.button(
        "← Previous photo",
        disabled=previous_source is None,
        width="stretch",
        on_click=queue_navigation,
        args=(previous_source,),
    )
    position.markdown(
        f"<div style='text-align:center'>Photo "
        f"<strong>{source_position + 1}</strong> of "
        f"<strong>{len(sources)}</strong> in this queue</div>",
        unsafe_allow_html=True,
    )
    next_column.button(
        "Next photo →",
        disabled=next_source is None,
        width="stretch",
        on_click=queue_navigation,
        args=(next_source,),
    )

    source_record = filtered[
        filtered["source_image"].astype(str) == source_image
    ].iloc[0]
    preview_column, details_column = st.columns([3, 2])
    with preview_column:
        st.subheader("Source photograph")
        try:
            st.image(
                annotated_source(source_rows),
                width="stretch",
            )
        except (OSError, ValueError) as exc:
            st.error(f"Could not render the source image: {exc}")
    with details_column:
        st.subheader(Path(source_image).name)
        st.write(
            {
                "folder_class": (
                    "goalkeeper"
                    if source_record["source_label"] == "goalkeeper"
                    else "outfield"
                ),
                "source_group": source_record["source_group"],
                "detected_people": int(source_record["crop_count"]),
                "unreviewed": int(source_record["remaining"]),
            }
        )
        st.info(
            "The folder class describes the intended subject, not every "
            "person in the photograph."
        )

    st.subheader("Fast labeling")
    fast_left, fast_middle, fast_right = st.columns(3)
    if fast_left.button(
        "Mark every crop outfield",
        width="stretch",
        help="Useful for an outfield source containing only field players.",
    ):
        labels = {
            int(row_index): "not_goalkeeper"
            for row_index in source_rows.index
        }
        frame = apply_source_labels(frame, source_image, labels)
        save_manifest(frame, path)
        clear_crop_widgets(source_rows)
        queue_navigation(next_source)
        st.rerun()

    crop_choices = list(range(1, len(source_rows) + 1))
    target_crop = fast_middle.selectbox(
        "Intended goalkeeper crop",
        crop_choices,
        format_func=lambda value: f"Crop {value}",
        help="Select the goalkeeper; the other detected people become outfield.",
    )
    if fast_middle.button(
        "Selected goalkeeper + others outfield",
        width="stretch",
    ):
        labels = {
            int(row_index): (
                "goalkeeper"
                if crop_number == target_crop
                else "not_goalkeeper"
            )
            for crop_number, row_index in enumerate(
                source_rows.index, start=1
            )
        }
        frame = apply_source_labels(frame, source_image, labels)
        save_manifest(frame, path)
        clear_crop_widgets(source_rows)
        queue_navigation(next_source)
        st.rerun()

    if fast_right.button(
        "Reject every crop",
        width="stretch",
        help="Use only when this photo cannot be labeled reliably.",
    ):
        labels = {
            int(row_index): "uncertain" for row_index in source_rows.index
        }
        frame = apply_source_labels(frame, source_image, labels)
        save_manifest(frame, path)
        clear_crop_widgets(source_rows)
        queue_navigation(next_source)
        st.rerun()

    st.subheader("Detected players")
    st.caption(
        "Use the individual controls when a referee, partial person, or "
        "multiple goalkeepers make the fast actions inappropriate."
    )
    selected_labels: dict[int, str] = {}
    crop_columns = st.columns(min(3, len(source_rows)))
    for crop_number, (row_index, row) in enumerate(
        source_rows.iterrows(), start=1
    ):
        with crop_columns[(crop_number - 1) % len(crop_columns)]:
            st.markdown(f"#### Crop {crop_number}")
            crop_path = Path(str(row["crop_path"]))
            if crop_path.is_file():
                st.image(str(crop_path), width="stretch")
            else:
                st.error(f"Missing crop: {crop_path}")
            current_label = str(row["review_label"]).strip()
            selected_labels[int(row_index)] = st.radio(
                f"Label crop {crop_number}",
                REVIEW_LABELS,
                index=REVIEW_LABELS.index(current_label),
                format_func=lambda value: LABEL_CAPTIONS[value],
                key=f"gk_label_{row['crop_id']}",
                horizontal=True,
                label_visibility="collapsed",
            )

    save_column, save_next_column, reset_column = st.columns(3)
    if save_column.button("Save", type="primary", width="stretch"):
        frame = apply_source_labels(frame, source_image, selected_labels)
        save_manifest(frame, path)
        st.success("Labels saved.")
    if save_next_column.button(
        "Save and next", type="primary", width="stretch"
    ):
        frame = apply_source_labels(frame, source_image, selected_labels)
        save_manifest(frame, path)
        clear_crop_widgets(source_rows)
        queue_navigation(next_source)
        st.rerun()
    if reset_column.button(
        "Reset this photo", width="stretch"
    ):
        labels = {int(row_index): "" for row_index in source_rows.index}
        frame = apply_source_labels(frame, source_image, labels)
        save_manifest(frame, path)
        clear_crop_widgets(source_rows)
        st.rerun()

    st.caption(
        f"Labels save to {path}. The original manifest is backed up once as "
        f"{path.name}.before_manual_review."
    )


if __name__ == "__main__":
    main()
