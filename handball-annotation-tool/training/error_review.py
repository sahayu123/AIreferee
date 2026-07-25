from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "artifacts/reports/gru_fold0_predictions.csv"
MANIFEST = ROOT / "artifacts/manifests/dataset.csv"
REVIEWS = ROOT / "artifacts/reviews/fold0_label_review.csv"
LABEL_NAME = {0: "not_handball", 1: "handball"}


@st.cache_data
def disagreements() -> pd.DataFrame:
    predictions = pd.read_csv(PREDICTIONS)
    predictions["predicted_label"] = (predictions["probability"] >= 0.5).astype(int)
    errors = predictions[predictions["label"] != predictions["predicted_label"]].copy()
    manifest = pd.read_csv(MANIFEST)[["example_id", "frames_dir"]]
    return errors.merge(manifest, on="example_id").reset_index(drop=True)


def existing_reviews() -> pd.DataFrame:
    if REVIEWS.is_file():
        return pd.read_csv(REVIEWS)
    return pd.DataFrame(columns=["example_id", "decision", "notes"])


def save_review(example_id: str, decision: str, notes: str) -> None:
    reviews = existing_reviews()
    reviews = reviews[reviews["example_id"] != example_id]
    reviews = pd.concat(
        [reviews, pd.DataFrame([{"example_id": example_id, "decision": decision, "notes": notes}])],
        ignore_index=True,
    )
    REVIEWS.parent.mkdir(parents=True, exist_ok=True)
    reviews.to_csv(REVIEWS, index=False)


st.set_page_config(page_title="Fold 0 error review", layout="wide")
st.title("Fold 0 model disagreements")
st.caption("These are model/label disagreements, not automatically incorrect annotations.")

items = disagreements()
reviews = existing_reviews()
reviewed_ids = set(reviews["example_id"].astype(str))
st.progress(len(reviewed_ids.intersection(items["example_id"])) / max(len(items), 1))
st.write(f"Reviewed {len(reviewed_ids.intersection(items['example_id']))} of {len(items)}")

index = st.selectbox(
    "Example",
    range(len(items)),
    format_func=lambda i: (
        f"{i + 1}/{len(items)} · {LABEL_NAME[int(items.iloc[i]['label'])]} → "
        f"{LABEL_NAME[int(items.iloc[i]['predicted_label'])]} · "
        f"{float(items.iloc[i]['probability']):.1%}"
    ),
)
item = items.iloc[index]
frames_dir = ROOT / str(item["frames_dir"])
clip_path = frames_dir.parent / "clip.mp4"

st.subheader(str(item["example_id"]))
left, middle, right = st.columns(3)
left.metric("Current label", LABEL_NAME[int(item["label"])])
middle.metric("Model prediction", LABEL_NAME[int(item["predicted_label"])])
right.metric("Handball probability", f"{float(item['probability']):.1%}")

if clip_path.is_file():
    st.video(str(clip_path))
else:
    st.warning(f"Clip missing: {clip_path}")

frames = sorted(frames_dir.glob("*.jpg"))
if frames:
    st.write("Frames")
    columns = st.columns(5)
    for frame_index, frame in enumerate(frames):
        columns[frame_index % 5].image(str(frame), caption=frame.name, use_container_width=True)

prior = reviews[reviews["example_id"] == item["example_id"]]
prior_decision = str(prior.iloc[0]["decision"]) if not prior.empty else "correct_as_is"
prior_notes = str(prior.iloc[0]["notes"]) if not prior.empty and pd.notna(prior.iloc[0]["notes"]) else ""
choices = ["correct_as_is", "change_to_handball", "change_to_not_handball", "uncertain"]
decision = st.radio("Review decision", choices, index=choices.index(prior_decision))
notes = st.text_area("Notes", value=prior_notes)
if st.button("Save review", type="primary"):
    save_review(str(item["example_id"]), decision, notes)
    st.success(f"Saved to {REVIEWS}")
    st.rerun()
