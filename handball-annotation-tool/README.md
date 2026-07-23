# Handball Annotation Tool

A local, human-in-the-loop application for mining possible ball-to-arm events from long soccer videos and building a labeled dataset. It does **not** make the final handball decision: it proposes visually plausible events and asks a person to label them.

## How the pipeline works

```text
Uploaded match video
        ↓
YOLO person + sports-ball detection
        ↓
ByteTrack persistent IDs and ball trail
        ↓
Pose estimation on players nearest the ball
        ↓
Ball-to-upper-arm/forearm distance
        ↓
20 frames before + contact frame + 20 frames after
        ↓
Human: Handball / Not handball / Uncertain
```

Distance is divided by the player bounding-box height. This is more robust than a fixed pixel threshold because broadcast camera zoom changes constantly. Candidates less than `arm_distance_threshold` apart are retained, and events within `candidate_cooldown_frames` are grouped to reduce duplicates.

The default detector is COCO-pretrained. It recognizes `person` and `sports ball`, but it is only a bootstrap model: small or blurred soccer balls are difficult, so a soccer-specific detector should eventually replace it. The model suggests candidates; missed detections do not mean contact did not occur.

## Install

Use Python 3.10 or newer from this folder:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For an NVIDIA GPU, install the CUDA-enabled PyTorch wheel appropriate for your system from the official PyTorch selector before installing the requirements. With `device: auto`, the app chooses CUDA, Apple MPS, or CPU in that order. Initial use downloads the YOLO model weights.

FFmpeg is recommended for browser-compatible H.264 previews. If it is not installed, the app keeps OpenCV MP4 output, which may not play in every browser. Install FFmpeg with your operating system’s package manager.

## Run

```bash
streamlit run app.py
```

The browser interface lets you upload MP4, MOV, MKV, AVI, or M4V footage. Select **Find contact candidates**, leave the page open during processing, and then review each result. Each source video has an independent review tab and remembers its candidate position. The frame viewer supports thumbnails, previous/next controls, keyboard arrow keys, and fullscreen navigation. CPU processing of a full match can take a long time; CUDA is strongly recommended for large videos.

To sample legal-play/non-handball windows progressively from full matches, run the separate application:

```bash
streamlit run negative_sampler_app.py
```

It presents one 41-frame window at a time at an adjustable interval (five seconds by default). Each clip autoplays in a continuous loop, with an optional manual frame-by-frame preview. Accepted windows join `dataset/not_handball` in the same format as annotations from the main app. Rejections, changes, stopping, and resuming are saved independently.

## YOLO + MediaPipe training pipeline

The training code is intentionally separate from both annotation interfaces. It uses YOLO person/ball detections, ByteTrack identities, MediaPipe shoulder/elbow/wrist landmarks, and a small temporal GRU. Imported `processed_frames_no_handball` data is filtered to auxiliary CSV label `1` only; label `0` actions and `dataset/uncertain` are excluded.

Install dependencies and download the official MediaPipe Pose Landmarker Full model:

```bash
python -m pip install -r requirements.txt
python -m training.download_models
```

Build the canonical manifest and fixed leakage-safe folds:

```bash
python -m training.manifest \
    --output artifacts/manifests/dataset.csv \
    --imported-label 1
```

Extract and cache YOLO/ByteTrack/MediaPipe features. This is the expensive step and resumes by skipping existing `.npz` files:

```bash
python -m training.features \
    --config configs/mediapipe_features.yaml
```

On macOS, MediaPipe needs access to the normal graphics runtime. If an IDE sandbox prevents that initialization, run the command from Terminal.

Inspect the generated contact sheets under `artifacts/overlays`, then create a detection-quality report before training:

```bash
python -m training.quality_report
```

Train the interpretable baselines:

```bash
python -m training.baseline --model logistic --fold 0
python -m training.baseline --model random_forest --fold 0
```

Train the temporal GRU:

```bash
python -m training.gru \
    --config configs/temporal_classifier.yaml \
    --fold 0
```

Repeat folds `0` through `4`, then summarize:

```bash
python -m training.evaluate --summarize
```

Resume an interrupted training run:

```bash
python -m training.gru \
    --fold 0 \
    --resume artifacts/checkpoints/gru_fold0_last.pt
```

Classify a labeled or workspace candidate and save an evidence overlay:

```bash
python -m training.inference \
    --input dataset/handball/CANDIDATE_ID \
    --checkpoint artifacts/checkpoints/gru_fold0_best.pt \
    --output outputs/prediction.json \
    --overlay outputs/prediction_overlay.jpg
```

Feature extraction must be audited before classifier results are trusted. Compare ball and pose detection rates across native positives, native negatives, and 224×224 imported negatives. A large domain gap means the classifier may learn detection quality rather than handball.

Current smoke-test warning: on the first imported 224×224 action, YOLO found a selected player in 11/12 frames, but MediaPipe Full recovered no usable arm pose and YOLO found the ball in only 1/12 frames. Full imported-data extraction and classifier training should remain gated until a representative quality sample is checked. Original-resolution imported footage, a soccer-specific ball detector, or a different pose estimator may be required; training with systematically missing imported features would create a class shortcut.

## Files and labels

```text
workspace/uploads/       uploaded source videos
workspace/candidates/    clean and evidence review artifacts
workspace/state/         resumable SQLite annotation state
dataset/handball/        clean labeled clips and frames
dataset/not_handball/    clean labeled clips and frames
dataset/uncertain/       clean labeled clips and frames
dataset/annotations.jsonl
```

Each labeled example contains `clip.mp4`, a `frames/` directory, and `metadata.json`. Detection boxes, skeletons, paths, and measurements are stored only under `workspace/candidates`; they are never copied into the labeled dataset. Selecting a different label later moves the logical example to the new class and updates the exported JSONL manifest.

## Configuration

Edit `config.yaml` to change models, tracker, device, thresholds, frame window, or directories. The current uploader accepts files up to 10 GB. The next ingestion extension can call the same `mine_video()` function for every video in a local directory without changing the annotation interface or dataset format.
