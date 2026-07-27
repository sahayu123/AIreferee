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

The training code is intentionally separate from both annotation interfaces. It uses YOLO person/ball detections, ByteTrack identities, MediaPipe shoulder/elbow/wrist landmarks, and a small temporal GRU. The canonical training manifest uses only `dataset/handball` and `dataset/not_handball`; `dataset/processed_frames_no_handball` and `dataset/uncertain` are excluded.

Install dependencies and download the official YOLO11n and MediaPipe Pose Landmarker Full models:

```bash
python -m pip install -r requirements.txt
python -m training.download_models
```

Build the canonical manifest and fixed leakage-safe folds:

```bash
python -m training.manifest \
    --output artifacts/manifests/dataset.csv
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

Feature extraction must be audited before classifier results are trusted. Compare ball and pose detection rates across the handball and not-handball classes. A large quality gap means the classifier may learn detection quality rather than handball.

### Independent goalkeeper detection

The PRTReID experiment uses the pretrained SoccerNet player-role model to
classify full player tracks as `goalkeeper`, `player`, `referee`, or `unknown`.
It reruns YOLO + ByteTrack over every available clip frame, scores every
retained person track, associates the handball actor using the original
center-frame arm location when available (otherwise the stored 12-frame actor
boxes), and uses jersey-colour difference only as secondary evidence. It
abstains when actor association or role evidence is weak.

PRTReID requires Python 3.9 and PyTorch 1.13, which are incompatible with the
main Python 3.13 environment on this server. Create the isolated worker
environment once:

```bash
conda create -n aireferee-prtreid python=3.9 -y
conda run -n aireferee-prtreid \
    python -m pip install --no-deps -r requirements-prtreid.txt
```

The checked-in configuration points to
`/home/cosmos32/anaconda3/envs/aireferee-prtreid/bin/python`. Change the first
entry under `worker.command` in `configs/prtreid_goalkeeper.yaml` if the
environment lives elsewhere.

Download and checksum-verify the official checkpoint:

```bash
python -m training.download_models --with-prtreid
```

Run a resumable 25-incident audit first:

```bash
python -m training.prtreid_audit \
    --config configs/prtreid_goalkeeper.yaml \
    --limit 25 \
    --verbose
```

Results are written under `artifacts/roles_prtreid`, contact sheets under
`artifacts/role_audits_prtreid`, and the summary CSV to
`artifacts/reports_prtreid/player_roles.csv`. Remove `--limit 25` to process all
manifest examples.

Add independent role output to single-candidate inference:

```bash
python -m training.inference \
    --input dataset/handball/CANDIDATE_ID \
    --checkpoint artifacts/checkpoints/gru_fold0_best.pt \
    --prtreid-config configs/prtreid_goalkeeper.yaml \
    --output outputs/prediction_with_role.json \
    --overlay outputs/prediction_with_role.jpg
```

This role output is deliberately separate from the 56-feature GRU: enabling it
does not retrain or change the handball probability. The dataset currently has
no goalkeeper ground-truth labels, so the audit cannot honestly report
goalkeeper accuracy. Treat `unknown` as a valid outcome and inspect the contact
sheets before using the result.

The pinned PRTReID source is distributed under the Hippocratic License 3.0 and
the SoccerNet checkpoint under CC BY 4.0. Review those terms before
redistributing or deploying the model. The implementation uses its own thin
worker and does not copy the GPL-3.0 SoccerNet game-state wrapper.

The earlier 12-frame Hugging Face YOLO role experiment remains available for
comparison with `training.role_audit` and `configs/hf_goalkeeper.yaml`, but it
does not provide full-track identity consistency.

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
