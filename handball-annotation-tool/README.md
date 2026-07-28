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

### Experimental visual-feature fusion

The visual-fusion experiment keeps the 56 YOLO/pose features and adds actual
appearance information. For each of the same 12 selected moments, it crops the
detected player together with the detected ball, runs a frozen
ImageNet-pretrained MobileNetV3-Small, and stores a 576-value embedding. A
small visual projection is pooled and concatenated with the existing GRU
representation.

Extract the resumable visual cache:

```bash
python -m training.visual_features \
    --config configs/visual_fusion.yaml
```

Train one fold or run the complete five-fold comparison:

```bash
python -m training.visual_fusion \
    --config configs/visual_fusion.yaml \
    --fold 0

python -m training.visual_fusion \
    --config configs/visual_fusion.yaml \
    --all-folds
```

The all-fold command writes predictions, per-fold histories, and a direct
comparison with the existing GRU under `artifacts/reports_visual_fusion`.
Generated embeddings and checkpoints are ignored by Git.

The first 286-clip experiment did not beat the existing GRU at threshold 0.5:

| Model | Accuracy | Precision | Recall | F1 | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing 56-feature GRU | 79.37% | 64.84% | 68.60% | 66.67% | 72.22% |
| MobileNet visual fusion | 78.67% | 62.89% | 70.93% | 66.67% | 70.72% |

The visual branch found two additional handballs but introduced four
additional false alarms, leaving two more total errors. It therefore remains
an isolated experiment and is not used by normal inference.

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

#### Supervised full-player goalkeeper classifier

This is the recommended goalkeeper pipeline. It fine-tunes a pretrained
MobileNetV3-Small on reviewed full-player crops, then applies that classifier
to multiple frames from the handball actor's ByteTrack track. Gloves, jersey
appearance, sleeves, and pose are learned together; no glove-only labels are
required.

Organize the incoming full photographs by class and, ideally, by match:

```text
incoming_goalkeeper_photos/
  match_001/
    image_001.jpg
    image_002.jpg
incoming_not_goalkeeper_photos/
  match_001/
    image_003.jpg
  match_002/
    image_004.jpg
```

Matching subdirectory names are treated as the same source group and can never
cross training/validation/test folds. This prevents nearly identical match
backgrounds and uniforms from leaking into test results.

Detect people and generate 15%-margin player crops:

```bash
python -m training.goalkeeper_dataset \
    --goalkeeper-source /path/to/incoming_goalkeeper_photos \
    --not-goalkeeper-source /path/to/incoming_not_goalkeeper_photos \
    --output-root workspace/goalkeeper_classifier \
    --detector yolo11n.pt
```

The command prints one live line per source image. A source containing exactly
one usable person is safely auto-labeled from its class folder. Every person
from a multi-player source remains unlabeled until reviewed:

```bash
streamlit run goalkeeper_review_app.py -- \
    --manifest workspace/goalkeeper_classifier/candidates.csv
```

The app reviews one source photograph at a time. It overlays a number on every
YOLO person box and shows all matching player crops together. Label crops
individually as `goalkeeper`, `outfield`, or `reject / uncertain`, or use the
fast actions to mark every crop as outfield or select one goalkeeper and mark
the remaining people as outfield. Progress is saved atomically to the
candidate manifest, and the first edit creates
`candidates.csv.before_manual_review` as a recoverable backup. The default
queue contains only photographs that still need review, so reopening the app
resumes unfinished work. Training ignores blank and uncertain rows.

When running on an SSH host, expose the app on port 8502:

```bash
streamlit run goalkeeper_review_app.py \
    --server.address 0.0.0.0 \
    --server.port 8502 -- \
    --manifest workspace/goalkeeper_classifier/candidates.csv
```

Forward port 8502 in the VS Code **Ports** panel, then open the forwarded URL.

Train and evaluate the classifier:

```bash
python -m training.goalkeeper_train \
    --config configs/goalkeeper_classifier.yaml
```

Training prints every epoch's loss, precision, recall, and F1. It uses
source-group-separated folds, a frozen-backbone warmup followed by fine-tuning,
weighted sampling, early stopping, and validation-tuned abstaining thresholds.
The final untouched fold produces accuracy, precision, recall, F1, PR-AUC, a
confusion matrix, a per-image predictions CSV, and copies of every mistake.
Artifacts are written to:

```text
artifacts/checkpoints/goalkeeper_mobilenet_v3_best.pt
artifacts/reports_goalkeeper_classifier/history.csv
artifacts/reports_goalkeeper_classifier/metrics.json
artifacts/reports_goalkeeper_classifier/test_predictions.csv
artifacts/reviews_goalkeeper_classifier/
```

After the checkpoint exists, run the complete handball pipeline with the
supervised goalkeeper model:

```bash
python -m training.inference \
    --input dataset/handball/CANDIDATE_ID \
    --checkpoint artifacts/checkpoints/gru_fold0_best.pt \
    --supervised-goalkeeper-config configs/supervised_goalkeeper.yaml \
    --output outputs/prediction_with_supervised_goalkeeper.json \
    --overlay outputs/prediction_with_supervised_goalkeeper.jpg
```

Once that trained checkpoint exists, the command-line inference entry point
also selects the supervised configuration automatically when no role option is
given. Supplying `--supervised-goalkeeper-config` explicitly is recommended for
recorded experiments because it makes the chosen backend obvious in the
command itself.

The GRU runs first. A raw `not_handball` result skips goalkeeper analysis
entirely. For a raw `handball`, the actor is associated with a YOLO + ByteTrack
identity, up to eight sharp full-player crops are classified, and their
probabilities are aggregated using the median plus a minimum frame-agreement
rule. Fewer than three usable crops, uncertain association, conflicting
probabilities, or an unavailable checkpoint produce `unknown` and preserve the
raw GRU decision. Only a consistent, high-confidence `goalkeeper` result can
veto a raw handball prediction. JSON output includes the explicit hierarchical
event label: `not_handball`, `handball_goalkeeper`, `handball_outfield`, or
`handball_actor_unknown`.

The source-photo test set measures the image classifier. Honest
goalkeeper-specific metrics on the existing 286 clips additionally require
goalkeeper labels for the involved actor in those clips.

After training, evaluate all clips while gating the supervised role stage to
raw-positive clips:

```bash
python -m training.combined_evaluate \
    --supervised-goalkeeper-config configs/supervised_goalkeeper.yaml \
    --role-cache-dir artifacts/roles_combined_supervised_gated \
    --output-csv artifacts/reports/combined_supervised_gated_predictions.csv \
    --output-json artifacts/reports/combined_supervised_gated_metrics.json
```

This produces raw-versus-final precision, recall, F1, binary clip-label
accuracy, confusion matrices, goalkeeper veto count, skipped/known/unknown/error
coverage, and one auditable row per clip. The role cache is checkpoint-aware
and resumable. Because the clip manifest does not label the actor as goalkeeper
or outfield, it cannot provide honest three-class actor-role accuracy.

Review every final mistake and goalkeeper-veto decision in a local UI:

```bash
streamlit run combined_mistake_review_app.py \
    --server.address 0.0.0.0 \
    --server.port 8503 -- \
    --predictions artifacts/reports/combined_supervised_gated_predictions.csv \
    --audit artifacts/reports/combined_supervised_gated_audit.csv
```

Forward port 8503 in the VS Code **Ports** panel and open its forwarded URL.
The default queue contains final combined mistakes. Filters separate false
positives, false negatives, base-model misses, surviving false alarms,
goalkeeper vetoes that helped, and goalkeeper vetoes that hurt. Each clip has
an animated preview, an exact-frame scrubber, base-model player/ball boxes,
the actor track and exact crops used by the goalkeeper classifier, and an
independent root-cause audit form. Reviews are saved atomically to the audit
CSV; the prediction report is never changed.

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

#### Jersey and wrist-localized glove experiment

The conservative pipeline now runs before the final handball label. It
preserves the GRU probability and raw label, associates the specific handball
actor with a YOLO + ByteTrack person track, compares the actor's center-torso
LAB color histogram with the two dominant field-team clusters, and optionally
scores MediaPipe wrist crops with a local MobileNetV3-Small glove classifier.
A validated `goalkeeper` result vetoes a raw handball label; `not_goalkeeper`,
`unknown`, or an unavailable role stage preserves the raw decision. It never
manufactures a fused probability.

Jersey histograms use a center mask and consistent Hellinger geometry. Up to
three candidate color groups are formed so a small referee/goalkeeper group can
be discarded, and overlapping duplicate actor tracks are excluded before team
clustering. Weak actor links or conflicting/missing evidence return `unknown`.
The older PRTReID classifier is disabled by default and is only an optional
cross-check.

Run a resumable audit on one labeled handball clip:

```bash
python -m training.jersey_glove_audit \
    --config configs/jersey_glove_goalkeeper.yaml \
    --limit 1 \
    --verbose
```

Remove `--limit 1` to audit all 86 labeled handball clips. JSON results are
written under `artifacts/roles_jersey_glove`, visual actor/hand contact sheets
under `artifacts/role_audits_jersey_glove`, individual accepted hand crops under
`artifacts/hand_crops_jersey_glove`, and the CSV report to
`artifacts/reports_jersey_glove/player_roles.csv`.

Add this independent result to single-candidate inference:

```bash
python -m training.inference \
    --input dataset/handball/CANDIDATE_ID \
    --checkpoint artifacts/checkpoints/gru_fold0_best.pt \
    --jersey-glove-config configs/jersey_glove_goalkeeper.yaml \
    --output outputs/prediction_with_jersey_glove.json \
    --overlay outputs/prediction_with_jersey_glove.jpg
```

The command-line inference entry point uses
`configs/jersey_glove_goalkeeper.yaml` by default when no legacy role option is
selected. Pass `--no-goalkeeper-filter` only when a raw-GRU-only diagnostic is
explicitly required. This project currently applies the requested
goalkeeper-veto policy without penalty-area geometry; that is an experimental
label policy, not the full football handball law.

The repository does not include a trained glove checkpoint. Therefore
`glove.enabled` is `false` by default: the audit still extracts hand crops, but
normal inference skips pose work and can return only a conservative
`not_goalkeeper` or `unknown`. A `goalkeeper` result requires a compatible
trained checkpoint at
`models/goalkeeper_glove_mobilenet_v3_small.pt` and `glove.enabled: true`.
Never enable it with random weights.

Run the complete five-fold gated evaluation on every manifest clip:

```bash
python -m training.combined_evaluate
```

The command prints one live line per clip, skips role inference for raw
negatives, caches resumable raw-positive role results under
`artifacts/roles_combined_oof`, writes per-clip raw and final decisions to
`artifacts/reports/combined_oof_predictions.csv`, and writes baseline/combined
precision, recall, F1, accuracy, confusion matrices, status coverage, and
fingerprints to `artifacts/reports/combined_oof_metrics.json`. The current
training loop selected each best epoch using that same outer fold, so these are
practical out-of-fold metrics but mildly optimistic—not a pristine nested-test
estimate. The manifest also has no goalkeeper ground-truth column, so the
command cannot honestly report goalkeeper-only accuracy.

The pinned PRTReID source is distributed under the Hippocratic License 3.0 and
the SoccerNet checkpoint under CC BY 4.0. Review those terms before
redistributing or deploying the model. The implementation uses its own thin
worker and does not copy the GPL-3.0 SoccerNet game-state wrapper.

The earlier 12-frame Hugging Face YOLO role experiment remains available for
comparison with `training.role_audit` and `configs/hf_goalkeeper.yaml`, but it
does not provide full-track identity consistency.

## Motion-enhanced GRU experiment

The isolated `trajectory-optical-flow-experiment` adds 84 engineered motion
channels to the original 56 features while preserving the same 12-frame GRU
architecture and split definitions. The added channels describe ball
velocity/acceleration/curvature, lower-frame bounce proxies, ball-to-arm
distance and relative motion, wrist motion, camera-compensated DIS optical
flow, and sparse ball-arm contact evidence. They are experimental signals, not
ground-truth contact or bounce labels.

Generate the `12 x 140` tensors and train all five folds:

```bash
python -m training.motion_features --config configs/motion_gru.yaml
python -m training.motion_gru --config configs/motion_gru.yaml --all-folds
```

The first five-fold run on 286 clips produced 212 correct and 74 incorrect
predictions: 74.13% accuracy, 56.00% precision, 65.12% recall, and 60.22% F1.
The original 56-feature GRU produced 227 correct and 59 incorrect predictions:
79.37% accuracy and 66.67% F1. Therefore this branch records a useful negative
experiment and does not replace the original GRU. Generated tensors,
checkpoints, and reports live under `artifacts/features_motion_flow`,
`artifacts/checkpoints_motion_flow`, and `artifacts/reports_motion_flow` and
are ignored by Git.

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
