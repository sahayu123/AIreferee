# AIreferee

AIreferee is a research prototype for reviewing football incidents with two
separate machine-learning specialists: a general-foul detector and a binary
handball detector. It accepts an image or video, gathers visual evidence,
combines the available model scores, and presents an explainable decision in a
browser interface.

The project is intended as an analysis and decision-support tool. It is not a
replacement for a match official, VAR system, or independently validated
officiating product.

## Current scope

- Video review for `handball`, `other foul`, `no foul`, or `needs review`.
- Image review with the general-foul specialist only.
- Player and ball tracking, pose/contact evidence, an evidence timeline, and a
  browser-compatible annotated review video.
- A Streamlit workflow for mining, reviewing, and labeling handball examples.
- Feature-extraction and five-fold GRU training code.

Goalkeeper identification, offside, and out-of-bounds classification are not
part of the current application.

## How the combined application works

The general-foul and handball models remain separate. For a video, the
general-foul model runs first so it can locate the most relevant incident time;
the handball model then evaluates a short window centered on that moment.

```text
Uploaded image or video
          |
          v
React review interface --> FastAPI analysis job
          |
          +---------------- image ----------------+
          |                                       |
          |                              General-foul specialist
          |                                       |
          +---------------- video ----------------+
                                                  |
                         Full-video general-foul analysis
                         + player and ball tracking
                                                  |
                                      Peak incident timestamp
                                                  |
                                   Centered 41-frame window
                                                  |
                           12 selected frames x 56 features
                                                  |
                              Five trained GRU fold models
                                                  |
                                 Mean handball probability
                                                  |
                          General-foul + handball decision rules
                                                  |
                Verdict, evidence, timeline, peak frame, and video
```

### General-foul specialist

The video pipeline analyzes every decoded frame. Its evidence stack includes:

- YOLO11m person and ball detection with BoT-SORT tracking.
- RTMW-X pose estimation on tracked player regions.
- A trained scene-level image MLP.
- Roboflow frame-level foul evidence.
- Separate contact and tackle classifiers.
- SAM 2 and MoGe-2 metric-depth verification for selected contact events.

The active backend builds this runtime from the modules embedded in
[`AI Referee Foul Checker Prototype (1).ipynb`](AI%20Referee%20Foul%20Checker%20Prototype%20%281%29.ipynb)
and loads three separately supplied foul checkpoints.

### Handball specialist

The handball model uses a centered 41-frame incident context. Feature
extraction examines the available sequence, divides it into temporal regions,
and selects 12 high-quality frames rather than simply taking the first or
middle 12 frames.

For every selected frame, YOLO11n, ByteTrack, and MediaPipe Pose produce 56
features covering:

- Ball and player position, size, confidence, and detection validity.
- Shoulder, elbow, and wrist coordinates and visibility.
- Ball distance to wrists, elbows, upper arms, and forearms.
- Arm angles and the ball position relative to the selected player.
- Ball velocity, speed, and direction change.
- Wrist motion, pose quality, and minimum normalized ball-to-arm distance.

The resulting `12 x 56` sequence is passed to five two-layer GRUs with 64
hidden units. Each checkpoint was trained with a different held-out fold and
keeps its own training-set normalization statistics. At inference time, all
five models evaluate the same sequence and their probabilities are averaged.
The models are not retrained when the application analyzes a video.

### Final decision rules

Let `P(handball)` be the five-GRU mean and `P(foul)` be the normalized output
from the general-foul specialist.

| Condition | Application result |
| --- | --- |
| `P(handball) >= 0.70` | `FOUL — HANDBALL` |
| Otherwise, `P(foul) >= 0.70` | General/other foul |
| General specialist says no-foul and both probabilities are `<= 0.30` | `NO FOUL` |
| Any other combination | `NEEDS REVIEW` |

The standalone handball specialist uses `0.50` as its binary classification
threshold, while the combined application requires the more conservative
`0.70` threshold before publishing a handball decision.

### Review output

The interface provides:

- Live queue, stage, and progress updates.
- A final verdict, confidence, and decision reason.
- Frame count and player/ball track counts.
- The peak incident frame and timestamp.
- A frame-by-frame evidence timeline and component model scores.
- A handball evidence contact sheet and per-fold probabilities through the API.
- An annotated H.264/`avc1`, YUV420p, fast-start MP4 for browser playback.

The backend validates the annotated video's codec, frame count, decodability,
and streaming metadata before publishing its artifact URL.

## Dataset

The binary handball dataset created for this project is publicly available on
Kaggle:

**[Football Handball vs No Handball](https://www.kaggle.com/datasets/sahayuraja/football-handball-vs-no-handball)**

The release contains 286 labeled examples:

| Class | Examples |
| --- | ---: |
| Handball | 86 |
| Not handball | 200 |
| **Total** | **286** |

We created the dataset curation and annotations from scratch for AIreferee.
Our work includes selecting the incidents, creating the temporal clips,
assigning the binary labels, extracting the frame sequences, and organizing
the metadata and class structure. The underlying match and broadcast footage
remains the property of its respective publishers and rights holders.

Each labeled example follows this structure:

```text
dataset/
  handball/<example_id>/
    clip.mp4
    frames/
    metadata.json
  not_handball/<example_id>/
    clip.mp4
    frames/
    metadata.json
```

Download it through the Kaggle page above or, from the repository root, with
an authenticated Kaggle CLI:

```bash
kaggle datasets download \
  -d sahayuraja/football-handball-vs-no-handball \
  --unzip \
  -p handball-annotation-tool/dataset
```

### Preliminary handball evaluation

The current local five-fold out-of-fold evaluation on all 286 examples
produced the following binary handball results:

| Metric | Result |
| --- | ---: |
| Correct / incorrect | 227 / 59 |
| Accuracy | 79.37% |
| Precision | 64.84% |
| Recall | 68.60% |
| F1 score | 66.67% |

These are fold-validation predictions from the five individual training runs,
not an untouched test-set evaluation of the deployed five-model ensemble. The
validation fold also participates in early stopping and checkpoint selection,
so these numbers should be treated as preliminary rather than as a claim of
real-world officiating accuracy. The generated reports and trained weights are
not committed to this repository.

## Repository layout

```text
AIreferee/
|-- README.md
|-- AI Referee Foul Checker Prototype (1).ipynb  # active foul source notebook
|-- *.ipynb                                      # earlier research prototypes
|-- handball-annotation-tool/
|   |-- app.py                                   # candidate review and labeling UI
|   |-- negative_sampler_app.py                  # not-handball sampling UI
|   |-- handball_annotator/                      # mining, geometry, and storage
|   |-- training/                                # manifest, features, GRU, evaluation
|   |-- configs/                                 # extraction and training settings
|   `-- tests/
`-- vair-gpu-ui/
    |-- app/                                     # React review interface
    |-- backend/                                 # FastAPI inference service
    |-- scripts/                                 # setup, startup, proxy, and pull helpers
    `-- tests/
```

Detailed component documentation is available in
[`vair-gpu-ui/README.md`](vair-gpu-ui/README.md) and
[`handball-annotation-tool/README.md`](handball-annotation-tool/README.md).

## Technology stack

| Area | Technology |
| --- | --- |
| Web interface | React 19, Next.js-compatible Vinext/Vite tooling |
| Model API | FastAPI and Uvicorn |
| Vision | OpenCV, Ultralytics YOLO, RTMLib, MediaPipe |
| Tracking and pose | BoT-SORT, ByteTrack, RTMW-X, MediaPipe Pose |
| Deep learning | PyTorch, torchvision, ONNX Runtime, Transformers |
| Handball sequence model | Five-fold temporal GRU ensemble |
| Review-video delivery | PyAV, H.264/`avc1`, YUV420p, MP4 fast start |
| Annotation tools | Streamlit |

## Requirements

For the combined review application:

- Node.js `>=22.13.0` and npm.
- Python 3.12 for the supplied setup script and backend environment.
- A machine capable of running the computer-vision models. Apple MPS is the
  currently validated acceleration path; CPU execution is substantially slower.
- Internet access for Roboflow inference and model downloads on first use.
- A private Roboflow API key supplied securely at runtime. Never commit it.

The annotation/training tool supports Python 3.10 or newer and selects CUDA,
Apple MPS, or CPU in that order. This does not mean that every component in the
combined UI has been validated on every CUDA configuration.

### Required model and runtime assets

Model files are intentionally excluded from Git. `npm run setup:models`
creates the Python environment and installs packages; it does not download the
project's trained checkpoints.

Place these general-foul checkpoints in `vair-gpu-ui/models/`, or point
`VAIR_CHECKPOINT_ROOT` to a directory containing them:

```text
image_foul_mlp_v408.pt
resnet18_mixed.pt
resnet_contact.pt
```

Full video inference also requires a complete handball runtime selected with
`VAIR_GRU_PROJECT_ROOT`. That directory must contain:

```text
combined_pipeline/
configs/mediapipe_features.yaml
configs/temporal_classifier.yaml
yolo11n.pt
models/pose_landmarker_full.task
artifacts/checkpoints/gru_fold0_best.pt
artifacts/checkpoints/gru_fold1_best.pt
artifacts/checkpoints/gru_fold2_best.pt
artifacts/checkpoints/gru_fold3_best.pt
artifacts/checkpoints/gru_fold4_best.pt
```

Important: the five GRU checkpoints and the deployed `combined_pipeline`
package are not currently included on `main`. A clean clone is therefore not a
complete inference deployment until those companion assets are supplied.

## Run the review application locally

Clone the repository and install the JavaScript and Python dependencies:

```bash
git clone https://github.com/sahayu123/AIreferee.git
cd AIreferee/vair-gpu-ui

npm ci
PYTHON_BIN="$(command -v python3.12)" npm run setup:models
```

Configure the source and model locations. Use absolute paths:

```bash
export VAIR_SOURCE_ROOT="/absolute/path/to/AIreferee"
export VAIR_CHECKPOINT_ROOT="/absolute/path/to/foul-checkpoints"
export VAIR_GRU_PROJECT_ROOT="/absolute/path/to/complete/handball-annotation-tool"
```

Start the backend and frontend together:

```bash
npm run dev:full
```

Open:

- Review UI: `http://localhost:3200`
- API health report: `http://localhost:8200/api/health`

The version currently committed to `main` asks for the private Roboflow key in
the interface and passes it to the local process for the active analysis. Do
not place a real key in source code, the README, or a committed `.env` file.

The FastAPI backend queues work in memory and serializes model execution, so
only one analysis runs at a time. Restarting the backend clears job state.

## Expose a temporary public demo with ngrok

Start the application first. In a second terminal, start the combined proxy:

```bash
cd AIreferee/vair-gpu-ui
npm run public:proxy
```

The proxy listens on port `8080`, sends `/api/*` to the FastAPI backend on
`8200`, and sends all other requests to the UI on `3200`. In a third terminal:

```bash
ngrok http 8080
```

Use the HTTPS URL printed by ngrok. Expose port `8080`, not `3200`, or artifact
and API requests will not reach the backend.

This is a development tunnel, not a production deployment. The application
does not currently include public-user authentication, rate limiting,
persistent job storage, or multi-worker GPU scheduling. Anyone with the URL
may be able to upload media and consume inference resources.

## Annotate handball data

Create the handball tool environment:

```bash
cd handball-annotation-tool
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start the candidate-mining and review interface:

```bash
streamlit run app.py
```

Start the separate not-handball sampler:

```bash
streamlit run negative_sampler_app.py
```

The miner detects players and the ball, tracks identities, estimates pose, and
proposes 41-frame ball-to-arm windows. A human still assigns the final
`handball`, `not_handball`, or `uncertain` label.

## Extract features and train the GRU

The commands below document the maintainer training stages; they are not yet a
clean-clone, end-to-end retraining command. First place the two Kaggle classes
under `handball-annotation-tool/dataset/` and supply a native-only
`artifacts/manifests/dataset.csv` matching that release. The current manifest
builder still assumes the older imported-negative dataset, and the manifest
committed on `main` is historical.

Once the correct manifest is in place, run the following from
`handball-annotation-tool/` with its virtual environment active:

```bash
python -m training.download_models
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

python -m training.features \
  --config configs/mediapipe_features.yaml \
  --verbose

python -m training.quality_report

for fold in 0 1 2 3 4; do
  python -m training.gru \
    --config configs/temporal_classifier.yaml \
    --fold "$fold"
done

python -m training.evaluate \
  --config configs/temporal_classifier.yaml \
  --summarize
```

Feature extraction is the expensive, resumable stage. Cached arrays are saved
under `artifacts/features/`; overlays, logs, checkpoints, and reports are saved
under their corresponding `artifacts/` directories and are ignored by Git.

Before extracting features, verify that every path in the manifest exists and
that its label and fold counts match the intended release. The current tracked
manifest includes legacy imported negatives and is not the exact manifest used
for the 286-example Kaggle release.

## Tests

Run the frontend build and rendered-interface test:

```bash
cd vair-gpu-ui
npm test
npm run lint
```

Run backend media-delivery tests:

```bash
./.venv/bin/python -m pytest backend/tests -q
```

Run annotation and training unit tests:

```bash
cd ../handball-annotation-tool
python -m pytest tests -q
```

## Known limitations

- A fresh clone needs private/untracked model checkpoints and the deployed
  handball integration package before full inference can run.
- The general-foul pipeline currently depends on Roboflow network requests for
  every video frame, which can dominate processing time.
- Generic COCO sports-ball detection can miss small, blurred, or occluded
  footballs in broadcast footage.
- Pose estimation can fail when players occupy very few pixels.
- The handball dataset is relatively small and varies by source footage.
- The current evaluation is cross-validation, not an untouched external test.
- The application runs one in-memory inference job at a time.
- Older notebooks and `backend/handball_project_detector.py` preserve previous
  experiments; the active combined server uses the five-fold GRU path.
- The repository does not currently include a software license, citation file,
  or production deployment configuration.

## Data and software rights

The AIreferee team created the dataset curation, temporal examples, labels,
frame extraction, metadata, and organization. Rights to the underlying match
and broadcast footage remain with their respective publishers and rights
holders.

No software license is currently included in this repository. Public access to
the source does not by itself grant permission to reuse, modify, or redistribute
the software or third-party footage. Consult the Kaggle dataset page for the
dataset's published terms.
