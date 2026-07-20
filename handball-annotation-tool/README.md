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
