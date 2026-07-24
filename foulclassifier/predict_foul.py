"""
Run the trained foul classifier.

Two modes:

1. Evaluate on a random slice of unseen data (default, no args):
   python predict_foul.py
   Each run randomly samples a fresh chunk of the held-out clips that
   train_foul_rf.py never trained on, and reports accuracy on it. The exact
   clips (and therefore the exact accuracy) change run to run, but since
   they're all drawn from the same held-out pool the numbers should stay
   close together -- that consistency is what shows the model is precise.

   Optional: python predict_foul.py --fraction 0.4   (default 0.5, i.e. half
   of the held-out set each run)

2. Predict a single clip:
   python predict_foul.py "C:\\path\\to\\folder_of_frames"
   The folder should contain the frame images (.jpg/.png) for one action/clip.
"""

import os
import sys
import argparse
import joblib
import numpy as np
import torch
import torchvision.models.video as tvv

from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

OUT_DIR = Path(os.environ.get("FOUL_OUT_DIR", str(Path(__file__).resolve().parent)))
MODEL_PATH = OUT_DIR / "model.pkl"
HOLDOUT_PATH = OUT_DIR / "holdout_test.pkl"
BACKBONE_CKPT = OUT_DIR / "finetuned_backbone.pt"
MAX_FRAMES_PER_ACTION = 8   # must match train_foul_rf.py


def build_extractor():
    """r3d_18 with its weights swapped for the fully fine-tuned backbone
    train_foul_rf.py trained end-to-end on our own foul/no-foul clips."""
    weights = tvv.R3D_18_Weights.KINETICS400_V1
    model = tvv.r3d_18(weights=weights)
    if BACKBONE_CKPT.exists():
        state = torch.load(BACKBONE_CKPT, map_location="cpu")
        backbone_state = {k: v for k, v in state.items()
                           if k.startswith(("stem.", "layer1.", "layer2.", "layer3.", "layer4."))}
        model.load_state_dict(backbone_state, strict=False)
    model.fc = torch.nn.Identity()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), weights.transforms(), device


def embed_folder(folder: Path, model, transform, device):
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not paths:
        raise ValueError(f"No images found in {folder}")

    idx = np.linspace(0, len(paths) - 1, MAX_FRAMES_PER_ACTION, dtype=int)
    paths = [paths[i] for i in idx]
    frames = [pil_to_tensor(Image.open(p).convert("RGB")) for p in paths]

    with torch.no_grad():
        vid = torch.stack(frames, dim=0)               # (T, C, H, W) uint8
        clip = transform(vid).unsqueeze(0).to(device)    # (1, C, T, H, W)
        emb = model(clip).cpu().numpy()
    return emb.reshape(1, -1)   # (1, 512)


def predict_single_clip(folder: Path):
    if not MODEL_PATH.exists():
        print(f"[!] {MODEL_PATH} not found. Run train_foul_rf.py first.")
        sys.exit(1)

    bundle = joblib.load(MODEL_PATH)
    clf, threshold = bundle["clf"], bundle["threshold"]
    model, preprocess, device = build_extractor()

    X = embed_folder(folder, model, preprocess, device)
    prob = clf.predict_proba(X)[0][1]
    pred = int(prob >= threshold)

    label = "FOUL" if pred == 1 else "NO FOUL"
    print(f"Prediction: {label}  (foul probability: {prob:.2%}, threshold: {threshold:.2%})")


def evaluate_random_holdout_slice(fraction: float):
    if not MODEL_PATH.exists():
        print(f"[!] {MODEL_PATH} not found. Run train_foul_rf.py first.")
        sys.exit(1)
    if not HOLDOUT_PATH.exists():
        print(f"[!] {HOLDOUT_PATH} not found. Re-run train_foul_rf.py to regenerate it.")
        sys.exit(1)

    bundle = joblib.load(MODEL_PATH)
    clf, threshold = bundle["clf"], bundle["threshold"]
    holdout = joblib.load(HOLDOUT_PATH)
    X_test, y_test = holdout["X_test"], holdout["y_test"]

    # No fixed seed here on purpose -- a fresh random slice every run.
    n = len(y_test)
    sample_size = max(1, int(round(n * fraction)))
    idx = np.random.choice(n, size=sample_size, replace=False)
    X_sample, y_sample = X_test[idx], y_test[idx]

    y_pred = (clf.predict_proba(X_sample)[:, 1] >= threshold).astype(int)
    acc = accuracy_score(y_sample, y_pred)
    bal_acc = balanced_accuracy_score(y_sample, y_pred)

    print(f"[->] Evaluating on {sample_size}/{n} randomly sampled unseen clips "
          f"({fraction*100:.0f}% of the held-out set, threshold={threshold:.2f})")
    print(f"\n=== Accuracy on this sample: {acc:.4f} ({acc*100:.1f}%) "
          f"| balanced accuracy: {bal_acc:.4f} ({bal_acc*100:.1f}%) ===")
    print(classification_report(y_sample, y_pred, target_names=["No Foul", "Foul"], zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(y_sample, y_pred))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("folder", nargs="?", default=None,
                         help="Folder of frames for a single clip prediction. "
                              "Omit to instead evaluate on a random unseen slice.")
    parser.add_argument("--fraction", type=float, default=0.5,
                         help="Fraction of the held-out set to sample each run (default 0.5).")
    args = parser.parse_args()

    if args.folder is None:
        evaluate_random_holdout_slice(args.fraction)
    else:
        predict_single_clip(Path(args.folder))


if __name__ == "__main__":
    main()
