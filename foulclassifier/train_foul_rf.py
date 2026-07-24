"""
Soccer Foul Random Forest Classifier

Pipeline:
  1. Fine-tune r3d_18 (a video CNN pretrained on Kinetics-400 action
     recognition) on our own labeled foul/no-foul clips: stem/layer1/layer2
     stay frozen (generic low-level motion features transfer fine and this
     roughly halves backward-pass compute on CPU), layer3/layer4/fc train
     with differential learning rates (small for the pretrained layers,
     larger for the fresh head), with data augmentation
     (random crop/flip/brightness/contrast) and dropout to fight overfitting
     on a small dataset. The training set is majority-class-undersampled to
     a 1:1 ratio before fine-tuning, since the raw data is ~8:1 Foul:No Foul
     -- both to keep the CNN's gradient signal from being dominated by the
     class it already gets right by default, and to cut per-epoch compute.
     Early stopping (on a dedicated, natural-ratio val split) keeps whichever
     epoch generalized best.
  2. Discard the classifier head, keep the now task-adapted backbone, and
     re-embed every clip with it.
  3. Train a RandomForest (with SMOTE-balanced training data and a
     cross-validated hyperparameter search) on top of those task-adapted
     embeddings, using a train/val/test split -- val drives backbone
     early-stopping and decision-threshold tuning, test is touched exactly
     once for the final reported numbers -- so predict_foul.py's random
     unseen-slice evaluation keeps working unchanged.

Usage: python train_foul_rf.py
"""

import copy
import json
import os
import random
import zipfile
import warnings
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as tvv
import torchvision.transforms.functional as TF

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import pil_to_tensor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────
EXTRACT_TO = Path(os.environ.get("FOUL_DATA_DIR", r"C:\Users\srawo\Downloads"))
ZIP_PATH   = Path(os.environ.get("FOUL_ZIP_PATH", str(EXTRACT_TO / "processed_frames.zip")))
DATA_ROOT  = EXTRACT_TO / "processed_frames"
OUT_DIR    = Path(os.environ.get("FOUL_OUT_DIR", str(Path(__file__).resolve().parent)))

BACKBONE_CKPT = OUT_DIR / "finetuned_backbone.pt"     # fully fine-tuned r3d_18 weights
EMBED_CACHE   = OUT_DIR / "embeddings_cache.npz"       # task-adapted 512-dim embeddings

ALL_SPLITS = ["train", "test", "valid"]
TRAIN_FRACTION = 0.75   # train on 75% of clips, test on the other 25%

MAX_FRAMES_PER_ACTION = 8
RANDOM_STATE = 42

FINETUNE_MAJORITY_RATIO = 2.0   # cap Foul clips in the fine-tune subset at this multiple
                                  # of the No Foul count -- more signal than a strict 1:1,
                                  # still far more balanced than the raw ~8:1 data

FINE_TUNE_MAX_EPOCHS = 15
FINE_TUNE_PATIENCE = 4       # stop if no val improvement for this many epochs
FINE_TUNE_BATCH_SIZE = 16
FINE_TUNE_LR = 1e-5          # small -- we're fine-tuning the pretrained backbone layers
FINE_TUNE_HEAD_LR = 1e-3     # larger -- the fc head starts from random init and needs to move faster
WEIGHT_DECAY = 1e-3
DROPOUT_P = 0.5
FREEZE_EARLY_LAYERS = True   # only fine-tune layer3/layer4/fc -- halves backward-pass
                              # compute on CPU and lowers overfitting risk on this small dataset

NUM_DATALOADER_WORKERS = min(6, os.cpu_count() or 1)   # overlap image decode/augment with CNN compute

KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD  = [0.22803, 0.22145, 0.216989]


# ── Get the raw data on disk ────────────────────────────────────────────────
def extract_if_needed():
    if DATA_ROOT.exists():
        return
    print(f"[->] Extracting {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(EXTRACT_TO)
    print("[OK] Extraction complete")


# ── Load the pretrained video backbone ──────────────────────────────────────
def build_backbone():
    weights = tvv.R3D_18_Weights.KINETICS400_V1
    model = tvv.r3d_18(weights=weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), weights.transforms(), device


# ── Turn one clip's frames into a raw (T, C, H, W) uint8 tensor ────────────
def load_clip_tensor(paths):
    frames = []
    for p in paths:
        try:
            frames.append(pil_to_tensor(Image.open(p).convert("RGB")))
        except Exception:
            pass
    if not frames:
        return None
    return torch.stack(frames, dim=0)


# ── List every clip's (label, frame paths) ──────────────────────────────────
def collect_clip_items(split: str):
    df = pd.read_csv(DATA_ROOT / split / "labels.csv")
    action_df = df.groupby("action_id").agg(label=("label", "first")).reset_index()

    items = []
    for _, row in action_df.iterrows():
        action_id = int(row["action_id"])
        label = int(row["label"])
        rows = df[df["action_id"] == action_id]
        paths = [DATA_ROOT / split / f"action_{action_id}" / r["clip"] / Path(r["image_path"]).name
                  for _, r in rows.iterrows()]
        if not paths:
            continue
        idx = np.linspace(0, len(paths) - 1, MAX_FRAMES_PER_ACTION, dtype=int)
        paths = [paths[i] for i in idx]
        items.append((label, paths))
    return items


# ── Random, temporally-consistent augmentation for one training clip ───────
def augment_clip(vid: torch.Tensor) -> torch.Tensor:
    """vid: (T, C, H, W) uint8. Same crop/flip/color-jitter applied to every
    frame in the clip, so motion between frames stays coherent."""
    T_, C_, H, W = vid.shape
    scale = random.uniform(0.7, 1.0)
    new_h, new_w = max(8, int(H * scale)), max(8, int(W * scale))
    top = random.randint(0, H - new_h)
    left = random.randint(0, W - new_w)
    vid = vid[..., top:top + new_h, left:left + new_w]
    vid = TF.resize(vid, [128, 171], antialias=False)
    vid = TF.center_crop(vid, [112, 112])
    if random.random() < 0.5:
        vid = torch.flip(vid, dims=[-1])
    vid = TF.convert_image_dtype(vid, torch.float)
    vid = TF.adjust_brightness(vid, random.uniform(0.8, 1.2))
    vid = TF.adjust_contrast(vid, random.uniform(0.8, 1.2))
    vid = TF.normalize(vid, mean=KINETICS_MEAN, std=KINETICS_STD)
    return vid.permute(1, 0, 2, 3)   # (T,C,H,W) -> (C,T,H,W)


# ── Keep DataLoader worker processes from each grabbing all 16 CPU threads
# for torch ops -- the heavy compute (CNN forward/backward) stays in the
# main process, workers just decode images and apply augmentation.
def _worker_init(_):
    torch.set_num_threads(1)


def make_loader(items, train: bool, eval_transform=None, shuffle=False):
    return DataLoader(
        ClipDataset(items, train=train, eval_transform=eval_transform),
        batch_size=FINE_TUNE_BATCH_SIZE, shuffle=shuffle,
        num_workers=NUM_DATALOADER_WORKERS, persistent_workers=NUM_DATALOADER_WORKERS > 0,
        worker_init_fn=_worker_init if NUM_DATALOADER_WORKERS > 0 else None,
    )


class ClipDataset(Dataset):
    def __init__(self, items, train: bool, eval_transform=None):
        self.items = items
        self.train = train
        self.eval_transform = eval_transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        label, paths = self.items[idx]
        vid = load_clip_tensor(paths)
        clip = augment_clip(vid) if self.train else self.eval_transform(vid)
        return clip, label


# ── The fine-tunable network: full backbone + dropout + fresh binary head ──
class FoulNet(nn.Module):
    def __init__(self, backbone, freeze_early: bool = FREEZE_EARLY_LAYERS):
        super().__init__()
        self.stem = backbone.stem
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.dropout = nn.Dropout(DROPOUT_P)
        self.fc = nn.Linear(512, 1)

        self.freeze_early = freeze_early
        if freeze_early:
            for m in (self.stem, self.layer1, self.layer2):
                for p in m.parameters():
                    p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_early and mode:
            # Keep frozen layers' BatchNorm running stats fixed too.
            self.stem.eval()
            self.layer1.eval()
            self.layer2.eval()
        return self

    def embed(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        return self.fc(self.dropout(self.embed(x))).squeeze(1)


def evaluate_net(net, items, transform, device, max_n=None):
    subset = items if max_n is None else items[:max_n]
    loader = make_loader(subset, train=False, eval_transform=transform)
    net.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = net(xb.to(device)).cpu()
            preds.append((torch.sigmoid(logits) >= 0.5).long())
            labels.append(yb)
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    return balanced_accuracy_score(labels, preds), preds, labels


# ── Fine-tune the whole backbone via real backprop, with early stopping ────
def fine_tune(net: FoulNet, train_items, val_items, transform, device):
    net.to(device)
    head_params = [p for n, p in net.named_parameters() if n.startswith("fc.") and p.requires_grad]
    backbone_params = [p for n, p in net.named_parameters() if not n.startswith("fc.") and p.requires_grad]
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": FINE_TUNE_LR},
        {"params": head_params, "lr": FINE_TUNE_HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)

    labels = np.array([label for label, _ in train_items])
    n = len(labels)
    n0, n1 = int((labels == 0).sum()), int((labels == 1).sum())
    class_weight = torch.tensor([n / (2.0 * n0), n / (2.0 * n1)], device=device)

    best_bal_acc = -1.0
    best_state = None
    epochs_since_improve = 0

    print(f"\n[->] Fine-tuning the full backbone for up to {FINE_TUNE_MAX_EPOCHS} epochs "
          f"(early stop after {FINE_TUNE_PATIENCE} epochs without improvement)...")
    loader = make_loader(train_items, train=True, shuffle=True)

    for epoch in range(1, FINE_TUNE_MAX_EPOCHS + 1):
        net.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            weight = class_weight[yb]

            optimizer.zero_grad()
            logits = net(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb.float(), weight=weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)

        bal_acc, _, _ = evaluate_net(net, val_items, transform, device)
        improved = bal_acc > best_bal_acc
        print(f"    epoch {epoch:2d}/{FINE_TUNE_MAX_EPOCHS}  loss={total_loss / n:.4f}  "
              f"val balanced_acc={bal_acc:.3f}{'  (best)' if improved else ''}")

        if improved:
            best_bal_acc = bal_acc
            best_state = copy.deepcopy(net.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= FINE_TUNE_PATIENCE:
                print(f"    [->] No improvement for {FINE_TUNE_PATIENCE} epochs, stopping early.")
                break

    net.load_state_dict(best_state)
    print(f"[OK] Restored best checkpoint (val balanced_acc={best_bal_acc:.3f})")
    return net


# ── Pick the decision threshold that maximizes balanced accuracy on val ────
def pick_threshold(clf, X_val, y_val):
    probs = clf.predict_proba(X_val)[:, 1]
    best_thr, best_score = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        preds = (probs >= thr).astype(int)
        score = balanced_accuracy_score(y_val, preds)
        if score > best_score:
            best_score, best_thr = score, float(thr)
    print(f"[OK] Picked decision threshold={best_thr:.2f} "
          f"(val balanced_acc={best_score:.4f} vs {balanced_accuracy_score(y_val, (probs >= 0.5).astype(int)):.4f} at 0.5)")
    return best_thr, best_score


# ── Evaluate the final classifier on unseen data ────────────────────────────
def evaluate(clf, X, y, split_name: str, threshold: float = 0.5):
    y_pred = (clf.predict_proba(X)[:, 1] >= threshold).astype(int)
    acc = accuracy_score(y, y_pred)
    bal_acc = balanced_accuracy_score(y, y_pred)
    print(f"\n=== {split_name} accuracy: {acc:.4f} ({acc*100:.1f}%) "
          f"| balanced accuracy: {bal_acc:.4f} ({bal_acc*100:.1f}%) | threshold={threshold:.2f} ===")
    print(classification_report(y, y_pred, target_names=["No Foul", "Foul"], zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(y, y_pred))
    return acc, bal_acc


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    extract_if_needed()

    backbone, transform, device = build_backbone()

    items = []
    for split in ALL_SPLITS:
        split_items = collect_clip_items(split)
        print(f"[OK] {split}: {len(split_items)} clips")
        items.extend(split_items)

    y_all_np = np.array([label for label, _ in items])
    print(f"[OK] Pooled {len(items)} clips total "
          f"(No Foul={np.sum(y_all_np==0)}, Foul={np.sum(y_all_np==1)})")

    # Three-way stratified split: train (fine-tune + RF fit) / val (backbone
    # early-stopping + RF threshold tuning) / test (touched once, at the end).
    idx_all = np.arange(len(items))
    train_idx, rest_idx = train_test_split(
        idx_all, train_size=TRAIN_FRACTION, stratify=y_all_np, random_state=RANDOM_STATE,
    )
    val_idx, test_idx = train_test_split(
        rest_idx, train_size=0.5, stratify=y_all_np[rest_idx], random_state=RANDOM_STATE,
    )
    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]
    test_items = [items[i] for i in test_idx]
    print(f"[OK] Train: {len(train_items)} clips | Val: {len(val_items)} clips | Test: {len(test_items)} clips")

    # Undersample the majority (Foul) class for fine-tuning only -- this is
    # the expensive per-epoch compute, and 8x more Foul than No Foul clips
    # both slows every epoch down and lets the CNN's gradient signal be
    # dominated by the class it already gets right by default. The RF stage
    # still trains on the full (SMOTE-balanced) train set below since that's
    # cheap and more real (non-synthetic) data there only helps.
    rng = np.random.RandomState(RANDOM_STATE)
    train_labels = y_all_np[train_idx]
    minority_class = min(np.unique(train_labels), key=lambda c: np.sum(train_labels == c))
    minority_count = int(np.sum(train_labels == minority_class))
    majority_cap = int(minority_count * FINETUNE_MAJORITY_RATIO)
    finetune_idx_parts = []
    for c in np.unique(train_labels):
        class_idx = train_idx[train_labels == c]
        cap = minority_count if c == minority_class else majority_cap
        if len(class_idx) > cap:
            class_idx = rng.choice(class_idx, size=cap, replace=False)
        finetune_idx_parts.append(class_idx)
    finetune_idx = np.concatenate(finetune_idx_parts)
    rng.shuffle(finetune_idx)
    finetune_items = [items[i] for i in finetune_idx]
    print(f"[OK] Fine-tune subset (majority capped at {FINETUNE_MAJORITY_RATIO:.0f}x minority): {len(finetune_items)} clips "
          f"(No Foul={int(np.sum(y_all_np[finetune_idx]==0))}, Foul={int(np.sum(y_all_np[finetune_idx]==1))})")

    # 1. Fine-tune the backbone (layer3/layer4/fc -- see FREEZE_EARLY_LAYERS)
    # on the class-balanced subset, with augmentation/dropout to fight
    # overfitting, and early stopping on held-out balanced accuracy.
    # Val (not test) drives early stopping so test stays untouched until the end.
    net = FoulNet(backbone)
    net = fine_tune(net, finetune_items, val_items, transform, device)
    torch.save(net.state_dict(), BACKBONE_CKPT)
    print(f"[OK] Fine-tuned backbone saved -> {BACKBONE_CKPT}")

    # 2. Re-embed every clip with the now task-adapted backbone (fc/dropout
    # discarded -- we just want the 512-dim representation).
    print("[->] Re-embedding all clips with the fine-tuned backbone...")
    net.eval()
    loader = make_loader(items, train=False, eval_transform=transform)
    embs = []
    with torch.no_grad():
        for xb, _ in loader:
            embs.append(net.embed(xb.to(device)).cpu())
    X_all = torch.cat(embs).numpy().astype(np.float32)
    np.savez(EMBED_CACHE, X=X_all, y=y_all_np)
    print(f"[OK] Task-adapted embeddings cached -> {EMBED_CACHE}")

    X_train, X_val, X_test = X_all[train_idx], X_all[val_idx], X_all[test_idx]
    y_train, y_val, y_test = y_all_np[train_idx], y_all_np[val_idx], y_all_np[test_idx]

    # 3. Balance the training set via SMOTE (synthetic interpolated minority
    # samples, not exact duplicates) for the RandomForest stage -- val/test
    # keep their natural ratio. SMOTE is fit *inside* the CV pipeline below
    # (resampled fresh per fold, training-portion-only) rather than once on
    # the whole train set up front -- doing it beforehand lets synthetic
    # points and the real neighbors they were interpolated from land in
    # different CV folds, which leaks information across folds and wildly
    # inflates the CV score.
    print(f"[OK] Train set before SMOTE: {len(y_train)} clips "
          f"(No Foul={np.sum(y_train==0)}, Foul={np.sum(y_train==1)})")

    # 4. Cross-validated hyperparameter search for the RandomForest on top of
    # the fine-tuned embeddings, to regularize against overfitting this
    # specific (small) dataset.
    print("\n[->] Searching RandomForest hyperparameters via 5-fold CV (SMOTE inside each fold)...")
    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("rf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)),  # outer CV parallelizes; avoid nested n_jobs=-1
    ])
    param_dist = {
        "rf__n_estimators": [200, 300, 400, 500],
        "rf__max_depth": [None, 10, 20, 30],
        "rf__min_samples_leaf": [1, 2, 4, 8],
        "rf__max_features": ["sqrt", "log2"],
    }
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, n_iter=15, scoring="balanced_accuracy",
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    clf = search.best_estimator_   # a SMOTE+RF pipeline; predict/predict_proba skip the SMOTE step automatically
    print(f"[OK] Best RF params: {search.best_params_} (cv balanced_acc={search.best_score_:.4f})")

    # 5. Tune the decision threshold on val, then report final numbers on
    # test -- the only time test is touched.
    threshold, val_bal_acc = pick_threshold(clf, X_val, y_val)
    test_acc, test_bal_acc = evaluate(clf, X_test, y_test, "TEST", threshold=threshold)

    # 6. Save the trained model + threshold + headline results + held-out set.
    joblib.dump({"clf": clf, "threshold": threshold}, OUT_DIR / "model.pkl")
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump({
            "test_accuracy": float(test_acc),
            "test_balanced_accuracy": float(test_bal_acc),
            "val_balanced_accuracy": float(val_bal_acc),
            "threshold": threshold,
            "best_rf_params": search.best_params_,
        }, f, indent=2)
    print(f"\n[OK] Model saved -> {OUT_DIR / 'model.pkl'}")

    joblib.dump({"X_test": X_test, "y_test": y_test}, OUT_DIR / "holdout_test.pkl")
    print(f"[OK] Held-out set saved -> {OUT_DIR / 'holdout_test.pkl'} ({len(y_test)} clips)")


if __name__ == "__main__":
    main()
