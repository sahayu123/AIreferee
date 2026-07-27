"""Isolated SoccerNet PRTReID role-classification worker.

This module intentionally targets Python 3.9 and the dependency versions in
``requirements-prtreid.txt``.  The rest of the application communicates with it
over newline-delimited JSON so its old PyTorch stack never enters the main
Python 3.13 process.

Request (one JSON object per line)::

    {
      "request_id": "clip-1",
      "crops": [
        {"frame_path": "/absolute/frame.jpg", "bbox": [x1, y1, x2, y2]}
      ]
    }

Exactly one response is emitted to stdout for every non-empty input line.
Diagnostics and library output are redirected to stderr.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence, Tuple

PRTREID_MD5 = "9633825232bc89f23a94522c5561650e"
ROLE_NAMES: Tuple[str, ...] = (
    "ball",
    "goalkeeper",
    "other",
    "player",
    "referee",
)


class RequestError(ValueError):
    """A recoverable protocol or crop error."""


def _diagnostic(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - verifies the checksum published upstream
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checkpoint(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"PRTReID checkpoint not found: {path}")
    actual = _md5(path)
    if actual != PRTREID_MD5:
        raise RuntimeError(
            f"PRTReID checkpoint checksum mismatch: expected {PRTREID_MD5}, "
            f"found {actual}"
        )
    _diagnostic(f"Verified PRTReID checkpoint (MD5 {actual}): {path}")


def _request_id(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return None
    value = payload.get("request_id")
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    return value


def _parse_bbox(value: Any, crop_index: int) -> Tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise RequestError(f"crops[{crop_index}].bbox must contain four numbers")
    coordinates: List[float] = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise RequestError(f"crops[{crop_index}].bbox must contain four numbers")
        number = float(coordinate)
        if not math.isfinite(number):
            raise RequestError(f"crops[{crop_index}].bbox contains a non-finite number")
        coordinates.append(number)
    x1, y1, x2, y2 = coordinates
    if x2 <= x1 or y2 <= y1:
        raise RequestError(f"crops[{crop_index}].bbox has no positive area")
    return x1, y1, x2, y2


def _validate_payload(payload: Any) -> Tuple[Any, List[Dict[str, Any]]]:
    if not isinstance(payload, dict):
        raise RequestError("request must be a JSON object")
    if "request_id" not in payload:
        raise RequestError("request_id is required")
    request_id = payload["request_id"]
    if isinstance(request_id, bool) or not isinstance(request_id, (str, int)):
        raise RequestError("request_id must be a string or integer")
    crops = payload.get("crops")
    if not isinstance(crops, list):
        raise RequestError("crops must be a list")

    normalized: List[Dict[str, Any]] = []
    for index, crop in enumerate(crops):
        if not isinstance(crop, dict):
            raise RequestError(f"crops[{index}] must be an object")
        frame_path = crop.get("frame_path")
        if not isinstance(frame_path, str) or not frame_path.strip():
            raise RequestError(f"crops[{index}].frame_path must be a non-empty string")
        normalized.append(
            {
                "frame_path": frame_path,
                "bbox": _parse_bbox(crop.get("bbox"), index),
            }
        )
    return request_id, normalized


class PRTReIDWorker:
    """Load the official role head once and classify crop batches."""

    def __init__(
        self,
        checkpoint: Path,
        device: str = "cpu",
        batch_size: int = 16,
        torch_threads: int = 0,
    ) -> None:
        _verify_checkpoint(checkpoint)
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        # Several PRTReID modules print during import/model construction.  Keep
        # stdout reserved exclusively for the JSON-lines protocol.
        with redirect_stdout(sys.stderr):
            import cv2
            import numpy as np
            import torch
            from prtreid.tools.feature_extractor import FeatureExtractor

        if torch_threads > 0:
            torch.set_num_threads(torch_threads)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")

        _diagnostic(f"Loading official PRTReID model on {device}...")
        with redirect_stdout(sys.stderr):
            # torch.load executes pickle data.  It is only reached after the
            # exact checksum of SoccerNet's published checkpoint is verified.
            checkpoint_data = torch.load(str(checkpoint), map_location="cpu")
        if not isinstance(checkpoint_data, dict):
            raise RuntimeError("PRTReID checkpoint is not a dictionary")
        config = checkpoint_data.get("config")
        state_dict = checkpoint_data.get("state_dict")
        if config is None or not isinstance(state_dict, dict):
            raise RuntimeError("PRTReID checkpoint is missing config or state_dict")

        role_weights = [
            tensor
            for key, tensor in state_dict.items()
            if key.endswith("global_Role_classifier.classifier.weight")
        ]
        if len(role_weights) != 1 or tuple(role_weights[0].shape)[0] != len(ROLE_NAMES):
            raise RuntimeError(
                "PRTReID checkpoint does not contain the expected five-class "
                "global role classifier"
            )

        config = config.clone()
        config.defrost()
        config.model.pretrained = False
        with redirect_stdout(sys.stderr):
            extractor = FeatureExtractor(
                config,
                model_path=str(checkpoint),
                image_size=(256, 128),
                device=device,
                verbose=False,
            )

        loaded_role_weights = [
            tensor
            for key, tensor in extractor.model.state_dict().items()
            if key.endswith("global_Role_classifier.classifier.weight")
        ]
        if (
            len(loaded_role_weights) != 1
            or tuple(loaded_role_weights[0].shape) != tuple(role_weights[0].shape)
            or not torch.equal(
                loaded_role_weights[0].detach().cpu(),
                role_weights[0].detach().cpu(),
            )
        ):
            raise RuntimeError(
                "The checkpoint's five-class global role weights were not loaded "
                "into PRTReID"
            )

        self.batch_size = batch_size
        self.cv2 = cv2
        self.np = np
        self.torch = torch
        self.extractor = extractor
        self._validate_model_output()
        _diagnostic("PRTReID worker ready.")

    def _validate_model_output(self) -> None:
        # This catches incompatible PRTReID revisions at startup instead of in
        # the middle of a long tracking run.
        probe = self.np.zeros((256, 128, 3), dtype=self.np.uint8)
        with redirect_stdout(sys.stderr):
            output = self.extractor([probe])
        self._role_logits(output, expected_batch=1)

    def _role_logits(self, output: Any, expected_batch: int) -> Any:
        try:
            role_heads = output[4]
            logits = role_heads["globl"]
        except (IndexError, KeyError, TypeError) as exc:
            raise RuntimeError("PRTReID output has no global role head") from exc
        if not self.torch.is_tensor(logits) or logits.ndim != 2:
            raise RuntimeError("PRTReID global role output is not a rank-2 tensor")
        if tuple(logits.shape) != (expected_batch, len(ROLE_NAMES)):
            raise RuntimeError(
                "PRTReID global role output has shape "
                f"{tuple(logits.shape)}; expected ({expected_batch}, {len(ROLE_NAMES)})"
            )
        return logits

    def _load_crop(
        self,
        crop: Dict[str, Any],
        crop_index: int,
    ) -> Tuple[Any, Dict[str, Any]]:
        frame_path = Path(crop["frame_path"]).expanduser()
        if not frame_path.is_file():
            raise RequestError(f"crops[{crop_index}] frame not found: {frame_path}")
        frame = self.cv2.imread(str(frame_path), self.cv2.IMREAD_COLOR)
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise RequestError(f"crops[{crop_index}] is not a readable color image")

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = crop["bbox"]
        left = max(0, min(width, math.floor(x1)))
        top = max(0, min(height, math.floor(y1)))
        right = max(0, min(width, math.ceil(x2)))
        bottom = max(0, min(height, math.ceil(y2)))
        if right <= left or bottom <= top:
            raise RequestError(
                f"crops[{crop_index}].bbox is empty after clamping to "
                f"{width}x{height}"
            )

        image = frame[top:bottom, left:right]
        image = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)
        image = self.np.ascontiguousarray(image, dtype=self.np.uint8)
        metadata = {
            "frame_path": str(frame_path),
            "bbox": [left, top, right, bottom],
        }
        return image, metadata

    def predict(self, crops: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        images: List[Any] = []
        metadata: List[Dict[str, Any]] = []
        for index, crop in enumerate(crops):
            image, item_metadata = self._load_crop(crop, index)
            images.append(image)
            metadata.append(item_metadata)

        predictions: List[Dict[str, Any]] = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            with redirect_stdout(sys.stderr):
                output = self.extractor(batch)
            logits = self._role_logits(output, expected_batch=len(batch))
            probabilities = self.torch.softmax(logits, dim=1).detach().cpu()
            top_values, top_indices = self.torch.topk(probabilities, k=2, dim=1)

            for row in range(len(batch)):
                values = [float(value) for value in probabilities[row].tolist()]
                predicted_index = int(top_indices[row, 0].item())
                confidence = float(top_values[row, 0].item())
                runner_up = float(top_values[row, 1].item())
                item = dict(metadata[start + row])
                item.update(
                    {
                        "role_probabilities": dict(zip(ROLE_NAMES, values)),
                        "predicted_role": ROLE_NAMES[predicted_index],
                        "confidence": confidence,
                        "margin": confidence - runner_up,
                    }
                )
                predictions.append(item)
        return predictions


def _response(payload: Dict[str, Any]) -> None:
    print(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        flush=True,
    )


def serve(worker: PRTReIDWorker) -> None:
    for raw_line in sys.stdin:
        if not raw_line.strip():
            _response({"request_id": None, "ok": False, "error": "empty request"})
            continue

        payload: Any = None
        try:
            payload = json.loads(raw_line)
            request_id, crops = _validate_payload(payload)
            predictions = worker.predict(crops)
            _response(
                {
                    "request_id": request_id,
                    "ok": True,
                    "predictions": predictions,
                }
            )
        except (RequestError, json.JSONDecodeError) as exc:
            request_id = _request_id(payload)
            _diagnostic(f"Rejected request {request_id!r}: {exc}")
            _response({"request_id": request_id, "ok": False, "error": str(exc)})
        except Exception as exc:  # keep the persistent worker usable after one bad request
            request_id = _request_id(payload)
            _diagnostic(f"Request {request_id!r} failed: {type(exc).__name__}: {exc}")
            _response(
                {
                    "request_id": request_id,
                    "ok": False,
                    "error": f"model inference failed: {type(exc).__name__}: {exc}",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve official SoccerNet PRTReID role inference over JSON lines."
    )
    parser.add_argument(
        "--checkpoint",
        default="models/prtreid-soccernet-baseline.pth.tar",
        type=Path,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="Limit PyTorch CPU threads; zero keeps PyTorch's default.",
    )
    args = parser.parse_args()

    if sys.version_info[:2] != (3, 9):
        raise RuntimeError(
            "The isolated PRTReID worker requires Python 3.9; "
            f"found {sys.version.split()[0]}"
        )
    worker = PRTReIDWorker(
        args.checkpoint.expanduser().resolve(),
        device=args.device,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
    )
    serve(worker)


if __name__ == "__main__":
    main()
