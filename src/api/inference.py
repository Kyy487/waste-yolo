"""ONNX inference helpers for the waste classifier."""

import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import onnxruntime as ort

from src.utils.io import read_yaml


def preprocess_frame(
    frame_bgr: np.ndarray, image_size: int = 224, padding_color: int = 0
) -> np.ndarray:
    """Resize with aspect-ratio padding, convert to RGB, and return NCHW float input."""
    if (
        frame_bgr is None
        or not isinstance(frame_bgr, np.ndarray)
        or frame_bgr.size == 0
    ):
        raise ValueError("frame must be a non-empty numpy array")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame must have three channels")

    height, width = frame_bgr.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("frame dimensions must be positive")

    scale = min(image_size / width, image_size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        frame_bgr,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.full((image_size, image_size, 3), padding_color, dtype=np.uint8)
    top = (image_size - resized_height) // 2
    left = (image_size - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized

    rgb = canvas[:, :, ::-1]
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(chw, axis=0)


def probabilities_from_output(values: Any, num_classes: int) -> np.ndarray:
    """Convert raw ONNX output to a normalized probability vector."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    raw = np.asarray(values, dtype=np.float32).reshape(-1)
    if raw.size != num_classes:
        raise ValueError(f"model output has {raw.size} values, expected {num_classes}")
    if not np.all(np.isfinite(raw)):
        raise ValueError("model output contains non-finite values")

    if np.all(raw >= 0) and np.all(raw <= 1) and abs(float(raw.sum()) - 1.0) <= 1e-3:
        return raw

    shifted = raw - float(np.max(raw))
    exponentials = np.exp(shifted)
    return exponentials / float(np.sum(exponentials))


def top_k_from_probabilities(
    probabilities: np.ndarray, classes: list[str], top_k: int
) -> dict[str, Any]:
    """Build the v1 prediction payload from a probability vector."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if len(classes) == 0:
        raise ValueError("classes must not be empty")

    count = min(top_k, len(classes))
    indices = np.argsort(probabilities)[::-1][:count]
    scores = [
        {
            "class": classes[int(index)],
            "confidence": float(probabilities[index]),
        }
        for index in indices
    ]
    return {
        "predicted_class": scores[0]["class"],
        "confidence": scores[0]["confidence"],
        "top_k": scores,
    }


class ONNXInference:
    """Small ONNX Runtime wrapper for YOLO-cls inference."""

    def __init__(
        self,
        session: ort.InferenceSession,
        input_name: str,
        output_name: str,
        classes: list[str],
        card: dict[str, Any],
        image_size: int,
        padding_color: int = 0,
    ) -> None:
        self.session = session
        self.input_name = input_name
        self.output_name = output_name
        self.classes = classes
        self.card = card
        self.image_size = image_size
        self.padding_color = padding_color
        self.available = True

    @property
    def version(self) -> str:
        """Return the active model version."""
        return str(self.card.get("version", "1.0.0"))

    @classmethod
    def load(cls, config: Mapping[str, Any], root: Path) -> "ONNXInference":
        """Load the ONNX model and class metadata from configuration."""
        model_path = root / str(config["model"]["path"])
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        card_path = root / str(config["model"]["card"])
        card: dict[str, Any] = {}
        if card_path.is_file():
            card = json.loads(card_path.read_text(encoding="utf-8"))

        classes_source = config["model"].get("classes_source")
        fallback_classes: list[str] = []
        if classes_source:
            fallback_classes = [
                str(name)
                for name in read_yaml(root / str(classes_source)).get("classes", [])
            ]
        classes = [str(name) for name in (card.get("classes") or fallback_classes)]
        if not classes:
            raise ValueError("no classes available for inference")

        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        return cls(
            session=session,
            input_name=session.get_inputs()[0].name,
            output_name=session.get_outputs()[0].name,
            classes=classes,
            card=card,
            image_size=int(config["model"]["input_size"]),
            padding_color=int(config["inference"].get("padding_color", 0)),
        )

    def predict(self, frame_bgr: np.ndarray, top_k: int) -> dict[str, Any]:
        """Predict one BGR frame and return the v1 payload without timing fields."""
        if not self.available:
            raise RuntimeError("model is not available")

        tensor = preprocess_frame(frame_bgr, self.image_size, self.padding_color)
        outputs = self.session.run([self.output_name], {self.input_name: tensor})[0]
        probabilities = probabilities_from_output(outputs[0], len(self.classes))
        return top_k_from_probabilities(probabilities, self.classes, top_k)
