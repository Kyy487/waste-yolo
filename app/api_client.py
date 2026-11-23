"""HTTP client used by the realtime camera demo."""

from typing import Any

import cv2
import numpy as np
import requests


class APIClientError(Exception):
    """Error returned by the inference API."""

    def __init__(self, code: str, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def encode_frame(frame_bgr: np.ndarray) -> tuple[bytes, str]:
    """Encode one BGR frame as JPEG bytes without writing to disk."""
    if (
        frame_bgr is None
        or not isinstance(frame_bgr, np.ndarray)
        or frame_bgr.size == 0
    ):
        raise ValueError("frame must be a non-empty numpy array")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame must have three channels")

    ok, buffer = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise ValueError("frame could not be encoded as JPEG")
    return buffer.tobytes(), "frame.jpg"


def map_error(response: requests.Response) -> APIClientError:
    """Map an HTTP error response to a safe client exception."""
    code = "INTERNAL_ERROR"
    message = "Kesalahan tak terduga."
    try:
        body = response.json()
        error = body.get("error", {})
        code = str(error.get("code", code))
        message = str(error.get("message", message))
    except ValueError:
        pass
    return APIClientError(code, message, response.status_code)


def predict_frame(
    session: requests.Session,
    base_url: str,
    frame_bgr: np.ndarray,
    top_k: int = 3,
    timeout: float = 2.0,
) -> dict[str, Any]:
    """Send one frame to the v1 predict endpoint and return its JSON payload."""
    payload, filename = encode_frame(frame_bgr)
    response = session.post(
        f"{base_url.rstrip('/')}/predict",
        files={"file": (filename, payload, "image/jpeg")},
        params={"top_k": top_k},
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise map_error(response)
    return response.json()
