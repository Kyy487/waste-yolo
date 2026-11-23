from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import requests

from app.api_client import APIClientError, encode_frame, map_error, predict_frame


def test_encode_frame_returns_jpeg_bytes() -> None:
    frame = np.full((16, 16, 3), 64, dtype=np.uint8)
    payload, filename = encode_frame(frame)
    assert filename == "frame.jpg"
    assert isinstance(payload, bytes)
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None


def test_encode_frame_rejects_invalid_frame() -> None:
    with pytest.raises(ValueError):
        encode_frame(np.zeros((0, 0, 3), dtype=np.uint8))


def test_predict_frame_success_calls_api() -> None:
    session = Mock(spec=requests.Session)
    response = Mock()
    response.status_code = 200
    response.json.return_value = {"predicted_class": "paper", "confidence": 0.88}
    session.post.return_value = response

    frame = np.full((16, 16, 3), 32, dtype=np.uint8)
    result = predict_frame(
        session, "http://localhost:8000/v1", frame, top_k=3, timeout=2.0
    )

    assert result["predicted_class"] == "paper"
    session.post.assert_called_once()
    args, kwargs = session.post.call_args
    assert args[0] == "http://localhost:8000/v1/predict"
    assert kwargs["params"] == {"top_k": 3}
    assert kwargs["timeout"] == 2.0
    assert "file" in kwargs["files"]


def test_predict_frame_maps_api_error() -> None:
    session = Mock(spec=requests.Session)
    response = Mock()
    response.status_code = 503
    response.json.return_value = {
        "error": {"code": "MODEL_NOT_LOADED", "message": "Model belum siap."}
    }
    session.post.return_value = response

    frame = np.full((16, 16, 3), 32, dtype=np.uint8)
    with pytest.raises(APIClientError) as excinfo:
        predict_frame(session, "http://localhost:8000/v1", frame)
    assert excinfo.value.code == "MODEL_NOT_LOADED"
    assert excinfo.value.status_code == 503


def test_map_error_handles_non_json_body() -> None:
    response = Mock()
    response.status_code = 500
    response.json.side_effect = ValueError("no json")
    error = map_error(response)
    assert error.code == "INTERNAL_ERROR"
    assert error.status_code == 500
