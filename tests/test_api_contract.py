from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

CLASSES = [
    "battery",
    "cardboard",
    "glass",
    "metal",
    "organic",
    "paper",
    "plastic",
    "textile",
    "trash",
]


class FakeEngine:
    """Deterministic engine used to test the API contract without ONNX."""

    available = True
    classes = CLASSES
    card = {
        "version": "1.0.0",
        "metrics": {"test_accuracy": None, "test_macro_f1": None},
    }
    version = "1.0.0"

    def predict(self, frame: np.ndarray, top_k: int) -> dict[str, Any]:
        scores = [
            {"class": "plastic", "confidence": 0.93},
            {"class": "glass", "confidence": 0.04},
            {"class": "metal", "confidence": 0.02},
        ][:top_k]
        return {
            "predicted_class": scores[0]["class"],
            "confidence": scores[0]["confidence"],
            "top_k": scores,
        }


@pytest.fixture
def image_bytes() -> bytes:
    frame = np.full((32, 32, 3), 128, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", frame)
    assert ok
    return buffer.tobytes()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        app.state.engine = None
        yield test_client


@pytest.fixture
def loaded_client():
    with TestClient(app) as test_client:
        app.state.engine = FakeEngine()
        yield test_client
        app.state.engine = None


def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["api_version"] == "v1"
    assert isinstance(body["model_loaded"], bool)


def test_classes_returns_nine_in_order(client: TestClient) -> None:
    response = client.get("/v1/classes")
    assert response.status_code == 200
    classes = response.json()["classes"]
    assert [item["name"] for item in classes] == CLASSES
    assert [item["index"] for item in classes] == list(range(9))


def test_model_metadata_shape(client: TestClient) -> None:
    response = client.get("/v1/model")
    assert response.status_code == 200
    body = response.json()
    assert body["num_classes"] == 9
    assert body["input_size"] == 224
    assert body["format"] == "onnx"
    assert "metrics" in body


def test_predict_valid_image_contract(
    loaded_client: TestClient, image_bytes: bytes
) -> None:
    response = loaded_client.post(
        "/v1/predict", files={"file": ("a.jpg", image_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in CLASSES
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model_version"] == "1.0.0"
    assert body["inference_ms"] >= 0.0
    confidences = [score["confidence"] for score in body["top_k"]]
    assert confidences == sorted(confidences, reverse=True)
    assert all("class" in score for score in body["top_k"])


def test_predict_invalid_image_returns_400(loaded_client: TestClient) -> None:
    response = loaded_client.post(
        "/v1/predict", files={"file": ("a.jpg", b"not an image", "image/jpeg")}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


def test_predict_too_large_returns_413(loaded_client: TestClient) -> None:
    payload = b"\xff\xd8" + b"0" * (5 * 1024 * 1024 + 1)
    response = loaded_client.post(
        "/v1/predict", files={"file": ("big.jpg", payload, "image/jpeg")}
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_predict_unsupported_media_returns_415(
    loaded_client: TestClient, image_bytes: bytes
) -> None:
    response = loaded_client.post(
        "/v1/predict", files={"file": ("a.gif", image_bytes, "image/gif")}
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_predict_invalid_top_k_returns_400(
    loaded_client: TestClient, image_bytes: bytes
) -> None:
    response = loaded_client.post(
        "/v1/predict?top_k=0", files={"file": ("a.jpg", image_bytes, "image/jpeg")}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_PARAMETER"


def test_predict_missing_file_returns_422(loaded_client: TestClient) -> None:
    response = loaded_client.post("/v1/predict")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_predict_without_model_returns_503(
    client: TestClient, image_bytes: bytes
) -> None:
    response = client.post(
        "/v1/predict", files={"file": ("a.jpg", image_bytes, "image/jpeg")}
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_NOT_LOADED"


def test_batch_keeps_alignment_and_per_file_errors(
    loaded_client: TestClient, image_bytes: bytes
) -> None:
    response = loaded_client.post(
        "/v1/predict/batch",
        files=[
            ("files", ("a.jpg", image_bytes, "image/jpeg")),
            ("files", ("b.jpg", b"broken", "image/jpeg")),
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 2
    assert body["results"][0]["filename"] == "a.jpg"
    assert body["results"][0]["predicted_class"] == "plastic"
    assert body["results"][1]["filename"] == "b.jpg"
    assert body["results"][1]["error"]["code"] == "INVALID_IMAGE"


def test_batch_too_many_files_returns_400(
    loaded_client: TestClient, image_bytes: bytes
) -> None:
    files = [("files", (f"{i}.jpg", image_bytes, "image/jpeg")) for i in range(17)]
    response = loaded_client.post("/v1/predict/batch", files=files)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_PARAMETER"
