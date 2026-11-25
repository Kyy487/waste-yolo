import asyncio
import json
from typing import Any

import cv2
import numpy as np
import pytest

from src.api.main import create_app, ensure_probabilities

EXPECTED_CLASSES = [
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


class FakePredictor:
    """Deterministic predictor returning fixed class logits/probabilities."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = np.asarray(scores, dtype=np.float64)

    def predict_proba(self, image_bgr: np.ndarray) -> np.ndarray:
        assert image_bgr.ndim == 3
        return self._scores


def encode_dummy_image(ext: str = ".jpg") -> bytes:
    """Encode a small solid BGR image into valid JPEG or PNG bytes."""
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    ok, encoded = cv2.imencode(ext, image)
    assert ok
    return encoded.tobytes()


def build_multipart_body(
    files: list[tuple[str, str, bytes, str]], boundary: str = "----WasteYoloBoundary"
) -> tuple[bytes, str]:
    """Build a multipart/form-data payload without external HTTP client libs."""
    chunks: list[bytes] = []
    for field_name, filename, content, content_type in files:
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(
            (
                f'Content-Disposition: form-data; name="{field_name}"; '
                f'filename="{filename}"\r\n'
            ).encode()
        )
        chunks.append(f"Content-Type: {content_type}\r\n\r\n".encode())
        chunks.append(content)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def asgi_request(
    app: Any,
    method: str,
    path: str,
    query_string: str = "",
    body: bytes = b"",
    content_type: str | None = None,
) -> tuple[int, dict[str, Any]]:
    """Invoke a FastAPI ASGI app in-process and return (status_code, json_body)."""

    async def _call() -> tuple[int, dict[str, Any]]:
        headers: list[tuple[bytes, bytes]] = [(b"host", b"testserver")]
        if content_type is not None:
            headers.append((b"content-type", content_type.encode("latin-1")))
            headers.append((b"content-length", str(len(body)).encode("latin-1")))

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": query_string.encode("ascii"),
            "headers": headers,
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
        }

        sent_body = False

        async def receive() -> dict[str, Any]:
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        status_code = 500
        response_chunks: list[bytes] = []

        async def send(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
            elif message["type"] == "http.response.body":
                response_chunks.append(message.get("body", b""))

        await app(scope, receive, send)
        payload = json.loads(b"".join(response_chunks).decode("utf-8"))
        return status_code, payload

    return asyncio.run(_call())


def test_health_endpoint_reports_model_status() -> None:
    """GET /v1/health returns 200 with model_loaded matching state."""
    loaded_app = create_app(predictor=FakePredictor([0.1] * 9))
    status, body = asgi_request(loaded_app, "GET", "/v1/health")
    assert status == 200
    assert body == {"status": "ok", "model_loaded": True, "api_version": "v1"}

    unloaded_app = create_app(predictor=None)
    status_unloaded, body_unloaded = asgi_request(unloaded_app, "GET", "/v1/health")
    assert status_unloaded == 200
    assert body_unloaded["model_loaded"] is False


def test_classes_endpoint_matches_contract_order() -> None:
    """GET /v1/classes returns exactly 9 classes in fixed alphabetical order."""
    app = create_app(predictor=FakePredictor([0.1] * 9))
    status, body = asgi_request(app, "GET", "/v1/classes")
    assert status == 200
    assert body["classes"] == [
        {"index": idx, "name": name} for idx, name in enumerate(EXPECTED_CLASSES)
    ]


def test_model_endpoint_returns_metadata_schema() -> None:
    """GET /v1/model returns active model metadata and null metrics before eval."""
    app = create_app(predictor=FakePredictor([0.1] * 9))
    status, body = asgi_request(app, "GET", "/v1/model")
    assert status == 200
    assert body["input_size"] == 224
    assert body["num_classes"] == 9
    assert body["format"] == "onnx"
    assert "test_accuracy" in body["metrics"]
    assert "test_macro_f1" in body["metrics"]


def test_predict_valid_image_returns_contract_response() -> None:
    """POST /v1/predict returns descending top_k scores summing to ~1 over all classes."""
    probs = [0.01, 0.01, 0.04, 0.02, 0.01, 0.01, 0.88, 0.01, 0.01]
    app = create_app(predictor=FakePredictor(probs))

    body_bytes, ct = build_multipart_body(
        [("file", "bottle.jpg", encode_dummy_image(".jpg"), "image/jpeg")]
    )
    status, payload = asgi_request(
        app, "POST", "/v1/predict", body=body_bytes, content_type=ct
    )
    assert status == 200
    assert payload["predicted_class"] == "plastic"
    assert payload["confidence"] == pytest.approx(0.88, abs=1e-4)
    assert len(payload["top_k"]) == 3
    assert [item["class"] for item in payload["top_k"]] == [
        "plastic",
        "glass",
        "metal",
    ]
    assert payload["inference_ms"] >= 0.0
    assert isinstance(payload["model_version"], str)

    status_all, payload_all = asgi_request(
        app,
        "POST",
        "/v1/predict",
        query_string="top_k=9",
        body=body_bytes,
        content_type=ct,
    )
    assert status_all == 200
    assert sum(item["confidence"] for item in payload_all["top_k"]) == pytest.approx(
        1.0, abs=1e-3
    )


def test_predict_error_contracts() -> None:
    """Invalid image, unsupported format, oversize file, bad top_k, and missing field."""
    app = create_app(predictor=FakePredictor([1.0] + [0.0] * 8))

    # 400 INVALID_IMAGE
    corrupt_body, corrupt_ct = build_multipart_body(
        [("file", "broken.jpg", b"not-an-image", "image/jpeg")]
    )
    status, payload = asgi_request(
        app, "POST", "/v1/predict", body=corrupt_body, content_type=corrupt_ct
    )
    assert status == 400
    assert payload["error"]["code"] == "INVALID_IMAGE"

    # 415 UNSUPPORTED_MEDIA_TYPE
    gif_body, gif_ct = build_multipart_body(
        [("file", "anim.gif", b"GIF89a", "image/gif")]
    )
    status, payload = asgi_request(
        app, "POST", "/v1/predict", body=gif_body, content_type=gif_ct
    )
    assert status == 415
    assert payload["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"

    # 413 FILE_TOO_LARGE (> 5 MB)
    huge_body, huge_ct = build_multipart_body(
        [("file", "big.jpg", b"\x00" * (5 * 1024 * 1024 + 1), "image/jpeg")]
    )
    status, payload = asgi_request(
        app, "POST", "/v1/predict", body=huge_body, content_type=huge_ct
    )
    assert status == 413
    assert payload["error"]["code"] == "FILE_TOO_LARGE"

    # 400 INVALID_PARAMETER (top_k out of 1..9)
    valid_body, valid_ct = build_multipart_body(
        [("file", "ok.jpg", encode_dummy_image(".jpg"), "image/jpeg")]
    )
    status, payload = asgi_request(
        app,
        "POST",
        "/v1/predict",
        query_string="top_k=10",
        body=valid_body,
        content_type=valid_ct,
    )
    assert status == 400
    assert payload["error"]["code"] == "INVALID_PARAMETER"

    # 422 VALIDATION_ERROR (missing required file field)
    status, payload = asgi_request(app, "POST", "/v1/predict")
    assert status == 422
    assert payload["error"]["code"] == "VALIDATION_ERROR"

    # 503 MODEL_NOT_LOADED
    unloaded_app = create_app(predictor=None)
    status, payload = asgi_request(
        unloaded_app, "POST", "/v1/predict", body=valid_body, content_type=valid_ct
    )
    assert status == 503
    assert payload["error"]["code"] == "MODEL_NOT_LOADED"


def test_predict_batch_partial_failure_and_limit() -> None:
    """Batch returns per-file errors in place and rejects >16 files."""
    app = create_app(
        predictor=FakePredictor([0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0])
    )
    valid_jpg = encode_dummy_image(".jpg")
    valid_png = encode_dummy_image(".png")

    batch_body, batch_ct = build_multipart_body(
        [
            ("files", "a.jpg", valid_jpg, "image/jpeg"),
            ("files", "bad.jpg", b"corrupt", "image/jpeg"),
            ("files", "c.png", valid_png, "image/png"),
        ]
    )
    status, payload = asgi_request(
        app,
        "POST",
        "/v1/predict/batch",
        query_string="top_k=1",
        body=batch_body,
        content_type=batch_ct,
    )
    assert status == 200
    assert len(payload["results"]) == 3
    assert payload["results"][0]["filename"] == "a.jpg"
    assert payload["results"][0]["predicted_class"] == "paper"
    assert payload["results"][1]["filename"] == "bad.jpg"
    assert payload["results"][1]["error"]["code"] == "INVALID_IMAGE"
    assert payload["results"][2]["filename"] == "c.png"
    assert payload["results"][2]["predicted_class"] == "paper"

    # Exceeding batch_limit (17 files) returns 400 INVALID_PARAMETER
    too_many_body, too_many_ct = build_multipart_body(
        [("files", f"f{i}.jpg", valid_jpg, "image/jpeg") for i in range(17)]
    )
    status_many, payload_many = asgi_request(
        app, "POST", "/v1/predict/batch", body=too_many_body, content_type=too_many_ct
    )
    assert status_many == 400
    assert payload_many["error"]["code"] == "INVALID_PARAMETER"


def test_ensure_probabilities_handles_logits_and_probabilities() -> None:
    """Both raw logits and probability vectors normalize to sum == 1."""
    from_logits = ensure_probabilities(np.array([2.0, 1.0, 0.0]))
    assert from_logits.sum() == pytest.approx(1.0)
    assert from_logits[0] > from_logits[1] > from_logits[2]

    from_probs = ensure_probabilities(np.array([0.7, 0.2, 0.1]))
    assert from_probs == pytest.approx(np.array([0.7, 0.2, 0.1]))
