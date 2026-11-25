from typing import Any

import pytest
import requests

from app import streamlit_app
from app.streamlit_app import (
    APIClientError,
    build_endpoint_url,
    default_api_base_url,
    fetch_health,
    fetch_model_info,
    format_metric_value,
    predict_image,
)


class FakeResponse:
    """Minimal stand-in for requests.Response used in unit tests."""

    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def test_url_builders_and_metric_formatting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Base URL reads config or env var, endpoint joins cleanly, null stays null."""
    monkeypatch.delenv("WASTE_API_BASE_URL", raising=False)
    cfg = {"api": {"host": "127.0.0.1", "port": 8000}}
    assert default_api_base_url(cfg) == "http://127.0.0.1:8000"
    assert (
        build_endpoint_url("http://127.0.0.1:8000/", "/v1/", "predict")
        == "http://127.0.0.1:8000/v1/predict"
    )

    monkeypatch.setenv("WASTE_API_BASE_URL", "http://localhost:9000/")
    assert default_api_base_url(cfg) == "http://localhost:9000"

    assert format_metric_value(None) == "Belum dievaluasi (null)"
    assert format_metric_value(0.91234) == "0.9123"


def test_fetch_health_and_model_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Health and model info return parsed JSON payloads on HTTP 200."""

    def fake_get(url: str, timeout: float) -> FakeResponse:
        if url.endswith("/v1/health"):
            return FakeResponse(
                200, {"status": "ok", "model_loaded": True, "api_version": "v1"}
            )
        return FakeResponse(
            200,
            {
                "name": "waste-yolo-cls",
                "version": "1.0.0",
                "metrics": {"test_accuracy": None, "test_macro_f1": None},
            },
        )

    monkeypatch.setattr(streamlit_app.requests, "get", fake_get)
    health = fetch_health("http://127.0.0.1:8000", "/v1")
    info = fetch_model_info("http://127.0.0.1:8000", "/v1")
    assert health["model_loaded"] is True
    assert info["name"] == "waste-yolo-cls"


def test_predict_image_happy_path_and_api_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Predict returns API JSON on 200 and raises APIClientError on error/offline."""

    def fake_post_ok(
        url: str, params: dict[str, Any], files: dict[str, Any], timeout: float
    ) -> FakeResponse:
        assert url == "http://127.0.0.1:8000/v1/predict"
        assert params == {"top_k": 3}
        assert "file" in files
        return FakeResponse(
            200,
            {
                "predicted_class": "plastic",
                "confidence": 0.93,
                "top_k": [{"class": "plastic", "confidence": 0.93}],
                "inference_ms": 25.0,
                "model_version": "1.0.0",
            },
        )

    monkeypatch.setattr(streamlit_app.requests, "post", fake_post_ok)
    result = predict_image(
        "http://127.0.0.1:8000", "/v1", "a.jpg", b"jpgbytes", "image/jpeg", top_k=3
    )
    assert result["predicted_class"] == "plastic"

    def fake_post_503(*args: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse(
            503,
            {
                "error": {
                    "code": "MODEL_NOT_LOADED",
                    "message": "Model belum siap.",
                    "details": None,
                }
            },
        )

    monkeypatch.setattr(streamlit_app.requests, "post", fake_post_503)
    with pytest.raises(APIClientError) as exc_info:
        predict_image(
            "http://127.0.0.1:8000", "/v1", "a.jpg", b"jpgbytes", "image/jpeg", top_k=3
        )
    assert exc_info.value.code == "MODEL_NOT_LOADED"

    def fake_post_offline(*args: Any, **kwargs: Any) -> FakeResponse:
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(streamlit_app.requests, "post", fake_post_offline)
    with pytest.raises(APIClientError):
        predict_image(
            "http://127.0.0.1:8000", "/v1", "a.jpg", b"jpgbytes", "image/jpeg", top_k=3
        )
