"""FastAPI v1 inference service for 9-class waste classification."""

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import time
from typing import Any

import cv2
from fastapi import APIRouter, FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import numpy as np
import onnxruntime as ort

from src.api.schemas import (
    BatchErrorItem,
    BatchPredictionItem,
    BatchPredictResponse,
    ClassesResponse,
    ClassItem,
    ClassScore,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    ModelMetrics,
    PredictResponse,
)
from src.data.prepare_data import preprocess_image
from src.utils.io import read_yaml
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_CONTENT_TYPES = frozenset({"image/jpeg", "image/jpg", "image/png"})


class APIError(Exception):
    """Domain error mapped to the uniform API error envelope."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


def ensure_probabilities(raw_scores: np.ndarray) -> np.ndarray:
    """Convert raw logits or probabilities into a valid probability vector."""
    vector = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        raise ValueError("Model returned an empty prediction vector")

    total = float(vector.sum())
    if np.all(vector >= 0.0) and np.all(vector <= 1.0) and abs(total - 1.0) <= 1e-3:
        probs = vector / total if total > 0.0 else vector
    else:
        shifted = vector - float(np.max(vector))
        exps = np.exp(shifted)
        probs = exps / float(exps.sum())
    return probs.astype(np.float64)


class OnnxClassifier:
    """ONNX Runtime wrapper matching the training preprocessing pipeline."""

    def __init__(
        self,
        model_path: Path,
        preprocess_config: Mapping[str, Any],
        num_classes: int,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        self.model_path = model_path
        self.preprocess_config = dict(preprocess_config)
        self.num_classes = num_classes
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict_proba(self, image_bgr: np.ndarray) -> np.ndarray:
        """Preprocess one BGR image and return class probabilities."""
        rgb = preprocess_image(image_bgr, self.preprocess_config)
        tensor = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
        batch = np.expand_dims(tensor, axis=0)
        outputs = self.session.run(None, {self.input_name: batch})
        probs = ensure_probabilities(np.asarray(outputs[0]))
        if probs.size != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} class scores, got {probs.size}"
            )
        return probs


def load_model_metadata(
    api_config: Mapping[str, Any],
    data_config: Mapping[str, Any],
    root: Path,
) -> ModelInfoResponse:
    """Load model metadata from model_card.json with safe null-metric defaults."""
    model_cfg = api_config["model"]
    card_path = root / str(model_cfg["card"])
    card: dict[str, Any] = {}
    if card_path.is_file():
        try:
            loaded = json.loads(card_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                card = loaded
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Failed to read model card %s: %s", card_path, exc)

    raw_metrics = card.get("metrics") if isinstance(card.get("metrics"), dict) else {}
    return ModelInfoResponse(
        name=str(card.get("name", "waste-yolo-cls")),
        version=str(card.get("version", "1.0.0")),
        architecture=str(card.get("architecture", "yolov8n-cls")),
        input_size=int(card.get("input_size", model_cfg["input_size"])),
        num_classes=len(data_config["classes"]),
        format=str(model_cfg.get("format", "onnx")),
        trained_on=str(card.get("dataset", data_config["dataset"]["name"])),
        metrics=ModelMetrics(
            test_accuracy=raw_metrics.get("test_accuracy"),
            test_macro_f1=raw_metrics.get("test_macro_f1"),
        ),
    )


def validate_top_k(top_k: int | None, inference_cfg: Mapping[str, Any]) -> int:
    """Validate and resolve the top_k query parameter."""
    default_k = int(inference_cfg["top_k_default"])
    min_k = int(inference_cfg["top_k_min"])
    max_k = int(inference_cfg["top_k_max"])
    resolved = default_k if top_k is None else top_k
    if resolved < min_k or resolved > max_k:
        raise APIError(
            status_code=400,
            code="INVALID_PARAMETER",
            message=f"Parameter top_k harus berada di antara {min_k} dan {max_k}.",
        )
    return resolved


def decode_uploaded_image(
    filename: str,
    content_type: str | None,
    payload: bytes,
    max_bytes: int,
    allowed_extensions: set[str],
) -> np.ndarray:
    """Validate file size, format, and decode image bytes with OpenCV."""
    if len(payload) > max_bytes:
        raise APIError(
            status_code=413,
            code="FILE_TOO_LARGE",
            message="Ukuran file melebihi batas maksimum.",
        )

    ext = Path(filename).suffix.lower() if filename else ""
    normalized_ct = (content_type or "").split(";")[0].strip().lower()

    if ext:
        if ext not in allowed_extensions and normalized_ct not in ALLOWED_CONTENT_TYPES:
            raise APIError(
                status_code=415,
                code="UNSUPPORTED_MEDIA_TYPE",
                message="Format file tidak didukung. Gunakan JPG atau PNG.",
            )
    elif normalized_ct and normalized_ct not in (
        ALLOWED_CONTENT_TYPES | {"application/octet-stream"}
    ):
        raise APIError(
            status_code=415,
            code="UNSUPPORTED_MEDIA_TYPE",
            message="Format file tidak didukung. Gunakan JPG atau PNG.",
        )

    if not payload:
        raise APIError(
            status_code=400,
            code="INVALID_IMAGE",
            message="File bukan gambar yang valid.",
        )

    buffer = np.frombuffer(payload, dtype=np.uint8)
    image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image_bgr is None or image_bgr.size == 0:
        raise APIError(
            status_code=400,
            code="INVALID_IMAGE",
            message="File bukan gambar yang valid.",
        )
    return image_bgr


def format_top_k(
    probs: np.ndarray, classes: Sequence[str], top_k: int
) -> tuple[str, float, list[ClassScore]]:
    """Build descending top_k predictions from a probability vector."""
    order = np.argsort(-probs, kind="stable")[:top_k]
    top_scores = [
        ClassScore(
            **{
                "class": str(classes[int(idx)]),
                "confidence": round(float(probs[int(idx)]), 6),
            }
        )
        for idx in order
    ]
    return top_scores[0].class_, top_scores[0].confidence, top_scores


def create_app(
    root: Path | None = None,
    api_config_path: Path | None = None,
    data_config_path: Path | None = None,
    predictor: Any | None = None,
) -> FastAPI:
    """Create and configure the FastAPI v1 application."""
    setup_logging()
    base_root = (root or REPO_ROOT).resolve()
    api_cfg = read_yaml(api_config_path or (base_root / "configs" / "api.yaml"))
    data_cfg = read_yaml(data_config_path or (base_root / "configs" / "data.yaml"))

    classes = [str(name) for name in data_cfg["classes"]]
    inference_cfg = api_cfg["inference"]
    max_bytes = int(inference_cfg["max_upload_mb"]) * 1024 * 1024
    allowed_extensions = {
        str(ext).lower() for ext in inference_cfg["allowed_extensions"]
    }
    batch_limit = int(inference_cfg["batch_limit"])
    api_version = str(api_cfg["api"]["version"])
    prefix = str(api_cfg["api"]["prefix"])
    log_requests = bool(api_cfg.get("logging", {}).get("requests", True))

    model_info = load_model_metadata(api_cfg, data_cfg, base_root)
    state: dict[str, Any] = {"predictor": predictor}

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if state["predictor"] is None:
            model_path = base_root / str(api_cfg["model"]["path"])
            if model_path.is_file():
                try:
                    state["predictor"] = OnnxClassifier(
                        model_path=model_path,
                        preprocess_config=data_cfg["preprocess"],
                        num_classes=len(classes),
                    )
                    LOGGER.info("Loaded ONNX model from %s", model_path)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error("Failed to load ONNX model: %s", exc)
            else:
                LOGGER.warning(
                    "ONNX model not found at %s; API started with model_loaded=false",
                    model_path,
                )
        yield

    application = FastAPI(
        title="Waste Classification API",
        version=model_info.version,
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def request_logging_middleware(request: Request, call_next: Any) -> Any:
        started = time.perf_counter()
        response = await call_next(request)
        if log_requests:
            duration_ms = (time.perf_counter() - started) * 1000.0
            LOGGER.info(
                "method=%s path=%s status=%d duration_ms=%.2f",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
            )
        return response

    @application.exception_handler(APIError)
    async def api_error_handler(_request: Request, exc: APIError) -> JSONResponse:
        body = ErrorResponse(
            error=ErrorDetail(code=exc.code, message=exc.message, details=exc.details)
        )
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        body = ErrorResponse(
            error=ErrorDetail(
                code="VALIDATION_ERROR",
                message="Permintaan tidak valid atau field wajib tidak ada.",
                details=None,
            )
        )
        return JSONResponse(status_code=422, content=body.model_dump())

    @application.exception_handler(Exception)
    async def unexpected_error_handler(
        _request: Request, exc: Exception
    ) -> JSONResponse:
        LOGGER.exception("Unhandled API error: %s", type(exc).__name__)
        body = ErrorResponse(
            error=ErrorDetail(
                code="INTERNAL_ERROR",
                message="Terjadi kesalahan internal pada server.",
                details=None,
            )
        )
        return JSONResponse(status_code=500, content=body.model_dump())

    router = APIRouter(prefix=prefix)

    @router.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_loaded=state["predictor"] is not None,
            api_version=api_version,
        )

    @router.get("/classes", response_model=ClassesResponse)
    async def get_classes() -> ClassesResponse:
        return ClassesResponse(
            classes=[
                ClassItem(index=idx, name=name) for idx, name in enumerate(classes)
            ]
        )

    @router.get("/model", response_model=ModelInfoResponse)
    async def get_model() -> ModelInfoResponse:
        return model_info

    @router.post(
        "/predict",
        response_model=PredictResponse,
        response_model_by_alias=True,
    )
    async def predict(
        file: UploadFile = File(...),
        top_k: int | None = None,
    ) -> PredictResponse:
        resolved_k = validate_top_k(top_k, inference_cfg)
        active_predictor = state["predictor"]
        if active_predictor is None:
            raise APIError(
                status_code=503,
                code="MODEL_NOT_LOADED",
                message="Model belum dimuat dan belum siap melayani prediksi.",
            )

        payload = await file.read()
        image_bgr = decode_uploaded_image(
            filename=file.filename or "",
            content_type=file.content_type,
            payload=payload,
            max_bytes=max_bytes,
            allowed_extensions=allowed_extensions,
        )

        started = time.perf_counter()
        probs = ensure_probabilities(active_predictor.predict_proba(image_bgr))
        inference_ms = round((time.perf_counter() - started) * 1000.0, 3)
        predicted_class, confidence, top_scores = format_top_k(
            probs, classes, resolved_k
        )
        return PredictResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            top_k=top_scores,
            inference_ms=inference_ms,
            model_version=model_info.version,
        )

    @router.post(
        "/predict/batch",
        response_model=BatchPredictResponse,
        response_model_by_alias=True,
    )
    async def predict_batch(
        files: list[UploadFile] = File(...),
        top_k: int | None = None,
    ) -> BatchPredictResponse:
        resolved_k = validate_top_k(top_k, inference_cfg)
        if not files or len(files) > batch_limit:
            raise APIError(
                status_code=400,
                code="INVALID_PARAMETER",
                message=f"Jumlah file batch harus antara 1 dan {batch_limit}.",
            )

        active_predictor = state["predictor"]
        if active_predictor is None:
            raise APIError(
                status_code=503,
                code="MODEL_NOT_LOADED",
                message="Model belum dimuat dan belum siap melayani prediksi.",
            )

        started = time.perf_counter()
        results: list[BatchPredictionItem | BatchErrorItem] = []

        for upload in files:
            filename = upload.filename or "unknown"
            try:
                payload = await upload.read()
                image_bgr = decode_uploaded_image(
                    filename=filename,
                    content_type=upload.content_type,
                    payload=payload,
                    max_bytes=max_bytes,
                    allowed_extensions=allowed_extensions,
                )
                probs = ensure_probabilities(active_predictor.predict_proba(image_bgr))
                predicted_class, confidence, top_scores = format_top_k(
                    probs, classes, resolved_k
                )
                results.append(
                    BatchPredictionItem(
                        filename=filename,
                        predicted_class=predicted_class,
                        confidence=confidence,
                        top_k=top_scores,
                    )
                )
            except APIError as exc:
                results.append(
                    BatchErrorItem(
                        filename=filename,
                        error=ErrorDetail(
                            code=exc.code, message=exc.message, details=exc.details
                        ),
                    )
                )

        inference_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return BatchPredictResponse(
            results=results,
            inference_ms=inference_ms,
            model_version=model_info.version,
        )

    application.include_router(router)
    return application


app = create_app()
