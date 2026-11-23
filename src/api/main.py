"""FastAPI inference service for waste classification."""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.inference import ONNXInference
from src.api.schemas import (
    BatchPredictResponse,
    ClassesResponse,
    HealthResponse,
    ModelResponse,
    PredictResponse,
)
from src.utils.io import read_yaml

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
CONFIG = read_yaml(ROOT / "configs/api.yaml")
PREFIX = str(CONFIG["api"]["prefix"])
MAX_UPLOAD_BYTES = int(CONFIG["inference"]["max_upload_mb"]) * 1024 * 1024
ALLOWED_EXTENSIONS = {
    str(extension).lower() for extension in CONFIG["inference"]["allowed_extensions"]
}
DEFAULT_TOP_K = int(CONFIG["inference"]["top_k_default"])
CLASS_FALLBACK = [
    str(name)
    for name in read_yaml(ROOT / str(CONFIG["model"]["classes_source"])).get(
        "classes", []
    )
]


class APIError(Exception):
    """Application error mapped to the uniform v1 error payload."""

    def __init__(
        self, status_code: int, code: str, message: str, details: Any = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details

    def payload(self) -> dict[str, Any]:
        """Return the uniform error body."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    """Build a JSON error response."""
    return JSONResponse(
        status_code=status_code,
        content=APIError(status_code, code, message).payload(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ONNX model once when the service starts."""
    try:
        app.state.engine = ONNXInference.load(CONFIG, ROOT)
        LOGGER.info("model loaded from %s", CONFIG["model"]["path"])
    except Exception as exc:
        app.state.engine = None
        LOGGER.warning("model not loaded: %s", exc)
    yield


app = FastAPI(
    title="Waste Classification API",
    version=str(CONFIG["api"]["version"]),
    lifespan=lifespan,
)
app.state.engine = None
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(CONFIG["api"].get("allowed_origins", [])),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(APIError)
async def handle_api_error(request: Request, exc: APIError) -> JSONResponse:
    """Return the uniform body for expected application errors."""
    return JSONResponse(status_code=exc.status_code, content=exc.payload())


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return a safe validation error without echoing uploaded content."""
    return error_response(
        422, "VALIDATION_ERROR", "Field wajib tidak ada atau tidak valid."
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Log unexpected failures and return a safe response."""
    LOGGER.exception("unhandled error on %s %s", request.method, request.url.path)
    return error_response(500, "INTERNAL_ERROR", "Kesalahan tak terduga.")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request metadata without storing images."""
    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - started) * 1000.0
    LOGGER.info(
        "method=%s path=%s status=%s duration_ms=%.1f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


def get_engine(request: Request) -> ONNXInference:
    """Return the loaded engine or raise MODEL_NOT_LOADED."""
    engine = getattr(request.app.state, "engine", None)
    if engine is None or not engine.available:
        raise APIError(503, "MODEL_NOT_LOADED", "Model belum siap.")
    return engine


def classes_for(request: Request) -> list[str]:
    """Return active classes, falling back to the data config before model load."""
    engine = getattr(request.app.state, "engine", None)
    if engine is not None and engine.classes:
        return engine.classes
    return CLASS_FALLBACK


def validate_upload(filename: str, data: bytes) -> np.ndarray:
    """Validate an uploaded image and decode it to a BGR frame."""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise APIError(
            415,
            "UNSUPPORTED_MEDIA_TYPE",
            "Gunakan gambar JPG atau PNG.",
        )
    if len(data) > MAX_UPLOAD_BYTES:
        raise APIError(413, "FILE_TOO_LARGE", "Ukuran file melebihi 5 MB.")

    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise APIError(400, "INVALID_IMAGE", "File bukan gambar yang valid.")
    return frame


async def read_upload(file: UploadFile) -> np.ndarray:
    """Read and validate one uploaded file."""
    data = await file.read()
    return validate_upload(file.filename or "", data)


@app.get(f"{PREFIX}/health", response_model=HealthResponse)
async def health(request: Request) -> dict[str, Any]:
    """Report service and model status."""
    engine = getattr(request.app.state, "engine", None)
    return {
        "status": "ok",
        "model_loaded": bool(engine is not None and engine.available),
        "api_version": str(CONFIG["api"]["version"]),
    }


@app.get(f"{PREFIX}/classes", response_model=ClassesResponse)
async def classes(request: Request) -> dict[str, Any]:
    """Return the fixed class order."""
    return {
        "classes": [
            {"index": index, "name": name}
            for index, name in enumerate(classes_for(request))
        ]
    }


@app.get(f"{PREFIX}/model", response_model=ModelResponse)
async def model_metadata(request: Request) -> dict[str, Any]:
    """Return metadata of the active model card."""
    engine = getattr(request.app.state, "engine", None)
    card = getattr(engine, "card", {}) if engine is not None else {}
    metrics = card.get("metrics") or {}
    dataset = str(card.get("dataset", "MWCD v1")).split(" (")[0]
    return {
        "name": str(card.get("name", "waste-yolo-cls")),
        "version": str(card.get("version", "1.0.0")),
        "architecture": str(card.get("architecture", "yolov8n-cls")),
        "input_size": int(card.get("input_size", CONFIG["model"]["input_size"])),
        "num_classes": len(classes_for(request)),
        "format": str(CONFIG["model"]["format"]),
        "trained_on": dataset,
        "metrics": {
            "test_accuracy": metrics.get("test_accuracy"),
            "test_macro_f1": metrics.get("test_macro_f1"),
        },
    }


@app.post(f"{PREFIX}/predict", response_model=PredictResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = Query(DEFAULT_TOP_K),
) -> dict[str, Any]:
    """Classify one uploaded image."""
    if not 1 <= top_k <= 9:
        raise APIError(400, "INVALID_PARAMETER", "top_k harus antara 1 dan 9.")

    engine = get_engine(request)
    frame = await read_upload(file)
    started = time.perf_counter()
    result = engine.predict(frame, top_k)
    inference_ms = (time.perf_counter() - started) * 1000.0
    return {
        **result,
        "inference_ms": round(inference_ms, 1),
        "model_version": engine.version,
    }


@app.post(f"{PREFIX}/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(
    request: Request,
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    """Classify up to 16 images and keep results aligned with inputs."""
    if not 1 <= len(files) <= 16:
        raise APIError(400, "INVALID_PARAMETER", "Jumlah file harus antara 1 dan 16.")

    engine = get_engine(request)
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for file in files:
        filename = file.filename or "unknown"
        try:
            frame = await read_upload(file)
            result = engine.predict(frame, DEFAULT_TOP_K)
            results.append({"filename": filename, **result})
        except APIError as exc:
            results.append({"filename": filename, "error": exc.payload()["error"]})
        except Exception:
            LOGGER.exception("batch item failed for %s", filename)
            failure = APIError(500, "INTERNAL_ERROR", "Kesalahan tak terduga.")
            results.append({"filename": filename, "error": failure.payload()["error"]})

    inference_ms = (time.perf_counter() - started) * 1000.0
    return {
        "results": results,
        "inference_ms": round(inference_ms, 1),
        "model_version": engine.version,
    }
