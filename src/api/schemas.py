"""Pydantic schemas for the Waste Classification API v1 contract."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorDetail(BaseModel):
    """Uniform error detail payload."""

    code: str
    message: str
    details: Any = None


class ErrorResponse(BaseModel):
    """Uniform error envelope returned on all API errors."""

    error: ErrorDetail


class HealthResponse(BaseModel):
    """Response payload for GET /v1/health."""

    status: str
    model_loaded: bool
    api_version: str


class ClassItem(BaseModel):
    """One class index and name entry."""

    index: int = Field(ge=0)
    name: str


class ClassesResponse(BaseModel):
    """Response payload for GET /v1/classes."""

    classes: list[ClassItem]


class ModelMetrics(BaseModel):
    """Reported evaluation metrics for the active model."""

    test_accuracy: float | None = None
    test_macro_f1: float | None = None


class ModelInfoResponse(BaseModel):
    """Response payload for GET /v1/model."""

    name: str
    version: str
    architecture: str
    input_size: int = Field(gt=0)
    num_classes: int = Field(gt=0)
    format: str
    trained_on: str
    metrics: ModelMetrics


class ClassScore(BaseModel):
    """Predicted class label and probability score."""

    model_config = ConfigDict(populate_by_name=True)

    class_: str = Field(alias="class")
    confidence: float = Field(ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    """Response payload for POST /v1/predict."""

    predicted_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    top_k: list[ClassScore]
    inference_ms: float = Field(ge=0.0)
    model_version: str


class BatchPredictionItem(BaseModel):
    """Successful prediction item inside POST /v1/predict/batch."""

    filename: str
    predicted_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    top_k: list[ClassScore]


class BatchErrorItem(BaseModel):
    """Per-file error item inside POST /v1/predict/batch."""

    filename: str
    error: ErrorDetail


class BatchPredictResponse(BaseModel):
    """Response payload for POST /v1/predict/batch."""

    results: list[BatchPredictionItem | BatchErrorItem]
    inference_ms: float = Field(ge=0.0)
    model_version: str
