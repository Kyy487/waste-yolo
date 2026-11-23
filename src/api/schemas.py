"""Pydantic schemas for the v1 inference API."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ClassScore(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_: str = Field(alias="class")
    confidence: float = Field(ge=0, le=1)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    api_version: str


class ClassItem(BaseModel):
    index: int
    name: str


class ClassesResponse(BaseModel):
    classes: list[ClassItem]


class ModelMetrics(BaseModel):
    test_accuracy: float | None = None
    test_macro_f1: float | None = None


class ModelResponse(BaseModel):
    name: str
    version: str
    architecture: str
    input_size: int
    num_classes: int
    format: str
    trained_on: str
    metrics: ModelMetrics


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float = Field(ge=0, le=1)
    top_k: list[ClassScore]
    inference_ms: float
    model_version: str


class BatchPredictResponse(BaseModel):
    results: list[dict[str, Any]]
    inference_ms: float
    model_version: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Any | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
