"""Streamlit demo application that calls the FastAPI v1 inference service."""

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_CONFIG = REPO_ROOT / "configs" / "api.yaml"


class APIClientError(RuntimeError):
    """Raised when the inference API returns an error or cannot be reached."""

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


def load_api_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load API configuration from ``configs/api.yaml``."""
    target = config_path or DEFAULT_API_CONFIG
    with target.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {target}")
    return data


def default_api_base_url(api_config: Mapping[str, Any]) -> str:
    """Resolve default API base URL from environment or ``configs/api.yaml``."""
    env_url = os.environ.get("WASTE_API_BASE_URL", "").strip()
    if env_url:
        return env_url.rstrip("/")
    api_section = api_config["api"]
    host = str(api_section["host"])
    port = int(api_section["port"])
    return f"http://{host}:{port}"


def build_endpoint_url(base_url: str, prefix: str, path: str) -> str:
    """Join base URL, API version prefix, and endpoint path cleanly."""
    clean_base = base_url.rstrip("/")
    clean_prefix = "/" + prefix.strip("/")
    clean_path = "/" + path.lstrip("/")
    return f"{clean_base}{clean_prefix}{clean_path}"


def parse_api_error(response: requests.Response) -> tuple[str | None, str]:
    """Extract uniform error code and message from a non-200 API response."""
    try:
        payload = response.json()
    except ValueError:
        return None, f"API mengembalikan status HTTP {response.status_code}."

    if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
        err = payload["error"]
        code = str(err.get("code", "API_ERROR"))
        message = str(err.get("message", f"HTTP {response.status_code}"))
        return code, message
    return None, f"API mengembalikan status HTTP {response.status_code}."


def fetch_health(
    base_url: str,
    prefix: str,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Call ``GET /v1/health`` and return the parsed JSON payload."""
    url = build_endpoint_url(base_url, prefix, "/health")
    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as exc:
        raise APIClientError(f"Tidak dapat terhubung ke API di {base_url}.") from exc

    if response.status_code != 200:
        code, message = parse_api_error(response)
        raise APIClientError(message, code=code)
    return dict(response.json())


def fetch_model_info(
    base_url: str,
    prefix: str,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Call ``GET /v1/model`` and return active model metadata."""
    url = build_endpoint_url(base_url, prefix, "/model")
    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as exc:
        raise APIClientError(f"Gagal mengambil info model dari {base_url}.") from exc

    if response.status_code != 200:
        code, message = parse_api_error(response)
        raise APIClientError(message, code=code)
    return dict(response.json())


def predict_image(
    base_url: str,
    prefix: str,
    filename: str,
    image_bytes: bytes,
    content_type: str,
    top_k: int,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Upload one image to ``POST /v1/predict`` and return the prediction."""
    url = build_endpoint_url(base_url, prefix, "/predict")
    try:
        response = requests.post(
            url,
            params={"top_k": top_k},
            files={"file": (filename, image_bytes, content_type)},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise APIClientError(
            f"Gagal mengirim gambar ke API di {base_url}. Pastikan server FastAPI aktif."
        ) from exc

    if response.status_code != 200:
        code, message = parse_api_error(response)
        raise APIClientError(message, code=code)
    return dict(response.json())


def format_metric_value(value: float | None) -> str:
    """Format an optional metric value without fabricating numbers."""
    if value is None:
        return "Belum dievaluasi (null)"
    return f"{value:.4f}"


def render_app() -> None:
    """Render the Streamlit demo interface."""
    api_config = load_api_config()
    prefix = str(api_config["api"]["prefix"])
    inference_cfg = api_config["inference"]
    top_k_min = int(inference_cfg["top_k_min"])
    top_k_max = int(inference_cfg["top_k_max"])
    top_k_default = int(inference_cfg["top_k_default"])
    max_upload_mb = int(inference_cfg["max_upload_mb"])
    allowed_types = [
        str(ext).lstrip(".").lower() for ext in inference_cfg["allowed_extensions"]
    ]

    st.set_page_config(
        page_title="Klasifikasi Sampah (YOLO-cls)",
        page_icon="♻️",
        layout="centered",
    )

    st.title("Klasifikasi Jenis Sampah (YOLO-cls)")
    st.caption(
        "Demo inferensi 9 kelas sampah (MWCD) yang memanggil layanan API FastAPI "
        f"`{prefix}` tanpa memuat model di aplikasi demo."
    )

    with st.sidebar:
        st.header("Pengaturan & Status API")
        base_url = st.text_input(
            "API Base URL",
            value=default_api_base_url(api_config),
            help="Alamat server FastAPI (default dari configs/api.yaml).",
        )
        top_k = st.slider(
            "Jumlah Top-K",
            min_value=top_k_min,
            max_value=top_k_max,
            value=top_k_default,
        )

        st.subheader("Status Layanan")
        api_ready = False
        try:
            health = fetch_health(base_url, prefix)
            model_loaded = bool(health.get("model_loaded", False))
            api_version = str(health.get("api_version", "-"))
            if model_loaded:
                st.success(f"API aktif ({api_version}) · Model siap")
                api_ready = True
            else:
                st.warning(f"API aktif ({api_version}) · Model belum dimuat")
        except APIClientError as exc:
            st.error(str(exc))

        st.subheader("Metadata Model")
        try:
            info = fetch_model_info(base_url, prefix)
            metrics = info.get("metrics") or {}
            st.markdown(
                "\n".join(
                    [
                        f"- **Nama:** `{info.get('name', '-')}`",
                        f"- **Versi:** `{info.get('version', '-')}`",
                        f"- **Arsitektur:** `{info.get('architecture', '-')}`",
                        f"- **Input:** `{info.get('input_size', '-')}` px (`{info.get('format', '-')}`)",
                        f"- **Dataset:** {info.get('trained_on', '-')}",
                        f"- **Test Accuracy:** {format_metric_value(metrics.get('test_accuracy'))}",
                        f"- **Test Macro-F1:** {format_metric_value(metrics.get('test_macro_f1'))}",
                    ]
                )
            )
        except APIClientError:
            st.info("Metadata model belum tersedia karena API belum terhubung.")

    uploaded = st.file_uploader(
        f"Unggah gambar sampah ({', '.join(ext.upper() for ext in allowed_types)}, maks. {max_upload_mb} MB)",
        type=allowed_types,
    )

    if uploaded is not None:
        image_bytes = uploaded.getvalue()
        st.image(image_bytes, caption=uploaded.name, use_column_width=True)

        size_mb = len(image_bytes) / (1024 * 1024)
        if size_mb > max_upload_mb:
            st.warning(
                f"Ukuran file {size_mb:.2f} MB melebihi batas {max_upload_mb} MB."
            )

        if st.button("Prediksi Jenis Sampah", type="primary", use_container_width=True):
            with st.spinner("Mengirim gambar ke API..."):
                try:
                    result = predict_image(
                        base_url=base_url,
                        prefix=prefix,
                        filename=uploaded.name,
                        image_bytes=image_bytes,
                        content_type=uploaded.type or "application/octet-stream",
                        top_k=top_k,
                    )
                except APIClientError as exc:
                    prefix_code = f"[{exc.code}] " if exc.code else ""
                    st.error(f"{prefix_code}{exc}")
                else:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Prediksi", str(result["predicted_class"]))
                    col2.metric(
                        "Confidence", f"{float(result['confidence']) * 100:.2f}%"
                    )
                    col3.metric("Inferensi", f"{float(result['inference_ms']):.1f} ms")

                    st.subheader(f"Top-{len(result['top_k'])} Confidence")
                    for item in result["top_k"]:
                        label = str(item["class"])
                        score = float(item["confidence"])
                        st.write(f"**{label}** — {score * 100:.2f}%")
                        st.progress(min(max(score, 0.0), 1.0))

                    st.caption(f"Model version: `{result.get('model_version', '-')}`")
    elif not api_ready:
        st.info(
            "Jalankan server API terlebih dahulu dengan "
            "`uvicorn src.api.main:app --reload --port 8000` lalu unggah gambar di atas."
        )

    st.divider()
    st.caption(
        "Dataset: Israt Jahan, Sanjida Tasneem, Fahmida Rifha. "
        "Merged Waste Classification Dataset (MWCD), Mendeley Data V1, "
        "DOI: 10.17632/x863v66fv3.1 (CC BY 4.0)."
    )


if __name__ == "__main__":
    render_app()
