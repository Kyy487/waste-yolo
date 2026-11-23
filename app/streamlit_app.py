"""Streamlit realtime camera demo that calls the v1 inference API."""

import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import requests
import streamlit as st

from app.api_client import APIClientError, predict_frame
from src.utils.io import read_yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = read_yaml(ROOT / "configs/api.yaml")
API = CONFIG["api"]
DEMO = CONFIG["demo"]
BASE_URL = f"http://{API['host']}:{API['port']}{API['prefix']}"
TARGET_FPS = float(DEMO["target_fps"])
SMOOTHING_WINDOW = int(DEMO["smoothing_window"])
TOP_K = int(DEMO["top_k"])
REQUEST_TIMEOUT_S = float(DEMO["request_timeout_s"])
SHOW_TOP_K = bool(DEMO["show_top_k"])


def overlay(
    frame_bgr: np.ndarray, label: str, confidence: float, fps: float
) -> np.ndarray:
    """Draw the predicted label, confidence, and FPS on a frame."""
    annotated = frame_bgr.copy()
    text = f"{label} {confidence:.0%} | {fps:.1f} FPS"
    cv2.putText(
        annotated,
        text,
        (16, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        text,
        (16, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return annotated


def majority_label(labels: list[str]) -> str:
    """Return the most frequent recent label."""
    if not labels:
        return ""
    return Counter(labels).most_common(1)[0][0]


def release_camera() -> None:
    """Release the camera and clear smoothing state."""
    capture = st.session_state.get("capture")
    if capture is not None:
        capture.release()
    st.session_state.capture = None
    st.session_state.running = False
    st.session_state.recent_labels = []


def main() -> None:
    """Run the realtime camera demo."""
    st.set_page_config(page_title="Klasifikasi Sampah Real-Time", layout="wide")
    st.title("Klasifikasi Sampah Real-Time")
    st.caption("Klasifikasi satu frame kamera lewat API v1, tanpa bounding box.")

    if "running" not in st.session_state:
        st.session_state.running = False
    if "capture" not in st.session_state:
        st.session_state.capture = None
    if "recent_labels" not in st.session_state:
        st.session_state.recent_labels = []
    if "http_session" not in st.session_state:
        st.session_state.http_session = requests.Session()

    start_column, stop_column = st.columns(2)
    with start_column:
        start_clicked = st.button(
            "Start kamera", type="primary", use_container_width=True
        )
    with stop_column:
        stop_clicked = st.button("Stop kamera", use_container_width=True)

    if start_clicked and not st.session_state.running:
        capture = cv2.VideoCapture(int(DEMO["camera_index"]))
        if not capture.isOpened():
            st.error(
                "Kamera tidak dapat dibuka. Periksa camera_index di configs/api.yaml."
            )
        else:
            st.session_state.capture = capture
            st.session_state.running = True
            st.session_state.recent_labels = []

    if stop_clicked:
        release_camera()

    if not st.session_state.running or st.session_state.capture is None:
        st.info("Klik Start kamera untuk memulai klasifikasi real-time.")
        return

    started = time.perf_counter()
    ok, frame = st.session_state.capture.read()
    if not ok or frame is None:
        st.warning("Frame kamera tidak terbaca.")
        release_camera()
        return

    result = None
    status = "Menunggu prediksi..."
    try:
        result = predict_frame(
            st.session_state.http_session,
            BASE_URL,
            frame,
            top_k=TOP_K,
            timeout=REQUEST_TIMEOUT_S,
        )
        status = "Prediksi diterima."
    except APIClientError as exc:
        status = f"API error {exc.code}: {exc.message}"
    except requests.RequestException:
        status = "API tidak tersedia. Jalankan uvicorn src.api.main:app --port 8000."

    label = ""
    confidence = 0.0
    if result is not None:
        st.session_state.recent_labels.append(str(result["predicted_class"]))
        st.session_state.recent_labels = st.session_state.recent_labels[
            -SMOOTHING_WINDOW:
        ]
        label = majority_label(st.session_state.recent_labels)
        confidence = float(result["confidence"])

    elapsed = time.perf_counter() - started
    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    annotated = overlay(frame, label or "unknown", confidence, fps)
    st.image(annotated, channels="BGR", use_container_width=True)
    st.caption(status)

    if SHOW_TOP_K and result is not None:
        st.write("Top-k:", result.get("top_k", []))

    delay = max(0.0, (1.0 / TARGET_FPS) - elapsed)
    time.sleep(delay)
    st.rerun()


if __name__ == "__main__":
    main()
