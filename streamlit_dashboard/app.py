from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.dont_write_bytecode = True

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.segmentation import (
        apply_mask,
        clean_mask,
        grabcut_segmentation,
        largest_connected_component,
    )

    SEGMENTATION_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - displayed in the dashboard.
    apply_mask = None
    clean_mask = None
    grabcut_segmentation = None
    largest_connected_component = None
    SEGMENTATION_IMPORT_ERROR = exc

APP_TITLE = "Waste Classification Dashboard"
APP_SUBTITLE = (
    "Analisis Pengaruh Teknik Image Enhancement terhadap Kinerja CNN Scratch dan "
    "MobileNetV2 pada Klasifikasi Jenis Sampah Berdasarkan Citra"
)
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
EDA_FIGURES_DIR = RESULTS_DIR / "eda" / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

CLASS_NAMES = ["biological", "cardboard", "glass", "metal", "paper", "plastic"]
CLASS_LABELS = {
    "biological": "Biological",
    "cardboard": "Cardboard",
    "glass": "Glass",
    "metal": "Metal",
    "paper": "Paper",
    "plastic": "Plastic",
}
GROUP_MEMBERS = [
    "Nama anggota kelompok 1",
    "Nama anggota kelompok 2",
    "Nama anggota kelompok 3",
]
GITHUB_REPOSITORY = "https://github.com/aauliaaz/waste-classification-cnn"
MODEL_RUNS = {
    "CNN Scratch": {
        "path": MODELS_DIR / "cnn_clahe_gaussian.keras",
        "input_size": (224, 224),
        "processing": "CLAHE + Gaussian",
        "model_type": "Custom CNN",
    },
    "MobileNetV2": {
        "path": MODELS_DIR / "original_finetune.keras",
        "input_size": (192, 192),
        "processing": "Original",
        "model_type": "Transfer Learning",
    },
}


def inject_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #071329;
            --muted: #667085;
            --line: #e8edf3;
            --accent: #ff5a45;
            --accent-soft: #fff2ef;
            --card: #ffffff;
            --page: #ffffff;
        }
        .stApp {
            background: var(--page);
            color: var(--ink);
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #dce3ee;
            width: 300px !important;
            min-width: 300px !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            padding: 4.2rem 1.45rem 2rem 1.45rem;
        }
        .block-container {
            max-width: 1500px;
            padding: 0 3.2rem 4rem 3.2rem;
        }
        .topbar {
            height: 54px;
            margin: 0 -3.2rem 4.8rem -3.2rem;
            background: #0c1016;
            display: flex;
            justify-content: flex-end;
            align-items: center;
            gap: 16px;
            padding: 0 22px;
            color: #7d8899;
            font-size: 0.88rem;
        }
        .brand-logo {
            width: 66px;
            height: 66px;
            margin: 0 auto 22px auto;
            border-radius: 18px;
            background:
                radial-gradient(circle at 68% 30%, #ff4545 0 34%, transparent 35%),
                radial-gradient(circle at 35% 55%, #a82bd5 0 36%, transparent 37%),
                linear-gradient(135deg, #36c275 0%, #ff6b44 58%, #5f2bd9 100%);
            box-shadow: 0 10px 26px rgba(255, 90, 69, 0.22);
            position: relative;
        }
        .brand-logo:before {
            content: "";
            position: absolute;
            left: 19px;
            top: -10px;
            width: 22px;
            height: 16px;
            border-radius: 14px 14px 2px 14px;
            background: #63c548;
            transform: rotate(20deg);
        }
        .sidebar-title {
            text-align: center;
            font-size: 1.34rem;
            line-height: 1.2;
            font-weight: 780;
            margin-bottom: 0.6rem;
            color: var(--ink);
        }
        .sidebar-subtitle {
            text-align: center;
            color: var(--ink);
            font-size: 0.82rem;
            margin-bottom: 2.2rem;
        }
        .sidebar-heading {
            color: var(--ink);
            font-size: 1.02rem;
            font-weight: 760;
            margin: 0 0 1rem 0;
        }
        .side-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 18px;
            margin-bottom: 14px;
            background: #ffffff;
            box-shadow: 0 7px 20px rgba(15, 23, 42, 0.035);
        }
        .side-card-title {
            color: var(--ink);
            font-size: 0.96rem;
            font-weight: 760;
            margin-bottom: 10px;
        }
        .side-card-copy {
            color: #24324a;
            font-size: 0.78rem;
            line-height: 1.45;
            margin-bottom: 14px;
        }
        .badge {
            display: inline-flex;
            align-items: center;
            border: 1px solid #ffd0c7;
            border-radius: 6px;
            padding: 7px 10px;
            background: var(--accent-soft);
            color: var(--ink);
            font-size: 0.76rem;
            font-weight: 760;
        }
        .main-title {
            color: var(--ink);
            font-size: 2.25rem;
            line-height: 1.12;
            font-weight: 820;
            letter-spacing: 0;
            margin: 0 0 1.6rem 0;
        }
        .main-subtitle {
            color: #111827;
            font-size: 1.02rem;
            line-height: 1.65;
            max-width: 1120px;
            margin: 0 0 3rem 0;
        }
        .metric-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.055);
            padding: 24px 26px;
            min-height: 134px;
        }
        .metric-label {
            color: #9aa4b8;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-size: 0.76rem;
            font-weight: 760;
            margin-bottom: 12px;
        }
        .metric-value {
            color: var(--ink);
            font-size: 1.75rem;
            line-height: 1.1;
            font-weight: 820;
            margin-bottom: 12px;
        }
        .metric-value.accent {
            color: var(--accent);
        }
        .metric-note {
            color: #24324a;
            font-size: 0.84rem;
            line-height: 1.4;
        }
        .section-title {
            color: var(--ink);
            font-size: 1.35rem;
            font-weight: 800;
            letter-spacing: 0;
            margin: 2.2rem 0 0.5rem 0;
        }
        .section-rule {
            height: 1px;
            background: var(--line);
            position: relative;
            margin: 0 0 1.35rem 0;
        }
        .section-rule:before {
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            width: 64px;
            height: 2px;
            background: var(--accent);
        }
        .upload-card {
            border: 1px dashed var(--accent);
            border-radius: 8px;
            padding: 18px 24px 24px 24px;
            background: #fffdfc;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.035);
        }
        [data-testid="stFileUploaderDropzone"] {
            min-height: 116px;
            border: 1px dashed var(--accent);
            border-radius: 8px;
            background: #fffdfc;
            padding: 18px 24px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.035);
        }
        [data-testid="stFileUploaderDropzone"] button {
            border-radius: 7px;
        }
        .preview-wrap {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px;
            background: #ffffff;
            margin-top: 16px;
        }
        .pipeline-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 9px 24px rgba(15, 23, 42, 0.05);
            padding: 16px 16px 18px 16px;
            min-height: 100%;
        }
        .pipeline-title {
            color: var(--ink);
            font-size: 0.86rem;
            font-weight: 820;
            text-align: center;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            padding-bottom: 12px;
            border-bottom: 2px solid var(--accent);
            margin-bottom: 14px;
        }
        .pipeline-copy {
            color: #8a94aa;
            font-size: 0.78rem;
            line-height: 1.45;
            text-align: center;
            margin-top: 12px;
            min-height: 34px;
        }
        .flow-arrow {
            color: var(--accent);
            font-size: 1.85rem;
            font-weight: 820;
            text-align: center;
            padding-top: 96px;
        }
        .prediction-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
            padding: 24px;
            min-height: 262px;
        }
        .prediction-head {
            color: var(--ink);
            font-size: 1.05rem;
            font-weight: 820;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--line);
            margin-bottom: 16px;
        }
        .prediction-pill {
            display: inline-flex;
            border: 1px solid #ffd0c7;
            border-radius: 999px;
            background: var(--accent-soft);
            color: #f04438;
            font-size: 0.86rem;
            font-weight: 820;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 10px 18px;
            margin-bottom: 18px;
        }
        .prediction-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
        }
        .prediction-k {
            color: #98a2b3;
            font-size: 0.74rem;
            font-weight: 760;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .prediction-v {
            color: var(--ink);
            font-size: 1.08rem;
            font-weight: 820;
            margin-top: 4px;
        }
        .prob-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.045);
            padding: 20px;
        }
        .prob-title {
            color: var(--ink);
            font-size: 1rem;
            font-weight: 820;
            margin-bottom: 16px;
        }
        .info-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.045);
            padding: 20px;
            height: 100%;
        }
        .info-card h4 {
            color: var(--ink);
            font-size: 1rem;
            margin: 0 0 12px 0;
        }
        .info-card p, .info-card li {
            color: #344054;
            font-size: 0.9rem;
            line-height: 1.55;
        }
        .stProgress > div > div > div > div {
            background-color: var(--accent);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-logo"></div>
            <div class="sidebar-title">Waste Classification</div>
            <div class="sidebar-subtitle">Deep Learning Research Dashboard</div>
            <div class="sidebar-heading">Model Summary</div>
            <div class="side-card">
                <div class="side-card-title">MobileNetV2</div>
                <div class="side-card-copy">Accuracy</div>
                <div class="badge">88.94% &nbsp; Transfer Learning</div>
            </div>
            <div class="side-card">
                <div class="side-card-title">CNN Scratch</div>
                <div class="side-card-copy">Accuracy</div>
                <div class="badge">64.65% &nbsp; Custom CNN</div>
            </div>
            <div class="side-card">
                <div class="side-card-title">Dataset</div>
                <div class="side-card-copy">Research image collection</div>
                <div class="badge">13348 Images &nbsp; 6 Classes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def section_title(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, note: str, accent: bool = False) -> None:
    value_class = "metric-value accent" if accent else "metric-value"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="{value_class}">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_uploaded_image(uploaded_file) -> np.ndarray:
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image)
        return np.array(image.convert("RGB"))
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("File gambar tidak dapat dibaca.") from exc


def resize_to(image_rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image_rgb, size, interpolation=cv2.INTER_AREA)


def resize_long_side(image_rgb: np.ndarray, max_side: int = 680) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    largest = max(height, width)
    if largest <= max_side:
        return image_rgb.copy()
    scale = max_side / float(largest)
    resized_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image_rgb, resized_size, interpolation=cv2.INTER_AREA)


def apply_clahe_rgb(image_rgb: np.ndarray) -> np.ndarray:
    # Parameters follow the existing notebook pipeline: LAB L-channel CLAHE.
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def apply_clahe_gaussian_rgb(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(apply_clahe_rgb(image_rgb), (3, 3), 0)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)


def build_enhancement_pipeline(image_rgb: np.ndarray) -> list[dict[str, object]]:
    original = resize_long_side(image_rgb)
    clahe = apply_clahe_rgb(original)
    clahe_gaussian = apply_clahe_gaussian_rgb(original)
    return [
        {
            "title": "Original",
            "image": original,
            "description": "Citra asli setelah dibaca dalam format RGB.",
        },
        {
            "title": "CLAHE",
            "image": clahe,
            "description": "Peningkatan kontras pada channel luminance LAB.",
        },
        {
            "title": "CLAHE + Gaussian",
            "image": clahe_gaussian,
            "description": "CLAHE dilanjutkan Gaussian blur untuk reduksi noise.",
        },
    ]


def build_grabcut_pipeline(image_rgb: np.ndarray) -> list[dict[str, object]]:
    if SEGMENTATION_IMPORT_ERROR is not None:
        raise RuntimeError(f"Gagal import src.segmentation: {SEGMENTATION_IMPORT_ERROR}")

    original = resize_to(image_rgb, (224, 224))
    image_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    raw_mask = grabcut_segmentation(image_bgr)
    kernel = np.ones((5, 5), np.uint8)
    morphology_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    morphology_mask = cv2.morphologyEx(morphology_mask, cv2.MORPH_CLOSE, kernel)
    largest_mask = largest_connected_component(morphology_mask)
    final_mask = clean_mask(raw_mask)
    final_bgr = apply_mask(image_bgr, final_mask)
    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
    return [
        {
            "title": "Original",
            "image": original,
            "description": "Citra input ukuran 224 x 224.",
        },
        {
            "title": "Raw GrabCut",
            "image": mask_to_rgb(raw_mask),
            "description": "Mask awal dari GrabCut berbasis rectangle.",
        },
        {
            "title": "Morphology",
            "image": mask_to_rgb(morphology_mask),
            "description": "Opening dan closing untuk membersihkan noise.",
        },
        {
            "title": "Largest Contour",
            "image": mask_to_rgb(largest_mask),
            "description": "Objek utama dipertahankan sebagai foreground.",
        },
        {
            "title": "Final Segmented",
            "image": final_rgb,
            "description": "Citra hasil mask final untuk skenario GrabCut.",
        },
    ]


def preprocess_for_model(image_rgb: np.ndarray, processing: str, input_size: tuple[int, int]) -> np.ndarray:
    if processing == "Original":
        processed = image_rgb
    elif processing == "CLAHE":
        processed = apply_clahe_rgb(image_rgb)
    elif processing == "CLAHE + Gaussian":
        processed = apply_clahe_gaussian_rgb(image_rgb)
    elif processing == "GrabCut":
        processed = build_grabcut_pipeline(image_rgb)[-1]["image"]
    else:
        raise ValueError(f"Metode preprocessing tidak dikenal: {processing}")
    return resize_to(processed, input_size).astype(np.uint8)


@st.cache_resource(show_spinner=False)
def load_tf_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {path}")
    if path.stat().st_size < 2048:
        raise RuntimeError(f"File {path.name} terlihat seperti pointer Git LFS, bukan model penuh.")

    import tensorflow as tf

    return tf.keras.models.load_model(path, compile=False)


def predict_with_model(image_rgb: np.ndarray, model_name: str) -> dict[str, object]:
    config = MODEL_RUNS[model_name]
    processed = preprocess_for_model(
        image_rgb,
        str(config["processing"]),
        tuple(config["input_size"]),
    )
    model = load_tf_model(str(config["path"]))
    batch = np.expand_dims(processed.astype("float32"), axis=0)

    start = time.perf_counter()
    prediction = model.predict(batch, verbose=0)
    inference_ms = (time.perf_counter() - start) * 1000

    probabilities = np.asarray(prediction[0], dtype=np.float64)
    if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0, atol=1e-3):
        exp_values = np.exp(probabilities - np.max(probabilities))
        probabilities = exp_values / exp_values.sum()

    best_index = int(np.argmax(probabilities))
    return {
        "model_name": model_name,
        "label": CLASS_NAMES[best_index],
        "confidence": float(probabilities[best_index]),
        "probabilities": probabilities,
        "inference_ms": inference_ms,
        "processing": config["processing"],
        "model_type": config["model_type"],
        "processed_image": processed,
    }


def run_analysis(image_rgb: np.ndarray) -> dict[str, object]:
    enhancement_pipeline = build_enhancement_pipeline(image_rgb)
    grabcut_pipeline = build_grabcut_pipeline(image_rgb)
    predictions = {
        "CNN Scratch": predict_with_model(image_rgb, "CNN Scratch"),
        "MobileNetV2": predict_with_model(image_rgb, "MobileNetV2"),
    }
    return {
        "enhancement_pipeline": enhancement_pipeline,
        "grabcut_pipeline": grabcut_pipeline,
        "predictions": predictions,
    }


def pipeline_card(item: dict[str, object]) -> None:
    with st.container(border=True):
        st.markdown(f'<div class="pipeline-title">{item["title"]}</div>', unsafe_allow_html=True)
        st.image(item["image"], use_container_width=True)
        st.markdown(
            f'<div class="pipeline-copy">{item["description"]}</div>',
            unsafe_allow_html=True,
        )


def flow_arrow() -> None:
    st.markdown('<div class="flow-arrow">&darr;</div>', unsafe_allow_html=True)


def show_pipeline_cards(items: list[dict[str, object]]) -> None:
    column_count = len(items) + max(0, len(items) - 1)
    ratios = []
    for index in range(column_count):
        ratios.append(0.16 if index % 2 else 1)
    cols = st.columns(ratios)
    item_index = 0
    for index, col in enumerate(cols):
        with col:
            if index % 2:
                flow_arrow()
            else:
                pipeline_card(items[item_index])
                item_index += 1


def prediction_card(result: dict[str, object]) -> None:
    label = CLASS_LABELS[str(result["label"])]
    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="prediction-head">{result["model_name"]}</div>
            <div class="prediction-pill">{label}</div>
            <div class="prediction-grid">
                <div>
                    <div class="prediction-k">Confidence</div>
                    <div class="prediction-v">{float(result["confidence"]) * 100:.2f}%</div>
                </div>
                <div>
                    <div class="prediction-k">Inference Time</div>
                    <div class="prediction-v">{float(result["inference_ms"]):.1f} ms</div>
                </div>
                <div>
                    <div class="prediction-k">Processing Method</div>
                    <div class="prediction-v">{result["processing"]}</div>
                </div>
                <div>
                    <div class="prediction-k">Model Type</div>
                    <div class="prediction-v">{result["model_type"]}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def probability_bars(probabilities: np.ndarray) -> None:
    sorted_rows = sorted(zip(CLASS_NAMES, probabilities), key=lambda row: row[1], reverse=True)
    for class_name, probability in sorted_rows:
        label_col, bar_col, value_col = st.columns([1.2, 4.2, 0.8])
        with label_col:
            st.write(CLASS_LABELS[class_name])
        with bar_col:
            st.progress(float(probability))
        with value_col:
            st.write(f"{float(probability) * 100:.2f}%")


def image_if_exists(path: Path, caption: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{caption}**")
        if path.exists():
            st.image(path, use_container_width=True)
        else:
            st.warning(f"Gambar tidak ditemukan: {path.relative_to(PROJECT_ROOT)}")


def show_metric_table(path: Path, title: str) -> None:
    if path.exists():
        st.markdown(f"**{title}**")
        st.dataframe(pd.read_csv(path), hide_index=True, use_container_width=True)


def show_model_evaluation() -> None:
    section_title("5. Model Evaluation")

    st.markdown("#### Accuracy Comparison")
    image_if_exists(FIGURES_DIR / "cnn_vs_mobilenet_comparison.png", "Accuracy Comparison")

    st.markdown("#### CNN Scratch")
    c1, c2 = st.columns(2)
    with c1:
        image_if_exists(FIGURES_DIR / "cnn_scratch_scenario_comparison.png", "CNN Scratch Scenario")
    with c2:
        image_if_exists(FIGURES_DIR / "cnn_scratch_confusion_comparison.png", "CNN Scratch Confusion Matrix")

    st.markdown("#### MobileNetV2")
    c1, c2 = st.columns(2)
    with c1:
        image_if_exists(FIGURES_DIR / "training_history_mobilenetv2_cpu.png", "MobileNetV2 Training History")
    with c2:
        image_if_exists(FIGURES_DIR / "confusion_matrix_mobilenetv2_cpu.png", "MobileNetV2 Confusion Matrix")

    st.markdown("#### Confusion Matrix")
    c1, c2 = st.columns(2)
    with c1:
        image_if_exists(FIGURES_DIR / "confusion_matrix_original.png", "Original")
    with c2:
        image_if_exists(FIGURES_DIR / "confusion_matrix_grabcut.png", "GrabCut")

    st.markdown("#### Histogram Enhancement")
    c1, c2 = st.columns(2)
    with c1:
        image_if_exists(FIGURES_DIR / "enhancement_comparison.png", "Enhancement Comparison")
    with c2:
        image_if_exists(FIGURES_DIR / "enhancement_confusion_matrices.png", "Enhancement Confusion Matrices")

    st.markdown("#### EDA")
    c1, c2 = st.columns(2)
    with c1:
        image_if_exists(EDA_FIGURES_DIR / "jumlah_gambar_per_kelas.png", "Jumlah Gambar per Kelas")
    with c2:
        image_if_exists(EDA_FIGURES_DIR / "distribusi_split_per_kelas.png", "Distribusi Split per Kelas")
    image_if_exists(EDA_FIGURES_DIR / "sample_gambar_dengan_ukuran.png", "Sample Gambar Dataset")

    with st.expander("Tabel metrik penelitian"):
        show_metric_table(METRICS_DIR / "cnn_scratch_results.csv", "CNN Scratch")
        show_metric_table(METRICS_DIR / "cnn_mobilenetv2_results.csv", "MobileNetV2")
        show_metric_table(METRICS_DIR / "enhancement_comparison_metrics.csv", "Image Enhancement")


def show_project_information() -> None:
    section_title("About This Research")
    st.markdown(
        """
        <div class="info-card">
            <h4>Tujuan Penelitian</h4>
            <p>
            Penelitian ini mengevaluasi pengaruh teknik image enhancement terhadap performa
            CNN Scratch dan MobileNetV2 dalam klasifikasi jenis sampah berbasis citra.
            Dashboard ini digunakan untuk mendemonstrasikan pipeline preprocessing,
            segmentasi, inference, dan hasil evaluasi model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="info-card">
                <h4>Workflow</h4>
                <p>Seleksi kelas, split dataset, enhancement, GrabCut, training model, evaluasi, dan visualisasi hasil.</p>
                <h4>Dataset</h4>
                <p>13348 images dengan 6 kelas: biological, cardboard, glass, metal, paper, dan plastic.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="info-card">
                <h4>Metode</h4>
                <p>CNN Scratch digunakan sebagai baseline custom CNN.</p>
                <p>MobileNetV2 digunakan sebagai pendekatan transfer learning.</p>
                <p>CLAHE, CLAHE + Gaussian, dan GrabCut digunakan untuk menganalisis pengaruh preprocessing.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        members = "".join(f"<li>{member}</li>" for member in GROUP_MEMBERS)
        st.markdown(
            f"""
            <div class="info-card">
                <h4>Kelompok</h4>
                <ul>{members}</ul>
                <p><strong>IPB University</strong><br>2026</p>
                <p>Repository GitHub:<br><a href="{GITHUB_REPOSITORY}" target="_blank">{GITHUB_REPOSITORY}</a></p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_style()
    sidebar()

    st.markdown(
        '<div class="topbar"><span>Research Dashboard</span><span>GitHub</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<h1 class="main-title">{APP_TITLE}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="main-subtitle">{APP_SUBTITLE}</p>', unsafe_allow_html=True)

    metric_cols = st.columns(4)
    with metric_cols[0]:
        metric_card("Best Accuracy", "88.94%", "MobileNetV2 Original", accent=True)
    with metric_cols[1]:
        metric_card("Best CNN Scratch", "64.65%", "CLAHE + Gaussian")
    with metric_cols[2]:
        metric_card("Dataset", "13348 Images", "Research dataset")
    with metric_cols[3]:
        metric_card("Classes", "6 Classes", "Biological, cardboard, glass, metal, paper, plastic")

    section_title("Unggah & Analisis Gambar")
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.info("Unggah gambar sampah untuk memperbarui pipeline preprocessing dan prediksi.")
        image_rgb = None
    else:
        try:
            image_rgb = load_uploaded_image(uploaded_file)
            st.markdown('<div class="preview-wrap">', unsafe_allow_html=True)
            st.image(image_rgb, caption="Preview gambar", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.session_state["uploaded_image_rgb"] = image_rgb
        except ValueError as exc:
            st.error(str(exc))
            image_rgb = None

    if image_rgb is None:
        section_title("1. Image Enhancement Pipeline")
        st.info("Pipeline akan muncul setelah gambar diunggah.")
        section_title("2. GrabCut Segmentation Pipeline")
        st.info("Tahapan GrabCut akan muncul setelah gambar diunggah.")
        section_title("3. Multi Model Prediction")
        st.info("Prediksi CNN Scratch dan MobileNetV2 akan muncul setelah gambar diunggah.")
        section_title("4. Probability Distribution")
        st.info("Distribusi probabilitas kelas akan muncul setelah gambar diunggah.")
    else:
        with st.spinner("Memproses image enhancement, GrabCut, dan inference model..."):
            analysis = run_analysis(image_rgb)

        section_title("1. Image Enhancement Pipeline")
        show_pipeline_cards(analysis["enhancement_pipeline"])

        section_title("2. GrabCut Segmentation Pipeline")
        show_pipeline_cards(analysis["grabcut_pipeline"])

        section_title("3. Multi Model Prediction")
        pred_cols = st.columns(2)
        with pred_cols[0]:
            prediction_card(analysis["predictions"]["CNN Scratch"])
        with pred_cols[1]:
            prediction_card(analysis["predictions"]["MobileNetV2"])

        section_title("4. Probability Distribution")
        prob_cols = st.columns(2)
        with prob_cols[0]:
            with st.container(border=True):
                st.markdown('<div class="prob-title">CNN Scratch</div>', unsafe_allow_html=True)
                probability_bars(analysis["predictions"]["CNN Scratch"]["probabilities"])
        with prob_cols[1]:
            with st.container(border=True):
                st.markdown('<div class="prob-title">MobileNetV2</div>', unsafe_allow_html=True)
                probability_bars(analysis["predictions"]["MobileNetV2"]["probabilities"])

    show_model_evaluation()
    show_project_information()


if __name__ == "__main__":
    main()
