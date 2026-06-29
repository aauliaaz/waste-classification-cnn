from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.segmentation import apply_mask, clean_mask, grabcut_segmentation

    SEGMENTATION_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - shown in the Streamlit UI.
    apply_mask = None
    clean_mask = None
    grabcut_segmentation = None
    SEGMENTATION_IMPORT_ERROR = exc


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

BEST_MOBILENET_ACCURACY = "88.94%"
BEST_CNN_ACCURACY = "64.65%"
GRABCUT_BASE_SIZE = (224, 224)

GROUP_MEMBERS = [
    "Fatiyya Ilmi Zahra",
    "Aulia Rahmasyifa Az Zahra",
    "Jelita Wahyuningtyas",
    "Felicia",
    "Johanna Davina Habeahan",
]
GITHUB_REPOSITORY = "https://github.com/aauliaaz/waste-classification-cnn"

PREPROCESSING_OPTIONS = ["Original", "CLAHE", "CLAHE + Gaussian", "GrabCut"]
SCENARIO_BY_PREPROCESSING = {
    "Original": "original",
    "CLAHE": "clahe",
    "CLAHE + Gaussian": "clahe_gaussian",
    "GrabCut": "original_grabcut",
}

MODEL_CONFIGS = {
    "CNN Scratch": {
        "input_size": (224, 224),
        "paths": {
            "Original": MODELS_DIR / "cnn_original_baseline.keras",
            "CLAHE": MODELS_DIR / "cnn_clahe.keras",
            "CLAHE + Gaussian": MODELS_DIR / "cnn_clahe_gaussian.keras",
            "GrabCut": MODELS_DIR / "cnn_grabcut.keras",
        },
    },
    "MobileNetV2": {
        "input_size": (192, 192),
        "paths": {
            "Original": MODELS_DIR / "original_finetune.keras",
            "CLAHE": MODELS_DIR / "clahe_finetune.keras",
            "CLAHE + Gaussian": MODELS_DIR / "clahe_gaussian_finetune.keras",
            "GrabCut": MODELS_DIR / "grabcut_original_finetune.keras",
        },
    },
}


st.set_page_config(
    page_title="Waste Classification Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .hero-card {
            background: #ffffff;
            border: 1px solid #e7ebf0;
            border-radius: 8px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
            padding: 30px 30px 22px 30px;
            margin-bottom: 18px;
        }
        .hero-title {
            color: #111827;
            font-size: 2.5rem;
            font-weight: 760;
            letter-spacing: 0;
            line-height: 1.08;
            margin: 0 0 12px 0;
        }
        .hero-subtitle {
            color: #4b5563;
            font-size: 1.05rem;
            line-height: 1.6;
            max-width: 900px;
            margin: 0;
        }
        .section-kicker {
            color: #0f766e;
            font-size: 0.78rem;
            font-weight: 760;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.2rem;
        }
        .section-title {
            color: #111827;
            font-size: 1.55rem;
            font-weight: 720;
            letter-spacing: 0;
            margin: 0 0 0.25rem 0;
        }
        .section-subtitle {
            color: #6b7280;
            font-size: 0.98rem;
            margin: 0 0 1rem 0;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #edf0f4;
            border-radius: 8px;
            padding: 16px 18px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.045);
        }
        div[data-testid="stMetricLabel"] p {
            color: #64748b;
            font-weight: 650;
        }
        div[data-testid="stMetricValue"] {
            color: #111827;
            font-weight: 760;
        }
        .stButton > button {
            width: 100%;
            border-radius: 8px;
            border: 1px solid #0f766e;
            background: #0f766e;
            color: white;
            font-weight: 700;
            min-height: 44px;
        }
        .stButton > button:hover {
            border-color: #115e59;
            background: #115e59;
            color: white;
        }
        [data-testid="stFileUploaderDropzone"] {
            border-radius: 8px;
            border: 1px dashed #94a3b8;
            background: #fbfcfd;
            padding: 18px;
        }
        .info-card {
            background: #ffffff;
            border: 1px solid #edf0f4;
            border-radius: 8px;
            padding: 18px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.045);
            height: 100%;
        }
        .small-muted {
            color: #64748b;
            font-size: 0.92rem;
            line-height: 1.55;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(index: int, title: str, subtitle: str | None = None) -> None:
    st.markdown(
        f"""
        <div class="section-kicker">Section {index}</div>
        <div class="section-title">{title}</div>
        """,
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(f'<p class="section-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def load_uploaded_image(uploaded_file) -> np.ndarray:
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image)
        return np.array(image.convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("File gambar tidak dapat dibaca.") from exc


def resize_to(image_rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image_rgb, size, interpolation=cv2.INTER_AREA)


def resize_long_side(image_rgb: np.ndarray, max_side: int = 640) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    largest = max(height, width)
    if largest <= max_side:
        return image_rgb.copy()

    scale = max_side / float(largest)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image_rgb, new_size, interpolation=cv2.INTER_AREA)


def apply_clahe_rgb(image_rgb: np.ndarray) -> np.ndarray:
    # CLAHE parameters follow the existing notebook pipeline.
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def apply_clahe_gaussian_rgb(image_rgb: np.ndarray) -> np.ndarray:
    enhanced = apply_clahe_rgb(image_rgb)
    return cv2.GaussianBlur(enhanced, (3, 3), 0)


def apply_grabcut_rgb(image_rgb: np.ndarray, target_size: tuple[int, int] = GRABCUT_BASE_SIZE) -> np.ndarray:
    if SEGMENTATION_IMPORT_ERROR is not None:
        raise RuntimeError(f"Gagal import src.segmentation: {SEGMENTATION_IMPORT_ERROR}")

    resized_rgb = resize_to(image_rgb, target_size)
    image_bgr = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
    mask = clean_mask(grabcut_segmentation(image_bgr))
    segmented_bgr = apply_mask(image_bgr, mask)
    return cv2.cvtColor(segmented_bgr, cv2.COLOR_BGR2RGB)


def build_pipeline_images(image_rgb: np.ndarray) -> dict[str, np.ndarray]:
    display_source = resize_long_side(image_rgb, max_side=640)
    clahe = apply_clahe_rgb(display_source)
    clahe_gaussian = apply_clahe_gaussian_rgb(display_source)
    grabcut = apply_grabcut_rgb(display_source, GRABCUT_BASE_SIZE)
    return {
        "Original": display_source,
        "CLAHE": clahe,
        "CLAHE + Gaussian": clahe_gaussian,
        "GrabCut": grabcut,
    }


def preprocess_for_model(
    image_rgb: np.ndarray,
    preprocessing_name: str,
    input_size: tuple[int, int],
) -> np.ndarray:
    if preprocessing_name == "Original":
        processed = image_rgb
    elif preprocessing_name == "CLAHE":
        processed = apply_clahe_rgb(image_rgb)
    elif preprocessing_name == "CLAHE + Gaussian":
        processed = apply_clahe_gaussian_rgb(image_rgb)
    elif preprocessing_name == "GrabCut":
        processed = apply_grabcut_rgb(image_rgb, GRABCUT_BASE_SIZE)
    else:
        raise ValueError(f"Metode preprocessing tidak dikenal: {preprocessing_name}")

    if processed.shape[:2] != (input_size[1], input_size[0]):
        processed = resize_to(processed, input_size)
    return processed.astype(np.uint8)


@st.cache_resource(show_spinner=False)
def load_model(model_name: str, preprocessing_name: str):
    model_path = MODEL_CONFIGS[model_name]["paths"][preprocessing_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
    if model_path.stat().st_size < 2048:
        raise RuntimeError(
            f"Model {model_path.name} terlihat seperti pointer Git LFS, bukan file model penuh."
        )
    return tf.keras.models.load_model(model_path, compile=False)


def predict_image(
    image_rgb: np.ndarray,
    model_name: str,
    preprocessing_name: str,
) -> dict[str, object]:
    input_size = MODEL_CONFIGS[model_name]["input_size"]
    processed = preprocess_for_model(image_rgb, preprocessing_name, input_size)
    model = load_model(model_name, preprocessing_name)
    batch = np.expand_dims(processed.astype("float32"), axis=0)

    start = time.perf_counter()
    prediction = model.predict(batch, verbose=0)
    elapsed_ms = (time.perf_counter() - start) * 1000

    probabilities = np.asarray(prediction[0], dtype=np.float64)
    if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0, atol=1e-3):
        exp_values = np.exp(probabilities - np.max(probabilities))
        probabilities = exp_values / exp_values.sum()

    best_index = int(np.argmax(probabilities))
    return {
        "processed": processed,
        "probabilities": probabilities,
        "label": CLASS_NAMES[best_index],
        "confidence": float(probabilities[best_index]),
        "inference_time_ms": elapsed_ms,
        "scenario": SCENARIO_BY_PREPROCESSING[preprocessing_name],
    }


def show_metric_cards() -> None:
    cols = st.columns(4)
    metrics = [
        ("Best MobileNetV2 Accuracy", BEST_MOBILENET_ACCURACY),
        ("Best CNN Scratch Accuracy", BEST_CNN_ACCURACY),
        ("Dataset", "6 Classes"),
        ("Models", "CNN Scratch + MobileNetV2"),
    ]
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label, value)


def show_pipeline(pipeline_images: dict[str, np.ndarray]) -> None:
    cols = st.columns(len(PREPROCESSING_OPTIONS))
    for col, step_name in zip(cols, PREPROCESSING_OPTIONS):
        with col:
            with st.container(border=True):
                st.markdown(f"**{step_name}**")
                st.image(pipeline_images[step_name], use_container_width=True)


def show_prediction_result(
    original_rgb: np.ndarray,
    result: dict[str, object],
    model_name: str,
    preprocessing_name: str,
) -> None:
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("**Original Image**")
            st.image(original_rgb, use_container_width=True)
    with right:
        with st.container(border=True):
            st.markdown("**Processed Image**")
            st.image(result["processed"], use_container_width=True)

    st.write("")
    cols = st.columns(5)
    cols[0].metric("Prediction Label", CLASS_LABELS[str(result["label"])])
    cols[1].metric("Confidence", f"{float(result['confidence']) * 100:.2f}%")
    cols[2].metric("Inference Time", f"{float(result['inference_time_ms']):.1f} ms")
    cols[3].metric("Processing Method", preprocessing_name)
    cols[4].metric("Model Used", model_name)


def show_class_probabilities(probabilities: np.ndarray) -> None:
    probability_rows = sorted(
        zip(CLASS_NAMES, probabilities),
        key=lambda item: item[1],
        reverse=True,
    )
    for class_name, probability in probability_rows:
        label_col, bar_col, value_col = st.columns([1.1, 4, 0.7])
        with label_col:
            st.write(CLASS_LABELS[class_name])
        with bar_col:
            st.progress(float(probability))
        with value_col:
            st.write(f"{float(probability) * 100:.2f}%")


def image_if_exists(path: Path, caption: str | None = None) -> None:
    if path.exists():
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Gambar tidak ditemukan: {path.relative_to(PROJECT_ROOT)}")


def show_metric_table(path: Path, title: str) -> None:
    if path.exists():
        st.markdown(f"**{title}**")
        st.dataframe(pd.read_csv(path), use_container_width=True, hide_index=True)


def show_model_performance() -> None:
    section_header(6, "Our Model Performance")

    st.markdown("#### Accuracy Comparison")
    image_if_exists(FIGURES_DIR / "cnn_vs_mobilenet_comparison.png")
    c1, c2 = st.columns(2)
    with c1:
        image_if_exists(FIGURES_DIR / "cnn_scratch_scenario_comparison.png")
    with c2:
        image_if_exists(FIGURES_DIR / "enhancement_comparison.png")

    st.markdown("#### Confusion Matrix")
    c1, c2 = st.columns(2)
    with c1:
        image_if_exists(FIGURES_DIR / "cnn_scratch_confusion_comparison.png", "CNN Scratch")
    with c2:
        image_if_exists(FIGURES_DIR / "confusion_matrix_mobilenetv2_cpu.png", "MobileNetV2")

    st.markdown("#### CNN Scratch")
    c1, c2 = st.columns(2)
    with c1:
        image_if_exists(RESULTS_DIR / "cnn" / "training_history_original.png", "CNN Scratch - Original")
    with c2:
        image_if_exists(RESULTS_DIR / "cnn" / "training_history_grabcut.png", "CNN Scratch - GrabCut")

    st.markdown("#### MobileNetV2")
    c1, c2 = st.columns(2)
    with c1:
        image_if_exists(FIGURES_DIR / "training_history_mobilenetv2_cpu.png")
    with c2:
        image_if_exists(FIGURES_DIR / "confusion_matrix_mobilenetv2_cpu.png")

    st.markdown("#### Histogram Enhancement")
    c1, c2 = st.columns(2)
    with c1:
        image_if_exists(FIGURES_DIR / "enhancement_comparison.png")
    with c2:
        image_if_exists(FIGURES_DIR / "enhancement_confusion_matrices.png")

    st.markdown("#### EDA")
    c1, c2 = st.columns(2)
    with c1:
        image_if_exists(EDA_FIGURES_DIR / "jumlah_gambar_per_kelas.png")
    with c2:
        image_if_exists(EDA_FIGURES_DIR / "distribusi_split_per_kelas.png")
    image_if_exists(EDA_FIGURES_DIR / "sample_gambar_dengan_ukuran.png")

    with st.expander("Tabel metrik"):
        show_metric_table(METRICS_DIR / "cnn_scratch_results.csv", "CNN Scratch")
        show_metric_table(METRICS_DIR / "cnn_mobilenetv2_results.csv", "MobileNetV2")
        show_metric_table(METRICS_DIR / "enhancement_comparison_metrics.csv", "Image Enhancement")


def show_project_information() -> None:
    section_header(7, "Project Information")

    st.markdown(
        """
        <div class="info-card">
        <p class="small-muted">
        Project ini menganalisis pengaruh image enhancement terhadap kinerja model
        klasifikasi sampah berbasis deep learning. Eksperimen membandingkan CNN Scratch
        dan MobileNetV2 pada enam kelas sampah dengan beberapa skenario preprocessing.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("**Workflow Project**")
            st.write("1. Seleksi 6 kelas sampah")
            st.write("2. Split dataset train, validation, test")
            st.write("3. Image enhancement")
            st.write("4. Segmentasi GrabCut")
            st.write("5. Training CNN Scratch dan MobileNetV2")
            st.write("6. Evaluasi dan komparasi hasil")
    with c2:
        with st.container(border=True):
            st.markdown("**Dataset dan Metode**")
            st.write("Dataset: 6 classes")
            st.write("Classes: Biological, Cardboard, Glass, Metal, Paper, Plastic")
            st.write("Image Enhancement: Original, CLAHE, CLAHE + Gaussian")
            st.write("Segmentation: GrabCut")
    with c3:
        with st.container(border=True):
            st.markdown("**Model**")
            st.write("CNN Scratch: baseline convolutional neural network")
            st.write("MobileNetV2: transfer learning dengan fine-tuning")
            st.write(f"Best CNN Scratch Accuracy: {BEST_CNN_ACCURACY}")
            st.write(f"Best MobileNetV2 Accuracy: {BEST_MOBILENET_ACCURACY}")

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("**Anggota Kelompok**")
            for member in GROUP_MEMBERS:
                st.write(member)
    with c2:
        with st.container(border=True):
            st.markdown("**IPB University**")
            st.write("2026")
            st.markdown(f"Repository GitHub: [{GITHUB_REPOSITORY}]({GITHUB_REPOSITORY})")


def main() -> None:
    inject_style()

    with st.container():
        st.markdown(
            """
            <div class="hero-card">
                <h1 class="hero-title">Waste Classification</h1>
                <p class="hero-subtitle">
                Analisis Pengaruh Teknik Image Enhancement terhadap Kinerja CNN Scratch
                dan MobileNetV2 pada Klasifikasi Jenis Sampah
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        show_metric_cards()

    st.divider()

    section_header(2, "Upload", "Upload image, pilih model, lalu jalankan prediksi.")
    upload_col, config_col = st.columns([1.4, 1])
    uploaded_file = None
    original_rgb = None

    with upload_col:
        with st.container(border=True):
            uploaded_file = st.file_uploader(
                "Upload Image",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                accept_multiple_files=False,
            )
            if uploaded_file is not None:
                try:
                    original_rgb = load_uploaded_image(uploaded_file)
                    st.image(original_rgb, caption="Preview gambar", use_container_width=True)
                except ValueError as exc:
                    st.error(str(exc))
                    original_rgb = None

    with config_col:
        with st.container(border=True):
            model_name = st.radio("Pilihan Model", ["CNN Scratch", "MobileNetV2"], horizontal=True)
            preprocessing_name = st.selectbox("Pilihan Preprocessing", PREPROCESSING_OPTIONS)
            predict_clicked = st.button("Predict", type="primary")

    result = None
    pipeline_images = None

    if predict_clicked:
        if original_rgb is None:
            st.warning("Upload image terlebih dahulu.")
        else:
            try:
                with st.spinner("Memproses gambar dan menjalankan inference..."):
                    pipeline_images = build_pipeline_images(original_rgb)
                    result = predict_image(original_rgb, model_name, preprocessing_name)
                st.session_state["last_prediction"] = {
                    "original_rgb": original_rgb,
                    "pipeline_images": pipeline_images,
                    "result": result,
                    "model_name": model_name,
                    "preprocessing_name": preprocessing_name,
                }
            except Exception as exc:
                st.error(f"Prediksi gagal: {exc}")

    last_prediction = st.session_state.get("last_prediction")
    if last_prediction is not None:
        st.divider()
        section_header(3, "Image Processing Pipeline")
        show_pipeline(last_prediction["pipeline_images"])

        st.divider()
        section_header(4, "Prediction Result")
        show_prediction_result(
            last_prediction["original_rgb"],
            last_prediction["result"],
            last_prediction["model_name"],
            last_prediction["preprocessing_name"],
        )

        st.divider()
        section_header(5, "Class Probability")
        show_class_probabilities(last_prediction["result"]["probabilities"])

    st.divider()
    show_model_performance()

    st.divider()
    show_project_information()


if __name__ == "__main__":
    main()
