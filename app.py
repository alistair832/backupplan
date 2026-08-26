from __future__ import annotations

from collections import Counter
from io import BytesIO
from pathlib import Path
import tempfile

import kagglehub
import pandas as pd
from PIL import Image
import streamlit as st
import yaml
from ultralytics import YOLO


# ============================================================
# PROJECT CONFIGURATION
# ============================================================
DATASET_HANDLE = "kapturovalexander/fruits-by-yolo-fruits-detection"
DATA_DIR = Path("data/fruits-yolo")
RUNS_DIR = Path("runs/fruit_detection")
TRAIN_RUN_NAME = "train"
DEFAULT_MODEL = "yolo11n.pt"
DEFAULT_BEST_MODEL = RUNS_DIR / TRAIN_RUN_NAME / "weights" / "best.pt"


# ============================================================
# STREAMLIT PAGE
# ============================================================
st.set_page_config(
    page_title="Smart Fruit Detection System",
    page_icon="🍎",
    layout="wide",
)

st.title("🍎 Smart Fruit Detection and Counting System")
st.caption(
    "One Python file for dataset download, YOLO training, image detection, "
    "fruit counting and result output in Streamlit."
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def find_dataset_yaml(search_root: Path) -> Path | None:
    """Find the YOLO data.yaml/dataset.yaml inside the downloaded dataset."""
    if not search_root.exists():
        return None

    candidates = sorted(search_root.rglob("data.yaml"))
    candidates += sorted(search_root.rglob("dataset.yaml"))

    return candidates[0] if candidates else None


def read_dataset_names(yaml_path: Path) -> list[str]:
    """Read class names from a YOLO dataset YAML file."""
    try:
        with yaml_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

        names = data.get("names", [])
        if isinstance(names, dict):
            return [str(names[key]) for key in sorted(names, key=lambda x: int(x))]
        if isinstance(names, list):
            return [str(name) for name in names]
    except Exception:
        pass

    return []


def download_dataset() -> Path:
    """Download the public Kaggle dataset and return the YOLO YAML path."""
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)

    kagglehub.dataset_download(
        DATASET_HANDLE,
        output_dir=str(DATA_DIR),
    )

    yaml_path = find_dataset_yaml(DATA_DIR)
    if yaml_path is None:
        raise FileNotFoundError(
            "Dataset downloaded, but no data.yaml or dataset.yaml was found."
        )

    return yaml_path


def get_or_download_dataset() -> Path:
    """Return an existing dataset YAML or download the dataset if missing."""
    yaml_path = find_dataset_yaml(DATA_DIR)
    if yaml_path is not None:
        return yaml_path
    return download_dataset()


def dataset_summary(yaml_path: Path) -> dict[str, object]:
    """Create a small dataset summary for the Streamlit interface."""
    class_names = read_dataset_names(yaml_path)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    image_count = sum(
        1
        for path in DATA_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in image_extensions
    )
    label_count = sum(1 for path in DATA_DIR.rglob("*.txt") if path.is_file())

    return {
        "yaml": str(yaml_path),
        "classes": class_names,
        "image_count": image_count,
        "label_count": label_count,
    }


def train_model(
    yaml_path: Path,
    epochs: int,
    image_size: int,
    batch_size: int,
    pretrained_model: str,
) -> Path:
    """Train YOLO using the Kaggle fruit dataset and return best.pt."""
    model = YOLO(pretrained_model)

    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        project=str(RUNS_DIR),
        name=TRAIN_RUN_NAME,
        exist_ok=True,
        patience=15,
        pretrained=True,
        plots=True,
    )

    if not DEFAULT_BEST_MODEL.exists():
        raise FileNotFoundError(
            f"Training finished but best.pt was not found at {DEFAULT_BEST_MODEL}."
        )

    return DEFAULT_BEST_MODEL


def save_uploaded_model(uploaded_model) -> Path:
    """Save an uploaded .pt model to a temporary file for YOLO inference."""
    suffix = Path(uploaded_model.name).suffix or ".pt"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(uploaded_model.getbuffer())
    temp_file.flush()
    temp_file.close()
    return Path(temp_file.name)


def run_detection(model_path: Path, image: Image.Image, confidence: float):
    """Run YOLO detection and return result data for Streamlit output."""
    model = YOLO(str(model_path))
    results = model.predict(
        source=image,
        conf=confidence,
        imgsz=640,
        verbose=False,
    )

    result = results[0]
    rows: list[dict[str, object]] = []
    detected_names: list[str] = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            fruit_name = str(result.names[class_id])

            detected_names.append(fruit_name)
            rows.append(
                {
                    "Fruit": fruit_name,
                    "Confidence (%)": round(score * 100, 2),
                    "X1": round(x1, 1),
                    "Y1": round(y1, 1),
                    "X2": round(x2, 1),
                    "Y2": round(y2, 1),
                }
            )

    counts = Counter(detected_names)
    details_df = pd.DataFrame(rows)

    annotated_bgr = result.plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]
    annotated_image = Image.fromarray(annotated_rgb)

    return counts, details_df, annotated_image


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose function",
    ["🏠 Home", "📥 Dataset & Training", "🔍 Image Detection"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Dataset")
st.sidebar.code(DATASET_HANDLE, language=None)
st.sidebar.caption("Single Streamlit entry point")
st.sidebar.code("streamlit run app.py", language=None)


# ============================================================
# HOME PAGE
# ============================================================
if page == "🏠 Home":
    st.subheader("Assignment Overview")

    st.markdown(
        """
        This assignment uses **Python, Ultralytics YOLO, KaggleHub and Streamlit**
        to build a fruit object-detection system.

        Everything is controlled from this single `app.py` file.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Application", "Streamlit")
    col2.metric("AI Model", "YOLO")
    col3.metric("Task", "Fruit Detection")

    st.subheader("System Functions")
    st.markdown(
        """
        1. Download the Kaggle fruit dataset directly.
        2. Locate the YOLO `data.yaml` automatically.
        3. Train a YOLO object-detection model.
        4. Use the trained `best.pt` model or upload another `.pt` model.
        5. Upload a fruit image.
        6. Detect and classify fruits in the image.
        7. Draw bounding boxes and confidence scores.
        8. Count each fruit type and total fruits.
        9. Display detailed detection results.
        10. Download the annotated image and CSV result.
        """
    )

    st.subheader("System Flow")
    st.code(
        """Kaggle Fruit Dataset
        ↓
Download with KaggleHub
        ↓
YOLO Training
        ↓
Trained best.pt
        ↓
Upload Fruit Image
        ↓
YOLO Detection
        ↓
Bounding Boxes + Fruit Class
        ↓
Confidence Score + Fruit Count
        ↓
Streamlit Output""",
        language=None,
    )


# ============================================================
# DATASET + TRAINING PAGE
# ============================================================
elif page == "📥 Dataset & Training":
    st.subheader("📥 Dataset Download and YOLO Training")

    st.info(
        "Dataset: Fruits by YOLO - Fruits Detection\n\n"
        "Kaggle handle: kapturovalexander/fruits-by-yolo-fruits-detection"
    )

    existing_yaml = find_dataset_yaml(DATA_DIR)

    if existing_yaml:
        st.success(f"Dataset is already available: {existing_yaml}")
        summary = dataset_summary(existing_yaml)

        col1, col2, col3 = st.columns(3)
        col1.metric("Images Found", summary["image_count"])
        col2.metric("Label Files", summary["label_count"])
        col3.metric("Classes", len(summary["classes"]))

        if summary["classes"]:
            st.write("**Detected classes:**")
            st.write(", ".join(summary["classes"]))
    else:
        st.warning("Dataset has not been downloaded yet.")

    if st.button("📥 Download / Check Dataset", type="primary"):
        try:
            with st.spinner("Downloading dataset from Kaggle..."):
                yaml_path = get_or_download_dataset()
            st.success("Dataset is ready.")
            st.code(str(yaml_path), language=None)
            st.rerun()
        except Exception as error:
            st.error(f"Dataset download failed: {error}")

    st.markdown("---")
    st.subheader("Train YOLO Model")

    col1, col2 = st.columns(2)
    with col1:
        epochs = st.number_input(
            "Epochs",
            min_value=1,
            max_value=300,
            value=30,
            step=1,
        )
        image_size = st.selectbox("Image size", [320, 416, 512, 640], index=3)

    with col2:
        batch_size = st.selectbox("Batch size", [2, 4, 8, 16], index=2)
        pretrained_model = st.selectbox(
            "Pretrained YOLO model",
            ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"],
            index=0,
        )

    st.caption(
        "Training can take significant time. A GPU is recommended. "
        "For a quick assignment test, start with 5-10 epochs; for a fuller run, use 30-50+ epochs."
    )

    if st.button("🚀 Start Training", type="primary"):
        try:
            with st.spinner("Preparing dataset..."):
                yaml_path = get_or_download_dataset()

            st.info(f"Training with: {yaml_path}")
            st.info("YOLO training has started. Training messages will also appear in the terminal.")

            with st.spinner("Training YOLO model..."):
                best_model = train_model(
                    yaml_path=yaml_path,
                    epochs=int(epochs),
                    image_size=int(image_size),
                    batch_size=int(batch_size),
                    pretrained_model=pretrained_model,
                )

            st.success("Training completed successfully.")
            st.code(str(best_model), language=None)

            model_size_mb = best_model.stat().st_size / (1024 * 1024)
            st.metric("best.pt size", f"{model_size_mb:.2f} MB")

            with best_model.open("rb") as file:
                st.download_button(
                    "⬇️ Download best.pt",
                    data=file.read(),
                    file_name="best.pt",
                    mime="application/octet-stream",
                )
        except Exception as error:
            st.error(f"Training failed: {error}")


# ============================================================
# IMAGE DETECTION PAGE
# ============================================================
elif page == "🔍 Image Detection":
    st.subheader("🔍 Fruit Image Detection")

    st.write(
        "Upload a fruit image and use either the model trained by this app "
        "or upload your own trained `best.pt`."
    )

    model_option = st.radio(
        "Model source",
        ["Use trained best.pt", "Upload .pt model"],
        horizontal=True,
    )

    selected_model_path: Path | None = None

    if model_option == "Use trained best.pt":
        if DEFAULT_BEST_MODEL.exists():
            selected_model_path = DEFAULT_BEST_MODEL
            st.success(f"Model found: {DEFAULT_BEST_MODEL}")
        else:
            st.warning(
                "No trained best.pt was found. Train the model under "
                "'Dataset & Training' or choose 'Upload .pt model'."
            )
    else:
        uploaded_model = st.file_uploader(
            "Upload trained YOLO model",
            type=["pt"],
            key="model_upload",
        )
        if uploaded_model is not None:
            selected_model_path = save_uploaded_model(uploaded_model)
            st.success(f"Uploaded model: {uploaded_model.name}")

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
    )

    uploaded_image = st.file_uploader(
        "Upload fruit image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_upload",
    )

    if uploaded_image is not None:
        input_image = Image.open(uploaded_image).convert("RGB")
        st.image(input_image, caption="Uploaded Image", width="stretch")

        if st.button("🍎 Detect Fruits", type="primary", disabled=selected_model_path is None):
            if selected_model_path is None:
                st.error("Please select or upload a trained YOLO model first.")
            else:
                try:
                    with st.spinner("Detecting fruits..."):
                        counts, details_df, annotated_image = run_detection(
                            selected_model_path,
                            input_image,
                            confidence,
                        )

                    st.success("Detection completed.")

                    left, right = st.columns(2)
                    with left:
                        st.subheader("Original Image")
                        st.image(input_image, width="stretch")
                    with right:
                        st.subheader("Detection Result")
                        st.image(annotated_image, width="stretch")

                    total_fruits = sum(counts.values())
                    fruit_types = len(counts)
                    average_confidence = (
                        float(details_df["Confidence (%)"].mean())
                        if not details_df.empty
                        else 0.0
                    )

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Fruits", total_fruits)
                    col2.metric("Fruit Types", fruit_types)
                    col3.metric("Average Confidence", f"{average_confidence:.2f}%")

                    st.subheader("Fruit Count")
                    if counts:
                        count_df = pd.DataFrame(
                            [
                                {"Fruit": fruit, "Count": count}
                                for fruit, count in sorted(counts.items())
                            ]
                        )
                        st.dataframe(count_df, width="stretch", hide_index=True)
                    else:
                        st.warning("No fruit was detected at the selected confidence threshold.")

                    st.subheader("Detection Details")
                    if not details_df.empty:
                        st.dataframe(details_df, width="stretch", hide_index=True)
                    else:
                        st.info("No detection details are available.")

                    csv_bytes = details_df.to_csv(index=False).encode("utf-8")
                    image_bytes = image_to_png_bytes(annotated_image)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "⬇️ Download Detection CSV",
                            data=csv_bytes,
                            file_name="fruit_detection_results.csv",
                            mime="text/csv",
                            width="stretch",
                        )
                    with col2:
                        st.download_button(
                            "⬇️ Download Detected Image",
                            data=image_bytes,
                            file_name="detected_fruits.png",
                            mime="image/png",
                            width="stretch",
                        )

                except Exception as error:
                    st.error(f"Detection failed: {error}")
