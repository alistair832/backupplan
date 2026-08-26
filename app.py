from __future__ import annotations

from collections import Counter
from io import BytesIO
from pathlib import Path
import shutil
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
RUNTIME_YAML = DATA_DIR / "streamlit_data.yaml"
RUNS_DIR = Path("runs/fruit_detection")
TRAIN_RUN_NAME = "train"
DEFAULT_BEST_MODEL = RUNS_DIR / TRAIN_RUN_NAME / "weights" / "best.pt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {
    "train": {"train", "training"},
    "val": {"val", "valid", "validation"},
    "test": {"test", "testing"},
}


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
    "One Python file for dataset download, dataset-path repair, YOLO training, "
    "image detection, fruit counting and result output in Streamlit."
)


# ============================================================
# DATASET HELPERS
# ============================================================
def find_source_yaml(search_root: Path) -> Path | None:
    """Find the original YOLO YAML supplied by the dataset."""
    if not search_root.exists():
        return None

    candidates = [
        path
        for path in sorted(search_root.rglob("data.yaml"))
        if path.resolve() != RUNTIME_YAML.resolve()
    ]
    candidates += [
        path
        for path in sorted(search_root.rglob("dataset.yaml"))
        if path.resolve() != RUNTIME_YAML.resolve()
    ]
    return candidates[0] if candidates else None


def read_yaml(yaml_path: Path) -> dict:
    with yaml_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def read_dataset_names(yaml_path: Path) -> list[str]:
    """Read class names from a YOLO dataset YAML file."""
    try:
        data = read_yaml(yaml_path)
        names = data.get("names", [])

        if isinstance(names, dict):
            try:
                keys = sorted(names, key=lambda item: int(item))
            except (TypeError, ValueError):
                keys = list(names)
            return [str(names[key]) for key in keys]

        if isinstance(names, list):
            return [str(name) for name in names]
    except Exception:
        pass

    return []


def directory_has_images(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    return any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in directory.iterdir()
    )


def classify_split_directory(directory: Path) -> str | None:
    """Support train/images and images/train style YOLO folder layouts."""
    name = directory.name.lower()
    parent = directory.parent.name.lower()

    if name == "images":
        split_name = parent
    elif parent == "images":
        split_name = name
    else:
        return None

    for canonical_name, aliases in SPLIT_ALIASES.items():
        if split_name in aliases:
            return canonical_name
    return None


def discover_split_image_dirs(search_root: Path) -> dict[str, Path]:
    """Find the real train/val/test image directories in the downloaded dataset."""
    found: dict[str, Path] = {}

    if not search_root.exists():
        return found

    for directory in search_root.rglob("*"):
        if not directory.is_dir():
            continue

        split = classify_split_directory(directory)
        if split and split not in found and directory_has_images(directory):
            found[split] = directory.resolve()

    return found


def count_images(directory: Path | None) -> int:
    if directory is None or not directory.exists():
        return 0
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def create_runtime_yaml(source_yaml: Path) -> Path:
    """Create a corrected YOLO YAML with absolute paths to real image folders."""
    splits = discover_split_image_dirs(DATA_DIR)

    if "train" not in splits or "val" not in splits:
        discovered = "\n".join(
            f"- {path.relative_to(DATA_DIR)}"
            for path in sorted(DATA_DIR.rglob("*"))
            if path.is_dir()
            and path.name.lower() in {"images", "train", "valid", "val", "test"}
        )
        raise FileNotFoundError(
            "The dataset YAML exists, but usable train/validation image folders "
            "could not be found. The download may be incomplete.\n\n"
            f"Folders discovered:\n{discovered or '- none'}"
        )

    source_data = read_yaml(source_yaml)
    names = source_data.get("names", [])

    runtime_data: dict[str, object] = {
        "train": str(splits["train"]),
        "val": str(splits["val"]),
        "names": names,
    }

    if "test" in splits:
        runtime_data["test"] = str(splits["test"])

    if isinstance(names, (list, dict)):
        runtime_data["nc"] = len(names)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with RUNTIME_YAML.open("w", encoding="utf-8") as file:
        yaml.safe_dump(runtime_data, file, sort_keys=False, allow_unicode=True)

    return RUNTIME_YAML


def validate_runtime_yaml(yaml_path: Path) -> tuple[bool, str]:
    """Check that train and validation image paths really exist."""
    try:
        data = read_yaml(yaml_path)
        for split in ("train", "val"):
            value = data.get(split)
            if not value:
                return False, f"Missing '{split}' path in {yaml_path}."

            path = Path(str(value))
            if not path.exists():
                return False, f"{split} images not found: {path}"
            if count_images(path) == 0:
                return False, f"No image files found inside: {path}"

        return True, "Dataset paths are valid."
    except Exception as error:
        return False, str(error)


def prepare_existing_dataset() -> Path | None:
    """Repair an already-downloaded dataset and return a valid runtime YAML."""
    source_yaml = find_source_yaml(DATA_DIR)
    if source_yaml is None:
        return None

    try:
        runtime_yaml = create_runtime_yaml(source_yaml)
        valid, _ = validate_runtime_yaml(runtime_yaml)
        return runtime_yaml if valid else None
    except Exception:
        return None


def download_dataset(force: bool = False) -> Path:
    """Download the complete Kaggle dataset and build a corrected local YAML."""
    if force and DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)

    kagglehub.dataset_download(
        DATASET_HANDLE,
        output_dir=str(DATA_DIR),
        force_download=force,
    )

    source_yaml = find_source_yaml(DATA_DIR)
    if source_yaml is None:
        raise FileNotFoundError(
            "Dataset download completed, but data.yaml/dataset.yaml was not found."
        )

    runtime_yaml = create_runtime_yaml(source_yaml)
    valid, message = validate_runtime_yaml(runtime_yaml)
    if not valid:
        raise FileNotFoundError(message)

    return runtime_yaml


def get_or_download_dataset() -> Path:
    """Return a verified local YAML, re-downloading an incomplete dataset if needed."""
    existing = prepare_existing_dataset()
    if existing is not None:
        return existing

    return download_dataset(force=True)


def dataset_summary(yaml_path: Path) -> dict[str, object]:
    class_names = read_dataset_names(yaml_path)
    data = read_yaml(yaml_path)

    train_path = Path(str(data["train"])) if data.get("train") else None
    val_path = Path(str(data["val"])) if data.get("val") else None
    test_path = Path(str(data["test"])) if data.get("test") else None

    return {
        "yaml": str(yaml_path),
        "classes": class_names,
        "train_images": count_images(train_path),
        "val_images": count_images(val_path),
        "test_images": count_images(test_path),
        "train_path": str(train_path) if train_path else "",
        "val_path": str(val_path) if val_path else "",
        "test_path": str(test_path) if test_path else "",
    }


# ============================================================
# YOLO HELPERS
# ============================================================
def train_model(
    yaml_path: Path,
    epochs: int,
    image_size: int,
    batch_size: int,
    pretrained_model: str,
) -> Path:
    """Train YOLO using the corrected absolute-path YAML and return best.pt."""
    valid, message = validate_runtime_yaml(yaml_path)
    if not valid:
        raise FileNotFoundError(message)

    model = YOLO(pretrained_model)
    model.train(
        data=str(yaml_path.resolve()),
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
    suffix = Path(uploaded_model.name).suffix or ".pt"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(uploaded_model.getbuffer())
    temp_file.flush()
    temp_file.close()
    return Path(temp_file.name)


def run_detection(model_path: Path, image: Image.Image, confidence: float):
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
# HOME
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
        1. Download and verify the Kaggle fruit dataset.
        2. Automatically repair YOLO dataset paths for the current environment.
        3. Train a YOLO object-detection model.
        4. Use the trained `best.pt` model or upload another `.pt` model.
        5. Upload a fruit image and run detection.
        6. Show bounding boxes, classes, confidence scores and fruit counts.
        7. Download the annotated image and CSV result.
        """
    )

    st.subheader("System Flow")
    st.code(
        """Kaggle Fruit Dataset
        ↓
Download + Verify Files
        ↓
Auto-detect Real train/valid/test Folders
        ↓
Create Corrected streamlit_data.yaml
        ↓
YOLO Training
        ↓
best.pt
        ↓
Upload Fruit Image
        ↓
Detection + Counting
        ↓
Streamlit Output""",
        language=None,
    )


# ============================================================
# DATASET + TRAINING
# ============================================================
elif page == "📥 Dataset & Training":
    st.subheader("📥 Dataset Download and YOLO Training")
    st.info(
        "Dataset: Fruits by YOLO - Fruits Detection\n\n"
        "Kaggle handle: kapturovalexander/fruits-by-yolo-fruits-detection"
    )

    existing_yaml = prepare_existing_dataset()

    if existing_yaml is not None:
        valid, validation_message = validate_runtime_yaml(existing_yaml)
        summary = dataset_summary(existing_yaml)

        if valid:
            st.success("Dataset is ready and the image paths are valid.")
        else:
            st.error(validation_message)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Train Images", summary["train_images"])
        col2.metric("Validation Images", summary["val_images"])
        col3.metric("Test Images", summary["test_images"])
        col4.metric("Classes", len(summary["classes"]))

        with st.expander("Show resolved dataset paths"):
            st.code(
                f"Training:   {summary['train_path']}\n"
                f"Validation: {summary['val_path']}\n"
                f"Test:       {summary['test_path'] or 'Not provided'}\n"
                f"YOLO YAML:  {summary['yaml']}",
                language=None,
            )

        if summary["classes"]:
            st.write("**Detected classes:**")
            st.write(", ".join(summary["classes"]))
    else:
        st.warning(
            "A complete dataset is not available yet, or the previous download "
            "is incomplete. Click the button below to download/repair it."
        )

    left_button, right_button = st.columns(2)

    with left_button:
        if st.button("📥 Download / Check Dataset", type="primary", width="stretch"):
            try:
                with st.spinner("Checking dataset and repairing paths..."):
                    yaml_path = get_or_download_dataset()
                st.success("Dataset is complete and ready for YOLO.")
                st.code(str(yaml_path), language=None)
                st.rerun()
            except Exception as error:
                st.error(f"Dataset preparation failed: {error}")

    with right_button:
        if st.button("🔧 Force Re-download Dataset", width="stretch"):
            try:
                with st.spinner("Deleting incomplete copy and downloading all files again..."):
                    yaml_path = download_dataset(force=True)
                st.success("Dataset was downloaded again and repaired successfully.")
                st.code(str(yaml_path), language=None)
                st.rerun()
            except Exception as error:
                st.error(f"Dataset re-download failed: {error}")

    st.markdown("---")
    st.subheader("Train YOLO Model")

    col1, col2 = st.columns(2)
    with col1:
        epochs = st.number_input(
            "Epochs",
            min_value=1,
            max_value=300,
            value=10,
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
        "For a quick assignment test, use 5-10 epochs. "
        "For a fuller training run, use 30-50+ epochs. A GPU is recommended."
    )

    if st.button("🚀 Start Training", type="primary"):
        try:
            with st.spinner("Preparing and validating dataset..."):
                yaml_path = get_or_download_dataset()
                valid, message = validate_runtime_yaml(yaml_path)

            if not valid:
                raise FileNotFoundError(message)

            st.success("Dataset paths validated.")
            st.info(f"Training with corrected YAML: {yaml_path.resolve()}")

            summary = dataset_summary(yaml_path)
            st.code(
                f"Train: {summary['train_path']}\n"
                f"Val:   {summary['val_path']}",
                language=None,
            )

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
            st.info(
                "If this is an 'images not found' error, click "
                "'Force Re-download Dataset' once, then start training again."
            )


# ============================================================
# IMAGE DETECTION
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

        if st.button(
            "🍎 Detect Fruits",
            type="primary",
            disabled=selected_model_path is None,
        ):
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
                        st.warning(
                            "No fruit was detected at the selected confidence threshold."
                        )

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
