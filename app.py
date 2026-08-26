from __future__ import annotations

from collections import Counter
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

import kagglehub
import pandas as pd
from PIL import Image
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================
DATASET = "kapturovalexander/fruits-by-yolo-fruits-detection"
ROOT = Path.cwd()
RUNTIME = ROOT / ".runtime"
RAW = RUNTIME / "raw"
PREPARED = RUNTIME / "fruit_dataset"
RUNS = RUNTIME / "runs"
REPORT_FILE = RUNTIME / "report.json"
ZIP_FILE = RUNTIME / "archive.zip"
BEST_MODEL = RUNS / "train" / "weights" / "best.pt"
UPLOADED_MODEL = RUNTIME / "uploaded_model.pt"

SPLITS = {
    "train": "train",
    "training": "train",
    "valid": "val",
    "validation": "val",
    "val": "val",
    "test": "test",
    "testing": "test",
}

PYTHON_VERSION = (
    f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
)


# ============================================================
# STREAMLIT PAGE
# ============================================================
st.set_page_config(
    page_title="Fruit Classification System",
    page_icon="🍎",
    layout="wide",
)

st.title("🍎 Fruit Image Classification System")
st.caption(
    "One Streamlit app: import dataset → prepare labels → train YOLO classifier → recognize fruit."
)

# Do not import Ultralytics at module startup. This keeps Streamlit alive even
# if an incompatible Python/OpenCV environment is selected on the deployment.
if sys.version_info >= (3, 14):
    st.error(
        "This deployment is currently using Python "
        f"{PYTHON_VERSION}. Ultralytics/OpenCV can fail to import in this "
        "environment. This repository is configured for Python 3.12. "
        "Reboot/redeploy the Streamlit app so the new runtime is used."
    )


# ============================================================
# ML IMPORT
# ============================================================
def get_yolo_class():
    """Import Ultralytics only when training or prediction is requested."""
    try:
        from ultralytics import YOLO

        return YOLO
    except Exception as error:
        raise RuntimeError(
            "Ultralytics/OpenCV could not be loaded. The project is configured "
            "for Python 3.12 and includes the Linux OpenCV libraries required by "
            "Streamlit Cloud. Reboot/redeploy the app after the latest GitHub "
            f"update. Current Python: {PYTHON_VERSION}. Original error: {error}"
        ) from error


# ============================================================
# DATASET HELPERS
# ============================================================
def safe_extract(zip_path: Path, destination: Path) -> None:
    """Extract a ZIP while blocking path traversal."""
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination_root and destination_root not in target.parents:
                raise ValueError(f"Unsafe ZIP path: {member.filename}")
        archive.extractall(destination)


def clear_runtime(clear_models: bool = False) -> None:
    """Remove imported/prepared data and optionally trained models."""
    for path in (RAW, PREPARED):
        if path.exists():
            shutil.rmtree(path)

    for path in (ZIP_FILE, REPORT_FILE):
        if path.exists():
            path.unlink()

    if clear_models and RUNS.exists():
        shutil.rmtree(RUNS)


def link_or_copy(source: Path, destination: Path) -> None:
    """Hard-link images when possible, otherwise copy them."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare_dataset(source: Path, source_name: str) -> dict:
    """
    Convert the supplied _classes.csv layout into YOLO classification folders:
    train/ClassName, val/ClassName and test/ClassName.
    """
    csv_files = sorted(source.rglob("_classes.csv"))
    if not csv_files:
        raise FileNotFoundError(
            "No _classes.csv files were found. Please upload the supplied archive.zip."
        )

    if PREPARED.exists():
        shutil.rmtree(PREPARED)
    PREPARED.mkdir(parents=True, exist_ok=True)

    counts: Counter[tuple[str, str]] = Counter()
    classes: list[str] = []
    skipped: list[dict[str, str]] = []

    for csv_path in csv_files:
        split = SPLITS.get(csv_path.parent.name.lower())
        if split is None:
            continue

        with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            headers = reader.fieldnames or []
            if len(headers) < 2:
                continue

            filename_column = headers[0]
            class_columns = [(header, header.strip()) for header in headers[1:]]

            for _, class_name in class_columns:
                if class_name and class_name not in classes:
                    classes.append(class_name)

            for row in reader:
                filename = (row.get(filename_column) or "").strip()
                active_classes: list[str] = []

                for raw_header, clean_name in class_columns:
                    try:
                        value = float((row.get(raw_header) or "0").strip() or 0)
                    except ValueError:
                        value = 0.0

                    if value > 0.5:
                        active_classes.append(clean_name)

                image_path = csv_path.parent / filename

                if not filename or not image_path.exists():
                    skipped.append(
                        {
                            "split": split,
                            "file": filename,
                            "reason": "missing image",
                        }
                    )
                    continue

                if len(active_classes) != 1:
                    reason = (
                        "no label"
                        if not active_classes
                        else "multiple labels: " + ", ".join(active_classes)
                    )
                    skipped.append(
                        {
                            "split": split,
                            "file": filename,
                            "reason": reason,
                        }
                    )
                    continue

                label = active_classes[0]
                output_image = PREPARED / split / label / filename
                link_or_copy(image_path, output_image)
                counts[(split, label)] += 1

    if not (PREPARED / "train").exists():
        raise RuntimeError("Training images could not be prepared from the CSV files.")

    if not (PREPARED / "val").exists():
        raise RuntimeError("Validation images could not be prepared from the CSV files.")

    report = {
        "source": source_name,
        "classes": classes,
        "usable": sum(counts.values()),
        "skipped": skipped,
        "totals": {
            split: sum(
                count for (current_split, _), count in counts.items() if current_split == split
            )
            for split in ("train", "val", "test")
        },
        "counts": {
            split: {class_name: counts.get((split, class_name), 0) for class_name in classes}
            for split in ("train", "val", "test")
        },
    }

    RUNTIME.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_report() -> dict | None:
    if not REPORT_FILE.exists():
        return None

    try:
        return json.loads(REPORT_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def dataset_ready() -> bool:
    return (PREPARED / "train").exists() and (PREPARED / "val").exists()


def distribution(report: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Fruit": class_name,
                "Train": report["counts"]["train"].get(class_name, 0),
                "Validation": report["counts"]["val"].get(class_name, 0),
                "Test": report["counts"]["test"].get(class_name, 0),
            }
            for class_name in report["classes"]
        ]
    )


# ============================================================
# MODEL HELPERS
# ============================================================
def train_model(epochs: int, image_size: int, batch_size: int, model_name: str) -> Path:
    if not dataset_ready():
        raise RuntimeError("Import and prepare the dataset first.")

    YOLO = get_yolo_class()

    training_run = RUNS / "train"
    if training_run.exists():
        shutil.rmtree(training_run)

    model = YOLO(model_name)
    model.train(
        data=str(PREPARED.resolve()),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        project=str(RUNS.resolve()),
        name="train",
        exist_ok=True,
        patience=10,
        pretrained=True,
        plots=True,
    )

    if not BEST_MODEL.exists():
        raise FileNotFoundError("Training finished but best.pt was not created.")

    return BEST_MODEL


def save_model(uploaded_file) -> Path:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with UPLOADED_MODEL.open("wb") as file:
        uploaded_file.seek(0)
        shutil.copyfileobj(uploaded_file, file)
    return UPLOADED_MODEL


def predict(model_path: Path, image: Image.Image) -> pd.DataFrame:
    YOLO = get_yolo_class()
    result = YOLO(str(model_path)).predict(
        source=image,
        imgsz=224,
        verbose=False,
    )[0]

    if result.probs is None:
        raise RuntimeError("This .pt file is not a YOLO classification model.")

    probabilities = result.probs.data.detach().cpu().tolist()
    top_indexes = sorted(
        range(len(probabilities)),
        key=lambda index: probabilities[index],
        reverse=True,
    )[:5]

    return pd.DataFrame(
        [
            {
                "Rank": rank,
                "Fruit": str(result.names[index]),
                "Confidence (%)": round(probabilities[index] * 100, 2),
            }
            for rank, index in enumerate(top_indexes, 1)
        ]
    )


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose function",
    ["🏠 Home", "📦 Dataset & Training", "🔎 Fruit Recognition"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Runtime")
st.sidebar.code(f"Python {PYTHON_VERSION}", language=None)
st.sidebar.caption("Single Streamlit file")
st.sidebar.code("streamlit run app.py", language=None)


# ============================================================
# HOME
# ============================================================
if page == "🏠 Home":
    st.subheader("Assignment Overview")
    st.markdown(
        """
        The supplied dataset contains fruit images with **image-level labels in
        `_classes.csv`**. It does not contain YOLO bounding-box text annotations,
        so this project correctly uses **YOLO image classification**.

        Everything runs through this single `app.py` Streamlit application.
        """
    )

    column1, column2, column3 = st.columns(3)
    column1.metric("Application", "Streamlit")
    column2.metric("AI Model", "YOLO Classification")
    column3.metric("Fruit Classes", "9")

    st.write(
        "**Classes:** Apple, Banana, Grapes, Kiwi, Mango, Orange, "
        "Pineapple, Sugerapple, Watermelon"
    )

    st.subheader("System Flow")
    st.code(
        """archive.zip / Kaggle dataset
        ↓
Read _classes.csv
        ↓
Prepare train/Class, val/Class, test/Class folders
        ↓
Train YOLO classification model
        ↓
best.pt
        ↓
Upload fruit image
        ↓
Predicted fruit + confidence + Top-5 results""",
        language=None,
    )


# ============================================================
# DATASET + TRAINING
# ============================================================
elif page == "📦 Dataset & Training":
    st.subheader("📦 Import and Prepare Dataset")
    st.info(
        "The app uses the real `_classes.csv` labels and converts them into "
        "the folder structure required for YOLO classification."
    )

    source = st.radio(
        "Dataset source",
        ["Upload supplied archive.zip", "Download same Kaggle dataset"],
    )

    if source == "Upload supplied archive.zip":
        uploaded_zip = st.file_uploader("Upload archive.zip", type=["zip"])

        if st.button(
            "📥 Import ZIP",
            type="primary",
            disabled=uploaded_zip is None,
        ):
            try:
                with st.spinner("Extracting and preparing dataset..."):
                    clear_runtime(False)
                    RUNTIME.mkdir(parents=True, exist_ok=True)

                    with ZIP_FILE.open("wb") as file:
                        uploaded_zip.seek(0)
                        shutil.copyfileobj(uploaded_zip, file)

                    safe_extract(ZIP_FILE, RAW)
                    report = prepare_dataset(RAW, "Uploaded archive.zip")

                st.success(
                    f"Prepared {report['usable']} usable images. "
                    f"Skipped {len(report['skipped'])} ambiguous/unlabelled rows."
                )
                st.rerun()
            except Exception as error:
                st.error(f"Import failed: {error}")

    else:
        st.code(DATASET, language=None)

        if st.button("🌐 Download and Prepare", type="primary"):
            try:
                with st.spinner("Downloading and preparing dataset..."):
                    clear_runtime(False)
                    RAW.mkdir(parents=True, exist_ok=True)

                    downloaded_path = Path(
                        kagglehub.dataset_download(
                            DATASET,
                            output_dir=str(RAW),
                            force_download=True,
                        )
                    )
                    report = prepare_dataset(downloaded_path, "KaggleHub")

                st.success(
                    f"Prepared {report['usable']} usable images. "
                    f"Skipped {len(report['skipped'])} ambiguous/unlabelled rows."
                )
                st.rerun()
            except Exception as error:
                st.error(f"Download/import failed: {error}")

    report = load_report()

    if report and dataset_ready():
        st.markdown("---")
        st.subheader("Dataset Summary")

        column1, column2, column3, column4 = st.columns(4)
        column1.metric("Train", report["totals"]["train"])
        column2.metric("Validation", report["totals"]["val"])
        column3.metric("Test", report["totals"]["test"])
        column4.metric("Skipped", len(report["skipped"]))

        st.dataframe(
            distribution(report),
            use_container_width=True,
            hide_index=True,
        )

        if report["skipped"]:
            with st.expander("Show skipped rows"):
                st.dataframe(
                    pd.DataFrame(report["skipped"]),
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown("---")
        st.subheader("🚀 Train YOLO Classifier")

        left, right = st.columns(2)
        with left:
            epochs = st.number_input(
                "Epochs",
                min_value=1,
                max_value=200,
                value=3,
                step=1,
            )
            image_size = st.selectbox("Image size", [160, 224, 320], index=1)

        with right:
            batch_size = st.selectbox("Batch size", [2, 4, 8, 16], index=2)
            model_name = st.selectbox(
                "Pretrained model",
                ["yolo11n-cls.pt", "yolo11s-cls.pt", "yolo11m-cls.pt"],
                index=0,
            )

        st.warning(
            "For Streamlit Cloud, start with yolo11n-cls.pt, 3 epochs, "
            "224 image size and batch size 8. Training runs on limited cloud resources."
        )

        if st.button("🚀 Start Training", type="primary"):
            try:
                with st.spinner("Training classifier..."):
                    best_model = train_model(
                        int(epochs),
                        int(image_size),
                        int(batch_size),
                        model_name,
                    )

                st.success("Training completed successfully.")
                with best_model.open("rb") as file:
                    st.download_button(
                        "⬇️ Download best.pt",
                        data=file.read(),
                        file_name="best.pt",
                        mime="application/octet-stream",
                    )
            except Exception as error:
                st.error(f"Training failed: {error}")

    st.markdown("---")
    if st.button("🧹 Clear Runtime Data and Models"):
        clear_runtime(True)
        if UPLOADED_MODEL.exists():
            UPLOADED_MODEL.unlink()
        st.rerun()


# ============================================================
# FRUIT RECOGNITION
# ============================================================
else:
    st.subheader("🔎 Fruit Recognition")

    model_source = st.radio(
        "Model source",
        ["Use trained best.pt", "Upload classification .pt"],
        horizontal=True,
    )

    model_path: Path | None = None

    if model_source == "Use trained best.pt":
        if BEST_MODEL.exists():
            model_path = BEST_MODEL
            st.success("Trained best.pt found.")
        else:
            st.warning("Train the classifier first or upload a classification model.")
    else:
        uploaded_model = st.file_uploader("Upload .pt model", type=["pt"])
        if uploaded_model is not None:
            model_path = save_model(uploaded_model)
            st.success(f"Loaded {uploaded_model.name}")

    uploaded_image = st.file_uploader(
        "Upload fruit image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="fruit_image",
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        left, right = st.columns(2)

        with left:
            st.image(image, caption="Input Image", use_container_width=True)

        if st.button(
            "🍎 Recognize Fruit",
            type="primary",
            disabled=model_path is None,
        ):
            try:
                with st.spinner("Classifying image..."):
                    predictions = predict(model_path, image)

                top_prediction = predictions.iloc[0]

                with right:
                    st.success("Prediction complete")
                    st.metric("Predicted Fruit", top_prediction["Fruit"])
                    st.metric(
                        "Confidence",
                        f"{top_prediction['Confidence (%)']:.2f}%",
                    )

                st.subheader("Top-5 Predictions")
                st.dataframe(
                    predictions,
                    use_container_width=True,
                    hide_index=True,
                )
                st.bar_chart(
                    predictions.set_index("Fruit")[["Confidence (%)"]]
                )

                st.download_button(
                    "⬇️ Download Prediction CSV",
                    data=predictions.to_csv(index=False).encode("utf-8"),
                    file_name="fruit_prediction.csv",
                    mime="text/csv",
                )

            except Exception as error:
                st.error(f"Recognition failed: {error}")
