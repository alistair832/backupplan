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
REPORT_FILE = RUNTIME / "report.json"
ZIP_FILE = RUNTIME / "archive.zip"
MODEL_DIR = RUNTIME / "model"
BEST_MODEL = MODEL_DIR / "best_model.pth"
UPLOADED_MODEL = RUNTIME / "uploaded_model.pth"

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
    "One Streamlit app: import dataset → prepare labels → train MobileNetV3 → recognize fruit."
)


# ============================================================
# PYTORCH IMPORT
# ============================================================
def get_ml():
    """Import PyTorch/Torchvision only when model work is requested."""
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import datasets, models, transforms

        return torch, nn, DataLoader, datasets, models, transforms
    except Exception as error:
        raise RuntimeError(
            "PyTorch/Torchvision could not be loaded. Reboot the Streamlit app "
            "after the latest GitHub dependency update. "
            f"Python: {PYTHON_VERSION}. Original error: {error}"
        ) from error


# ============================================================
# DATASET HELPERS
# ============================================================
def safe_extract(zip_path: Path, destination: Path) -> None:
    """Extract ZIP safely, blocking path traversal."""
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination_root and destination_root not in target.parents:
                raise ValueError(f"Unsafe ZIP path: {member.filename}")
        archive.extractall(destination)


def clear_runtime(clear_model: bool = False) -> None:
    """Clear imported/prepared data and optionally the trained model."""
    for path in (RAW, PREPARED):
        if path.exists():
            shutil.rmtree(path)

    for path in (ZIP_FILE, REPORT_FILE):
        if path.exists():
            path.unlink()

    if clear_model and MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)

    if clear_model and UPLOADED_MODEL.exists():
        UPLOADED_MODEL.unlink()


def link_or_copy(source: Path, destination: Path) -> None:
    """Hard-link images when possible to save space; otherwise copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare_dataset(source: Path, source_name: str) -> dict:
    """Convert _classes.csv into train/ClassName, val/ClassName, test/ClassName."""
    csv_files = sorted(source.rglob("_classes.csv"))
    if not csv_files:
        raise FileNotFoundError(
            "No _classes.csv files were found. Upload the supplied archive.zip "
            "or use the Kaggle download option."
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
                        {"split": split, "file": filename, "reason": "missing image"}
                    )
                    continue

                if len(active_classes) != 1:
                    reason = (
                        "no label"
                        if not active_classes
                        else "multiple labels: " + ", ".join(active_classes)
                    )
                    skipped.append(
                        {"split": split, "file": filename, "reason": reason}
                    )
                    continue

                label = active_classes[0]
                output_image = PREPARED / split / label / filename
                link_or_copy(image_path, output_image)
                counts[(split, label)] += 1

    if not (PREPARED / "train").exists():
        raise RuntimeError("Training images could not be prepared.")
    if not (PREPARED / "val").exists():
        raise RuntimeError("Validation images could not be prepared.")

    report = {
        "source": source_name,
        "classes": sorted(classes),
        "usable": sum(counts.values()),
        "skipped": skipped,
        "totals": {
            split: sum(
                count
                for (current_split, _), count in counts.items()
                if current_split == split
            )
            for split in ("train", "val", "test")
        },
        "counts": {
            split: {
                class_name: counts.get((split, class_name), 0)
                for class_name in sorted(classes)
            }
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
def create_model(num_classes: int, use_pretrained: bool = True):
    _, nn, _, _, models, _ = get_ml()

    weights = None
    if use_pretrained:
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
        except Exception:
            weights = None

    try:
        model = models.mobilenet_v3_small(weights=weights)
    except Exception:
        model = models.mobilenet_v3_small(weights=None)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_transforms(image_size: int):
    _, _, _, _, _, transforms = get_ml()

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_transform, eval_transform


def train_model(
    epochs: int,
    image_size: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[Path, list[dict[str, float]]]:
    if not dataset_ready():
        raise RuntimeError("Import and prepare the dataset first.")

    torch, nn, DataLoader, datasets, _, _ = get_ml()
    train_transform, eval_transform = build_transforms(image_size)

    train_set = datasets.ImageFolder(PREPARED / "train", transform=train_transform)
    val_set = datasets.ImageFolder(PREPARED / "val", transform=eval_transform)

    if train_set.classes != val_set.classes:
        raise RuntimeError("Training and validation class folders do not match.")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(len(train_set.classes), use_pretrained=True).to(device)

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * inputs.size(0)
            correct += int((outputs.argmax(1) == labels).sum().item())
            total += int(labels.size(0))

        train_loss = running_loss / max(total, 1)
        train_accuracy = correct / max(total, 1)

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.inference_mode():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss_sum += float(loss.item()) * inputs.size(0)
                val_correct += int((outputs.argmax(1) == labels).sum().item())
                val_total += int(labels.size(0))

        val_loss = val_loss_sum / max(val_total, 1)
        val_accuracy = val_correct / max(val_total, 1)

        history.append(
            {
                "Epoch": epoch,
                "Train Loss": round(train_loss, 4),
                "Train Accuracy (%)": round(train_accuracy * 100, 2),
                "Validation Loss": round(val_loss, 4),
                "Validation Accuracy (%)": round(val_accuracy * 100, 2),
            }
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                {
                    "architecture": "mobilenet_v3_small",
                    "model_state_dict": model.state_dict(),
                    "classes": train_set.classes,
                    "image_size": image_size,
                    "validation_accuracy": val_accuracy,
                },
                BEST_MODEL,
            )

    return BEST_MODEL, history


def save_uploaded_model(uploaded_file) -> Path:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with UPLOADED_MODEL.open("wb") as file:
        uploaded_file.seek(0)
        shutil.copyfileobj(uploaded_file, file)
    return UPLOADED_MODEL


def load_checkpoint(model_path: Path):
    torch, _, _, _, _, _ = get_ml()
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    classes = checkpoint.get("classes")
    image_size = int(checkpoint.get("image_size", 224))
    if not classes:
        raise RuntimeError("Model file does not contain fruit class names.")

    model = create_model(len(classes), use_pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, list(classes), image_size


def predict(model_path: Path, image: Image.Image) -> pd.DataFrame:
    torch, _, _, _, _, transforms = get_ml()
    model, classes, image_size = load_checkpoint(model_path)

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    tensor = transform(image).unsqueeze(0)
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    k = min(5, len(classes))
    values, indexes = torch.topk(probabilities, k=k)

    rows = []
    for rank, (value, index) in enumerate(zip(values.tolist(), indexes.tolist()), 1):
        rows.append(
            {
                "Rank": rank,
                "Fruit": classes[index],
                "Confidence (%)": round(float(value) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


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
st.sidebar.caption("Single Streamlit entry point")
st.sidebar.code("streamlit run app.py", language=None)


# ============================================================
# HOME
# ============================================================
if page == "🏠 Home":
    st.subheader("Assignment Overview")
    st.markdown(
        """
        The supplied fruit dataset contains **image-level labels in `_classes.csv`**.
        Because it does not contain object bounding-box coordinates, the technically
        correct task is **fruit image classification**.

        This rebuild uses **PyTorch + Torchvision MobileNetV3** and does not use
        OpenCV or Ultralytics, avoiding the `cv2` deployment error.
        """
    )

    column1, column2, column3 = st.columns(3)
    column1.metric("Application", "Streamlit")
    column2.metric("Model", "MobileNetV3-Small")
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
Prepare train/Class, val/Class, test/Class
        ↓
Train MobileNetV3 classifier
        ↓
best_model.pth
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
        "Upload the supplied archive.zip or download the same Kaggle dataset. "
        "The app reads `_classes.csv` and prepares classification folders automatically."
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
        st.subheader("🚀 Train Fruit Classifier")

        left, right = st.columns(2)
        with left:
            epochs = st.number_input(
                "Epochs",
                min_value=1,
                max_value=50,
                value=3,
                step=1,
            )
            image_size = st.selectbox("Image size", [160, 192, 224], index=2)

        with right:
            batch_size = st.selectbox("Batch size", [4, 8, 16], index=1)
            learning_rate = st.selectbox(
                "Learning rate",
                [0.0005, 0.001, 0.002],
                index=1,
            )

        st.caption(
            "Streamlit Cloud is CPU-based. Start with 1-3 epochs and batch size 8. "
            "The feature extractor is frozen to make training lighter."
        )

        if st.button("🚀 Start Training", type="primary"):
            try:
                with st.spinner("Training MobileNetV3 classifier..."):
                    best_model, history = train_model(
                        int(epochs),
                        int(image_size),
                        int(batch_size),
                        float(learning_rate),
                    )

                st.success("Training completed.")
                history_df = pd.DataFrame(history)
                st.dataframe(history_df, use_container_width=True, hide_index=True)

                if not history_df.empty:
                    st.line_chart(
                        history_df.set_index("Epoch")[
                            ["Train Accuracy (%)", "Validation Accuracy (%)"]
                        ]
                    )

                with best_model.open("rb") as file:
                    st.download_button(
                        "⬇️ Download best_model.pth",
                        data=file.read(),
                        file_name="best_model.pth",
                        mime="application/octet-stream",
                    )
            except Exception as error:
                st.error(f"Training failed: {error}")

    if st.button("🧹 Clear Runtime Data and Model"):
        clear_runtime(True)
        st.rerun()


# ============================================================
# FRUIT RECOGNITION
# ============================================================
else:
    st.subheader("🔎 Fruit Recognition")

    model_source = st.radio(
        "Model source",
        ["Use trained best_model.pth", "Upload .pth model"],
        horizontal=True,
    )

    model_path: Path | None = None

    if model_source == "Use trained best_model.pth":
        if BEST_MODEL.exists():
            model_path = BEST_MODEL
            st.success("Trained model found.")
        else:
            st.warning(
                "No trained model found. Train the classifier first or upload a .pth model."
            )
    else:
        uploaded_model = st.file_uploader(
            "Upload PyTorch model",
            type=["pth"],
            key="model_upload",
        )
        if uploaded_model is not None:
            model_path = save_uploaded_model(uploaded_model)
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
                    result_df = predict(model_path, image)

                top = result_df.iloc[0]

                with right:
                    st.success("Prediction complete")
                    st.metric("Predicted Fruit", top["Fruit"])
                    st.metric(
                        "Confidence",
                        f"{float(top['Confidence (%)']):.2f}%",
                    )

                st.subheader("Top-5 Predictions")
                st.dataframe(
                    result_df,
                    use_container_width=True,
                    hide_index=True,
                )
                st.bar_chart(result_df.set_index("Fruit")[["Confidence (%)"]])

                st.download_button(
                    "⬇️ Download Prediction CSV",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="fruit_prediction.csv",
                    mime="text/csv",
                )
            except Exception as error:
                st.error(f"Recognition failed: {error}")
