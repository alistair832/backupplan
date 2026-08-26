from __future__ import annotations

from collections import Counter
import csv
import json
import os
from pathlib import Path
import shutil
import zipfile

import kagglehub
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

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
SPLITS = {"train": "train", "training": "train", "valid": "val", "validation": "val", "val": "val", "test": "test", "testing": "test"}

st.set_page_config(page_title="Fruit Classification System", page_icon="🍎", layout="wide")
st.title("🍎 Fruit Image Classification System")
st.caption("One Streamlit app: import dataset → prepare labels → train YOLO classifier → recognize fruit.")


def safe_extract(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for member in z.infolist():
            target = (dest / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe ZIP path: {member.filename}")
        z.extractall(dest)


def clear_runtime(clear_models: bool = False) -> None:
    for p in (RAW, PREPARED):
        if p.exists():
            shutil.rmtree(p)
    for p in (ZIP_FILE, REPORT_FILE):
        if p.exists():
            p.unlink()
    if clear_models and RUNS.exists():
        shutil.rmtree(RUNS)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_dataset(source: Path, source_name: str) -> dict:
    csv_files = sorted(source.rglob("_classes.csv"))
    if not csv_files:
        raise FileNotFoundError("No _classes.csv files were found in this dataset.")

    if PREPARED.exists():
        shutil.rmtree(PREPARED)
    PREPARED.mkdir(parents=True, exist_ok=True)

    counts: Counter[tuple[str, str]] = Counter()
    classes: list[str] = []
    skipped: list[dict] = []

    for csv_path in csv_files:
        split = SPLITS.get(csv_path.parent.name.lower())
        if not split:
            continue

        with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            if len(headers) < 2:
                continue

            filename_col = headers[0]
            class_cols = [(h, h.strip()) for h in headers[1:]]
            for _, name in class_cols:
                if name and name not in classes:
                    classes.append(name)

            for row in reader:
                filename = (row.get(filename_col) or "").strip()
                active = []
                for raw_header, clean_name in class_cols:
                    try:
                        value = float((row.get(raw_header) or "0").strip() or 0)
                    except ValueError:
                        value = 0
                    if value > 0.5:
                        active.append(clean_name)

                src = csv_path.parent / filename
                if not filename or not src.exists():
                    skipped.append({"split": split, "file": filename, "reason": "missing image"})
                    continue
                if len(active) != 1:
                    reason = "no label" if not active else "multiple labels: " + ", ".join(active)
                    skipped.append({"split": split, "file": filename, "reason": reason})
                    continue

                label = active[0]
                link_or_copy(src, PREPARED / split / label / filename)
                counts[(split, label)] += 1

    if not (PREPARED / "train").exists() or not (PREPARED / "val").exists():
        raise RuntimeError("Train/validation folders could not be prepared.")

    report = {
        "source": source_name,
        "classes": classes,
        "usable": sum(counts.values()),
        "skipped": skipped,
        "totals": {s: sum(v for (sp, _), v in counts.items() if sp == s) for s in ("train", "val", "test")},
        "counts": {s: {c: counts.get((s, c), 0) for c in classes} for s in ("train", "val", "test")},
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
    return pd.DataFrame([
        {
            "Fruit": c,
            "Train": report["counts"]["train"].get(c, 0),
            "Validation": report["counts"]["val"].get(c, 0),
            "Test": report["counts"]["test"].get(c, 0),
        }
        for c in report["classes"]
    ])


def train_model(epochs: int, imgsz: int, batch: int, model_name: str) -> Path:
    if not dataset_ready():
        raise RuntimeError("Import the dataset first.")
    run = RUNS / "train"
    if run.exists():
        shutil.rmtree(run)
    model = YOLO(model_name)
    model.train(
        data=str(PREPARED.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
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


def save_model(upload) -> Path:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with UPLOADED_MODEL.open("wb") as f:
        upload.seek(0)
        shutil.copyfileobj(upload, f)
    return UPLOADED_MODEL


def predict(model_path: Path, image: Image.Image) -> pd.DataFrame:
    result = YOLO(str(model_path)).predict(source=image, imgsz=224, verbose=False)[0]
    if result.probs is None:
        raise RuntimeError("This .pt file is not a YOLO classification model.")
    probs = result.probs.data.detach().cpu().tolist()
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:5]
    return pd.DataFrame([
        {"Rank": rank, "Fruit": str(result.names[i]), "Confidence (%)": round(probs[i] * 100, 2)}
        for rank, i in enumerate(order, 1)
    ])


st.sidebar.header("Navigation")
page = st.sidebar.radio("Choose function", ["🏠 Home", "📦 Dataset & Training", "🔎 Fruit Recognition"])
st.sidebar.markdown("---")
st.sidebar.caption("Single Streamlit file")
st.sidebar.code("streamlit run app.py", language=None)

if page == "🏠 Home":
    st.subheader("Assignment Overview")
    st.markdown(
        """
        The supplied ZIP contains **2,974 fruit images** and image-level labels in `_classes.csv`.
        It does **not** contain YOLO bounding-box `.txt` annotations, so this rebuild correctly uses
        **YOLO image classification** instead of object detection.
        """
    )
    a, b, c = st.columns(3)
    a.metric("Raw Images", "2,974")
    b.metric("Fruit Classes", "9")
    c.metric("Task", "Classification")
    st.write("**Classes:** Apple, Banana, Grapes, Kiwi, Mango, Orange, Pineapple, Sugerapple, Watermelon")
    st.code(
        """archive.zip / Kaggle dataset
        ↓
Read _classes.csv
        ↓
Prepare train/Class, val/Class, test/Class
        ↓
Train yolo11*-cls.pt
        ↓
best.pt
        ↓
Upload fruit image
        ↓
Fruit class + confidence + Top-5 results""",
        language=None,
    )

elif page == "📦 Dataset & Training":
    st.subheader("📦 Import and Prepare Dataset")
    st.info("The app ignores the archive's incorrect detection-style data.yaml and uses the real _classes.csv labels.")

    source = st.radio("Dataset source", ["Upload supplied archive.zip", "Download same Kaggle dataset"])
    if source == "Upload supplied archive.zip":
        uploaded_zip = st.file_uploader("Upload archive.zip", type=["zip"])
        if st.button("📥 Import ZIP", type="primary", disabled=uploaded_zip is None):
            try:
                with st.spinner("Extracting and preparing dataset..."):
                    clear_runtime(False)
                    RUNTIME.mkdir(parents=True, exist_ok=True)
                    with ZIP_FILE.open("wb") as f:
                        uploaded_zip.seek(0)
                        shutil.copyfileobj(uploaded_zip, f)
                    safe_extract(ZIP_FILE, RAW)
                    report = prepare_dataset(RAW, "Uploaded archive.zip")
                st.success(f"Prepared {report['usable']} usable images. Skipped {len(report['skipped'])} ambiguous/unlabelled rows.")
                st.rerun()
            except Exception as e:
                st.error(f"Import failed: {e}")
    else:
        st.code(DATASET, language=None)
        if st.button("🌐 Download and Prepare", type="primary"):
            try:
                with st.spinner("Downloading and preparing dataset..."):
                    clear_runtime(False)
                    RAW.mkdir(parents=True, exist_ok=True)
                    downloaded = Path(kagglehub.dataset_download(DATASET, output_dir=str(RAW), force_download=True))
                    report = prepare_dataset(downloaded, "KaggleHub")
                st.success(f"Prepared {report['usable']} usable images. Skipped {len(report['skipped'])} ambiguous/unlabelled rows.")
                st.rerun()
            except Exception as e:
                st.error(f"Download/import failed: {e}")

    report = load_report()
    if report and dataset_ready():
        st.markdown("---")
        st.subheader("Dataset Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train", report["totals"]["train"])
        c2.metric("Validation", report["totals"]["val"])
        c3.metric("Test", report["totals"]["test"])
        c4.metric("Skipped", len(report["skipped"]))
        st.dataframe(distribution(report), width="stretch", hide_index=True)
        if report["skipped"]:
            with st.expander("Show skipped rows"):
                st.dataframe(pd.DataFrame(report["skipped"]), width="stretch", hide_index=True)

        st.markdown("---")
        st.subheader("🚀 Train YOLO Classifier")
        left, right = st.columns(2)
        with left:
            epochs = st.number_input("Epochs", 1, 200, 5)
            imgsz = st.selectbox("Image size", [160, 224, 320], index=1)
        with right:
            batch = st.selectbox("Batch size", [4, 8, 16, 32], index=1)
            model_name = st.selectbox("Pretrained model", ["yolo11n-cls.pt", "yolo11s-cls.pt", "yolo11m-cls.pt"])
        st.warning("For Streamlit Cloud, start with yolo11n-cls.pt and 3-5 epochs. Training can be slow on CPU.")

        if st.button("🚀 Start Training", type="primary"):
            try:
                with st.spinner("Training classifier..."):
                    best = train_model(int(epochs), int(imgsz), int(batch), model_name)
                st.success("Training completed.")
                with best.open("rb") as f:
                    st.download_button("⬇️ Download best.pt", f.read(), "best.pt", "application/octet-stream")
            except Exception as e:
                st.error(f"Training failed: {e}")

    if st.button("🧹 Clear Runtime Data and Models"):
        clear_runtime(True)
        if UPLOADED_MODEL.exists():
            UPLOADED_MODEL.unlink()
        st.rerun()

else:
    st.subheader("🔎 Fruit Recognition")
    model_source = st.radio("Model source", ["Use trained best.pt", "Upload classification .pt"], horizontal=True)
    model_path: Path | None = None
    if model_source == "Use trained best.pt":
        if BEST_MODEL.exists():
            model_path = BEST_MODEL
            st.success("Trained best.pt found.")
        else:
            st.warning("Train the classifier first or upload a classification model.")
    else:
        uploaded_model = st.file_uploader("Upload .pt model", type=["pt"])
        if uploaded_model:
            model_path = save_model(uploaded_model)
            st.success(f"Loaded {uploaded_model.name}")

    uploaded_image = st.file_uploader("Upload fruit image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="fruit_image")
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        l, r = st.columns(2)
        with l:
            st.image(image, caption="Input Image", width="stretch")
        if st.button("🍎 Recognize Fruit", type="primary", disabled=model_path is None):
            try:
                with st.spinner("Classifying image..."):
                    df = predict(model_path, image)
                top = df.iloc[0]
                with r:
                    st.success("Prediction complete")
                    st.metric("Predicted Fruit", top["Fruit"])
                    st.metric("Confidence", f"{top['Confidence (%)']:.2f}%")
                st.subheader("Top-5 Predictions")
                st.dataframe(df, width="stretch", hide_index=True)
                st.bar_chart(df.set_index("Fruit")[["Confidence (%)"]])
                st.download_button(
                    "⬇️ Download Prediction CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    "fruit_prediction.csv",
                    "text/csv",
                )
            except Exception as e:
                st.error(f"Recognition failed: {e}")
