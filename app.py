from __future__ import annotations

from collections import Counter
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


APP_TITLE = "Smart Fruit Detection & Counting System"
DEFAULT_MODEL = Path("best.pt")
CACHE_DIR = Path(".cache_models")
FRUIT_CLASSES = [
    "Apple",
    "Banana",
    "Grapes",
    "Kiwi",
    "Mango",
    "Orange",
    "Pineapple",
    "Sugerapple",
    "Watermelon",
]


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🍎",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model(model_path: str) -> YOLO:
    """Load and cache a YOLO model from disk."""
    return YOLO(model_path)


def save_uploaded_model(uploaded_model) -> Path:
    """Save an uploaded .pt model to a stable local cache path."""
    model_bytes = uploaded_model.getvalue()
    digest = sha256(model_bytes).hexdigest()[:12]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CACHE_DIR / f"model_{digest}.pt"

    if not model_path.exists():
        model_path.write_bytes(model_bytes)

    return model_path


def detections_to_dataframe(result) -> pd.DataFrame:
    """Convert Ultralytics detections into a table for Streamlit."""
    rows = []

    if result.boxes is None:
        return pd.DataFrame(
            columns=["Fruit", "Confidence (%)", "X1", "Y1", "X2", "Y2"]
        )

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]

        rows.append(
            {
                "Fruit": str(result.names[class_id]),
                "Confidence (%)": round(confidence * 100, 2),
                "X1": round(x1, 1),
                "Y1": round(y1, 1),
                "X2": round(x2, 1),
                "Y2": round(y2, 1),
            }
        )

    return pd.DataFrame(rows)


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


st.title("🍎 Smart Fruit Detection & Counting System")
st.write(
    "Upload a trained YOLO model and a fruit image. The system detects the fruits, "
    "draws bounding boxes, shows confidence scores, and counts each detected class."
)

with st.sidebar:
    st.header("Detection Settings")

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Higher values make the detector more strict.",
    )

    model_source = st.radio(
        "Model source",
        options=["Use best.pt from repository", "Upload trained .pt model"],
    )

    uploaded_model = None
    if model_source == "Upload trained .pt model":
        uploaded_model = st.file_uploader(
            "Upload YOLO model",
            type=["pt"],
            help="Upload the best.pt generated after training.",
        )

    st.divider()
    st.subheader("Dataset Classes")
    st.caption(" • ".join(FRUIT_CLASSES))
    st.markdown(
        "[Kaggle dataset](https://www.kaggle.com/datasets/"
        "kapturovalexander/fruits-by-yolo-fruits-detection)"
    )

uploaded_image = st.file_uploader(
    "Upload a fruit image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_image is None:
    st.info("Upload an image to start fruit detection.")
    st.stop()

image = Image.open(uploaded_image).convert("RGB")

original_col, result_col = st.columns(2)

with original_col:
    st.subheader("Original Image")
    st.image(image, width="stretch")

if model_source == "Use best.pt from repository":
    model_path = DEFAULT_MODEL
else:
    model_path = save_uploaded_model(uploaded_model) if uploaded_model is not None else None

if st.button("🔍 Detect Fruits", type="primary", width="stretch"):
    if model_path is None:
        st.error("Upload a trained YOLO .pt model before running detection.")
        st.stop()

    if not Path(model_path).exists():
        st.error(
            "best.pt was not found. Run `python train.py` first, copy the trained "
            "best.pt into the project folder, or choose 'Upload trained .pt model'."
        )
        st.stop()

    try:
        model = load_model(str(model_path))

        with st.spinner("Detecting fruits..."):
            results = model.predict(
                source=image,
                conf=confidence,
                verbose=False,
            )

        result = results[0]
        detections = detections_to_dataframe(result)
        annotated_image = result.plot(pil=True)

        with result_col:
            st.subheader("Detection Result")
            st.image(annotated_image, width="stretch")

        st.subheader("Detection Summary")

        if detections.empty:
            st.warning("No fruits were detected at the selected confidence threshold.")
        else:
            fruit_counts = Counter(detections["Fruit"].tolist())
            count_table = pd.DataFrame(
                [
                    {"Fruit": fruit, "Count": count}
                    for fruit, count in sorted(fruit_counts.items())
                ]
            )

            metric1, metric2, metric3 = st.columns(3)
            metric1.metric("Total Fruits", int(len(detections)))
            metric2.metric("Fruit Types", int(len(fruit_counts)))
            metric3.metric(
                "Average Confidence",
                f"{detections['Confidence (%)'].mean():.2f}%",
            )

            left_table, right_table = st.columns([1, 2])

            with left_table:
                st.markdown("#### Fruit Count")
                st.dataframe(count_table, hide_index=True, width="stretch")

            with right_table:
                st.markdown("#### Detection Details")
                st.dataframe(detections, hide_index=True, width="stretch")

            csv_bytes = detections.to_csv(index=False).encode("utf-8")
            annotated_bytes = image_to_png_bytes(annotated_image)

            download1, download2 = st.columns(2)
            with download1:
                st.download_button(
                    "⬇️ Download Detection CSV",
                    data=csv_bytes,
                    file_name="fruit_detections.csv",
                    mime="text/csv",
                    width="stretch",
                )

            with download2:
                st.download_button(
                    "⬇️ Download Detected Image",
                    data=annotated_bytes,
                    file_name="detected_fruits.png",
                    mime="image/png",
                    width="stretch",
                )

    except Exception as error:
        st.exception(error)

st.divider()
st.caption(
    "Assignment project: Fruit Image Detection and Counting System using Python, "
    "Ultralytics YOLO and Streamlit."
)
