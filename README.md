# Fruit Image Classification System

This project is rebuilt around the supplied `archive.zip` dataset and runs from **one Streamlit Python file: `app.py`**.

## Important dataset finding

The supplied archive contains 2,974 fruit images and image-level labels in three `_classes.csv` files. It does **not** contain YOLO bounding-box `.txt` annotations. Although the archive includes a detection-style `data.yaml`, the actual label format is classification data.

Therefore this project correctly uses **YOLO image classification** rather than bounding-box object detection.

Classes:

- Apple
- Banana
- Grapes
- Kiwi
- Mango
- Orange
- Pineapple
- Sugerapple
- Watermelon

The importer converts the CSV export into the folder layout expected by Ultralytics classification. The supplied archive has 2,960 directly usable single-label images; 14 zero-label or multi-label rows are skipped and shown in the Streamlit interface.

## Files

```text
backupplan/
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

Only `app.py` is executed by Streamlit.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Dataset workflow

Open **Dataset & Training** in the sidebar. You can either:

1. Upload the supplied `archive.zip`, or
2. Download the same public Kaggle dataset from `kapturovalexander/fruits-by-yolo-fruits-detection`.

The app does not train from the archive's incorrect detection YAML. Instead it:

```text
_classes.csv
    ↓
Read image-level class labels
    ↓
Prepare train/ClassName, val/ClassName, test/ClassName folders
    ↓
YOLO classification training
    ↓
best.pt
```

For a quick test, use:

- Model: `yolo11n-cls.pt`
- Epochs: 3-5
- Image size: 224
- Batch size: 8

For a fuller experiment, increase the epochs.

## Recognition workflow

After training, open **Fruit Recognition**:

1. Use the trained `best.pt` or upload a classification `.pt` model.
2. Upload a fruit image.
3. Click **Recognize Fruit**.
4. Streamlit displays the predicted fruit, confidence, and Top-5 probabilities.
5. Download the prediction results as CSV if needed.

## Why there are no bounding boxes

Bounding boxes require object-detection annotations such as YOLO `.txt` files containing class and box coordinates. The supplied archive has no such files; its `_classes.csv` files contain only image-level class labels. Creating fake bounding boxes would make the assignment technically incorrect.
