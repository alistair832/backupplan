# Fruit Image Classification System

This project runs from **one Streamlit Python file: `app.py`** and is rebuilt around the supplied fruit dataset.

## Dataset format

The supplied archive contains fruit images and image-level labels in `_classes.csv` files. It does **not** contain YOLO bounding-box `.txt` annotations, so this project correctly uses **YOLO image classification** rather than bounding-box detection.

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

The importer converts the CSV labels into the folder structure expected by Ultralytics classification:

```text
train/ClassName/
val/ClassName/
test/ClassName/
```

## Files

```text
backupplan/
├── app.py
├── requirements.txt
├── packages.txt
├── runtime.txt
├── .python-version
├── .gitignore
└── README.md
```

Only `app.py` is executed by Streamlit.

## Streamlit Cloud runtime

This repository is configured for **Python 3.12** because the Ultralytics/OpenCV stack may fail under Python 3.14.

- `runtime.txt` requests Python 3.12.
- `.python-version` also declares Python 3.12.
- `packages.txt` installs Linux libraries required by OpenCV.
- `app.py` imports Ultralytics lazily so a bad ML runtime no longer crashes the entire Streamlit page at startup.

If an already-deployed Streamlit app is still using Python 3.14, reboot/redeploy it. If the old runtime is retained, open the Streamlit app settings and redeploy using Python 3.12.

## Install locally

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

The app reads `_classes.csv`, prepares the class folders, and then trains a YOLO classification model.

For a quick Streamlit Cloud test, use:

- Model: `yolo11n-cls.pt`
- Epochs: `3`
- Image size: `224`
- Batch size: `8`

## Recognition workflow

After training, open **Fruit Recognition**:

1. Use the trained `best.pt` or upload a classification `.pt` model.
2. Upload a fruit image.
3. Click **Recognize Fruit**.
4. Streamlit displays the predicted fruit, confidence, and Top-5 probabilities.
5. Download the prediction results as CSV if needed.
