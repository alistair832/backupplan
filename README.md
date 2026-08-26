# Smart Fruit Detection and Counting System

This assignment uses **one Python file only (`app.py`)** as the Streamlit application.

The app combines:

- Kaggle dataset download
- YOLO dataset discovery
- YOLO model training
- trained model loading
- fruit image upload
- fruit object detection
- bounding boxes and confidence scores
- per-class fruit counting
- detection details table
- CSV result download
- annotated image download

## Dataset

**Fruits by YOLO - Fruits Detection**

https://www.kaggle.com/datasets/kapturovalexander/fruits-by-yolo-fruits-detection

Kaggle dataset handle used in `app.py`:

```text
kapturovalexander/fruits-by-yolo-fruits-detection
```

## Project Structure

```text
backupplan/
├── app.py
├── requirements.txt
└── README.md
```

Generated files such as the Kaggle dataset, YOLO training runs and `best.pt` are created when the application runs and do not need to be manually prepared first.

## Install

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
```

## Run Streamlit

Only one Python file needs to be started:

```bash
streamlit run app.py
```

The browser will open the Smart Fruit Detection and Counting System.

## Functions Inside `app.py`

### 1. Home

Shows the assignment overview, system functions and processing flow.

### 2. Dataset & Training

This page can:

1. Check whether the fruit dataset already exists.
2. Download the Kaggle dataset directly using `kagglehub`.
3. Automatically locate `data.yaml` or `dataset.yaml`.
4. Display the number of images, label files and fruit classes found.
5. Select training epochs, image size, batch size and YOLO model.
6. Train the fruit detector directly from Streamlit.
7. Save the trained model as `best.pt`.
8. Download `best.pt` after training.

Default training model:

```text
yolo11n.pt
```

Default trained model location:

```text
runs/fruit_detection/train/weights/best.pt
```

### 3. Image Detection

This page can:

1. Use the `best.pt` produced by the training page, or upload another `.pt` model.
2. Upload a JPG, JPEG, PNG, BMP or WEBP image.
3. Change the confidence threshold.
4. Detect fruits using YOLO.
5. Show the original image and annotated result side by side.
6. Display total fruits detected.
7. Display the number of different fruit types.
8. Display average confidence.
9. Display a fruit-count table.
10. Display bounding-box detection details.
11. Download results as CSV.
12. Download the annotated image as PNG.

## Application Flow

```text
Kaggle Fruit Dataset
        ↓
Single Streamlit app.py
        ↓
Dataset Download / Check
        ↓
YOLO Training
        ↓
Trained best.pt
        ↓
Upload Fruit Image
        ↓
YOLO Detection
        ↓
Bounding Boxes + Fruit Classes
        ↓
Confidence Scores + Fruit Counts
        ↓
Streamlit Result + Download
```

## Important

Streamlit is started with only:

```bash
streamlit run app.py
```

You do **not** need to run separate Python files for downloading, training or detection. All assignment functions are included in `app.py` and selected through the Streamlit sidebar.
