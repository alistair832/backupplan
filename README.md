# Fruit Image Detection Using Python and YOLO

This project uses Python and Ultralytics YOLO to detect and count fruits in an image.

Dataset: [Fruits by YOLO - Fruits Detection](https://www.kaggle.com/datasets/kapturovalexander/fruits-by-yolo-fruits-detection)

## Functions

- Download the Kaggle fruit dataset with Python
- Train a YOLO fruit detection model
- Detect multiple fruits from an image
- Draw bounding boxes around detected fruits
- Show the fruit class and confidence score
- Count how many fruits of each class are detected
- Save the detected image to the `output` folder

## Project Files

```text
backupplan/
├── detect.py
├── train.py
├── download_dataset.py
├── requirements.txt
├── .gitignore
├── images/
│   └── README.md
└── README.md
```

## 1. Install Python Packages

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
```

## 2. Download the Dataset

```bash
python download_dataset.py
```

The dataset will be downloaded into:

```text
data/fruits-yolo/
```

The downloader will also print the location of the dataset YAML file. If the YAML path is different from the default path, update `DATA_YAML` inside `train.py`.

## 3. Train the YOLO Model

```bash
python train.py
```

The trained model is normally saved at:

```text
runs/fruit_detection/train/weights/best.pt
```

Before detection, copy `best.pt` into the main project folder:

```text
backupplan/best.pt
```

## 4. Add a Test Image

Place an image inside the `images` folder and name it:

```text
fruit.jpg
```

Example:

```text
images/fruit.jpg
```

You can also change `IMAGE_PATH` in `detect.py` if you want to use another image.

## 5. Run Fruit Image Detection

```bash
python detect.py
```

The program will:

1. Load the trained `best.pt` model.
2. Read `images/fruit.jpg`.
3. Detect fruits using YOLO.
4. Print each fruit and confidence score.
5. Count each detected fruit type.
6. Draw bounding boxes on the image.
7. Save the result to `output/detected_fruit.jpg`.
8. Display the result in an OpenCV window.

## Example Terminal Output

```text
===== Detection Result =====
Detected: Apple        Confidence: 94.18%
Detected: Banana       Confidence: 91.52%
Detected: Apple        Confidence: 89.73%

===== Fruit Count =====
Apple: 2
Banana: 1
Total Fruits: 3
```

## Main Detection Flow

```text
Fruit Image
    ↓
Python Program
    ↓
Load YOLO Model
    ↓
Detect Objects
    ↓
Draw Bounding Boxes
    ↓
Identify Fruit Class
    ↓
Confidence Score
    ↓
Count Fruits
    ↓
Save and Display Result
```

## Change the Detection Image

Open `detect.py` and change:

```python
IMAGE_PATH = "images/fruit.jpg"
```

For example:

```python
IMAGE_PATH = "images/apple_banana.jpg"
```

## Change Confidence Threshold

The default confidence threshold is:

```python
CONFIDENCE = 0.25
```

A higher value gives stricter detections. For example:

```python
CONFIDENCE = 0.50
```

## Project Title

**Fruit Image Detection and Counting System Using Python and YOLO**
