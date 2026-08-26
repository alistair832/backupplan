# Smart Fruit Detection & Counting System

A Python computer-vision assignment that uses **Ultralytics YOLO** for fruit object detection and **Streamlit** for the user interface.

**Dataset:** [Fruits by YOLO - Fruits Detection](https://www.kaggle.com/datasets/kapturovalexander/fruits-by-yolo-fruits-detection)

The project connects directly to the Kaggle dataset with `kagglehub`, trains a custom YOLO detector, and provides both a command-line detector and a Streamlit web application.

## Main Features

- Download the Kaggle dataset automatically with Python
- Automatically locate the YOLO `data.yaml`
- Train a custom YOLO fruit detection model
- Detect multiple fruits in one image
- Draw bounding boxes around detected fruits
- Show fruit class names and confidence scores
- Count each detected fruit class
- Show the total number of detected fruits
- Display detection results in Streamlit
- Show detection details in tables
- Download detection results as CSV
- Download the annotated result image

## Dataset Classes

The project is designed for these fruit classes:

1. Apple
2. Banana
3. Grapes
4. Kiwi
5. Mango
6. Orange
7. Pineapple
8. Sugerapple
9. Watermelon

## Project Structure

```text
backupplan/
├── app.py                 # Streamlit web application
├── detect.py              # Command-line image detection
├── train.py               # YOLO model training
├── download_dataset.py    # Direct Kaggle dataset downloader
├── requirements.txt       # Python packages
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

The project is directly linked to:

```text
https://www.kaggle.com/datasets/kapturovalexander/fruits-by-yolo-fruits-detection
```

Download it with:

```bash
python download_dataset.py
```

The dataset is stored under:

```text
data/fruits-yolo/
```

You can also skip this step and run `train.py`. The training program downloads the dataset automatically if it is missing.

## 3. Train the YOLO Model

Run:

```bash
python train.py
```

Training flow:

```text
Kaggle Dataset
      ↓
KaggleHub Download
      ↓
Find data.yaml
      ↓
Load Pretrained YOLO Model
      ↓
Train on Fruit Images
      ↓
Validate Model
      ↓
Generate best.pt
```

The trained model is normally created at:

```text
runs/fruit_detection/train/weights/best.pt
```

Copy the trained model into the project root if you want the Streamlit app and `detect.py` to load it automatically:

```text
backupplan/best.pt
```

`best.pt` is ignored by Git because trained model files are normally too large to commit directly.

## 4. Run the Streamlit Application

Start the web application with:

```bash
streamlit run app.py
```

The Streamlit interface supports two model options:

- **Use best.pt from repository folder** — place `best.pt` in the project root.
- **Upload trained .pt model** — upload your trained `best.pt` directly through the web interface.

Then upload a fruit image and click:

```text
Detect Fruits
```

The application displays:

- Original image
- Detected image with YOLO bounding boxes
- Fruit class names
- Detection confidence scores
- Total fruit count
- Number of different fruit types
- Average detection confidence
- Fruit-count table
- Full detection-details table
- CSV download button
- Detected-image download button

## Streamlit Detection Flow

```text
User Uploads Image
        ↓
Load Trained best.pt
        ↓
YOLO Image Prediction
        ↓
Detect Bounding Boxes
        ↓
Identify Fruit Classes
        ↓
Calculate Confidence Scores
        ↓
Count Detected Fruits
        ↓
Streamlit Dashboard
        ↓
Display + Download Results
```

## 5. Run Command-Line Image Detection

Place a test image at:

```text
images/fruit.jpg
```

Then run:

```bash
python detect.py
```

The command-line program will:

1. Load `best.pt`.
2. Read `images/fruit.jpg`.
3. Detect the fruits.
4. Print each fruit and confidence score.
5. Count each fruit class.
6. Draw bounding boxes.
7. Save the annotated image to `output/detected_fruit.jpg`.

Example output:

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

## 6. Confidence Threshold

In Streamlit, change the confidence threshold with the sidebar slider.

For command-line detection, edit:

```python
CONFIDENCE = 0.25
```

A higher value makes the model stricter. For example:

```python
CONFIDENCE = 0.50
```

## 7. Deploy to Streamlit Community Cloud

After the project is merged into the GitHub `main` branch:

1. Open Streamlit Community Cloud.
2. Choose **Create app**.
3. Select the GitHub repository `alistair832/backupplan`.
4. Select branch `main`.
5. Set the main file path to:

```text
app.py
```

6. Deploy the application.

Because `best.pt` is ignored by Git, for a simple assignment demonstration you can use the **Upload trained .pt model** option inside the deployed Streamlit application.

## Technologies Used

- Python
- Ultralytics YOLO
- Streamlit
- OpenCV
- Pillow
- Pandas
- KaggleHub

## Project Title

**Smart Fruit Detection and Automatic Counting System Using Python, YOLO and Streamlit**

## Project Objective

To develop a computer-vision application that can automatically detect, classify and count different fruit types from uploaded images using a trained YOLO object-detection model and display the results through an interactive Streamlit web interface.
