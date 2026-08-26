# Fruit Image Classification System

This project runs from **one Streamlit Python file: `app.py`**.

The supplied fruit dataset contains image-level labels in `_classes.csv`. It does not contain bounding-box coordinates, so this rebuild uses **image classification** rather than object detection.

## Model stack

- Streamlit
- PyTorch
- Torchvision
- MobileNetV3-Small
- KaggleHub
- Pandas
- Pillow

OpenCV and Ultralytics are intentionally not used, which removes the previous `cv2` deployment problem.

## Fruit classes

Apple, Banana, Grapes, Kiwi, Mango, Orange, Pineapple, Sugerapple, Watermelon.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit workflow

### Dataset & Training

1. Upload the supplied `archive.zip`, or download the same Kaggle dataset.
2. The app reads `_classes.csv`.
3. It automatically creates `train/ClassName`, `val/ClassName`, and `test/ClassName` folders.
4. Choose epochs, image size, batch size, and learning rate.
5. Train MobileNetV3-Small.
6. Download `best_model.pth`.

For a quick Streamlit Cloud test, start with:

- Epochs: 1-3
- Image size: 160 or 224
- Batch size: 8
- Learning rate: 0.001

### Fruit Recognition

1. Use the trained `best_model.pth` or upload a compatible `.pth` model.
2. Upload a fruit image.
3. Click **Recognize Fruit**.
4. View the predicted fruit, confidence, and Top-5 probabilities.
5. Download the prediction CSV if required.

## Deployment note

Streamlit Community Cloud chooses the Python version in its deployment settings. The repository therefore does not use `.python-version` or `runtime.txt`. The current dependency set supports modern Streamlit Cloud Python environments and does not require OpenCV system packages.
