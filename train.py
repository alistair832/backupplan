from pathlib import Path

from ultralytics import YOLO


DATA_YAML = "data/fruits-yolo/data.yaml"
MODEL = "yolo11n.pt"
EPOCHS = 50
IMAGE_SIZE = 640


def main():
    data_file = Path(DATA_YAML)

    if not data_file.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATA_YAML}. Run download_dataset.py first and update DATA_YAML if needed."
        )

    model = YOLO(MODEL)

    model.train(
        data=str(data_file),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        project="runs/fruit_detection",
        name="train",
        patience=15,
        plots=True,
    )

    print("\nTraining completed.")
    print("Best model should be inside:")
    print("runs/fruit_detection/train/weights/best.pt")


if __name__ == "__main__":
    main()
