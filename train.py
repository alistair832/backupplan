from pathlib import Path

from ultralytics import YOLO

from download_dataset import OUTPUT_DIR, download_dataset, find_dataset_yaml


MODEL = "yolo11n.pt"
EPOCHS = 50
IMAGE_SIZE = 640


def get_data_yaml():
    """Use the local Kaggle dataset if available; otherwise download it automatically."""
    output_dir = Path(OUTPUT_DIR)

    if output_dir.exists():
        yaml_files = find_dataset_yaml(output_dir)
        if yaml_files:
            print(f"Using existing dataset: {yaml_files[0]}")
            return yaml_files[0]

    print("Fruit dataset not found locally. Downloading it from Kaggle automatically...")
    return download_dataset()


def main():
    data_file = get_data_yaml()

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
