from pathlib import Path

import kagglehub


DATASET = "kapturovalexander/fruits-by-yolo-fruits-detection"
DATASET_URL = "https://www.kaggle.com/datasets/kapturovalexander/fruits-by-yolo-fruits-detection"
OUTPUT_DIR = "data/fruits-yolo"


def find_dataset_yaml(root: Path):
    yaml_files = list(root.rglob("data.yaml")) + list(root.rglob("dataset.yaml"))
    return sorted(set(yaml_files))


def download_dataset():
    """Download the Kaggle fruit dataset and return its YOLO YAML path."""
    output_dir = Path(OUTPUT_DIR)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Dataset page: {DATASET_URL}")
    print(f"Downloading Kaggle dataset: {DATASET}")

    dataset_path = Path(
        kagglehub.dataset_download(
            DATASET,
            output_dir=OUTPUT_DIR,
        )
    )

    print(f"Dataset available at: {dataset_path.resolve()}")

    yaml_files = find_dataset_yaml(dataset_path)
    if not yaml_files:
        yaml_files = find_dataset_yaml(output_dir)

    if not yaml_files:
        raise FileNotFoundError(
            "Dataset downloaded, but no data.yaml or dataset.yaml was found. "
            "Check the dataset folder structure."
        )

    print(f"YOLO dataset YAML: {yaml_files[0]}")
    return yaml_files[0]


def main():
    download_dataset()


if __name__ == "__main__":
    main()
