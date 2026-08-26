from pathlib import Path

import kagglehub


DATASET = "kapturovalexander/fruits-by-yolo-fruits-detection"
OUTPUT_DIR = "data/fruits-yolo"


def main():
    Path(OUTPUT_DIR).parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Kaggle dataset: {DATASET}")
    dataset_path = Path(
        kagglehub.dataset_download(
            DATASET,
            output_dir=OUTPUT_DIR,
        )
    )

    print(f"\nDataset downloaded to: {dataset_path.resolve()}")

    yaml_files = list(dataset_path.rglob("data.yaml")) + list(dataset_path.rglob("dataset.yaml"))

    if yaml_files:
        print("\nYOLO dataset YAML found:")
        for yaml_file in yaml_files:
            print(yaml_file)
        print("\nCopy the correct path into DATA_YAML in train.py if it is different.")
    else:
        print("\nNo data.yaml was found automatically. Check the downloaded dataset folders.")


if __name__ == "__main__":
    main()
