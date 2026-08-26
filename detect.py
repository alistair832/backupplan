from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


MODEL_PATH = "best.pt"
IMAGE_PATH = "images/fruit.jpg"
CONFIDENCE = 0.25
OUTPUT_PATH = "output/detected_fruit.jpg"


def main():
    model_file = Path(MODEL_PATH)
    image_file = Path(IMAGE_PATH)

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Train the model first and place best.pt in the project folder."
        )

    if not image_file.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}. Put a test image inside the images folder."
        )

    model = YOLO(str(model_file))

    results = model.predict(
        source=str(image_file),
        conf=CONFIDENCE,
        verbose=False,
    )

    result = results[0]
    detected_fruits = []

    print("\n===== Detection Result =====")

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        fruit_name = result.names[class_id]

        detected_fruits.append(fruit_name)

        print(
            f"Detected: {fruit_name:<12} "
            f"Confidence: {confidence * 100:.2f}%"
        )

    counts = Counter(detected_fruits)

    print("\n===== Fruit Count =====")
    if counts:
        for fruit, count in sorted(counts.items()):
            print(f"{fruit}: {count}")
        print(f"Total Fruits: {sum(counts.values())}")
    else:
        print("No fruit detected.")

    annotated_image = result.plot()

    output_file = Path(OUTPUT_PATH)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_file), annotated_image)

    print(f"\nDetected image saved to: {output_file}")

    cv2.imshow("Fruit Detection", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
