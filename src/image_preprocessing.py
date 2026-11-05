
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

# =====================================
# 1. PROJECT PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = BASE_DIR / "data" / "images"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "images_224"

REPORT_PATH = BASE_DIR / "data" / "processed" / "preprocessing_report.csv"

IMAGE_SIZE = (224, 224)


# =====================================
# 2. IMAGE PREPROCESSING
# =====================================

def preprocess_images():

    print("\nStarting image preprocessing...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_extensions = [".jpg", ".jpeg", ".png", ".webp"]

    report = []

    processed_count = 0
    failed_count = 0

    image_files = [
        f for f in IMAGE_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    for image_path in image_files:

        try:

            # Open image
            with Image.open(image_path) as image:

                # Convert to RGB
                image = image.convert("RGB")

                original_width, original_height = image.size

                # Resize while preserving aspect ratio
                image.thumbnail(
                    IMAGE_SIZE,
                    Image.Resampling.LANCZOS
                )

                # Create 224x224 canvas
                canvas = Image.new(
                    "RGB",
                    IMAGE_SIZE,
                    (0, 0, 0)
                )

                x = (224 - image.width) // 2
                y = (224 - image.height) // 2

                canvas.paste(image, (x, y))

                # Save processed image
                output_path = (
                    OUTPUT_DIR /
                    f"{image_path.stem}_224.png"
                )

                canvas.save(output_path)

                # Normalize pixels to 0-1
                image_array = (
                    np.asarray(canvas, dtype=np.float32) / 255.0
                )

                processed_count += 1

                report.append({
                    "Image": image_path.name,
                    "Processed_Image": output_path.name,
                    "Original_Width": original_width,
                    "Original_Height": original_height,
                    "Processed_Shape": str(image_array.shape),
                    "Pixel_Min": float(image_array.min()),
                    "Pixel_Max": float(image_array.max()),
                    "Status": "Success"
                })

        except Exception as error:

            failed_count += 1

            report.append({
                "Image": image_path.name,
                "Status": f"Failed: {error}"
            })

    # Save report
    pd.DataFrame(report).to_csv(
        REPORT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========== PREPROCESSING SUMMARY ==========")

    print("Total images:", len(image_files))
    print("Successfully processed:", processed_count)
    print("Failed images:", failed_count)

    print("\nImage dimensions: 224 x 224")
    print("Colour channels: RGB")
    print("Normalization range: 0 to 1")

    print("\nProcessed images saved at:")
    print(OUTPUT_DIR)

    print("\nReport saved at:")
    print(REPORT_PATH)


# =====================================
# 3. MAIN
# =====================================

if __name__ == "__main__":
    preprocess_images()
