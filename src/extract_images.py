
import zipfile
import re
import shutil
from pathlib import Path
import pandas as pd
from PIL import Image

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

ZIP_PATH = BASE_DIR / "data" / "raw" / "media.zip"
DATASET_PATH = BASE_DIR / "data" / "processed" / "validated_dataset.csv"
IMAGE_DIR = BASE_DIR / "data" / "images"
REPORT_PATH = BASE_DIR / "data" / "processed" / "image_report.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_sample_id(filename):
    match = re.search(
        r"sample[\s_-]*0*(\d+)",
        Path(filename).name,
        re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    return None


def extract_images():

    print("\nStarting image extraction...")

    if not ZIP_PATH.exists():
        raise FileNotFoundError("media.zip not found!")

    if not DATASET_PATH.exists():
        raise FileNotFoundError("Validated dataset not found!")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(DATASET_PATH)

    valid_ids = set(
        dataset["Sample ID"]
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .dropna()
        .astype(int)
    )

    report = []
    extracted_ids = set()

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:

        for file in archive.namelist():

            path = Path(file)

            if path.is_dir():
                continue

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            sample_id = get_sample_id(file)

            if sample_id is None:
                continue

            if sample_id not in valid_ids:
                continue

            output_name = (
                f"SAMPLE_{sample_id:03d}{path.suffix.lower()}"
            )

            output_path = IMAGE_DIR / output_name

            try:

                with archive.open(file) as source:
                    with Image.open(source) as image:
                        image.verify()

                with archive.open(file) as source:
                    with Image.open(source) as image:
                        image = image.convert("RGB")
                        image.save(output_path)

                extracted_ids.add(sample_id)

                report.append({
                    "Sample_ID": f"SAMPLE_{sample_id:03d}",
                    "Image_File": output_name,
                    "Status": "Extracted",
                    "Width": image.width,
                    "Height": image.height
                })

            except Exception as error:

                report.append({
                    "Sample_ID": f"SAMPLE_{sample_id:03d}",
                    "Image_File": path.name,
                    "Status": f"Error: {error}",
                    "Width": None,
                    "Height": None
                })

    # Identify missing images
    for sample_id in sorted(valid_ids - extracted_ids):

        report.append({
            "Sample_ID": f"SAMPLE_{sample_id:03d}",
            "Image_File": "",
            "Status": "Missing",
            "Width": None,
            "Height": None
        })

    report_df = pd.DataFrame(report)

    report_df.to_csv(
        REPORT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========== EXTRACTION SUMMARY ==========")
    print("Dataset samples:", len(valid_ids))
    print("Images extracted:", len(extracted_ids))
    print("Images missing:", len(valid_ids - extracted_ids))

    print("\nReport saved:", REPORT_PATH)
    print("Images saved:", IMAGE_DIR)


if __name__ == "__main__":
    extract_images()
