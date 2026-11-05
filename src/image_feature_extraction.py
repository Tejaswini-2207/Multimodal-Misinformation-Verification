
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from skimage.feature import hog

# =====================================
# 1. PROJECT PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = BASE_DIR / "data" / "processed" / "images_224"

DATASET_PATH = BASE_DIR / "data" / "processed" / "cleaned_dataset.csv"

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "image_features"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================
# 2. IMAGE FEATURE EXTRACTION
# =====================================

def extract_image_features():

    print("\nStarting image feature extraction...")

    df = pd.read_csv(DATASET_PATH)

    feature_vectors = []
    report = []

    for _, row in df.iterrows():

        sample_id = str(row["Sample ID"])

        number = int(
            "".join(filter(str.isdigit, sample_id))
        )

        image_path = (
            IMAGE_DIR / f"SAMPLE_{number:03d}_224.png"
        )

        if not image_path.exists():

            print(f"Image missing: {sample_id}")

            report.append({
                "Sample_ID": sample_id,
                "Status": "Missing"
            })

            continue

        try:

            # Read image
            image = Image.open(image_path).convert("RGB")

            image_array = (
                np.asarray(image, dtype=np.float32) / 255.0
            )

            # Convert to grayscale for HOG
            gray_image = np.asarray(
                image.convert("L")
            )

            # HOG features
            hog_features = hog(
                gray_image,
                orientations=9,
                pixels_per_cell=(16, 16),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
                feature_vector=True
            )

            # RGB histograms
            rgb_histograms = []

            for channel in range(3):

                histogram, _ = np.histogram(
                    image_array[:, :, channel],
                    bins=16,
                    range=(0, 1),
                    density=True
                )

                rgb_histograms.extend(histogram)

            # Combine features
            combined_features = np.concatenate([
                hog_features,
                np.array(rgb_histograms)
            ]).astype(np.float32)

            feature_vectors.append(combined_features)

            report.append({
                "Sample_ID": sample_id,
                "Feature_Dimension": len(combined_features),
                "Status": "Success"
            })

        except Exception as error:

            print(f"Error processing {sample_id}: {error}")

            report.append({
                "Sample_ID": sample_id,
                "Status": f"Failed: {error}"
            })

    # Save feature vectors
    if feature_vectors:

        X_image = np.vstack(feature_vectors)

        np.save(
            OUTPUT_DIR / "image_features.npy",
            X_image
        )

        print("\nImage feature shape:", X_image.shape)

    # Save report
    pd.DataFrame(report).to_csv(
        OUTPUT_DIR / "image_feature_report.csv",
        index=False
    )

    print("\n========== IMAGE FEATURE SUMMARY ==========")

    print("Total dataset samples:", len(df))
    print("Images with features:", len(feature_vectors))
    print("Images without features:", len(df) - len(feature_vectors))

    print("\nOutput directory:", OUTPUT_DIR)

    print("\nIMAGE FEATURE EXTRACTION COMPLETED!")


# =====================================
# 3. MAIN
# =====================================

if __name__ == "__main__":
    extract_image_features()
