
import numpy as np
import pandas as pd
from pathlib import Path

from scipy.sparse import load_npz, csr_matrix, hstack, save_npz


# =====================================
# 1. PROJECT PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEXT_DIR = BASE_DIR / "data" / "processed" / "text_features"

IMAGE_DIR = BASE_DIR / "data" / "processed" / "image_features"

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "fused_features"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================
# 2. LOAD FEATURES
# =====================================

def load_features():

    print("\nLoading extracted features...")

    claim_features = load_npz(
        TEXT_DIR / "claim_tfidf.npz"
    )

    evidence_features = load_npz(
        TEXT_DIR / "evidence_tfidf.npz"
    )

    image_features = np.load(
        IMAGE_DIR / "image_features.npy"
    )

    text_index = pd.read_csv(
        TEXT_DIR / "feature_index.csv"
    )

    image_report = pd.read_csv(
        IMAGE_DIR / "image_feature_report.csv"
    )

    print("Claim features:", claim_features.shape)
    print("Evidence features:", evidence_features.shape)
    print("Image features:", image_features.shape)

    return (
        claim_features,
        evidence_features,
        image_features,
        text_index,
        image_report
    )


# =====================================
# 3. ALIGN FEATURES BY SAMPLE ID
# =====================================

def align_features(
    claim_features,
    evidence_features,
    image_features,
    text_index,
    image_report
):

    print("\nAligning features by Sample ID...")

    # Image feature rows were saved only for successful images.
    successful_images = image_report[
        image_report["Status"] == "Success"
    ].reset_index(drop=True)

    if len(successful_images) != image_features.shape[0]:
        raise ValueError(
            "Image report and image feature rows do not match."
        )

    if len(text_index) != claim_features.shape[0]:
        raise ValueError(
            "Claim feature rows and text index do not match."
        )

    if len(text_index) != evidence_features.shape[0]:
        raise ValueError(
            "Evidence feature rows and text index do not match."
        )

    # Create Sample ID to feature-row mappings.
    image_row_map = {
        str(sample_id): row
        for row, sample_id in enumerate(
            successful_images["Sample_ID"]
        )
    }

    text_row_map = {
        str(sample_id): row
        for row, sample_id in enumerate(
            text_index["Sample_ID"]
        )
    }

    # Keep only samples having both text and image features.
    common_ids = [
        str(sample_id)
        for sample_id in text_index["Sample_ID"]
        if str(sample_id) in image_row_map
    ]

    text_rows = [
        text_row_map[sample_id]
        for sample_id in common_ids
    ]

    image_rows = [
        image_row_map[sample_id]
        for sample_id in common_ids
    ]

    X_claim = claim_features[text_rows]
    X_evidence = evidence_features[text_rows]

    X_image = csr_matrix(
        image_features[image_rows]
    )

    # Combine claim, evidence and image features.
    X_fused = hstack(
        [X_claim, X_evidence, X_image],
        format="csr"
    )

    # Preserve matching labels and IDs.
    aligned_index = (
        text_index.iloc[text_rows]
        .reset_index(drop=True)
        .copy()
    )

    aligned_index["Image_Feature_Available"] = True

    return X_fused, aligned_index


# =====================================
# 4. SAVE FUSED FEATURES
# =====================================

def save_features(X_fused, aligned_index):

    save_npz(
        OUTPUT_DIR / "fused_features.npz",
        X_fused
    )

    aligned_index.to_csv(
        OUTPUT_DIR / "fused_feature_index.csv",
        index=False
    )

    print("\nFused features saved successfully!")


# =====================================
# 5. MAIN
# =====================================

def main():

    (
        claim_features,
        evidence_features,
        image_features,
        text_index,
        image_report
    ) = load_features()

    X_fused, aligned_index = align_features(
        claim_features,
        evidence_features,
        image_features,
        text_index,
        image_report
    )

    save_features(X_fused, aligned_index)

    print("\n========== FUSION SUMMARY ==========")

    print("Samples with all modalities:", X_fused.shape[0])

    print("Claim feature dimensions:", claim_features.shape[1])

    print("Evidence feature dimensions:", evidence_features.shape[1])

    print("Image feature dimensions:", image_features.shape[1])

    print("Combined feature dimensions:", X_fused.shape[1])

    print("\nMissing image samples are excluded from fusion.")

    print("\nMULTIMODAL FEATURE FUSION COMPLETED!")


if __name__ == "__main__":
    main()
