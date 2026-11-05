
import pandas as pd
import re
from pathlib import Path

# =====================================================
# 1. CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR / "data" / "raw" / "dataset.csv"
)

OUTPUT_DIR = (
    BASE_DIR / "data" / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR / "normalized_labels.csv"
)


# =====================================================
# 2. LABEL MAPPING
# =====================================================

LABEL_MAPPING = {

    # TRUE
    "true": "TRUE",

    # FALSE
    "fake": "FALSE",
    "false": "FALSE",
    "false / fabricated": "FALSE",
    "false/fabricated": "FALSE",
    "fake / fabricated": "FALSE",
    "false / scam": "FALSE",
    "false / unsupported": "FALSE",
    "false / scripted video": "FALSE",
    "false (scripted video)": "FALSE",
    "false / ai-generated": "FALSE",
    "false-ai generated": "FALSE",
    "false / edited video": "FALSE",
    "false / satire": "FALSE",

    # MOSTLY FALSE
    "mostly false": "MOSTLY_FALSE",

    # MISLEADING
    "misleading": "MISLEADING",
    "false / misleading context": "MISLEADING",
    "false (old video)": "MISLEADING",
    "misleading (old footage)": "MISLEADING",
    "misleading / distorted": "MISLEADING",
    "misleading / needs qualification": "MISLEADING",
    "misleading - old footage": "MISLEADING",
    "misleading - old video": "MISLEADING",
    "misleading - unrelated footage": "MISLEADING",
    "old and unrelated visuals": "MISLEADING",
    "partly true / needs qualification": "MISLEADING",

    # MANIPULATED / SYNTHETIC MEDIA
    "ai - generated": "MANIPULATED",
    "ai-generated": "MANIPULATED",
    "ai-altered": "MANIPULATED",
    "altered": "MANIPULATED",
    "altered / deepfake": "MANIPULATED",
    "deepfake / manipulated": "MANIPULATED"

}


# =====================================================
# 3. NORMALIZE LABEL
# =====================================================

def normalize_label(label):

    if pd.isna(label):
        return None

    normalized = str(label).strip().lower()

    # Standardize different dash characters.
    normalized = normalized.replace("–", "-")
    normalized = normalized.replace("—", "-")

    # Remove accidental leading punctuation.
    normalized = re.sub(
        r"^[^a-z]+",
        "",
        normalized
    )

    # Remove extra spaces.
    normalized = " ".join(
        normalized.split()
    )

    return LABEL_MAPPING.get(normalized)


# =====================================================
# 4. MAIN FUNCTION
# =====================================================

def main():

    print("\nLoading original dataset...")

    df = pd.read_csv(DATASET_PATH)

    print("Total records:", len(df))

    # Preserve original labels.
    df["Original_Label"] = df["Claim_Label"]

    # Normalize labels.
    df["Normalized_Label"] = (
        df["Original_Label"].apply(normalize_label)
    )

    # ================================================
    # LABEL DISTRIBUTION
    # ================================================

    print("\n========== NORMALIZED LABEL DISTRIBUTION ==========")

    print(
        df["Normalized_Label"].value_counts(
            dropna=False
        )
    )

    # ================================================
    # UNMAPPED LABELS
    # ================================================

    unmapped = df[
        df["Normalized_Label"].isna()
        & df["Original_Label"].notna()
    ]

    print("\n========== UNMAPPED LABELS ==========")

    if unmapped.empty:

        print("All non-empty labels were mapped.")

    else:

        print(
            unmapped["Original_Label"].value_counts()
        )

    # ================================================
    # SAVE NORMALIZED DATASET
    # ================================================

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n========== FINAL SUMMARY ==========")

    print("Total records:", len(df))

    print(
        "Successfully normalized:",
        df["Normalized_Label"].notna().sum()
    )

    print(
        "Unmapped or missing:",
        df["Normalized_Label"].isna().sum()
    )

    print("\nSaved successfully:")
    print(OUTPUT_PATH)


# =====================================================
# 5. EXECUTION
# =====================================================

if __name__ == "__main__":
    main()
