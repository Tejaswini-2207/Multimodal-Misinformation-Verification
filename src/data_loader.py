
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "data" / "raw" / "dataset.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "validated_dataset.csv"

REQUIRED_COLUMNS = [
    "Sample ID",
    "Claim_text",
    "Claim_Label",
    "Evidence_Text",
    "Evidence_URL",
    "Reasoning",
    "Modality_Type"
]


def load_dataset():

    print("\nLoading dataset...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    print("Dataset loaded successfully!")
    print("Dataset shape:", df.shape)

    return df


def validate_dataset(df):

    print("\n========== DATA VALIDATION ==========")

    # Check columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print("Required columns verified.")

    # Check duplicate IDs
    print(
        "Duplicate Sample IDs:",
        df["Sample ID"].duplicated().sum()
    )

    # Check missing values
    print("\nMissing values:")
    print(df[REQUIRED_COLUMNS].isnull().sum())

    # Label distribution
    print("\nClaim label distribution:")
    print(df["Claim_Label"].value_counts())

    # Modality distribution
    print("\nModality distribution:")
    print(df["Modality_Type"].value_counts())

    return df


def save_dataset(df):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nValidated dataset saved:")
    print(OUTPUT_PATH)


def main():

    try:

        df = load_dataset()

        df = validate_dataset(df)

        save_dataset(df)

        print("\n================================")
        print("DATASET VALIDATION COMPLETED!")
        print("================================")

    except Exception as error:

        print("\nERROR:", error)


if __name__ == "__main__":
    main()
