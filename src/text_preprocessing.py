
import pandas as pd
import re
from pathlib import Path

# =====================================
# 1. PROJECT PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "processed" / "validated_dataset.csv"

OUTPUT_PATH = BASE_DIR / "data" / "processed" / "cleaned_dataset.csv"


# =====================================
# 2. TEXT CLEANING FUNCTION
# =====================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove user mentions
    text = re.sub(r"@\w+", " ", text)

    # Remove hashtag symbol, preserve word
    text = text.replace("#", "")

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =====================================
# 3. PREPROCESS DATASET
# =====================================

def preprocess_dataset():

    print("\nStarting text preprocessing...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError("Validated dataset not found!")

    df = pd.read_csv(INPUT_PATH)

    print("Dataset loaded:", df.shape)

    text_columns = {
        "Claim_text": "Claim_text_clean",
        "Evidence_Text": "Evidence_Text_clean",
        "Reasoning": "Reasoning_clean"
    }

    for original, cleaned in text_columns.items():

        if original not in df.columns:
            raise ValueError(f"Missing column: {original}")

        df[cleaned] = df[original].apply(clean_text)

        print(f"Processed: {original}")

    # Save cleaned dataset
    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========== TEXT PREPROCESSING SUMMARY ==========")

    print("Total records:", len(df))

    for column in text_columns.values():
        print(
            column,
            "missing/empty:",
            (df[column] == "").sum()
        )

    print("\nCleaned dataset saved at:")
    print(OUTPUT_PATH)

    print("\nTEXT PREPROCESSING COMPLETED!")


# =====================================
# 4. MAIN
# =====================================

if __name__ == "__main__":
    preprocess_dataset()
