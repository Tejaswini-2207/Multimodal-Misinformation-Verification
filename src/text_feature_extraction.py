
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz


# =====================================
# 1. PROJECT PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    BASE_DIR / "data" / "processed" / "cleaned_dataset.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "text_features"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================
# 2. TF-IDF CONFIGURATION
# =====================================

MAX_FEATURES = 5000

def create_vectorizer():

    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=1,
        sublinear_tf=True
    )


# =====================================
# 3. TEXT FEATURE EXTRACTION
# =====================================

def extract_features():

    print("\nStarting TF-IDF feature extraction...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Cleaned dataset not found!"
        )

    df = pd.read_csv(INPUT_PATH)

    # Preserve row alignment
    sample_ids = df["Sample ID"].astype(str)

    # Get cleaned text
    claim_text = df["Claim_text_clean"].fillna("").astype(str)

    evidence_text = df["Evidence_Text_clean"].fillna("").astype(str)

    # Create separate vectorizers
    claim_vectorizer = create_vectorizer()
    evidence_vectorizer = create_vectorizer()

    # Transform claim text
    X_claim = claim_vectorizer.fit_transform(claim_text)

    # Transform evidence text
    X_evidence = evidence_vectorizer.fit_transform(evidence_text)

    # Save sparse matrices
    save_npz(
        OUTPUT_DIR / "claim_tfidf.npz",
        X_claim
    )

    save_npz(
        OUTPUT_DIR / "evidence_tfidf.npz",
        X_evidence
    )

    # Save vectorizers
    joblib.dump(
        claim_vectorizer,
        OUTPUT_DIR / "claim_vectorizer.joblib"
    )

    joblib.dump(
        evidence_vectorizer,
        OUTPUT_DIR / "evidence_vectorizer.joblib"
    )

    # Save sample alignment
    feature_index = pd.DataFrame({
        "Row_Index": np.arange(len(df)),
        "Sample_ID": sample_ids,
        "Claim_Label": df["Claim_Label"]
    })

    feature_index.to_csv(
        OUTPUT_DIR / "feature_index.csv",
        index=False
    )

    # Display results
    print("\n========== TF-IDF SUMMARY ==========")

    print("Total samples:", len(df))

    print("Claim feature shape:", X_claim.shape)

    print("Evidence feature shape:", X_evidence.shape)

    print("Claim vocabulary:", len(claim_vectorizer.vocabulary_))

    print("Evidence vocabulary:", len(evidence_vectorizer.vocabulary_))

    print("\nFiles saved in:", OUTPUT_DIR)

    print("\nTF-IDF FEATURE EXTRACTION COMPLETED!")


# =====================================
# 4. MAIN
# =====================================

if __name__ == "__main__":
    extract_features()
