
import re
import numpy as np
import pandas as pd

from pathlib import Path
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR / "data" / "processed" / "cleaned_dataset.csv"
)

MIN_RELEVANCE_SCORE = 0.08


# ==================================================
# 2. SOURCE URL VALIDATION
# ==================================================

def is_valid_url(value):

    if not value or pd.isna(value):
        return False

    value = str(value).strip()

    try:
        parsed = urlparse(value)

        return (
            parsed.scheme in ("http", "https")
            and bool(parsed.netloc)
            and "." in parsed.netloc
        )

    except Exception:
        return False


# ==================================================
# 3. RAG EVIDENCE RETRIEVER
# ==================================================

class RAGEvidenceRetriever:

    def __init__(self, dataset_path):

        self.dataset_path = Path(dataset_path)

        self.df = None

        self.claim_vectorizer = None
        self.evidence_vectorizer = None

        self.claim_matrix = None
        self.evidence_matrix = None

    # ----------------------------------------------
    # Load dataset
    # ----------------------------------------------

    def load_data(self):

        print("\nLoading misinformation dataset...")

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}"
            )

        self.df = pd.read_csv(self.dataset_path)

        required_columns = [
            "Sample ID",
            "Claim_text_clean",
            "Evidence_Text_clean",
            "Evidence_URL",
            "Fact_Check_Article_URL"
        ]

        missing = [
            column for column in required_columns
            if column not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"Dataset is missing columns: {missing}"
            )

        # Clean text columns
        for column in [
            "Claim_text_clean",
            "Evidence_Text_clean",
            "Evidence_URL",
            "Fact_Check_Article_URL"
        ]:

            self.df[column] = (
                self.df[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

        self.df["Sample ID"] = (
            self.df["Sample ID"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        # Keep records containing searchable text
        self.df = self.df[
            (
                self.df["Claim_text_clean"] != ""
            ) |
            (
                self.df["Evidence_Text_clean"] != ""
            )
        ].reset_index(drop=True)

        # Remove duplicate sample IDs
        self.df = self.df.drop_duplicates(
            subset=["Sample ID"],
            keep="first"
        ).reset_index(drop=True)

        print("Records loaded:", len(self.df))

        print(
            "Records with evidence text:",
            (self.df["Evidence_Text_clean"] != "").sum()
        )

        print(
            "Records with valid evidence URLs:",
            self.df["Evidence_URL"].apply(is_valid_url).sum()
        )

    # ----------------------------------------------
    # Build TF-IDF indexes
    # ----------------------------------------------

    def build_index(self):

        print("\nBuilding retrieval indexes...")

        if self.df is None:
            self.load_data()

        # Claim index
        claim_texts = self.df["Claim_text_clean"]

        self.claim_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            strip_accents="unicode"
        )

        self.claim_matrix = (
            self.claim_vectorizer.fit_transform(claim_texts)
        )

        # Evidence index
        evidence_texts = self.df["Evidence_Text_clean"]

        non_empty_evidence = evidence_texts.str.strip() != ""

        if non_empty_evidence.any():

            self.evidence_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words="english",
                sublinear_tf=True,
                strip_accents="unicode"
            )

            self.evidence_matrix = (
                self.evidence_vectorizer.fit_transform(
                    evidence_texts
                )
            )

        else:

            self.evidence_vectorizer = None
            self.evidence_matrix = None

        print(
            "Claim index:",
            self.claim_matrix.shape
        )

        if self.evidence_matrix is not None:

            print(
                "Evidence index:",
                self.evidence_matrix.shape
            )

        else:

            print("Evidence index: unavailable (no evidence text)")

        print("Retrieval indexes built successfully!")

    # ----------------------------------------------
    # Retrieve relevant evidence
    # ----------------------------------------------

    def retrieve(
        self,
        input_claim,
        top_k=5,
        exclude_sample_id=None,
        min_score=MIN_RELEVANCE_SCORE
    ):

        if self.claim_matrix is None:
            self.build_index()

        input_claim = str(input_claim).strip()

        if not input_claim:
            raise ValueError("Enter a valid claim.")

        # Transform query
        query_claim = self.claim_vectorizer.transform(
            [input_claim]
        )

        # Claim similarity
        claim_scores = cosine_similarity(
            query_claim,
            self.claim_matrix
        ).flatten()

        # Evidence similarity
        if self.evidence_vectorizer is not None:

            query_evidence = (
                self.evidence_vectorizer.transform(
                    [input_claim]
                )
            )

            evidence_scores = cosine_similarity(
                query_evidence,
                self.evidence_matrix
            ).flatten()

        else:

            evidence_scores = np.zeros(len(self.df))

        # Weighted relevance
        # Claim similarity: 55%
        # Evidence similarity: 45%

        combined_scores = (
            0.55 * claim_scores +
            0.45 * evidence_scores
        )

        # Exclude query sample when requested
        eligible_indices = np.arange(len(self.df))

        if exclude_sample_id:

            target_id = (
                str(exclude_sample_id)
                .strip()
                .upper()
                .replace("_", "-")
            )

            normalized_ids = (
                self.df["Sample ID"]
                .str.upper()
                .str.replace("_", "-", regex=False)
            )

            eligible_indices = eligible_indices[
                normalized_ids.to_numpy() != target_id
            ]

        # Rank remaining records
        ranked_indices = eligible_indices[
            np.argsort(
                combined_scores[eligible_indices]
            )[::-1]
        ]

        results = []

        for index in ranked_indices:

            score = float(combined_scores[index])

            # Reject weak matches
            if score < min_score:
                break

            row = self.df.iloc[index]

            claim_text = row["Claim_text_clean"]
            evidence_text = row["Evidence_Text_clean"]

            # Avoid records without any evidence text
            if not evidence_text.strip():
                continue

            evidence_url = row["Evidence_URL"]
            fact_check_url = row["Fact_Check_Article_URL"]

            results.append({

                "Sample_ID": row["Sample ID"],

                "Claim": claim_text,

                "Evidence": evidence_text,

                "Claim_Similarity": round(
                    float(claim_scores[index]), 4
                ),

                "Evidence_Similarity": round(
                    float(evidence_scores[index]), 4
                ),

                "Combined_Score": round(score, 4),

                "Evidence_URL": evidence_url,

                "Fact_Check_URL": fact_check_url,

                "Evidence_URL_Available": is_valid_url(
                    evidence_url
                ),

                "Fact_Check_URL_Available": is_valid_url(
                    fact_check_url
                )
            })

            if len(results) >= top_k:
                break

        return results


# ==================================================
# 4. DISPLAY RETRIEVAL RESULTS
# ==================================================

def display_results(results):

    print("\n========== SOURCE-AWARE RAG RESULTS ==========")

    if not results:

        print("\nNo sufficiently relevant evidence found.")

        print(
            "The system should return INSUFFICIENT EVIDENCE "
            "unless other reliable evidence is available."
        )

        return

    for rank, result in enumerate(results, start=1):

        print(f"\nRESULT {rank}")

        print("Sample ID:", result["Sample_ID"])

        print(
            "Combined relevance:",
            result["Combined_Score"]
        )

        print(
            "Claim similarity:",
            result["Claim_Similarity"]
        )

        print(
            "Evidence similarity:",
            result["Evidence_Similarity"]
        )

        print("\nRelated claim:")
        print(result["Claim"])

        print("\nRetrieved evidence:")
        print(result["Evidence"])

        print("\nEvidence URL:")
        print(
            result["Evidence_URL"]
            if result["Evidence_URL_Available"]
            else "Missing or invalid URL"
        )

        print("\nFact-check URL:")
        print(
            result["Fact_Check_URL"]
            if result["Fact_Check_URL_Available"]
            else "Missing or invalid URL"
        )

        print("-" * 60)


# ==================================================
# 5. MAIN FUNCTION
# ==================================================


def main():

    retriever = RAGEvidenceRetriever(DATASET_PATH)

    retriever.load_data()
    retriever.build_index()

    print("\n====================================")
    print("       RAG EVALUATION TEST")
    print("====================================")

    sample_id = input(
        "\nEnter Sample ID (example SAMPLE-052): "
    ).strip()

    normalized_ids = (
        retriever.df["Sample ID"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace("_", "-", regex=False)
    )

    target_id = (
        sample_id.upper().replace("_", "-")
    )

    matching = retriever.df[
        normalized_ids == target_id
    ]

    if matching.empty:
        print("Sample ID not found.")
        return

    claim = matching.iloc[0]["Claim_text_clean"]

    print("\nSelected Sample:", sample_id)
    print("\nOriginal Claim:")
    print(claim)

    print("\nRetrieving independent evidence...")

    results = retriever.retrieve(
        claim,
        top_k=5,
        exclude_sample_id=sample_id
    )

    display_results(results)


if __name__ == "__main__":
    main()
