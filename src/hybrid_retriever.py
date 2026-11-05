
import re
import hashlib
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR / "data" / "processed" / "cleaned_dataset.csv"
)

CACHE_DIR = BASE_DIR / "data" / "processed" / "hybrid_index"

CACHE_PATH = CACHE_DIR / "hybrid_index.joblib"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5

# Initial pilot threshold. Must be calibrated using
# manually judged relevant and irrelevant claim pairs.
MIN_HYBRID_SCORE = 0.32

# Weights for the hybrid retrieval score.
SEMANTIC_WEIGHT = 0.45
TFIDF_WEIGHT = 0.30
KEYWORD_WEIGHT = 0.25

# Remove generic misinformation-related words from
# keyword overlap so they do not dominate matching.
CUSTOM_STOPWORDS = {
    "claim", "image", "images", "video", "videos",
    "viral", "shared", "shows", "show", "allegedly",
    "social", "media", "post", "posts", "picture",
    "photo", "photos", "circulated", "internet"
}

STOPWORDS = set(ENGLISH_STOP_WORDS) | CUSTOM_STOPWORDS


# ==================================================
# 2. HYBRID RAG RETRIEVER
# ==================================================

class HybridRAGEvidenceRetriever:

    def __init__(self, dataset_path=DATASET_PATH):

        self.dataset_path = Path(dataset_path)

        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None

        self.embedding_model = None
        self.embedding_matrix = None

        self.ready = False

    # ------------------------------------------------
    # LOAD DATASET
    # ------------------------------------------------

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

        for column in required_columns:

            if column not in self.df.columns:
                raise ValueError(
                    f"Required column missing: {column}"
                )

        for column in required_columns:

            self.df[column] = (
                self.df[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

        self.df = self.df[
            self.df["Claim_text_clean"].str.len() > 0
        ].reset_index(drop=True)

        self.df["Normalized_ID"] = (
            self.df["Sample ID"]
            .str.replace("_", "-", regex=False)
            .str.upper()
        )

        print("Records loaded:", len(self.df))

    # ------------------------------------------------
    # NORMALIZE KEYWORDS
    # ------------------------------------------------

    @staticmethod
    def extract_keywords(text):

        words = re.findall(
            r"[a-zA-Z0-9]+",
            str(text).lower()
        )

        return {
            word for word in words
            if len(word) > 1 and word not in STOPWORDS
        }

    # ------------------------------------------------
    # DATASET HASH FOR CACHE VALIDATION
    # ------------------------------------------------

    def get_dataset_hash(self):

        content = "\n".join(
            self.df["Sample ID"].tolist()
            + self.df["Claim_text_clean"].tolist()
        )

        return hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

    # ------------------------------------------------
    # LOAD OR BUILD RETRIEVAL INDEX
    # ------------------------------------------------

    def build_index(self):

        if self.df is None:
            self.load_data()

        CACHE_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        current_hash = self.get_dataset_hash()

        # --------------------------------------------
        # TRY LOADING EXISTING CACHE
        # --------------------------------------------

        if CACHE_PATH.exists():

            print("\nChecking saved retrieval index...")

            try:

                cached = joblib.load(CACHE_PATH)

                if (
                    cached.get("dataset_hash") == current_hash
                    and cached.get("model_name") == MODEL_NAME
                ):

                    self.vectorizer = cached["vectorizer"]
                    self.tfidf_matrix = cached["tfidf_matrix"]
                    self.embedding_matrix = cached["embedding_matrix"]

                    print("Saved retrieval index loaded.")
                    print("Dataset embeddings will NOT be regenerated.")

                    self.embedding_model = SentenceTransformer(
                        MODEL_NAME
                    )

                    self.ready = True

                    return

            except Exception as error:

                print("Cache could not be loaded:", error)

        # --------------------------------------------
        # BUILD NEW INDEX
        # --------------------------------------------

        print("\nBuilding new retrieval index...")

        print("\nBuilding TF-IDF claim index...")

        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )

        self.tfidf_matrix = (
            self.vectorizer.fit_transform(
                self.df["Claim_text_clean"].tolist()
            )
        )

        print("TF-IDF matrix:", self.tfidf_matrix.shape)

        print("\nLoading sentence embedding model...")

        self.embedding_model = SentenceTransformer(
            MODEL_NAME
        )

        print("\nGenerating claim embeddings...")

        self.embedding_matrix = (
            self.embedding_model.encode(
                self.df["Claim_text_clean"].tolist(),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
                batch_size=16
            )
        )

        print(
            "Embedding matrix:",
            self.embedding_matrix.shape
        )

        # --------------------------------------------
        # SAVE INDEX
        # --------------------------------------------

        cache_data = {
            "dataset_hash": current_hash,
            "model_name": MODEL_NAME,
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "embedding_matrix": self.embedding_matrix
        }

        joblib.dump(
            cache_data,
            CACHE_PATH
        )

        print("\nRetrieval index saved successfully.")

        self.ready = True

    # ------------------------------------------------
    # URL VALIDATION
    # ------------------------------------------------

    @staticmethod
    def is_valid_url(value):

        value = str(value).strip()

        return bool(
            re.match(
                r"^https?://\S+$",
                value,
                flags=re.IGNORECASE
            )
        )

    # ------------------------------------------------
    # HYBRID EVIDENCE RETRIEVAL
    # ------------------------------------------------

    def retrieve(
        self,
        input_claim,
        top_k=TOP_K,
        exclude_sample_id=None
    ):

        if not self.ready:
            self.build_index()

        input_claim = str(input_claim).strip()

        if not input_claim:
            raise ValueError(
                "Input claim cannot be empty."
            )

        # --------------------------------------------
        # 1. TF-IDF SIMILARITY
        # --------------------------------------------

        query_tfidf = self.vectorizer.transform(
            [input_claim]
        )

        tfidf_scores = cosine_similarity(
            query_tfidf,
            self.tfidf_matrix
        ).flatten()

        # --------------------------------------------
        # 2. SEMANTIC SIMILARITY
        # --------------------------------------------

        query_embedding = (
            self.embedding_model.encode(
                [input_claim],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        )

        semantic_scores = (
            self.embedding_matrix @ query_embedding[0]
        )

        semantic_scores = np.clip(
            semantic_scores,
            0.0,
            1.0
        )

        # --------------------------------------------
        # 3. KEYWORD OVERLAP
        # --------------------------------------------

        query_keywords = self.extract_keywords(
            input_claim
        )

        keyword_scores = []

        for candidate_claim in self.df[
            "Claim_text_clean"
        ]:

            candidate_keywords = self.extract_keywords(
                candidate_claim
            )

            if not query_keywords:
                overlap = 0.0

            else:
                common_keywords = (
                    query_keywords & candidate_keywords
                )

                overlap = (
                    len(common_keywords)
                    / len(query_keywords)
                )

            keyword_scores.append(overlap)

        keyword_scores = np.array(keyword_scores)

        # --------------------------------------------
        # 4. FINAL HYBRID SCORE
        # --------------------------------------------

        hybrid_scores = (
            SEMANTIC_WEIGHT * semantic_scores
            + TFIDF_WEIGHT * tfidf_scores
            + KEYWORD_WEIGHT * keyword_scores
        )

        # --------------------------------------------
        # 5. EXCLUDE ORIGINAL SAMPLE
        # --------------------------------------------

        eligible_indices = np.arange(len(self.df))

        if exclude_sample_id is not None:

            normalized_exclusion = (
                str(exclude_sample_id)
                .strip()
                .replace("_", "-")
                .upper()
            )

            eligible_indices = eligible_indices[
                self.df["Normalized_ID"].to_numpy()
                != normalized_exclusion
            ]

        ranked_indices = eligible_indices[
            np.argsort(
                hybrid_scores[eligible_indices]
            )[::-1]
        ]

        # --------------------------------------------
        # 6. PREPARE RESULTS
        # --------------------------------------------

        results = []

        for index in ranked_indices:

            score = float(hybrid_scores[index])

            if score < MIN_HYBRID_SCORE:
                break

            row = self.df.iloc[index]

            evidence_url = row["Evidence_URL"]
            fact_check_url = row["Fact_Check_Article_URL"]

            results.append({

                "Sample_ID": row["Sample ID"],

                "Claim": row["Claim_text_clean"],

                "Evidence": row["Evidence_Text_clean"],

                "TFIDF_Score": round(
                    float(tfidf_scores[index]), 4
                ),

                "Semantic_Score": round(
                    float(semantic_scores[index]), 4
                ),

                "Keyword_Overlap": round(
                    float(keyword_scores[index]), 4
                ),

                "Hybrid_Score": round(score, 4),

                "Evidence_URL": evidence_url,

                "Fact_Check_URL": fact_check_url,

                "Evidence_URL_Available": (
                    self.is_valid_url(evidence_url)
                ),

                "Fact_Check_URL_Available": (
                    self.is_valid_url(fact_check_url)
                )
            })

            if len(results) >= top_k:
                break

        return results

    # ------------------------------------------------
    # CLOSE MODEL
    # ------------------------------------------------

    def close(self):

        self.embedding_model = None


# ==================================================
# 3. DISPLAY RESULTS
# ==================================================

def display_results(results):

    print("\n========== HYBRID RAG RESULTS ==========")

    if not results:

        print("\nNo sufficiently relevant evidence found.")

        print(
            "System decision: INSUFFICIENT EVIDENCE"
        )

        return

    for rank, result in enumerate(results, start=1):

        print(f"\nRESULT {rank}")

        print("Sample ID:", result["Sample_ID"])

        print("Hybrid score:", result["Hybrid_Score"])

        print("TF-IDF score:", result["TFIDF_Score"])

        print(
            "Semantic score:",
            result["Semantic_Score"]
        )

        print(
            "Keyword overlap:",
            result["Keyword_Overlap"]
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

        print("-" * 65)

    print(
        "\nNote: Similarity scores measure retrieval relevance, "
        "not claim truth or probability."
    )


# ==================================================
# 4. MAIN PROGRAM
# ==================================================

def main():

    retriever = HybridRAGEvidenceRetriever()

    try:

        retriever.build_index()

        print("\n======================================")
        print("          HYBRID RAG SYSTEM")
        print("======================================")

        sample_id = input(
            "\nEnter Sample ID (example SAMPLE-052): "
        ).strip()

        normalized_sample_id = (
            sample_id.replace("_", "-").upper()
        )

        matching_rows = retriever.df[
            retriever.df["Normalized_ID"]
            == normalized_sample_id
        ]

        if matching_rows.empty:

            print("\nSample ID not found.")

            return

        selected_row = matching_rows.iloc[0]

        claim = selected_row["Claim_text_clean"]

        print("\nSelected Sample:", sample_id)

        print("\nOriginal claim:")
        print(claim)

        print("\nRetrieving independent evidence...")

        results = retriever.retrieve(
            claim,
            top_k=TOP_K,
            exclude_sample_id=sample_id
        )

        display_results(results)

    finally:

        retriever.close()


if __name__ == "__main__":
    main()
