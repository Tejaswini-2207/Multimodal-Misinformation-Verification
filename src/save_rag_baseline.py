
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from rag_retriever import RAGEvidenceRetriever, DATASET_PATH


# ==========================================
# 1. OUTPUT CONFIGURATION
# ==========================================

OUTPUT_DIR = BASE_DIR / "outputs" / "rag_baseline"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANKINGS_PATH = (
    OUTPUT_DIR / "tfidf_baseline_rankings.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR / "tfidf_baseline_summary.csv"
)

TOP_K = 5


# ==========================================
# 2. INITIALIZE RETRIEVER
# ==========================================

print("\n======================================")
print("       TF-IDF BASELINE EVALUATION")
print("======================================")

retriever = RAGEvidenceRetriever(DATASET_PATH)

retriever.load_data()
retriever.build_index()


# ==========================================
# 3. EVALUATE ALL DATASET SAMPLES
# ==========================================

ranking_records = []
summary_records = []

total_samples = len(retriever.df)

for position, row in retriever.df.iterrows():

    sample_id = str(row["Sample ID"])
    claim = str(row["Claim_text_clean"])

    print(
        f"\nProcessing {position + 1}/{total_samples}: "
        f"{sample_id}"
    )

    results = retriever.retrieve(
        claim,
        top_k=TOP_K,
        exclude_sample_id=sample_id
    )

    summary_records.append({

        "Query_Sample_ID": sample_id,

        "Retrieved_Count": len(results),

        "Retrieval_Status": (
            "MATCHES_FOUND"
            if results
            else "NO_RELEVANT_MATCH"
        )
    })

    for rank, result in enumerate(results, start=1):

        ranking_records.append({

            "Query_Sample_ID": sample_id,

            "Rank": rank,

            "Retrieved_Sample_ID": result["Sample_ID"],

            "Claim_Similarity": result["Claim_Similarity"],

            "Evidence_Similarity": result["Evidence_Similarity"],

            "Combined_Score": result["Combined_Score"],

            "Retrieved_Claim": result["Claim"],

            "Retrieved_Evidence": result["Evidence"],

            "Evidence_URL": result["Evidence_URL"],

            "Fact_Check_URL": result["Fact_Check_URL"],

            "Evidence_URL_Available": (
                result["Evidence_URL_Available"]
            ),

            "Fact_Check_URL_Available": (
                result["Fact_Check_URL_Available"]
            )
        })


# ==========================================
# 4. SAVE BASELINE RESULTS
# ==========================================

rankings_df = pd.DataFrame(ranking_records)

summary_df = pd.DataFrame(summary_records)

rankings_df.to_csv(
    RANKINGS_PATH,
    index=False,
    encoding="utf-8-sig"
)

summary_df.to_csv(
    SUMMARY_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 5. DISPLAY SUMMARY
# ==========================================

matched_queries = (
    summary_df["Retrieval_Status"] == "MATCHES_FOUND"
).sum()

no_match_queries = (
    summary_df["Retrieval_Status"] == "NO_RELEVANT_MATCH"
).sum()

print("\n======================================")
print("         BASELINE SUMMARY")
print("======================================")

print("Total queries:", total_samples)

print("Queries with retrieved results:", matched_queries)

print("Queries without results:", no_match_queries)

print("Total retrieved records:", len(rankings_df))

print("\nSaved files:")

print(RANKINGS_PATH)

print(SUMMARY_PATH)

print("\nBaseline evaluation completed!")
