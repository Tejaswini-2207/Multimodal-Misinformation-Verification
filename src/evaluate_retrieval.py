
import pandas as pd
from pathlib import Path

from rag_retriever import RAGEvidenceRetriever


# =====================================
# 1. PROJECT PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR / "data" / "processed" / "cleaned_dataset.csv"
)

OUTPUT_PATH = (
    BASE_DIR / "outputs" / "retrieval_evaluation.csv"
)


# =====================================
# 2. TEST CLAIMS
# =====================================

TEST_CASES = [
    {
        "query": (
            "A viral picture allegedly shows Pakistan's "
            "women's cricket captain touching the feet "
            "of Indian cricketer Smriti Mandhana."
        ),
        "expected_id": "SAMPLE-052"
    },
    {
        "query": (
            "A viral photograph claims that British Prime "
            "Minister Keir Starmer and King Charles were "
            "praying together inside a mosque."
        ),
        "expected_id": "SAMPLE-094"
    }
]


# =====================================
# 3. EVALUATE RETRIEVAL
# =====================================

def evaluate_retrieval():

    print("\nStarting RAG retrieval evaluation...")

    df = pd.read_csv(DATASET_PATH)

    retriever = RAGEvidenceRetriever(DATASET_PATH)

    retriever.load_data()
    retriever.build_index()

    evaluation_results = []

    reciprocal_ranks = []
    hits_at_5 = []

    for case in TEST_CASES:

        query = case["query"]
        expected_id = case["expected_id"]

        print("\n" + "=" * 60)
        print("QUERY:", query)
        print("EXPECTED SAMPLE:", expected_id)

        # Retrieve all records to determine the target rank
        results = retriever.retrieve(
            query,
            top_k=len(retriever.df)
        )

        rank = None

        for position, result in enumerate(results, start=1):

            if str(result["Sample_ID"]).strip() == expected_id:
                rank = position
                break

        if rank is not None:

            reciprocal_rank = 1 / rank
            hit_at_5 = int(rank <= 5)

            reciprocal_ranks.append(reciprocal_rank)
            hits_at_5.append(hit_at_5)

            print("\nTarget found at rank:", rank)

        else:

            reciprocal_rank = 0
            hit_at_5 = 0

            reciprocal_ranks.append(0)
            hits_at_5.append(0)

            print("\nTarget sample was not retrieved.")

        # Display top five results
        print("\nTop 5 retrieved records:")

        for position, result in enumerate(results[:5], start=1):

            print(
                position,
                result["Sample_ID"],
                "Score:",
                result["Combined_Score"]
            )

        evaluation_results.append({
            "Query": query,
            "Expected_Sample_ID": expected_id,
            "Retrieved_Rank": rank if rank else "Not Found",
            "Hit_at_5": hit_at_5,
            "Reciprocal_Rank": reciprocal_rank
        })

    # =================================
    # 4. CALCULATE METRICS
    # =================================

    mean_reciprocal_rank = (
        sum(reciprocal_ranks) / len(reciprocal_ranks)
    )

    hit_rate_at_5 = (
        sum(hits_at_5) / len(hits_at_5)
    )

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 60)

    print("Number of test queries:", len(TEST_CASES))

    print("Hit Rate @ 5:", round(hit_rate_at_5, 4))

    print("MRR:", round(mean_reciprocal_rank, 4))

    # =================================
    # 5. SAVE RESULTS
    # =================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    pd.DataFrame(evaluation_results).to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nEvaluation report saved at:")
    print(OUTPUT_PATH)

    print("\nRETRIEVAL EVALUATION COMPLETED!")


if __name__ == "__main__":
    evaluate_retrieval()
