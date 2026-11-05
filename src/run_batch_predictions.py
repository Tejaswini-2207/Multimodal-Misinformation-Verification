import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from llm_verifier import verify_claim, PREDICTIONS_PATH

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "normalized_labels.csv"
)

BATCH_SIZE = 100


def normalize_id(value):
    return (
        str(value)
        .strip()
        .upper()
        .replace("_", "-")
    )


def main():

    print("\n========== BATCH PREDICTION ==========")

    dataset = pd.read_csv(DATASET_PATH)

    dataset["Sample_ID"] = (
        dataset["Sample ID"].apply(normalize_id)
    )

    # Identify all previously attempted samples.
    if PREDICTIONS_PATH.exists():

        existing = pd.read_csv(PREDICTIONS_PATH)

        existing["Sample_ID"] = (
            existing["Sample_ID"].apply(normalize_id)
        )

        completed = set(existing["Sample_ID"])

    else:

        completed = set()

    # Select only samples not previously attempted.
    pending = dataset[
        ~dataset["Sample_ID"].isin(completed)
    ]

    batch = pending.head(BATCH_SIZE)

    print("Total dataset samples:", len(dataset))
    print("Previously attempted samples:", len(completed))
    print("Remaining new samples:", len(pending))
    print("This batch:", len(batch))

    if batch.empty:

        print("\nAll samples have been attempted.")
        print("We can now review and retry abstentions separately.")
        return

    for index, (_, row) in enumerate(
        batch.iterrows(),
        start=1
    ):

        sample_id = row["Sample_ID"]

        print("\n-----------------------------------")
        print(f"Processing {index}/{len(batch)}")
        print("Sample:", sample_id)
        print("-----------------------------------")

        try:

            result = verify_claim(sample_id)

            if result:
                print("Prediction completed.")

        except Exception as error:

            print(f"Failed for {sample_id}: {error}")
            print("Continuing with the next sample...")

    print("\n========== BATCH COMPLETED ==========")

    if PREDICTIONS_PATH.exists():

        final_df = pd.read_csv(PREDICTIONS_PATH)

        print(
            "Total saved predictions:",
            len(final_df)
        )

        print(
            "Prediction file:",
            PREDICTIONS_PATH
        )


if __name__ == "__main__":
    main()