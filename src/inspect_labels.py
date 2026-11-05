
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "data" / "raw" / "dataset.csv"

df = pd.read_csv(DATASET_PATH)

print("\n========== ORIGINAL LABEL VALUES ==========")

print(df["Claim_Label"].value_counts(dropna=False))

print("\n========== LABELLED SAMPLES ==========")

labelled = df[df["Claim_Label"].notna()][
    ["Sample ID", "Claim_text", "Claim_Label"]
]

print(labelled.to_string(index=False))

print("\n========== UNLABELLED SAMPLES ==========")

unlabelled = df[df["Claim_Label"].isna()]

print("Unlabelled records:", len(unlabelled))

print("\n========== LABEL SUMMARY ==========")

print(
    df.groupby("Claim_Label", dropna=False)
    .size()
    .reset_index(name="Record_Count")
)
