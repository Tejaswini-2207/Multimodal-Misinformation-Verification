
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    BASE_DIR / "data" / "processed" / "cleaned_dataset.csv"
)

df["Evidence_URL"] = df["Evidence_URL"].fillna("").str.strip()
df["Fact_Check_Article_URL"] = (
    df["Fact_Check_Article_URL"].fillna("").str.strip()
)

different = df[
    df["Evidence_URL"] != df["Fact_Check_Article_URL"]
]

output_path = BASE_DIR / "outputs" / "different_urls.csv"

different[
    [
        "Sample ID",
        "Claim_text_clean",
        "Evidence_URL",
        "Fact_Check_Article_URL"
    ]
].to_csv(output_path, index=False)

print("Different URL records:", len(different))
print("Report saved:", output_path)

for _, row in different.iterrows():
    print("\n", row["Sample ID"])
    print("Claim:", row["Claim_text_clean"])
    print("Evidence URL:", row["Evidence_URL"])
    print("Fact-check URL:", row["Fact_Check_Article_URL"])
