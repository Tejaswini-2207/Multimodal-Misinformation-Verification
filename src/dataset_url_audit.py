
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    BASE_DIR / "data" / "processed" / "cleaned_dataset.csv"
)

def classify_url(value):

    value = str(value).strip()

    if not value or value.lower() in ["nan", "none"]:
        return "Missing"

    if not value.startswith(("http://", "https://")):
        return "Not a URL"

    domain = urlparse(value).netloc.lower()

    if "youtube.com" in domain or "youtu.be" in domain:
        return "Video platform"

    if "facebook.com" in domain or "x.com" in domain:
        return "Social media"

    if any(x in domain for x in [
        "who.int", "gov", "un.org"
    ]):
        return "Official source"

    return "Web article / other source"


df["Evidence_URL_Type"] = df["Evidence_URL"].apply(
    classify_url
)

df["Fact_Check_URL_Type"] = df[
    "Fact_Check_Article_URL"
].apply(classify_url)

output = BASE_DIR / "outputs" / "dataset_url_audit.csv"

df[
    [
        "Sample ID",
        "Evidence_URL",
        "Evidence_URL_Type",
        "Fact_Check_Article_URL",
        "Fact_Check_URL_Type"
    ]
].to_csv(output, index=False)

print("\nURL AUDIT COMPLETE")
print("\nEvidence URL distribution:")
print(df["Evidence_URL_Type"].value_counts())

print("\nFact-check URL distribution:")
print(df["Fact_Check_URL_Type"].value_counts())

print("\nReport saved:", output)
