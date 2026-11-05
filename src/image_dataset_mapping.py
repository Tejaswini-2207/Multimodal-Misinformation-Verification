
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET = BASE_DIR / "data" / "processed" / "cleaned_dataset.csv"
IMAGE_DIR = BASE_DIR / "data" / "images"
OUTPUT = BASE_DIR / "outputs" / "image_dataset_mapping.csv"


def normalize_sample_id(value):
    return str(value).strip().upper().replace("_", "-")


# Load dataset
df = pd.read_csv(DATASET)

# Create normalized image mapping
image_files = {}

for path in IMAGE_DIR.rglob("*"):

    if path.is_file():

        normalized_id = normalize_sample_id(path.stem)

        image_files[normalized_id] = path.name


rows = []

for _, record in df.iterrows():

    sample_id = normalize_sample_id(record["Sample ID"])

    image_name = image_files.get(sample_id)

    rows.append({
        "Sample_ID": sample_id,
        "Claim": record["Claim_text_clean"],
        "Image_File": image_name if image_name else "",
        "Image_Status": "FOUND" if image_name else "MISSING",
        "Evidence_URL": record["Evidence_URL"],
        "Fact_Check_URL": record["Fact_Check_Article_URL"]
    })


mapping = pd.DataFrame(rows)

mapping.to_csv(OUTPUT, index=False)

print("\nIMAGE DATASET MAPPING COMPLETE")
print("Total records:", len(mapping))

print(
    "Images found:",
    (mapping["Image_Status"] == "FOUND").sum()
)

print(
    "Images missing:",
    (mapping["Image_Status"] == "MISSING").sum()
)

print("\nMissing samples:")

print(
    mapping[mapping["Image_Status"] == "MISSING"]["Sample_ID"].tolist()
)

print("\nReport saved:", OUTPUT)
