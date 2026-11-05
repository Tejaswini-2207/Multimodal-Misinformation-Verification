import os
import base64
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv
from groq import Groq


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = BASE_DIR / "data" / "images"

DATASET_PATH = (
    BASE_DIR / "data" / "processed" / "cleaned_dataset.csv"
)

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Check your project's .env file."
    )

client = Groq(api_key=API_KEY)

VISION_MODEL = "qwen/qwen3.8-27b"

SUPPORTED_FORMATS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


# ============================================================
# 2. FIND DATASET IMAGE USING SAMPLE ID
# ============================================================

def find_image(sample_id):

    normalized_id = (
        str(sample_id)
        .strip()
        .upper()
        .replace("_", "-")
    )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_DIR}"
        )

    for path in IMAGE_DIR.rglob("*"):

        if not path.is_file():
            continue

        file_id = (
            path.stem
            .strip()
            .upper()
            .replace("_", "-")
        )

        if (
            file_id == normalized_id
            and path.suffix.lower() in SUPPORTED_FORMATS
        ):
            return path

    raise FileNotFoundError(
        f"No image found for Sample ID: {sample_id}"
    )


# ============================================================
# 3. LOAD DATASET CLAIM
# ============================================================

def load_claim(sample_id):

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    dataset = pd.read_csv(DATASET_PATH)

    required_columns = {
        "Sample ID",
        "Claim_text_clean"
    }

    missing_columns = (
        required_columns - set(dataset.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: {missing_columns}"
        )

    normalized_id = (
        str(sample_id)
        .strip()
        .upper()
        .replace("_", "-")
    )

    dataset_ids = (
        dataset["Sample ID"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace("_", "-", regex=False)
    )

    matching_rows = dataset[
        dataset_ids == normalized_id
    ]

    if matching_rows.empty:
        raise ValueError(
            f"{sample_id} not found in dataset."
        )

    claim = matching_rows.iloc[0]["Claim_text_clean"]

    if pd.isna(claim) or not str(claim).strip():
        raise ValueError(
            f"No claim text available for {sample_id}."
        )

    return str(claim).strip()


# ============================================================
# 4. ENCODE IMAGE
# ============================================================

def encode_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if image_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            "Unsupported image format. "
            "Use JPG, JPEG, PNG or WEBP."
        )

    try:

        with Image.open(image_path) as image:

            image = image.convert("RGB")

            image.thumbnail((1280, 1280))

            buffer = BytesIO()

            image.save(
                buffer,
                format="JPEG",
                quality=85
            )

            encoded_image = base64.b64encode(
                buffer.getvalue()
            ).decode("utf-8")

    except UnidentifiedImageError:
        raise ValueError(
            "The uploaded file is not a valid image."
        )

    except Exception as error:
        raise RuntimeError(
            f"Unable to process image: {error}"
        )

    return (
        f"data:image/jpeg;base64,{encoded_image}"
    )


# ============================================================
# 5. ANALYZE ANY IMAGE
# ============================================================

def analyze_image_file(image_path, claim):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not str(claim).strip():
        raise ValueError(
            "Claim text cannot be empty."
        )

    print("\nAnalyzing supplied image...")
    print("Image:", image_path.name)

    image_data = encode_image(image_path)

    prompt = f"""
You are an image analysis component in an
Explainable Multimodal Misinformation Verification System.

CLAIM TO INVESTIGATE:

{claim}

Analyze the supplied image carefully.

Return your analysis using the following sections:

1. VISUAL DESCRIPTION

Describe the visible people, objects, aircraft,
buildings, locations, activities and background.

2. VISIBLE TEXT

Transcribe readable text appearing inside the image.

3. WATERMARKS AND OVERLAYS

Identify visible logos, watermarks, labels,
captions, timestamps and editing marks.

4. CLAIM-IMAGE COMPARISON

Explain:

- Which parts of the claim are visually consistent
  with the supplied image.
- Which parts are inconsistent.
- Which details cannot be verified visually.

5. IMAGE CONTEXT

Explain what the image appears to represent.
Do not assume its date, location or original source
unless this information is visibly established.

6. LIMITATIONS

Mention what cannot be established from the image alone.

STRICT RULES:

- Do not independently classify the claim as TRUE or FALSE.
- Do not assume that an image or video screenshot is authentic.
- Do not identify aircraft models or locations unless
  the visual evidence supports that identification.
- Do not claim an image is AI-generated without forensic evidence.
- Do not invent visible text, events or visual details.
- Distinguish direct observations from interpretations.
- If the image does not directly address the claim,
  explicitly state that it is unrelated or insufficient
  for visual verification.
- Explain uncertainty clearly.

Return a structured and evidence-based analysis.
"""

    try:

        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data
                            }
                        }
                    ]
                }
            ],
            temperature=0.2,
            max_completion_tokens=1500
        )

        result = response.choices[0].message.content

        if not result:
            raise RuntimeError(
                "Vision model returned an empty response."
            )

        return result.strip()

    except Exception as error:

        raise RuntimeError(
            f"Image analysis failed: {error}"
        )


# ============================================================
# 6. ANALYZE DATASET IMAGE
# ============================================================

def analyze_image(sample_id):

    sample_id = str(sample_id).strip().upper()

    claim = load_claim(sample_id)

    image_path = find_image(sample_id)

    print("\nSample ID:", sample_id)
    print("Claim:", claim)
    print("Image:", image_path.name)

    return analyze_image_file(
        image_path=image_path,
        claim=claim
    )


# ============================================================
# 7. MAIN FUNCTION - STANDALONE TESTING
# ============================================================

def main():

    print("\n========================================")
    print("       IMAGE UNDERSTANDING MODULE")
    print("========================================")

    print("1. Analyze dataset image")
    print("2. Analyze custom image")

    choice = input(
        "\nChoose option (1 or 2): "
    ).strip()

    try:

        if choice == "1":

            sample_id = input(
                "Enter Sample ID: "
            ).strip()

            result = analyze_image(sample_id)

        elif choice == "2":

            image_path = input(
                "Enter image file path: "
            ).strip().strip('"')

            claim = input(
                "Enter the claim: "
            ).strip()

            result = analyze_image_file(
                image_path=image_path,
                claim=claim
            )

        else:

            print("Invalid option.")
            return

        print("\n========================================")
        print("             IMAGE ANALYSIS")
        print("========================================")

        print(result)

        print("\n========================================")
        print("          ANALYSIS COMPLETED")
        print("========================================")

    except Exception as error:

        print("\nIMAGE ANALYSIS ERROR:")
        print(error)


if __name__ == "__main__":
    main()