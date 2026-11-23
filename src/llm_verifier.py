import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from rag_retriever import RAGEvidenceRetriever
from image_understanding import analyze_image, analyze_image_file
from external_retriever import ExternalEvidenceRetriever
from source_validator import validate_sources

from confidence_scorer import (
    calculate_confidence_score,
    display_confidence_score
)

from source_similarity import (
    analyze_source_similarity,
    display_source_similarity
)


# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "data" / "processed" / "cleaned_dataset.csv"

PREDICTIONS_PATH = (
    BASE_DIR / "outputs" / "evaluation" / "model_predictions.csv"
)

load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    try:
        import streamlit as st
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        api_key = None

if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment or Streamlit Secrets.")

client = Groq(api_key=api_key)

TEXT_MODEL = "openai/gpt-oss-120b"

MIN_LOCAL_SCORE = 0.30
MAX_ARTICLE_CHARS = 1500

VALID_LABELS = {
    "TRUE",
    "FALSE",
    "MOSTLY_FALSE",
    "MISLEADING",
    "MANIPULATED",
    "INSUFFICIENT_EVIDENCE"
}

retriever = RAGEvidenceRetriever(DATASET_PATH)
external_retriever = ExternalEvidenceRetriever()


# ==========================================================
# LOAD CLAIM
# ==========================================================

def load_claim(sample_id):

    dataset = pd.read_csv(DATASET_PATH)

    normalized_ids = (
        dataset["Sample ID"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace("_", "-", regex=False)
    )

    target_id = sample_id.strip().upper().replace("_", "-")

    matching_rows = dataset[normalized_ids == target_id]

    if matching_rows.empty:
        raise ValueError(f"{sample_id} not found in dataset.")

    claim = matching_rows.iloc[0]["Claim_text_clean"]

    if pd.isna(claim):
        raise ValueError(f"Claim text is missing for {sample_id}.")

    return str(claim)


# ==========================================================
# SAVE MODEL PREDICTION
# ==========================================================

def save_prediction(sample_id, predicted_label, notes=""):

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    predicted_label = predicted_label.strip().upper()

    if predicted_label not in VALID_LABELS:
        raise ValueError(f"Invalid prediction label: {predicted_label}")

    sample_id = sample_id.strip().upper().replace("_", "-")

    columns = ["Sample_ID", "Predicted_Label", "Notes"]

    if PREDICTIONS_PATH.exists():

        existing = pd.read_csv(PREDICTIONS_PATH)

        for column in columns:
            if column not in existing.columns:
                existing[column] = ""

        rows = existing[columns].to_dict(orient="records")

    else:
        rows = []

    updated = False

    for row in rows:

        existing_id = (
            str(row.get("Sample_ID", ""))
            .strip()
            .upper()
            .replace("_", "-")
        )

        if existing_id == sample_id:

            row["Predicted_Label"] = predicted_label
            row["Notes"] = notes

            updated = True
            break

    if not updated:

        rows.append({
            "Sample_ID": sample_id,
            "Predicted_Label": predicted_label,
            "Notes": notes
        })

    output_df = pd.DataFrame(rows, columns=columns)

    output_df.to_csv(PREDICTIONS_PATH, index=False)

    print("\n========== PREDICTION SAVED ==========")
    print("Sample ID:", sample_id)
    print("Predicted label:", predicted_label)
    print("Saved file:", PREDICTIONS_PATH)


# ==========================================================
# LOCAL RAG EVIDENCE RETRIEVAL
# ==========================================================

def retrieve_local_evidence(claim, current_sample_id, top_k=5):

    print("\nSearching local dataset...")

    # Retrieve evidence while excluding the current sample directly
    # inside the RAG retriever.
    results = retriever.retrieve(
        claim,
        top_k=15,
        exclude_sample_id=current_sample_id
    )

    if results is None:
        return []

    if isinstance(results, list):
        results = pd.DataFrame(results)

    if not isinstance(results, pd.DataFrame):
        raise TypeError(
            "Unexpected local RAG output: " + str(type(results))
        )

    if results.empty:
        return []

    required_columns = [
        "Sample_ID",
        "Claim",
        "Evidence",
        "Combined_Score",
        "Evidence_URL",
        "Fact_Check_URL"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in results.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing local RAG columns: {missing_columns}"
        )

    current_id = (
        str(current_sample_id)
        .strip()
        .upper()
        .replace("_", "-")
    )

    results = results.copy()

    # Normalize IDs for an additional self-retrieval safety check.
    results["Normalized_ID"] = (
        results["Sample_ID"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace("_", "-", regex=False)
    )

    results = results[
        results["Normalized_ID"] != current_id
    ]

    # Safely convert relevance scores.
    results["Combined_Score"] = pd.to_numeric(
        results["Combined_Score"],
        errors="coerce"
    )

    results = results.dropna(
        subset=["Combined_Score"]
    )

    results = results.sort_values(
        by="Combined_Score",
        ascending=False
    )

    evidence_list = []

    for _, row in results.iterrows():

        score = float(row["Combined_Score"])

        # Reject weak matches.
        if score < MIN_LOCAL_SCORE:
            continue

        evidence_list.append({
            "sample_id": str(row["Sample_ID"]),
            "claim": str(row["Claim"]),
            "evidence": str(row["Evidence"]),
            "similarity": score,
            "evidence_url": str(row["Evidence_URL"]),
            "fact_check_url": str(row["Fact_Check_URL"])
        })

        if len(evidence_list) >= top_k:
            break

    print(
        "Relevant local evidence records:",
        len(evidence_list)
    )

    return evidence_list
# ==========================================================
# EXTERNAL EVIDENCE RETRIEVAL
# ==========================================================

def retrieve_external_evidence(claim, max_results=5):

    print("\nSearching external fact-check sources...")

    search_results = external_retriever.retrieve(
        claim,
        max_results=max_results
    )

    validated_results = validate_sources(search_results)

    external_evidence = []

    for item in validated_results:

        external_evidence.append({

            "title": item.get("Title", ""),
            "domain": item.get("Domain", ""),
            "url": item.get("URL", ""),
            "snippet": item.get("Snippet", ""),

            "article_text": item.get("Article_Text", ""),

            "extraction_status": item.get(
                "Extraction_Status", "FAILED"
            ),

            "source_type": item.get(
                "Source_Type", "UNKNOWN"
            ),

            "credibility_category": item.get(
                "Credibility_Category", "UNCLASSIFIED"
            ),

            "classification_basis": item.get(
                "Classification_Basis", "Not classified"
            ),

            "claim_relevance_score": float(
                item.get("Claim_Relevance_Score", 0.0)
            ),

            "keyword_overlap": float(
                item.get("Keyword_Overlap", 0.0)
            )
        })

    print("\nExternal evidence summary")
    print("Relevant articles:", len(external_evidence))

    for item in external_evidence:

        print(
            f"{item['domain']} | "
            f"Relevance: {item['claim_relevance_score']:.4f}"
        )

    return external_evidence


# ==========================================================
# PREPARE LOCAL EVIDENCE
# ==========================================================

def prepare_local_evidence(evidence):

    if not evidence:
        return "No relevant local evidence found."

    text = ""

    for index, item in enumerate(evidence, start=1):

        text += f"""

LOCAL EVIDENCE {index}

Sample ID:
{item['sample_id']}

Related Claim:
{item['claim']}

Evidence:
{item['evidence']}

Retrieval Similarity:
{item['similarity']:.4f}

Evidence URL:
{item['evidence_url']}

Fact Check URL:
{item['fact_check_url']}

----------------------------------------

"""

    return text


# ==========================================================
# PREPARE EXTERNAL EVIDENCE
# ==========================================================

def prepare_external_evidence(evidence):

    if not evidence:
        return "No sufficiently relevant external evidence found."

    text = ""

    for index, item in enumerate(evidence, start=1):

        if item["extraction_status"] == "SUCCESS":

            content_type = "EXTRACTED ARTICLE TEXT"

            content = item["article_text"][:MAX_ARTICLE_CHARS]

        else:

            content_type = "SEARCH SNIPPET ONLY"

            content = item["snippet"]

        text += f"""

EXTERNAL SOURCE {index}

Title:
{item['title']}

Domain:
{item['domain']}

URL:
{item['url']}

Claim Relevance Score:
{item['claim_relevance_score']}

Keyword Overlap:
{item['keyword_overlap']}

Source Type:
{item['source_type']}

Source Category:
{item['credibility_category']}

Classification Basis:
{item['classification_basis']}

Extraction Status:
{item['extraction_status']}

Content Type:
{content_type}

Article Content:
{content}

----------------------------------------

"""

    return text


# ==========================================================
# SOURCE SIMILARITY ANALYSIS
# ==========================================================

def compare_external_sources(external_evidence):

    print("\nAnalysing article similarity...")

    report = analyze_source_similarity(external_evidence)

    display_source_similarity(report)

    return report


def prepare_similarity_text(report):

    if not report.get("source_relationships"):

        return (
            "Insufficient article text for "
            "pairwise similarity analysis."
        )

    text = f"""

SOURCES ANALYSED:
{report.get('unique_sources', 0)}

ESTIMATED TEXT-SIMILARITY CLUSTERS:
{report.get('independent_source_count', 0)}

LIKELY RELATED GROUPS:
"""

    duplicate_groups = report.get("duplicate_groups", [])

    if duplicate_groups:

        for number, group in enumerate(duplicate_groups, start=1):

            domains = [
                source.get("domain", "Unknown")
                for source in group
            ]

            text += f"\nGroup {number}: " + ", ".join(domains)

    else:

        text += "\nNo highly similar groups detected."

    text += "\n\nPAIRWISE SIMILARITY SCORES:\n"

    for item in report.get("source_relationships", []):

        text += (
            f"\n{item['source_1']} vs "
            f"{item['source_2']}: "
            f"{item['similarity']:.4f}"
        )

    text += """

IMPORTANT:
TF-IDF similarity is a textual heuristic.
It does not prove source independence,
copying or syndication.
"""

    return text
# ==========================================================
# HEURISTIC EVIDENCE QUALITY SCORING
# ==========================================================

def calculate_claim_evidence_quality(
    local_evidence,
    external_evidence
):

    scoring_records = []

    # Process local evidence.
    for item in local_evidence:

        url = item.get("fact_check_url", "")

        if not url or url.lower() in ("nan", "none"):
            url = item.get("evidence_url", "")

        if not url or url.lower() in ("nan", "none"):
            continue

        validated = validate_sources([
            {
                "Title": item.get("claim", ""),
                "URL": url
            }
        ])

        if not validated:
            continue

        source = validated[0]

        scoring_records.append({

            "domain": source.get("Domain", ""),

            "source_type": source.get(
                "Source_Type", "UNKNOWN"
            ),

            "extraction_status": (
                "SUCCESS" if item.get("evidence") else "FAILED"
            )
        })

    # Process external evidence.
    for item in external_evidence:

        scoring_records.append({

            "domain": item.get("domain", ""),

            "source_type": item.get(
                "source_type", "UNKNOWN"
            ),

            "extraction_status": item.get(
                "extraction_status", "FAILED"
            )
        })

    result = calculate_confidence_score(scoring_records)

    display_confidence_score(result)

    return result


# ==========================================================
# FORMAT QUALITY SCORE
# ==========================================================

def prepare_quality_text(result):

    return f"""

HEURISTIC EVIDENCE QUALITY SCORE:
{result.get('percentage', 'N/A')}%

SOURCE QUALITY:
{result.get('source_quality', 'N/A')}

EVIDENCE AGREEMENT:
{result.get('evidence_agreement', 'N/A')}

UNIQUE SOURCES:
{result.get('unique_sources', 'N/A')}

INTERPRETATION:
{result.get('interpretation', 'Not available')}

IMPORTANT:
This is a heuristic evidence-quality score.
It is not a calibrated probability that the claim
is true or false.
"""


# ==========================================================
# EXTRACT DATASET-COMPATIBLE LABEL
# ==========================================================

def extract_predicted_label(final_result):

    if not isinstance(final_result, str) or not final_result.strip():

        print("Warning: Empty LLM response. Saving abstention.")

        return "INSUFFICIENT_EVIDENCE"

    valid_labels = {
        "TRUE",
        "FALSE",
        "MOSTLY_FALSE",
        "MISLEADING",
        "MANIPULATED",
        "INSUFFICIENT_EVIDENCE"
    }

    for line in final_result.splitlines():

        if "DATASET_LABEL" not in line.upper():
            continue

        key_match = re.search(
            r"DATASET_LABEL\s*\*{0,2}\s*:",
            line,
            re.IGNORECASE
        )

        if not key_match:
            continue

        value = line[key_match.end():]

        value = re.sub(
            r"[*`_\[\]{}]",
            "",
            value
        ).strip().upper()

        value = re.sub(r"\s+", "_", value)

        value = value.strip(".,;:()")

        if value:
            value = value.split()[0].strip(".,;:()")

        if value in valid_labels:

            print("Extracted DATASET_LABEL:", value)

            return value

    print(
        "Warning: No valid explicit DATASET_LABEL found. "
        "Saving abstention."
    )

    return "INSUFFICIENT_EVIDENCE"
# ==========================================================
# MAIN VERIFICATION PIPELINE
# ==========================================================

def verify_claim(sample_id=None, claim_override=None, image_path=None):

    print("\n==========================================")
    print("   MULTIMODAL MISINFORMATION VERIFIER")
    print("==========================================")

    # STEP 1: Load claim or use custom claim.
    if claim_override and claim_override.strip():
        claim = claim_override.strip()
        print("\nCUSTOM CLAIM:")
    else:
        if not sample_id:
            raise ValueError("Provide a Sample ID or custom claim.")
        claim = load_claim(sample_id)
        print("\nORIGINAL CLAIM:")

    print(claim)

        # STEP 2: Image analysis.
    print("\nStep 1: Analysing image...")

    if image_path:
        image_analysis = analyze_image_file(
            image_path=image_path,
            claim=claim
        )

    elif sample_id:
        image_analysis = analyze_image(sample_id)

    else:
        image_analysis = "No image was provided."

    image_analysis = str(image_analysis)[:4000]

    print("Image analysis completed.")
    print(image_analysis)


    # STEP 3: Local RAG retrieval.
    print("\nStep 2: Retrieving local evidence...")

    retrieval_sample_id = (
        sample_id if sample_id else "CUSTOM_CLAIM"
    )

    local_evidence = retrieve_local_evidence(
        claim,
        retrieval_sample_id,
        top_k=3
    )

    # STEP 4: External retrieval.
    external_evidence = []

    if len(local_evidence) < 2:

        print(
            "\nInsufficient local evidence. "
            "Activating external retrieval..."
        )

        external_evidence = retrieve_external_evidence(
            claim,
            max_results=3
        )

    else:

        print(
            "\nLocal evidence candidates found. "
            "Skipping external fallback."
        )

    # STEP 5: Source similarity.
    similarity_report = compare_external_sources(
        external_evidence
    )

    similarity_text = prepare_similarity_text(
        similarity_report
    )

    # STEP 6: Evidence quality.
    print("\nCalculating evidence quality...")

    quality_result = calculate_claim_evidence_quality(
        local_evidence,
        external_evidence
    )

    quality_text = prepare_quality_text(quality_result)

    # STEP 7: Prepare evidence for LLM.
    local_text = prepare_local_evidence(local_evidence)

    external_text = prepare_external_evidence(external_evidence)

    # STEP 8: Build LLM prompt.
    prompt = f"""

You are an evidence-grounded multimodal
misinformation verification assistant.

==================================================
ORIGINAL CLAIM
==================================================

{claim}

==================================================
IMAGE ANALYSIS
==================================================

{image_analysis}

==================================================
LOCAL DATASET EVIDENCE
==================================================

{local_text}

==================================================
EXTERNAL WEB EVIDENCE
==================================================

{external_text}

==================================================
SOURCE SIMILARITY ANALYSIS
==================================================

{similarity_text}

==================================================
HEURISTIC EVIDENCE QUALITY
==================================================

{quality_text}

==================================================
DATASET-COMPATIBLE CLASSIFICATION
==================================================

Select exactly ONE primary category:

TRUE
FALSE
MOSTLY_FALSE
MISLEADING
MANIPULATED
INSUFFICIENT_EVIDENCE

CATEGORY DEFINITIONS:

TRUE:
The central claim is supported by reliable evidence.

FALSE:
The central claim is directly contradicted by reliable evidence.

MOSTLY_FALSE:
The claim contains some accurate information but its
central assertion is predominantly false.

MISLEADING:
The information uses genuine or potentially genuine
material in an incorrect context, such as old footage,
unrelated visuals or missing context.

MANIPULATED:
The primary evidence concerns media that is altered,
AI-generated, fabricated or deepfake. Do not select
this category merely because the claim is false.

INSUFFICIENT_EVIDENCE:
Available evidence cannot establish a defensible category.

==================================================
CLASSIFICATION RULES
==================================================

1. Evaluate the original claim using retrieved evidence.

2. Do not use Claim_Label or Normalized_Label.

3. Do not assume every refuted claim is MANIPULATED.

4. Do not assume every AI-generated image makes the
   associated claim false.

5. Distinguish image observations from factual conclusions.

6. Distinguish extracted article text from snippets.

7. Consider claim relevance scores.

8. Multiple articles repeating the same fact-check
   are not automatically independent investigations.

9. Source classifications and text similarity
   are metadata, not proof of factual accuracy.

10. The evidence-quality score is not a probability.

11. If evidence is insufficient, abstain.

12. Do not invent facts, dates or sources.

13. Explain why the selected category fits the
    original claim and the available evidence.

14. If a claim is false because its central assertion
    is fabricated, classify it as FALSE rather than
    automatically classifying it as MANIPULATED.

15. Use MANIPULATED when the principal finding
    concerns the authenticity or alteration of media.

==================================================
REQUIRED OUTPUT FORMAT
==================================================

DATASET_LABEL: [ONE CATEGORY]

VERDICT: [SUPPORTED / REFUTED / INSUFFICIENT EVIDENCE]

CLAIM ANALYSIS:
Explain the original claim.

IMAGE ANALYSIS:
Describe visible content and limitations.

LOCAL EVIDENCE ANALYSIS:
Discuss relevant local evidence.

EXTERNAL EVIDENCE ANALYSIS:
Discuss relevant articles, relevance scores
and extraction status.

SOURCE SIMILARITY ANALYSIS:
Explain article similarity results.

EVIDENCE QUALITY:
Report the heuristic score and explain its limitations.

FINAL EXPLANATION:
Give a step-by-step justification.

LIMITATIONS:
Explain what cannot be established.

SOURCES:
List only supplied URLs and source classifications.

"""

    # STEP 9: Groq LLM reasoning.
    print("\nStep 3: Performing final verification...")

    response = client.chat.completions.create(

        model=TEXT_MODEL,

        messages=[

            {
                "role": "system",

                "content": (
                    "You are a careful multimodal "
                    "misinformation verification assistant. "
                    "Use supplied evidence only. "
                    "Never invent facts or sources. "
                    "Return exactly one valid DATASET_LABEL. "
                    "Abstain when evidence is insufficient."
                )
            },

            {
                "role": "user",
                "content": prompt
            }

        ],

        temperature=0.2,

        max_completion_tokens=1500
    )

    final_result = (
        response.choices[0].message.content or ""
    )

    if not final_result.strip():
        raise ValueError("Groq returned an empty response.")

        # STEP 10: Extract predicted category.
    predicted_label = extract_predicted_label(final_result)

    # STEP 11: Save prediction only for dataset samples.
    if sample_id:
        save_prediction(
            sample_id,
            predicted_label,
            notes=final_result[:1000]
        )

    # STEP 12: Return result to frontend.
    return final_result


# ==========================================================
# MAIN FUNCTION
# ==========================================================

def main():

    print("\n==========================================")
    print("       MULTIMODAL VERIFICATION")
    print("==========================================")

    sample_id = input(
        "\nEnter Sample ID (example SAMPLE-099): "
    ).strip()

    try:

        result = verify_claim(sample_id)

        print("\n==========================================")
        print("              FINAL RESULT")
        print("==========================================")

        print(result)

        print("\n==========================================")
        print("          VERIFICATION COMPLETED")
        print("==========================================")

    except Exception as error:

        print("\nVERIFICATION ERROR:")
        print(error)


if __name__ == "__main__":
    main()