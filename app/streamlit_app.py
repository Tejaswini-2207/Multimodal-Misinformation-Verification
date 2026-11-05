
import sys
import uuid
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st


# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from llm_verifier import verify_claim


UPLOAD_DIR = ROOT_DIR / "outputs" / "streamlit_uploads"


# =====================================================
# 2. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Misinformation Verifier",
    page_icon="🔎",
    layout="wide"
)


# =====================================================
# 3. CUSTOM STYLING
# =====================================================

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 750;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #888888;
    font-size: 17px;
}

.result-card {
    padding: 18px;
    border-radius: 12px;
    background-color: rgba(128,128,128,0.10);
    margin-bottom: 12px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 4. HEADER
# =====================================================

st.markdown(
    '<div class="main-title">🔎 AI Misinformation Verifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Explainable Multimodal Misinformation Verification '
    'using RAG and LLMs'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =====================================================
# 5. CLAIM INPUT
# =====================================================

st.header("Verify a Social Media Claim")

st.write(
    "Enter a claim from social media. "
    "You can optionally upload the image associated with it."
)

claim = st.text_area(
    "Enter your claim",
    placeholder=(
        "Example: A viral post claims that a public figure "
        "made a particular statement..."
    ),
    height=130,
    key="claim_input"
)


# =====================================================
# 6. IMAGE UPLOAD
# =====================================================

uploaded_image = st.file_uploader(
    "Upload an associated image (optional)",
    type=["jpg", "jpeg", "png", "webp"],
    key="claim_image"
)

image_path = None

if uploaded_image:

    st.image(
        uploaded_image,
        caption="Uploaded claim image",
        width=400
    )


# =====================================================
# 7. SOURCE DISPLAY HELPER
# =====================================================

def safe_url(url):

    if not url:
        return False

    try:
        parsed = urlparse(str(url))

        return parsed.scheme in ("http", "https")

    except Exception:
        return False


def display_source(url):

    if safe_url(url):

        st.markdown(
            f"[Open original source]({url})"
        )

    else:

        st.caption("Source URL unavailable or invalid.")


# =====================================================
# 8. DISPLAY LOCAL EVIDENCE
# =====================================================

def display_local_evidence(evidence):

    st.subheader("Local Dataset Evidence")

    if not evidence:

        st.info(
            "No sufficiently relevant local evidence found."
        )

        return

    st.write(
        f"Retrieved {len(evidence)} local evidence records."
    )

    for index, item in enumerate(evidence, start=1):

        with st.expander(
            f"Local Evidence {index} — "
            f"{item.get('sample_id', 'Unknown')}"
        ):

            st.markdown("**Related Claim**")

            st.write(
                item.get("claim", "Not available")
            )

            st.markdown("**Evidence Text**")

            st.write(
                item.get("evidence", "Not available")
            )

            st.metric(
                "Retrieval Similarity",
                f"{item.get('similarity', 0):.4f}"
            )

            st.markdown("**Evidence URL**")

            display_source(
                item.get("evidence_url")
            )

            st.markdown("**Fact-check URL**")

            display_source(
                item.get("fact_check_url")
            )


# =====================================================
# 9. DISPLAY EXTERNAL EVIDENCE
# =====================================================

def display_external_evidence(evidence):

    st.subheader("External Fact-Checking Evidence")

    if not evidence:

        st.info(
            "No relevant external articles were retrieved."
        )

        return

    st.write(
        f"Retrieved {len(evidence)} external sources."
    )

    for index, item in enumerate(evidence, start=1):

        title = item.get(
            "title",
            f"External Source {index}"
        )

        with st.expander(
            f"External Evidence {index}: {title}"
        ):

            st.markdown("**Source**")

            st.write(
                item.get("domain", "Unknown")
            )

            st.markdown("**Article Title**")

            st.write(title)

            st.markdown("**Extracted Article Evidence**")

            article_text = item.get(
                "article_text",
                ""
            )

            snippet = item.get(
                "snippet",
                ""
            )

            extraction_status = item.get(
                "extraction_status",
                "UNKNOWN"
            )

            if extraction_status == "SUCCESS" and article_text:

                st.write(article_text[:4000])

            elif snippet:

                st.write(snippet)

                st.caption(
                    "Search snippet only; full article extraction "
                    "was not successful."
                )

            else:

                st.info(
                    "No article text or search snippet available."
                )

            st.markdown("**Retrieval Relevance**")

            st.write(
                item.get(
                    "claim_relevance_score",
                    "Not available"
                )
            )

            st.markdown("**Source Classification**")

            st.write(
                item.get(
                    "credibility_category",
                    "Unclassified"
                )
            )

            st.write(
                item.get(
                    "classification_basis",
                    ""
                )
            )

            st.markdown("**Extraction Status**")

            st.write(extraction_status)

            st.markdown("**Original Article**")

            display_source(
                item.get("url")
            )


# =====================================================
# 10. DISPLAY IMAGE ANALYSIS
# =====================================================

def display_image_analysis(result):

    st.subheader("Image Understanding")

    analysis = result.get("image_analysis", "")

    if analysis:

        st.write(analysis)

    else:

        st.info(
            "Image analysis was not available."
        )


# =====================================================
# 11. DISPLAY QUALITY AND SIMILARITY
# =====================================================

def display_quality(result):

    st.subheader("Evidence Quality Analysis")

    quality = result.get("quality")

    if not quality:

        st.info(
            "Evidence-quality score was not calculated."
        )

        return

    st.write(
        "Heuristic evidence-quality score:"
    )

    percentage = quality.get("percentage")

    if percentage is not None:

        st.metric(
            "Evidence Quality",
            f"{percentage}%"
        )

    st.json(quality)


def display_similarity(result):

    st.subheader("Source Similarity Analysis")

    similarity = result.get("similarity")

    if not similarity:

        st.info(
            "Source similarity information unavailable."
        )

        return

    st.json(similarity)

    st.caption(
        "Text similarity is a heuristic. It does not establish "
        "actual source independence."
    )


# =====================================================
# 12. VERIFY CLAIM BUTTON
# =====================================================

if st.button(
    "🔍 Verify Claim",
    type="primary",
    use_container_width=True,
    key="verify_claim_button"
):

    if not claim.strip():

        st.error(
            "Please enter a claim before verification."
        )

    else:

        try:

            # Save uploaded image if provided.

            image_path = None

            if uploaded_image is not None:

                UPLOAD_DIR.mkdir(
                    parents=True,
                    exist_ok=True
                )

                filename = (
                    uuid.uuid4().hex
                    + "_"
                    + Path(uploaded_image.name).name
                )

                image_path = UPLOAD_DIR / filename

                image_path.write_bytes(
                    uploaded_image.getvalue()
                )

            # Generate a unique ID for this custom query.

            custom_id = (
                "CUSTOM-"
                + uuid.uuid4().hex[:10].upper()
            )

            st.divider()

            st.header("Verification in Progress")

            with st.spinner(
                "Analysing claim, retrieving evidence "
                "and generating an explanation..."
            ):

                result = verify_claim(
                    sample_id=custom_id,
                    claim_override=claim,
                    image_path=image_path,
                    return_details=True
                )

            # =====================================
            # VERDICT
            # =====================================

            st.divider()

            st.header("Final Verification Result")

            verdict = result.get(
                "verdict",
                "INSUFFICIENT_EVIDENCE"
            )

            if verdict == "TRUE":

                st.success(f"Verdict: {verdict}")

            elif verdict == "INSUFFICIENT_EVIDENCE":

                st.info(f"Verdict: {verdict}")

            else:

                st.warning(f"Verdict: {verdict}")

            st.caption(
                "Verification mode: "
                + result.get("verification_mode", "Unknown")
            )

            # =====================================
            # RETRIEVED EVIDENCE
            # =====================================

            st.divider()

            st.header("Retrieved Evidence")

            display_local_evidence(
                result.get("local_evidence", [])
            )

            display_external_evidence(
                result.get("external_evidence", [])
            )

            # =====================================
            # IMAGE ANALYSIS
            # =====================================

            st.divider()

            display_image_analysis(result)

            # =====================================
            # EXPLAINABILITY
            # =====================================

            st.divider()

            display_quality(result)

            display_similarity(result)

            # =====================================
            # FINAL LLM EXPLANATION
            # =====================================

            st.divider()

            st.header("AI Verification Explanation")

            report = result.get("report", "")

            st.text_area(
                "Detailed reasoning",
                value=report,
                height=450,
                key="verification_report"
            )

            st.download_button(
                "Download Verification Report",
                data=report,
                file_name="verification_report.txt",
                mime="text/plain",
                key="download_verification_report"
            )

            st.success(
                "Claim verification completed."
            )

        except Exception as error:

            st.error(
                f"Verification failed: {error}"
            )

            st.exception(error)


# =====================================================
# 13. FOOTER
# =====================================================

st.divider()

st.caption(
    "Research Prototype | AI-Driven Explainable "
    "Multimodal Misinformation Verification using RAG"
)
