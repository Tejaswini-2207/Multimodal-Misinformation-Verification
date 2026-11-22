# ============================================================
# TRACEVERIFY
# AI-Driven Explainable Multimodal Misinformation Verification
# ============================================================

import streamlit as st
import json
import csv
import io
import sys
import re
import uuid
from pathlib import Path
from datetime import datetime

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
OUTPUT_DIR = BASE_DIR / "outputs"
APP_DIR = BASE_DIR / "app"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
APP_DIR.mkdir(parents=True, exist_ok=True)

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

HISTORY_FILE = OUTPUT_DIR / "traceverify_new_history.json"

# ------------------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="TraceVerify",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------

DEFAULTS = {
    "tv_page": "Home",
    "tv_theme": "Light",
    "tv_history": [],
    "tv_last_result": None,
    "tv_last_claim": "",
    "tv_last_image": None
}

for key, value in DEFAULTS.items():

    if key not in st.session_state:
        st.session_state[key] = value

# ------------------------------------------------------------
# JSON UTILITIES
# ------------------------------------------------------------

def load_json(path, default=None):

    if default is None:
        default = []

    try:

        if path.exists():

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

            return data

    except (
        json.JSONDecodeError,
        OSError,
        UnicodeDecodeError
    ):

        return default

    return default


def save_json(path, data):

    try:

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False,
                default=str
            )

        return True

    except OSError:

        return False


# ------------------------------------------------------------
# LOAD HISTORY SAFELY
# ------------------------------------------------------------

if not isinstance(
    st.session_state.tv_history,
    list
):

    st.session_state.tv_history = []

loaded_history = load_json(
    HISTORY_FILE,
    []
)

if isinstance(loaded_history, list):

    st.session_state.tv_history = loaded_history

else:

    st.session_state.tv_history = []

# ------------------------------------------------------------
# BACKEND IMPORT
# ------------------------------------------------------------

verify_function = None
backend_error = None

try:

    from src.llm_verifier import verify_claim

    verify_function = verify_claim

except Exception as first_error:

    try:

        from llm_verifier import verify_claim

        verify_function = verify_claim

    except Exception as second_error:

        backend_error = (
            "Backend import failed.\n\n"
            "Primary import error:\n"
            + str(first_error)
            + "\n\nFallback import error:\n"
            + str(second_error)
        )

# ------------------------------------------------------------
# THEME CONFIGURATION
# ------------------------------------------------------------

THEMES = {

    "Dark": {

        "background": "#080D19",
        "sidebar": "#101827",
        "card": "#141E30",
        "card_secondary": "#1C2940",
        "text": "#F1F5F9",
        "muted": "#9CAAC0",
        "border": "#293750",
        "accent": "#8B5CF6",
        "accent_secondary": "#6366F1",
        "cyan": "#22D3EE",
        "input_background": "#050505",
        "input_text": "#FFFFFF",
        "placeholder": "#A0A0A0"
    },

    "Light": {

        "background": "#FAF7FF",
        "sidebar": "#F0E9FF",
        "card": "#FFFFFF",
        "card_secondary": "#F1EAFE",
        "text": "#2E1A47",
        "muted": "#76658F",
        "border": "#DED0F7",
        "accent": "#7C3AED",
        "accent_secondary": "#A855F7",
        "cyan": "#0891B2",
        "input_background": "#FFFFFF",
        "input_text": "#2E1A47",
        "placeholder": "#8B7AA5"
    }
}

theme_name = st.session_state.tv_theme

if theme_name not in THEMES:
    theme_name = "Light"
    st.session_state.tv_theme = "Light"

theme = THEMES[theme_name]

# ------------------------------------------------------------
# CSS STYLING
# ------------------------------------------------------------

st.markdown(
    f"""
    <style>

    .stApp {{
        background-color: {theme["background"]};
        color: {theme["text"]};
    }}

    [data-testid="stSidebar"] {{
        background-color: {theme["sidebar"]};
        border-right: 1px solid {theme["border"]};
    }}

    .block-container {{
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}

    h1, h2, h3, h4 {{
        color: {theme["text"]} !important;
    }}

    p, label {{
        color: {theme["text"]};
    }}

    [data-testid="stMetric"] {{
        background-color: {theme["card"]};
        border: 1px solid {theme["border"]};
        border-radius: 14px;
        padding: 16px;
    }}

    [data-testid="stMetricLabel"] {{
        color: {theme["muted"]} !important;
    }}

    [data-testid="stMetricValue"] {{
        color: {theme["text"]} !important;
    }}

    div[data-testid="stTextArea"] textarea,
    div[data-testid="stTextInput"] input {{
        background-color: {theme["input_background"]} !important;
        color: {theme["input_text"]} !important;
        border: 1px solid {theme["border"]} !important;
        border-radius: 10px !important;
    }}

    div[data-testid="stTextArea"] textarea::placeholder,
    div[data-testid="stTextInput"] input::placeholder {{
        color: {theme["placeholder"]} !important;
    }}

    div[data-testid="stButton"] button {{
        border-radius: 10px;
        min-height: 42px;
        font-weight: 600;
    }}

    div[data-testid="stFormSubmitButton"] button {{
        border-radius: 10px;
        min-height: 44px;
        font-weight: 700;
    }}

    div[data-testid="stButton"] button[kind="primary"],
    div[data-testid="stFormSubmitButton"] button {{
        background: linear-gradient(
            135deg,
            {theme["accent"]},
            {theme["accent_secondary"]}
        ) !important;
        color: #FFFFFF !important;
        border: 0 !important;
    }}

    div[data-testid="stButton"] button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {{
        border-color: {theme["accent"]} !important;
        box-shadow: 0 4px 14px rgba(124, 58, 237, 0.18);
        transform: translateY(-1px);
    }}

    [data-testid="stProgressBar"] > div > div > div > div {{
        background-color: {theme["accent"]} !important;
    }}

    [data-testid="stExpander"] {{
        border: 1px solid {theme["border"]};
        border-radius: 12px;
    }}

    hr {{
        border-color: {theme["border"]};
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# RESULT HELPERS
# ============================================================

VALID_VERDICTS = {
    "TRUE",
    "FALSE",
    "MOSTLY_FALSE",
    "MISLEADING",
    "MANIPULATED",
    "INSUFFICIENT_EVIDENCE"
}


def normalize_verdict(value):

    if value is None:
        return "UNKNOWN"

    text = str(value).strip().upper()

    text = text.replace("-", "_")
    text = text.replace(" ", "_")

    # Exact verdict matching first
    if text in VALID_VERDICTS:
        return text

    # Support common response wording
    if "INSUFFICIENT_EVIDENCE" in text:
        return "INSUFFICIENT_EVIDENCE"

    if "MOSTLY_FALSE" in text:
        return "MOSTLY_FALSE"

    if "MISLEADING" in text:
        return "MISLEADING"

    if "MANIPULATED" in text:
        return "MANIPULATED"

    if "REFUTED" in text:
        return "FALSE"

    if "SUPPORTED" in text:
        return "TRUE"

    if re.search(r"\bFALSE\b", text):
        return "FALSE"

    if re.search(r"\bTRUE\b", text):
        return "TRUE"

    return "UNKNOWN"


def get_verdict(result):

    # --------------------------------------------------------
    # DICTIONARY RESULT
    # --------------------------------------------------------

    if isinstance(result, dict):

        keys = [
            "dataset_label",
            "verdict",
            "final_verdict",
            "prediction",
            "classification",
            "label"
        ]

        for key in keys:

            value = result.get(key)

            if value is not None:

                verdict = normalize_verdict(value)

                if verdict != "UNKNOWN":
                    return verdict

    # --------------------------------------------------------
    # TEXT RESULT
    # --------------------------------------------------------

    if isinstance(result, str):

        text = result.strip()

        # Explicit dataset label
        match = re.search(
            r"DATASET_LABEL\s*:\s*([A-Z_]+)",
            text,
            re.IGNORECASE
        )

        if match:

            verdict = normalize_verdict(
                match.group(1)
            )

            if verdict != "UNKNOWN":
                return verdict

        # Explicit verdict line
        match = re.search(
            r"VERDICT\s*:\s*([^\n]+)",
            text,
            re.IGNORECASE
        )

        if match:

            verdict = normalize_verdict(
                match.group(1)
            )

            if verdict != "UNKNOWN":
                return verdict

    return "UNKNOWN"


def get_confidence(result):

    if isinstance(result, dict):

        keys = [
            "confidence",
            "confidence_score",
            "final_confidence",
            "score"
        ]

        for key in keys:

            value = result.get(key)

            if value is not None:
                return value

    return None


# ============================================================
# HISTORY
# ============================================================

def save_history_record(record):

    if not isinstance(
        st.session_state.tv_history,
        list
    ):

        st.session_state.tv_history = []

    st.session_state.tv_history.insert(
        0,
        record
    )

    save_json(
        HISTORY_FILE,
        st.session_state.tv_history
    )


def create_history_record(
    claim,
    result,
    image_name=None,
    sample_id=None
):

    return {

        "id": str(uuid.uuid4()),

        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "claim": claim,

        "sample_id": sample_id,

        "image": image_name,

        "verdict": get_verdict(result),

        "confidence": get_confidence(result),

        "result": result
    }


# ============================================================
# CSV EXPORT
# ============================================================

def history_to_csv(records):

    output = io.StringIO()

    columns = [
        "timestamp",
        "sample_id",
        "claim",
        "verdict",
        "confidence",
        "image"
    ]

    writer = csv.DictWriter(
        output,
        fieldnames=columns
    )

    writer.writeheader()

    for record in records:

        writer.writerow({

            "timestamp": record.get(
                "timestamp",
                ""
            ),

            "sample_id": record.get(
                "sample_id",
                ""
            ),

            "claim": record.get(
                "claim",
                ""
            ),

            "verdict": record.get(
                "verdict",
                ""
            ),

            "confidence": record.get(
                "confidence",
                ""
            ),

            "image": record.get(
                "image",
                ""
            )
        })

    return output.getvalue()


# ============================================================
# BACKEND EXECUTION
# ============================================================

def run_backend(
    claim=None,
    sample_id=None,
    image_path=None
):

    if verify_function is None:

        raise RuntimeError(
            "Verification backend is unavailable.\n\n"
            + str(backend_error)
        )

    claim = (
        claim.strip()
        if isinstance(claim, str)
        else None
    )

    sample_id = (
        sample_id.strip()
        if isinstance(sample_id, str)
        else None
    )

    if not claim and not sample_id:

        raise ValueError(
            "Enter a claim or a dataset sample ID."
        )

    # --------------------------------------------------------
    # ACTUAL BACKEND SIGNATURE
    # verify_claim(sample_id=None,
    #             claim_override=None,
    #             image_path=None)
    # --------------------------------------------------------

    return verify_function(
        sample_id=sample_id or None,
        claim_override=claim or None,
        image_path=image_path
    )
# ============================================================
# PART 2 — RESULT DISPLAY, SIDEBAR & HOME PAGE
# ============================================================

# ------------------------------------------------------------
# NAVIGATION CALLBACK
# ------------------------------------------------------------

def navigate_to(page):

    st.session_state.tv_page = page
    st.session_state.tv_navigation = page


# ------------------------------------------------------------
# THEME CALLBACK
# ------------------------------------------------------------

def change_theme():

    st.session_state.tv_theme = (
        st.session_state.tv_theme_selector
    )


# ------------------------------------------------------------
# DISPLAY VERIFICATION RESULT
# ------------------------------------------------------------

def display_result(result):

    st.subheader("Verification Result")

    verdict = get_verdict(result)

    st.metric(
        label="Verdict",
        value=verdict.replace("_", " ")
    )

    # --------------------------------------------------------
    # VERDICT MESSAGE
    # --------------------------------------------------------

    if verdict == "TRUE":

        st.success(
            "The system classified this claim as TRUE."
        )

    elif verdict == "FALSE":

        st.error(
            "The system classified this claim as FALSE."
        )

    elif verdict == "MOSTLY_FALSE":

        st.error(
            "The system classified this claim as MOSTLY FALSE."
        )

    elif verdict == "MISLEADING":

        st.warning(
            "The system classified this claim as MISLEADING."
        )

    elif verdict == "MANIPULATED":

        st.warning(
            "The system detected potentially manipulated content."
        )

    elif verdict == "INSUFFICIENT_EVIDENCE":

        st.warning(
            "Insufficient evidence was available for a definitive verdict."
        )

    else:

        st.warning(
            "The system could not determine a recognized verdict."
        )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence = get_confidence(result)

    if confidence is not None:

        try:

            confidence_value = float(
                str(confidence)
                .replace("%", "")
                .strip()
            )

            if confidence_value <= 1:

                confidence_value *= 100

            if 0 <= confidence_value <= 100:

                st.metric(
                    "Confidence",
                    f"{confidence_value:.1f}%"
                )

        except (ValueError, TypeError):

            pass

    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    explanation = None

    if isinstance(result, dict):

        for key in [
            "explanation",
            "reasoning",
            "summary",
            "analysis",
            "llm_response",
            "response",
            "answer"
        ]:

            if result.get(key):

                explanation = result[key]
                break

    elif isinstance(result, str):

        explanation = result

    if explanation:

        st.markdown("### Explanation")

        if isinstance(explanation, (dict, list)):

            st.json(explanation)

        else:

            st.write(explanation)

    # --------------------------------------------------------
    # EVIDENCE
    # --------------------------------------------------------

    evidence = None

    if isinstance(result, dict):

        for key in [
            "evidence",
            "sources",
            "retrieved_evidence",
            "references",
            "retrieved_sources",
            "search_results"
        ]:

            if result.get(key):

                evidence = result[key]
                break

    if evidence:

        st.markdown("### Evidence & Sources")

        if isinstance(evidence, dict):

            evidence = [evidence]

        if isinstance(evidence, list):

            for index, item in enumerate(
                evidence,
                start=1
            ):

                with st.container(border=True):

                    if isinstance(item, dict):

                        title = (
                            item.get("title")
                            or item.get("headline")
                            or item.get("name")
                            or f"Evidence {index}"
                        )

                        url = (
                            item.get("url")
                            or item.get("link")
                            or item.get("source_url")
                            or item.get("article_url")
                        )

                        source = (
                            item.get("domain")
                            or item.get("website")
                            or item.get("source")
                        )

                        description = (
                            item.get("summary")
                            or item.get("snippet")
                            or item.get("description")
                            or item.get("content")
                            or item.get("text")
                        )

                        st.markdown(
                            f"**{index}. {title}**"
                        )

                        if source:

                            st.caption(str(source))

                        if (
                            isinstance(url, str)
                            and url.startswith(
                                ("http://", "https://")
                            )
                        ):

                            st.markdown(
                                f"[Open Source]({url})"
                            )

                        if description:

                            st.write(description)

                    else:

                        st.write(item)

        else:

            st.write(evidence)

    # --------------------------------------------------------
    # COMPLETE OUTPUT
    # --------------------------------------------------------

    if isinstance(result, dict):

        with st.expander(
            "View Complete Technical Output"
        ):

            st.json(result)

    # --------------------------------------------------------
    # DOWNLOAD RESULT
    # --------------------------------------------------------

    try:

        report_data = json.dumps(
            result,
            indent=4,
            ensure_ascii=False,
            default=str
        )

        st.download_button(
            label="Download Verification Report",
            data=report_data,
            file_name="traceverify_report.json",
            mime="application/json",
            key="download_verification_report"
        )

    except Exception as error:

        st.warning(
            f"Could not generate report: {error}"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("TraceVerify")

    st.caption(
        "MULTIMODAL MISINFORMATION VERIFICATION"
    )

    st.divider()

    navigation_pages = [
        "Home",
        "Verify",
        "History",
        "Research",
        "Team"
    ]

    # Keep the navigation widget synchronized
    # with the current application page.

    if "tv_navigation" not in st.session_state:

        st.session_state.tv_navigation = (
            st.session_state.tv_page
        )

    if (
        st.session_state.tv_navigation
        not in navigation_pages
    ):

        st.session_state.tv_navigation = "Home"

    selected_page = st.radio(
        "Navigation",
        navigation_pages,
        key="tv_navigation"
    )

    st.session_state.tv_page = selected_page

    st.divider()

    # --------------------------------------------------------
    # THEME
    # --------------------------------------------------------

    st.subheader("Appearance")

    if "tv_theme_selector" not in st.session_state:

        st.session_state.tv_theme_selector = (
            st.session_state.tv_theme
        )

    st.radio(
        "Theme",
        ["Dark", "Light"],
        horizontal=True,
        key="tv_theme_selector",
        on_change=change_theme
    )

    st.divider()

    # --------------------------------------------------------
    # BACKEND STATUS
    # --------------------------------------------------------

    st.subheader("System Status")

    if verify_function is not None:

        st.success(
            "Backend Connected"
        )

    else:

        st.error(
            "Backend Unavailable"
        )

        if backend_error:

            with st.expander(
                "Error Details"
            ):

                st.code(backend_error)

    st.divider()

    st.caption(
        "M.Tech Artificial Intelligence"
    )

    st.caption(
        "Amrita Vishwa Vidyapeetham"
    )


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.tv_page == "Home":

    st.title("TraceVerify")

    st.subheader(
        "AI-Driven Explainable Multimodal "
        "Misinformation Verification Using RAG"
    )

    st.write(
        "Verify claims. Examine evidence. Understand the reasoning."
    )

    st.divider()

    # --------------------------------------------------------
    # FEATURE CARDS
    # --------------------------------------------------------

    feature1, feature2, feature3 = st.columns(3)

    with feature1:

        with st.container(border=True):

            st.subheader("Explainable AI")

            st.write(
                "Generate verification results with "
                "reasoning and supporting context."
            )

    with feature2:

        with st.container(border=True):

            st.subheader("Evidence Retrieval")

            st.write(
                "Retrieve relevant information to help "
                "evaluate claims."
            )

    with feature3:

        with st.container(border=True):

            st.subheader("Multimodal Research")

            st.write(
                "Investigate misinformation involving "
                "textual and visual information."
            )

    st.divider()

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    st.subheader("Start Verification")

    st.write(
        "Submit a claim or select a dataset sample "
        "to begin an investigation."
    )

    st.button(
        "Start Verification",
        type="primary",
        use_container_width=True,
        on_click=navigate_to,
        args=("Verify",),
        key="home_start_verification"
    )

    st.divider()

    # --------------------------------------------------------
    # PROJECT OVERVIEW
    # --------------------------------------------------------

    st.header("Project Overview")

    st.write("""
    TraceVerify is an AI-driven research prototype for
    explainable multimodal misinformation verification.

    The project investigates misinformation presented through
    textual claims and associated visual content. It combines
    machine learning, multimodal processing, evidence retrieval,
    and language-model-based reasoning within a verification
    workflow.

    The goal is to retrieve relevant evidence, examine the
    information associated with a claim, and produce a
    transparent verification result supported by the available
    context.
    """)

    # --------------------------------------------------------
    # PROBLEM STATEMENT
    # --------------------------------------------------------

    st.header("Problem Statement")

    st.write("""
    Misinformation may be distributed through text, images,
    or combinations of both. Images can be reused in unrelated
    contexts, while textual claims may contain misleading,
    incomplete, or fabricated information.

    Text-only verification approaches may fail to identify
    inconsistencies between a claim and its associated visual
    content.

    TraceVerify investigates a multimodal verification approach
    that combines textual information, visual information,
    evidence retrieval, and explainable reasoning to support
    misinformation analysis.
    """)

    # --------------------------------------------------------
    # OBJECTIVES
    # --------------------------------------------------------

    st.header("Research Objectives")

    objectives = [
        "Analyse textual claims associated with misinformation.",
        "Investigate multimodal information involving text and images.",
        "Retrieve relevant evidence for claim verification.",
        "Use retrieval-augmented reasoning.",
        "Generate explainable verification results.",
        "Identify situations where available evidence is insufficient."
    ]

    for objective in objectives:

        st.markdown(
            f"- {objective}"
        )

    st.divider()

    # --------------------------------------------------------
    # QUICK NAVIGATION
    # --------------------------------------------------------

    st.header("Explore the Platform")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.button(
            "Verify a Claim",
            use_container_width=True,
            on_click=navigate_to,
            args=("Verify",),
            key="home_verify_button"
        )

    with col2:

        st.button(
            "View History",
            use_container_width=True,
            on_click=navigate_to,
            args=("History",),
            key="home_history_button"
        )

    with col3:

        st.button(
            "Architecture",
            use_container_width=True,
            on_click=navigate_to,
            args=("Architecture",),
            key="home_architecture_button"
        )
        # ============================================================
# PART 3 — VERIFICATION PAGE
# ============================================================

elif st.session_state.tv_page == "Verify":

    st.title("Claim Verification")

    st.write(
        "Submit a claim or dataset sample to investigate "
        "potential misinformation."
    )

    st.divider()

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    st.subheader("Verification Input")

    sample_id = st.text_input(
        "Dataset Sample ID (Optional)",
        placeholder="Enter dataset sample ID",
        help=(
            "Use this to retrieve a claim from the dataset. "
            "A custom claim takes priority if both are provided."
        ),
        key="tv_sample_id"
    )

    claim_text = st.text_area(
        "Claim to Verify",
        placeholder="Enter the claim you want to verify...",
        height=160,
        key="tv_claim_text"
    )

    st.caption(
        "Enter a custom claim, a dataset Sample ID, or both."
    )

    st.divider()

    # --------------------------------------------------------
    # IMAGE UPLOAD
    # --------------------------------------------------------

    st.subheader("Related Image (Optional)")

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "webp"],
        key="tv_uploaded_image"
    )

    if uploaded_image is not None:

        st.image(
            uploaded_image,
            caption="Uploaded image preview",
            use_container_width=True
        )

        st.info(
            "The image can be attached to the verification request. "
            "Visual analysis of newly uploaded images is not yet "
            "integrated into the current backend."
        )

    st.divider()

    # --------------------------------------------------------
    # RUN VERIFICATION
    # --------------------------------------------------------

    verify_clicked = st.button(
        "Run Verification",
        type="primary",
        use_container_width=True,
        key="tv_run_verification"
    )

    if verify_clicked:

        clean_claim = claim_text.strip()
        clean_sample_id = sample_id.strip()

        # Clear the previous result before starting a new request.
        st.session_state.tv_last_result = None

        st.session_state.tv_last_claim = ""
        st.session_state.tv_last_image = None
        st.session_state.tv_last_sample_id = None

        # ----------------------------------------------------
        # INPUT VALIDATION
        # ----------------------------------------------------

        if not clean_claim and not clean_sample_id:

            st.warning(
                "Enter a claim or a Dataset Sample ID "
                "before running verification."
            )

        elif verify_function is None:

            st.error(
                "The verification backend is unavailable."
            )

            if backend_error:

                with st.expander("Backend Error Details"):

                    st.code(backend_error)

        else:

            # ------------------------------------------------
            # IMAGE STORAGE
            # ------------------------------------------------

            image_path = None
            image_name = None

            try:

                if uploaded_image is not None:

                    upload_dir = (
                        OUTPUT_DIR / "traceverify_uploads"
                    )

                    upload_dir.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    image_name = uploaded_image.name

                    allowed_extensions = {
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".webp"
                    }

                    extension = Path(
                        uploaded_image.name
                    ).suffix.lower()

                    if extension not in allowed_extensions:

                        raise ValueError(
                            "Unsupported image file format."
                        )

                    unique_filename = (
                        f"{uuid.uuid4().hex}{extension}"
                    )

                    saved_image_path = (
                        upload_dir / unique_filename
                    )

                    with open(
                        saved_image_path,
                        "wb"
                    ) as image_file:

                        image_file.write(
                            uploaded_image.getbuffer()
                        )

                    image_path = str(
                        saved_image_path.resolve()
                    )

                # ------------------------------------------------
                # BACKEND EXECUTION
                # ------------------------------------------------

                with st.spinner(
                    "Verifying claim and processing evidence..."
                ):

                    result = run_backend(
                        claim=clean_claim or None,
                        sample_id=clean_sample_id or None,
                        image_path=image_path
                    )

                # ------------------------------------------------
                # RESULT VALIDATION
                # ------------------------------------------------

                if result is None:

                    st.error(
                        "The backend returned no verification result."
                    )

                else:

                    st.session_state.tv_last_result = result

                    st.session_state.tv_last_claim = (
                        clean_claim
                        if clean_claim
                        else clean_sample_id
                    )

                    st.session_state.tv_last_image = image_name

                    st.session_state.tv_last_sample_id = (
                        clean_sample_id or None
                    )

                    st.success(
                        "Verification process completed."
                    )

                    # ------------------------------------------------
                    # SAVE HISTORY
                    # ------------------------------------------------

                    try:

                        history_record = create_history_record(
                            claim=(
                                clean_claim
                                if clean_claim
                                else clean_sample_id
                            ),
                            result=result,
                            image_name=image_name,
                            sample_id=clean_sample_id or None
                        )

                        save_history_record(
                            history_record
                        )

                    except Exception as history_error:

                        st.warning(
                            "The verification completed, but "
                            "the result could not be saved "
                            "to history."
                        )

                        with st.expander(
                            "History Save Error"
                        ):

                            st.exception(
                                history_error
                            )

                    # ------------------------------------------------
                    # DISPLAY RESULT
                    # ------------------------------------------------

                    st.divider()

                    display_result(result)

            except Exception as error:

                st.session_state.tv_last_result = None

                st.error(
                    "Verification could not be completed."
                )

                with st.expander(
                    "Error Details"
                ):

                    st.exception(error)

    # --------------------------------------------------------
    # DISPLAY PREVIOUS RESULT
    # --------------------------------------------------------

    elif st.session_state.get(
        "tv_last_result"
    ) is not None:

        st.divider()

        st.subheader(
            "Previous Verification"
        )

        previous_claim = (
            st.session_state.get(
                "tv_last_claim",
                ""
            )
        )

        if previous_claim:

            st.markdown("**Claim:**")

            st.write(previous_claim)

        display_result(
            st.session_state.tv_last_result
        )
        # ============================================================
# PART 4 — HISTORY PAGE
# ============================================================

elif st.session_state.tv_page == "History":

    st.title("Verification History")

    st.write(
        "Review previous investigations, search saved claims, "
        "filter results, and export verification records."
    )

    st.divider()

    # --------------------------------------------------------
    # LOAD & VALIDATE HISTORY
    # --------------------------------------------------------

    history = st.session_state.get(
        "tv_history",
        []
    )

    if not isinstance(history, list):

        history = []

        st.session_state.tv_history = []

    valid_history = [
        item
        for item in history
        if isinstance(item, dict)
    ]

    # --------------------------------------------------------
    # EMPTY HISTORY
    # --------------------------------------------------------

    if not valid_history:

        st.info(
            "No verification history is available yet."
        )

        st.write(
            "Your completed verifications will appear here."
        )

        st.button(
            "Go to Verification",
            type="primary",
            use_container_width=True,
            on_click=navigate_to,
            args=("Verify",),
            key="tv_history_go_verify"
        )

    else:

        # ----------------------------------------------------
        # NORMALIZE HISTORY RECORDS FOR DISPLAY
        # ----------------------------------------------------

        normalized_records = []

        for record in valid_history:

            stored_result = record.get(
                "result",
                {}
            )

            if stored_result is None:

                stored_result = {}

            # Get verdict from the stored record first.
            verdict = record.get(
                "verdict"
            )

            if not verdict:

                verdict = get_verdict(
                    stored_result
                )

            verdict = str(
                verdict
            ).strip().upper().replace(
                " ",
                "_"
            )

            # Get confidence from stored record first.
            confidence = record.get(
                "confidence"
            )

            if confidence is None:

                confidence = get_confidence(
                    stored_result
                )

            normalized_records.append(
                {
                    "original": record,
                    "claim": record.get(
                        "claim",
                        "Claim unavailable"
                    ),
                    "sample_id": record.get(
                        "sample_id"
                    ),
                    "timestamp": record.get(
                        "timestamp",
                        "Date unavailable"
                    ),
                    "verdict": verdict,
                    "confidence": confidence,
                    "image": record.get(
                        "image"
                    ),
                    "result": stored_result
                }
            )

        # ----------------------------------------------------
        # SUMMARY METRICS
        # ----------------------------------------------------

        total_count = len(
            normalized_records
        )

        true_count = sum(
            1
            for item in normalized_records
            if item["verdict"] == "TRUE"
        )

        false_count = sum(
            1
            for item in normalized_records
            if item["verdict"] in [
                "FALSE",
                "MOSTLY_FALSE",
                "MISLEADING",
                "MANIPULATED"
            ]
        )

        uncertain_count = sum(
            1
            for item in normalized_records
            if item["verdict"] in [
                "UNKNOWN",
                "INSUFFICIENT_EVIDENCE"
            ]
        )

        metric1, metric2, metric3, metric4 = st.columns(4)

        with metric1:

            st.metric(
                "Total",
                total_count
            )

        with metric2:

            st.metric(
                "True",
                true_count
            )

        with metric3:

            st.metric(
                "False / Misleading",
                false_count
            )

        with metric4:

            st.metric(
                "Uncertain",
                uncertain_count
            )

        st.divider()

        # ----------------------------------------------------
        # SEARCH & FILTER
        # ----------------------------------------------------

        st.subheader("Search & Filter")

        search_col, filter_col = st.columns(
            [2, 1]
        )

        with search_col:

            search_query = st.text_input(
                "Search claim or Sample ID",
                placeholder="Type to search...",
                key="tv_history_search"
            )

        with filter_col:

            verdict_options = [
                "All",
                "TRUE",
                "FALSE",
                "MOSTLY_FALSE",
                "MISLEADING",
                "MANIPULATED",
                "INSUFFICIENT_EVIDENCE",
                "UNKNOWN"
            ]

            selected_verdict = st.selectbox(
                "Verdict",
                verdict_options,
                key="tv_history_filter"
            )

        # ----------------------------------------------------
        # APPLY FILTERS
        # ----------------------------------------------------

        filtered_records = []

        query = search_query.strip().lower()

        for item in normalized_records:

            searchable_text = (
                str(item["claim"])
                + " "
                + str(item["sample_id"] or "")
            ).lower()

            if query and query not in searchable_text:

                continue

            if selected_verdict != "All":

                if item["verdict"] != selected_verdict:

                    continue

            filtered_records.append(
                item
            )

        st.caption(
            f"{len(filtered_records)} matching record(s) "
            f"out of {total_count}."
        )

        # ----------------------------------------------------
        # EXPORT FILTERED HISTORY
        # ----------------------------------------------------

        st.download_button(
            label="Download Filtered History as CSV",
            data=history_to_csv(
                [
                    item["original"]
                    for item in filtered_records
                ]
            ),
            file_name="traceverify_history.csv",
            mime="text/csv",
            use_container_width=True,
            key="tv_history_download"
        )

        # ----------------------------------------------------
        # NO SEARCH RESULTS
        # ----------------------------------------------------

        if not filtered_records:

            st.info(
                "No records match your current search or filter."
            )

        # ----------------------------------------------------
        # DISPLAY HISTORY
        # ----------------------------------------------------

        else:

            st.subheader("Saved Investigations")

            for index, item in enumerate(
                filtered_records,
                start=1
            ):

                verdict_display = item[
                    "verdict"
                ].replace(
                    "_",
                    " "
                )

                with st.container(border=True):

                    st.markdown(
                        f"### {index}. {verdict_display}"
                    )

                    st.caption(
                        f"Date: {item['timestamp']}"
                    )

                    st.markdown("**Claim**")

                    st.write(
                        item["claim"]
                    )

                    detail_col1, detail_col2 = st.columns(2)

                    with detail_col1:

                        if item["sample_id"]:

                            st.caption(
                                f"Sample ID: {item['sample_id']}"
                            )

                    with detail_col2:

                        if item["confidence"] is not None:

                            st.caption(
                                f"Confidence: {item['confidence']}"
                            )

                    if item["image"]:

                        st.caption(
                            f"Image: {item['image']}"
                        )

                    # ------------------------------------------------
                    # DETAILS
                    # ------------------------------------------------

                    with st.expander(
                        "View Full Verification Details"
                    ):

                        if isinstance(
                            item["result"],
                            dict
                        ):

                            st.json(
                                item["result"]
                            )

                        elif item["result"]:

                            st.write(
                                item["result"]
                            )

                        else:

                            st.info(
                                "No additional result details were saved."
                            )

        st.divider()

        # ----------------------------------------------------
        # CLEAR HISTORY
        # ----------------------------------------------------

        st.subheader("History Management")

        st.warning(
            "This action permanently clears all saved verification "
            "history from the application."
        )

        confirm_clear = st.checkbox(
            "Confirm that I want to clear all history.",
            key="tv_confirm_clear_history"
        )

        if st.button(
            "Clear All History",
            disabled=not confirm_clear,
            key="tv_clear_history_button"
        ):

            st.session_state.tv_history = []

            try:

                save_json(
                    HISTORY_FILE,
                    []
                )

                st.session_state.tv_confirm_clear_history = False

                st.success(
                    "History cleared successfully."
                )

                st.rerun()

            except Exception as error:

                st.error(
                    "Failed to clear history."
                )

                st.exception(
                    error
                )
                # ============================================================
# PART 5 — RESEARCH & METHODOLOGY PAGE
# ============================================================

elif st.session_state.tv_page == "Research":

    import pandas as pd

    st.title("Research & Methodology")

    st.write(
        "Complete research overview of TraceVerify, including "
        "motivation, dataset, methodology, algorithms, mathematical "
        "formulation, implementation, evaluation, and limitations."
    )

    st.divider()
         # ========================================================
    # 1. PROJECT OVERVIEW
    # ========================================================

    st.header("1. Project Overview")

    st.markdown("""
    **TraceVerify** is an AI-driven misinformation verification
    platform designed to analyse claims and provide evidence-based
    verification results.

    The system integrates information retrieval techniques with
    Large Language Model reasoning to retrieve relevant evidence,
    analyse claims, and generate explainable outputs.

    Unlike systems that provide only a prediction, TraceVerify
    focuses on presenting retrieved evidence alongside the
    verification result to improve transparency.
    """)

    st.markdown("---")

    # PROJECT HIGHLIGHTS

    st.subheader("Project Highlights")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="AI-Powered",
            value="Verification",
            help="AI-assisted claim verification"
        )

    with col2:
        st.metric(
            label="Retrieval",
            value="RAG",
            help="Retrieval-Augmented Generation"
        )

    with col3:
        st.metric(
            label="Explainability",
            value="Evidence-Based",
            help="Results supported by retrieved evidence"
        )

    with col4:
        st.metric(
            label="Approach",
            value="Multimodal",
            help="Designed for misinformation analysis"
        )

    st.markdown("---")

    # PROJECT OBJECTIVE VISUAL

    st.subheader("Core Objective")

    st.markdown("""
    <div style="
        padding:25px;
        border-radius:15px;
        background:linear-gradient(135deg, #6d28d9, #9333ea);
        color:white;
        text-align:center;
        margin:10px 0;
    ">
        <h3 style="color:white;">
            From Claims to Evidence-Based Verification
        </h3>
        <p style="font-size:16px;">
            Retrieve relevant information → Analyse evidence →
            Generate explainable verification
        </p>
    </div>
    """, unsafe_allow_html=True)

    # PROJECT COMPONENTS

    st.subheader("Key Components")

    overview_col1, overview_col2 = st.columns(2)

    with overview_col1:

        st.markdown("""
        #### 🔎 Evidence Retrieval

        Retrieves relevant information from available
        evidence sources using text similarity.
        """)

        st.markdown("""
        #### 📊 Evidence Ranking

        Ranks retrieved evidence based on its relevance
        to the submitted claim.
        """)

    with overview_col2:

        st.markdown("""
        #### 🤖 AI Reasoning

        Analyses the relationship between the claim
        and retrieved evidence using an LLM.
        """)

        st.markdown("""
        #### 📋 Explainable Results

        Presents verification results together with
        supporting evidence and explanations.
        """)
            # ========================================================
    # 2. DATASET
    # ========================================================

    st.header("2. Dataset")

    st.markdown("""
    TraceVerify uses a misinformation-related dataset containing
    claims labelled according to their verification category.

    The dataset is used to support claim analysis, verification,
    and evaluation of the system.
    """)

    # DATASET SUMMARY CARDS

    st.subheader("Dataset Statistics")

    dcol1, dcol2, dcol3, dcol4 = st.columns(4)

    with dcol1:
        st.metric("Total Samples", "100")

    with dcol2:
        st.metric("False / Misleading", "93")

    with dcol3:
        st.metric("True", "5")

    with dcol4:
        st.metric("Mixed / Qualified", "2")

    st.markdown("---")

    # DATA DISTRIBUTION

    st.subheader("Dataset Class Distribution")

    dataset_counts = pd.DataFrame({
        "Label": [
            "False / Misleading",
            "True",
            "Mixed / Qualified"
        ],
        "Count": [93, 5, 2]
    })

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        st.markdown("#### Class Distribution")

        st.bar_chart(
            dataset_counts.set_index("Label"),
            y="Count"
        )

    with chart_col2:

        st.markdown("#### Dataset Composition")

        try:
            import plotly.express as px

            fig_dataset = px.pie(
                dataset_counts,
                names="Label",
                values="Count",
                hole=0.5,
                title="Dataset Label Distribution"
            )

            fig_dataset.update_layout(
                height=400,
                legend_title="Label"
            )

            st.plotly_chart(
                fig_dataset,
                use_container_width=True
            )

        except ImportError:

            st.bar_chart(
                dataset_counts.set_index("Label")
            )

    # CLASS DISTRIBUTION PERCENTAGES

    st.subheader("Class Proportion")

    st.markdown("**False / Misleading — 93%**")
    st.progress(0.93)

    st.markdown("**True — 5%**")
    st.progress(0.05)

    st.markdown("**Mixed / Qualified — 2%**")
    st.progress(0.02)

    # DATASET OBSERVATION

    st.warning("""
    **Dataset Observation**

    The dataset exhibits significant class imbalance.
    False/Misleading samples constitute the majority of the
    dataset, while True and Mixed/Qualified samples are
    comparatively underrepresented.
    """)

    # DATASET TABLE

    st.subheader("Dataset Summary")

    st.dataframe(
        dataset_counts,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Dataset statistics are based on the 100 samples "
        "currently considered in the project."
    )

    st.markdown("---")
        # ========================================================
    # 3. PROBLEM STATEMENT
    # ========================================================

    st.header("3. Problem Statement")

    st.markdown("""
    The rapid dissemination of misinformation across digital
    platforms poses significant challenges to information
    credibility, public awareness, and decision-making.

    Traditional fact-checking approaches often require
    substantial manual effort, making large-scale and
    real-time verification difficult. Existing automated
    systems may also produce predictions without providing
    sufficient supporting evidence or transparent explanations.

    TraceVerify addresses these challenges by integrating
    information retrieval, evidence ranking, and Large
    Language Model reasoning to analyse claims and generate
    evidence-supported verification results.
    """)

    st.markdown("---")

    st.subheader("Key Challenges")

    challenge_col1, challenge_col2 = st.columns(2)

    with challenge_col1:

        st.markdown("""
        #### 📰 Information Overload

        The large volume of online information makes
        manual verification time-consuming and difficult.
        """)

        st.markdown("""
        #### ⚠️ Misleading Content

        False, misleading, and contextually inaccurate
        claims can spread rapidly across digital platforms.
        """)

    with challenge_col2:

        st.markdown("""
        #### 🔍 Lack of Evidence

        Predictions without supporting evidence reduce
        transparency and user trust.
        """)

        st.markdown("""
        #### 🤖 Explainability

        Automated systems should provide understandable
        explanations for their verification decisions.
        """)

    st.subheader("Research Problem")

    st.markdown("""
    <div style="
        padding:22px;
        border-radius:12px;
        border:1px solid rgba(139,92,246,0.5);
        background:rgba(139,92,246,0.08);
        text-align:center;
    ">
        <h4>
            How can misinformation verification be automated
            while ensuring evidence retrieval, contextual
            reasoning, and explainable results?
        </h4>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
        # ========================================================
    # 4. RESEARCH OBJECTIVES
    # ========================================================

    st.header("4. Research Objectives")

    st.markdown("""
    The primary objective of TraceVerify is to develop an
    AI-driven misinformation verification framework that
    combines evidence retrieval with intelligent reasoning
    to provide transparent and explainable verification results.
    """)

    st.subheader("Objectives")

    objectives = [
        {
            "title": "Automated Claim Verification",
            "description": "Develop a system capable of analysing claims and generating automated verification results."
        },
        {
            "title": "Evidence Retrieval",
            "description": "Retrieve relevant information from available evidence sources to support claim analysis."
        },
        {
            "title": "Similarity-Based Ranking",
            "description": "Rank retrieved evidence using textual similarity techniques."
        },
        {
            "title": "RAG Integration",
            "description": "Integrate retrieval with Large Language Model reasoning to provide contextual information."
        },
        {
            "title": "Explainable Verification",
            "description": "Generate explanations and present supporting evidence alongside the predicted verdict."
        },
        {
            "title": "Interactive Platform",
            "description": "Develop an interactive interface for claim submission, verification, and result analysis."
        }
    ]

    for i in range(0, len(objectives), 2):

        cols = st.columns(2)

        for j, col in enumerate(cols):

            idx = i + j

            if idx < len(objectives):

                with col:

                    obj = objectives[idx]

                    st.markdown(
                        f"""
                        <div style="
                            padding:20px;
                            border-radius:12px;
                            border:1px solid rgba(139,92,246,0.35);
                            margin-bottom:12px;
                            min-height:150px;
                        ">
                            <h4>{idx + 1}. {obj['title']}</h4>
                            <p>{obj['description']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    st.subheader("Research Goal")

    st.markdown("""
    <div style="
        padding:22px;
        border-radius:14px;
        background:linear-gradient(135deg, #6d28d9, #9333ea);
        color:white;
        text-align:center;
    ">
        <h3 style="color:white;">
            Evidence-Grounded and Explainable Misinformation Verification
        </h3>
        <p>
            Combining information retrieval and AI reasoning
            to support transparent claim analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
        # ========================================================
    # 5. SYSTEM METHODOLOGY
    # ========================================================

    st.header("5. System Methodology")

    st.markdown("""
    TraceVerify follows a structured methodology that combines
    claim processing, evidence retrieval, similarity-based
    ranking, and Large Language Model reasoning to generate
    explainable misinformation verification results.
    """)

    # METHODOLOGY OVERVIEW

    methodology_steps = [
        ("01", "Input Processing",
         "The system receives a user-submitted claim or a selected dataset sample for verification."),

        ("02", "Claim Representation",
         "The claim is processed and represented in a suitable textual format for retrieval."),

        ("03", "Evidence Retrieval",
         "Relevant evidence is retrieved from the available dataset or knowledge sources."),

        ("04", "Similarity Calculation",
         "Textual similarity between the claim and retrieved evidence is calculated."),

        ("05", "Evidence Ranking",
         "Evidence is ranked according to its similarity and relevance to the claim."),

        ("06", "Top-K Selection",
         "The highest-ranked evidence items are selected as contextual information."),

        ("07", "LLM-Based Analysis",
         "The selected evidence is provided to the Large Language Model for contextual reasoning."),

        ("08", "Verdict Generation",
         "The system generates a verification verdict and an explanation based on the analysis.")
    ]

    st.subheader("Methodological Stages")

    for number, title, description in methodology_steps:

        with st.container(border=True):

            col1, col2 = st.columns([1, 5])

            with col1:
                st.markdown(
                    f"""
                    <div style="
                        background:rgba(139,92,246,0.15);
                        padding:12px;
                        border-radius:10px;
                        text-align:center;
                        font-size:22px;
                        font-weight:bold;
                        color:#a78bfa;
                    ">
                        {number}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(f"### {title}")
                st.write(description)

    # METHODOLOGY FLOW

    st.subheader("Methodology Workflow")

    st.code("""
    Claim Input
        |
        v
    Input Processing
        |
        v
    Claim Representation
        |
        v
    Evidence Retrieval
        |
        v
    Similarity Calculation
        |
        v
    Evidence Ranking
        |
        v
    Top-K Selection
        |
        v
    LLM-Based Analysis
        |
        v
    Verdict Generation
        |
        v
    Explainable Output
    """, language="text")

    # METHODOLOGY SUMMARY

    st.subheader("Methodology Summary")

    method_col1, method_col2, method_col3 = st.columns(3)

    with method_col1:
        st.metric(
            "Retrieval",
            "Evidence-Based",
            help="Retrieval of relevant supporting information"
        )

    with method_col2:
        st.metric(
            "Ranking",
            "Similarity-Based",
            help="Ranking evidence based on relevance"
        )

    with method_col3:
        st.metric(
            "Reasoning",
            "LLM-Based",
            help="Contextual reasoning using retrieved evidence"
        )

    st.markdown("---")
        # ========================================================
    # 6. SYSTEM PIPELINE
    # ========================================================

    st.header("6. System Pipeline")

    st.markdown("""
    The TraceVerify system pipeline illustrates the sequence
    of processing stages through which a claim is analysed,
    relevant evidence is retrieved, and a verification result
    is generated.
    """)

    # PIPELINE VISUALIZATION

    pipeline_stages = [
        ("📝", "Claim Input", "User submits a claim or selects a dataset sample."),
        ("⚙️", "Input Processing", "The input is prepared for analysis."),
        ("🔎", "Evidence Retrieval", "Relevant evidence is retrieved."),
        ("📊", "TF-IDF & Similarity", "Text is represented and similarity is calculated."),
        ("📑", "Evidence Ranking", "Evidence is sorted according to relevance."),
        ("🎯", "Top-K Selection", "Relevant evidence is selected."),
        ("🤖", "LLM Reasoning", "The claim is analysed using retrieved context."),
        ("✅", "Verdict Generation", "The system generates a verification result."),
        ("📋", "Explainable Output", "The verdict, evidence, and explanation are presented.")
    ]

    for i, (emoji, title, description) in enumerate(pipeline_stages):

        st.markdown(
            f"""
            <div style="
                border:1px solid rgba(139,92,246,0.4);
                border-radius:14px;
                padding:18px;
                margin:5px 0;
                background:rgba(139,92,246,0.05);
            ">
                <div style="display:flex;align-items:center;gap:15px;">
                    <div style="font-size:30px;">{emoji}</div>
                    <div>
                        <h4 style="margin:0;">{i + 1}. {title}</h4>
                        <p style="margin:5px 0 0 0;color:gray;">
                            {description}
                        </p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if i < len(pipeline_stages) - 1:

            st.markdown(
                """
                <div style="
                    text-align:center;
                    font-size:22px;
                    color:#a78bfa;
                    margin:0;
                ">
                    ↓
                </div>
                """,
                unsafe_allow_html=True
            )

    # PIPELINE SUMMARY

    st.subheader("Pipeline Summary")

    pcol1, pcol2, pcol3 = st.columns(3)

    with pcol1:
        st.metric("Input", "Claim")

    with pcol2:
        st.metric("Processing", "Retrieval + Reasoning")

    with pcol3:
        st.metric("Output", "Verdict + Evidence")

    st.markdown("---")
        # ========================================================
    # 7. RETRIEVAL-AUGMENTED GENERATION (RAG)
    # ========================================================

    st.header("7. Retrieval-Augmented Generation (RAG)")

    st.markdown("""
    Retrieval-Augmented Generation (RAG) combines information
    retrieval with Large Language Model generation.

    Instead of relying entirely on the model's internal
    knowledge, RAG retrieves relevant information from an
    external knowledge source and provides it as context
    during generation.

    In TraceVerify, retrieved evidence is used to support
    the analysis of submitted claims and the generation
    of verification explanations.
    """)

    # RAG CONCEPT

    st.subheader("How RAG Works")

    rag_stages = [
        ("1", "User Claim", "The claim is submitted for verification."),
        ("2", "Retrieval", "Relevant evidence is retrieved from available sources."),
        ("3", "Relevance Ranking", "Retrieved evidence is ranked based on similarity."),
        ("4", "Context Construction", "Selected evidence is prepared as context."),
        ("5", "LLM Generation", "The language model analyses the claim using the retrieved context."),
        ("6", "Response", "The system generates a verdict and explanation.")
    ]

    for number, title, description in rag_stages:

        with st.container(border=True):

            col1, col2 = st.columns([1, 5])

            with col1:
                st.markdown(
                    f"""
                    <div style="
                        padding:12px;
                        border-radius:50%;
                        background:rgba(139,92,246,0.15);
                        text-align:center;
                        font-size:20px;
                        font-weight:bold;
                        color:#a78bfa;
                    ">
                        {number}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(f"### {title}")
                st.write(description)

    # RAG FLOW DIAGRAM

    st.subheader("RAG Workflow")

    st.code("""
    User Claim
        |
        v
    Retrieval Query
        |
        v
    Evidence Search
        |
        v
    Similarity Ranking
        |
        v
    Relevant Evidence Selection
        |
        v
    Context Preparation
        |
        v
    Large Language Model
        |
        v
    Evidence-Grounded Response
    """, language="text")

    # RAG COMPONENTS

    st.subheader("Core RAG Components")

    rag_col1, rag_col2 = st.columns(2)

    with rag_col1:

        st.markdown("""
        #### 🔎 Retriever

        Searches the available evidence sources and
        identifies information relevant to the claim.
        """)

        st.markdown("""
        #### 📚 Knowledge Source

        Provides the information used during retrieval
        and verification.
        """)

    with rag_col2:

        st.markdown("""
        #### 📊 Ranking Mechanism

        Evaluates retrieved evidence based on its
        relevance to the submitted claim.
        """)

        st.markdown("""
        #### 🤖 Generator

        Uses the retrieved contextual information
        to generate a verification response.
        """)

    # RAG BENEFITS

    st.subheader("Advantages of RAG")

    benefits = [
        "Grounds model responses in retrieved information.",
        "Provides contextual evidence for claim analysis.",
        "Reduces dependence on internal model knowledge alone.",
        "Supports evidence-based explanations.",
        "Improves transparency of the verification process."
    ]

    for benefit in benefits:
        st.markdown(f"- {benefit}")

    st.markdown("---")
        # ========================================================
    # 8. ALGORITHMS USED
    # ========================================================

    st.header("8. Algorithms Used")

    st.markdown("""
    TraceVerify uses information retrieval and similarity-based
    algorithms to identify relevant evidence for claim verification.
    """)

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    st.subheader("8.1 TF-IDF (Term Frequency–Inverse Document Frequency)")

    st.markdown("""
    TF-IDF is a text representation technique used to convert
    textual information into numerical vectors.

    It assigns importance to a term based on its frequency
    within a document and its rarity across the document collection.
    """)

    st.latex(r"TF(t,d)= \frac{f_{t,d}}{\sum_{t'} f_{t',d}}")

    st.latex(r"IDF(t)= \log\left(\frac{N}{DF(t)}\right)")

    st.latex(r"TFIDF(t,d)=TF(t,d)\times IDF(t)")

    tfidf_col1, tfidf_col2 = st.columns(2)

    with tfidf_col1:

        st.markdown("""
        **Term Frequency (TF)**

        Measures how frequently a term occurs
        within a particular document.
        """)

    with tfidf_col2:

        st.markdown("""
        **Inverse Document Frequency (IDF)**

        Reduces the importance of terms that
        occur frequently across documents.
        """)

    # --------------------------------------------------------
    # COSINE SIMILARITY
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("8.2 Cosine Similarity")

    st.markdown("""
    Cosine similarity measures the similarity between
    two vector representations by calculating the cosine
    of the angle between them.

    It is used to measure the relevance of retrieved
    evidence to the submitted claim.
    """)

    st.latex(
        r"Cosine(A,B)=\frac{A\cdot B}{\|A\|\|B\|}"
    )

    similarity_values = pd.DataFrame({
        "Similarity": [0.95, 0.75, 0.50, 0.25, 0.10],
        "Interpretation": [
            "Very High",
            "High",
            "Moderate",
            "Low",
            "Very Low"
        ]
    })

    st.markdown("#### Illustrative Similarity Scores")

    st.bar_chart(
        similarity_values.set_index("Interpretation"),
        y="Similarity"
    )

    st.caption(
        "Illustrative values for explaining cosine similarity; "
        "not actual experimental results."
    )

    # --------------------------------------------------------
    # EVIDENCE RANKING
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("8.3 Evidence Ranking")

    st.markdown("""
    Retrieved evidence is sorted based on its similarity
    score with the submitted claim.

    Higher similarity scores indicate greater textual
    relevance to the claim.
    """)

    st.latex(
        r"S(C,E_i)=\frac{C\cdot E_i}{\|C\|\|E_i\|}"
    )

    ranking_demo = pd.DataFrame({
        "Evidence": [
            "Evidence A",
            "Evidence B",
            "Evidence C",
            "Evidence D",
            "Evidence E"
        ],
        "Similarity Score": [0.92, 0.81, 0.67, 0.43, 0.21]
    })

    st.markdown("#### Illustrative Evidence Ranking")

    st.dataframe(
        ranking_demo,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        ranking_demo.set_index("Evidence"),
        y="Similarity Score"
    )

    st.caption(
        "Illustrative ranking scores; not actual retrieval results."
    )

    # --------------------------------------------------------
    # TOP-K SELECTION
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("8.4 Top-K Evidence Selection")

    st.markdown("""
    Top-K selection retains the highest-ranked evidence items
    for further analysis.

    This reduces the amount of irrelevant information supplied
    to the language model and focuses reasoning on the most
    relevant retrieved context.
    """)

    st.latex(
        r"E_{selected}=\operatorname{TopK}(E_{ranked})"
    )

    topk_col1, topk_col2 = st.columns(2)

    with topk_col1:

        st.markdown("""
        **Before Selection**

        All retrieved evidence items are considered
        during the ranking process.
        """)

    with topk_col2:

        st.markdown("""
        **After Selection**

        The highest-ranked evidence items are selected
        as contextual input.
        """)

    st.markdown("---")
        # ========================================================
    # 9. LLM-BASED REASONING
    # ========================================================

    st.header("9. LLM-Based Reasoning")

    st.markdown("""
    TraceVerify integrates a Large Language Model (LLM) to
    analyse claims using retrieved evidence and contextual
    information.

    The retrieved evidence is provided to the model as context
    to support its reasoning and verification process.
    """)

    # REASONING PROCESS

    st.subheader("Reasoning Process")

    reasoning_steps = [
        ("Claim Understanding",
         "The LLM interprets the submitted claim and identifies its main assertion."),

        ("Evidence Context",
         "Retrieved evidence is provided as contextual information."),

        ("Evidence Analysis",
         "The model analyses the relationship between the claim and retrieved evidence."),

        ("Consistency Analysis",
         "The model examines whether the evidence supports, contradicts, or provides context to the claim."),

        ("Verdict Generation",
         "The model generates a verification result based on the claim and available evidence."),

        ("Explanation Generation",
         "The model produces a natural-language explanation describing the verification result.")
    ]

    for i, (title, description) in enumerate(reasoning_steps, 1):

        with st.container(border=True):

            st.markdown(f"### {i}. {title}")
            st.write(description)

    # REASONING PIPELINE

    st.subheader("LLM Reasoning Pipeline")

    st.code("""
    Submitted Claim
          |
          v
    Retrieved Evidence
          |
          v
    Context Construction
          |
          v
    LLM Input
          |
          v
    Claim-Evidence Analysis
          |
          v
    Verdict Generation
          |
          v
    Explanation Generation
          |
          v
    Final Verification Response
    """, language="text")

    # CLAIM-EVIDENCE RELATIONSHIP

    st.subheader("Claim–Evidence Relationship")

    relation_col1, relation_col2, relation_col3 = st.columns(3)

    with relation_col1:

        st.markdown("""
        <div style="
            padding:20px;
            border-radius:12px;
            border:1px solid #22c55e;
            text-align:center;
        ">
            <h3>Supporting</h3>
            <p>Evidence supports the claim.</p>
        </div>
        """, unsafe_allow_html=True)

    with relation_col2:

        st.markdown("""
        <div style="
            padding:20px;
            border-radius:12px;
            border:1px solid #ef4444;
            text-align:center;
        ">
            <h3>Contradicting</h3>
            <p>Evidence contradicts the claim.</p>
        </div>
        """, unsafe_allow_html=True)

    with relation_col3:

        st.markdown("""
        <div style="
            padding:20px;
            border-radius:12px;
            border:1px solid #eab308;
            text-align:center;
        ">
            <h3>Contextual</h3>
            <p>Evidence provides additional context.</p>
        </div>
        """, unsafe_allow_html=True)

    # ROLE OF LLM

    st.subheader("Role of the Large Language Model")

    llm_col1, llm_col2 = st.columns(2)

    with llm_col1:

        st.markdown("""
        #### 🧠 Contextual Analysis

        Analyses the claim in relation to retrieved
        information rather than treating it in isolation.
        """)

        st.markdown("""
        #### 🔍 Evidence Interpretation

        Interprets supporting, contradictory,
        and contextual evidence.
        """)

    with llm_col2:

        st.markdown("""
        #### ⚖️ Verification

        Uses the available context to generate
        a verification assessment.
        """)

        st.markdown("""
        #### 📝 Explanation

        Generates a natural-language explanation
        of the verification result.
        """)

    st.info("""
    **Important:** LLM-generated reasoning depends on the
    quality, relevance, and reliability of the retrieved
    evidence. The model's output should not be considered
    independent proof of factual correctness.
    """)

    st.markdown("---")
        # ========================================================
    # 10. MATHEMATICAL FORMULATION
    # ========================================================

    st.header("10. Mathematical Formulation")

    st.markdown("""
    This section presents the mathematical formulations
    underlying the text representation, evidence retrieval,
    similarity computation, ranking, and selection process
    used in TraceVerify.
    """)

    # --------------------------------------------------------
    # TF-IDF FORMULATION
    # --------------------------------------------------------

    st.subheader("10.1 TF-IDF Representation")

    st.markdown("""
    TF-IDF represents textual documents as numerical vectors
    by combining term frequency and inverse document frequency.
    """)

    st.latex(
        r"TF(t,d)=\frac{f_{t,d}}{\sum_{t'}f_{t',d}}"
    )

    st.latex(
        r"IDF(t)=\log\left(\frac{N}{DF(t)}\right)"
    )

    st.latex(
        r"TFIDF(t,d)=TF(t,d)\times IDF(t)"
    )

    st.markdown("""
    **Notation:**

    - \(t\): term
    - \(d\): document
    - \(f_{t,d}\): frequency of term \(t\) in document \(d\)
    - \(N\): total number of documents
    - \(DF(t)\): number of documents containing term \(t\)
    """)

    # --------------------------------------------------------
    # COSINE SIMILARITY
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("10.2 Cosine Similarity")

    st.markdown("""
    Cosine similarity measures the similarity between
    a claim vector and an evidence vector.
    """)

    st.latex(
        r"\operatorname{Sim}(C,E_i)"
        r"=\frac{C\cdot E_i}{\|C\|\|E_i\|}"
    )

    st.markdown("""
    **Notation:**

    - \(C\): vector representation of the claim
    - \(E_i\): vector representation of the \(i\)-th evidence item
    - \(C\cdot E_i\): dot product
    - \(\|C\|\), \(\|E_i\|\): vector magnitudes
    """)

    # --------------------------------------------------------
    # EVIDENCE RANKING
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("10.3 Evidence Ranking")

    st.markdown("""
    Evidence items are ranked in descending order
    according to their similarity scores.
    """)

    st.latex(
        r"E_{\text{ranked}}="
        r"\operatorname{sort}_{\text{descending}}"
        r"\left(E,\operatorname{Sim}(C,E_i)\right)"
    )

    # --------------------------------------------------------
    # TOP-K SELECTION
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("10.4 Top-K Evidence Selection")

    st.markdown("""
    The Top-K strategy selects the highest-ranked
    evidence items from the retrieved collection.
    """)

    st.latex(
        r"E_{\text{selected}}="
        r"\operatorname{TopK}(E_{\text{ranked}})"
    )

    # --------------------------------------------------------
    # RAG CONTEXT
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("10.5 Context Construction")

    st.markdown("""
    The selected evidence is combined with the input claim
    to form the context supplied to the language model.
    """)

    st.latex(
        r"Context=C\cup E_{\text{selected}}"
    )

    # --------------------------------------------------------
    # VERIFICATION FUNCTION
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("10.6 Verification Function")

    st.markdown("""
    The verification process can be represented abstractly
    as a function that receives a claim and its retrieved
    contextual evidence and generates a verification output.
    """)

    st.latex(
        r"V=f(C,E_{\text{selected}},LLM)"
    )

    st.markdown("""
    Where:

    - \(V\): verification output
    - \(C\): input claim
    - \(E_{\text{selected}}\): selected evidence
    - \(LLM\): language model reasoning function
    """)

    # MATHEMATICAL SUMMARY

    st.markdown("---")

    st.subheader("Mathematical Process Summary")

    st.code("""
    Claim C
       |
       v
    TF-IDF Vectorization
       |
       v
    Similarity(C, Evidence)
       |
       v
    Evidence Ranking
       |
       v
    Top-K Selection
       |
       v
    Context Construction
       |
       v
    LLM Function
       |
       v
    Verification Output
    """, language="text")

    st.markdown("---")
        # ========================================================
    # 11. VERIFICATION OUTPUT
    # ========================================================

    st.header("11. Verification Output")

    st.markdown("""
    TraceVerify presents the verification results through an
    interactive interface that combines the predicted verdict,
    retrieved evidence, and AI-generated explanation.
    """)

    # OUTPUT COMPONENTS

    st.subheader("Components of Verification Output")

    output_col1, output_col2 = st.columns(2)

    with output_col1:

        st.markdown("""
        #### 📝 Submitted Claim

        Displays the claim submitted by the user
        or selected from the dataset.
        """)

        st.markdown("""
        #### ⚖️ Verification Verdict

        Presents the verification classification
        generated by the system.
        """)

        st.markdown("""
        #### 📊 Similarity Information

        Displays relevant similarity information
        associated with retrieved evidence.
        """)

    with output_col2:

        st.markdown("""
        #### 🔎 Retrieved Evidence

        Displays information retrieved to support
        the verification process.
        """)

        st.markdown("""
        #### 🤖 AI Explanation

        Provides a natural-language explanation
        of the verification result.
        """)

        st.markdown("""
        #### 📚 Evidence Sources

        Displays available source information
        associated with the retrieved evidence.
        """)

    # EXAMPLE OUTPUT STRUCTURE

    st.subheader("Verification Result Structure")

    st.code("""
    ┌─────────────────────────────────────┐
    │         CLAIM VERIFICATION          │
    ├─────────────────────────────────────┤
    │ Submitted Claim                     │
    │                                     │
    │ Verification Verdict                │
    │                                     │
    │ Retrieved Evidence                  │
    │                                     │
    │ Evidence Relevance                  │
    │                                     │
    │ AI-Generated Explanation             │
    │                                     │
    │ Supporting Source Information        │
    └─────────────────────────────────────┘
    """, language="text")

    # EXPLAINABILITY

    st.subheader("Explainability")

    st.markdown("""
    Explainability is an important component of TraceVerify.

    Instead of presenting only a classification result,
    the system aims to provide supporting evidence and
    a contextual explanation to help users understand
    the basis of the verification result.
    """)

    explain_col1, explain_col2, explain_col3 = st.columns(3)

    with explain_col1:
        st.metric("Result", "Verdict")

    with explain_col2:
        st.metric("Context", "Evidence")

    with explain_col3:
        st.metric("Transparency", "Explanation")

    st.info("""
    Verification outputs should be interpreted in relation
    to the retrieved evidence and the limitations of automated
    reasoning. An AI-generated verdict is not a substitute
    for independent fact-checking.
    """)

    st.markdown("---")
        # ========================================================
    # 12. EVALUATION METRICS
    # ========================================================

    st.header("12. Evaluation Metrics")

    st.markdown("""
    TraceVerify can be evaluated using standard classification
    metrics to assess the performance of its verification
    system. These metrics measure the correctness of predictions
    and the system's ability to identify different classes.
    """)

    # METRIC CARDS

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:

        st.markdown("""
        ### Accuracy

        Measures the proportion of correctly classified
        samples among all evaluated samples.
        """)

        st.latex(
            r"Accuracy = \frac{TP + TN}{TP + TN + FP + FN}"
        )

        st.markdown("""
        ### Precision

        Measures how many predicted positive samples
        are actually positive.
        """)

        st.latex(
            r"Precision = \frac{TP}{TP + FP}"
        )

    with metric_col2:

        st.markdown("""
        ### Recall

        Measures the proportion of actual positive samples
        correctly identified by the model.
        """)

        st.latex(
            r"Recall = \frac{TP}{TP + FN}"
        )

        st.markdown("""
        ### F1-Score

        Combines precision and recall using their
        harmonic mean.
        """)

        st.latex(
            r"F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}"
        )

    st.divider()

    # CONFUSION MATRIX

    st.subheader("Confusion Matrix")

    st.markdown("""
    A confusion matrix provides a detailed comparison
    between actual labels and predicted labels.
    It helps identify correct classifications and
    classification errors.
    """)

    st.code("""
                    Predicted
                 Positive  Negative

    Actual
    Positive      TP        FN
    Negative      FP        TN
    """, language="text")

    confusion_col1, confusion_col2 = st.columns(2)

    with confusion_col1:

        st.markdown("""
        **True Positive (TP)**

        Positive samples correctly predicted as positive.
        """)

        st.markdown("""
        **False Positive (FP)**

        Negative samples incorrectly predicted as positive.
        """)

    with confusion_col2:

        st.markdown("""
        **True Negative (TN)**

        Negative samples correctly predicted as negative.
        """)

        st.markdown("""
        **False Negative (FN)**

        Positive samples incorrectly predicted as negative.
        """)

    st.divider()

    # EVALUATION APPROACH

    st.subheader("Evaluation Approach")

    evaluation_steps = [
        "Prepare the ground-truth labels.",
        "Generate predictions for the evaluated samples.",
        "Compare predicted labels with actual labels.",
        "Calculate accuracy, precision, recall and F1-score.",
        "Analyze classification errors and limitations."
    ]

    for i, step in enumerate(evaluation_steps, 1):

        st.markdown(f"""
        <div style="
            padding:12px;
            margin:6px 0;
            border-radius:10px;
            border:1px solid rgba(128,128,128,0.3);
        ">
            <b>Step {i}</b><br>
            {step}
        </div>
        """, unsafe_allow_html=True)

    st.info("""
    Actual evaluation scores should be calculated using
    the genuine prediction outputs and ground-truth labels.
    No performance values are displayed here because
    verified experimental results have not been inserted.
    """)

    st.markdown("---")
        # ========================================================
    # 13. RESULTS & ANALYSIS
    # ========================================================

    st.header("13. Results & Analysis")

    st.markdown("""
    This section presents the analysis of the TraceVerify
    verification process, including dataset characteristics,
    evidence retrieval behavior, classification outcomes,
    and system observations.
    """)

    # DATASET ANALYSIS

    st.subheader("Dataset Analysis")

    st.markdown("""
    The dataset contains samples belonging to different
    verification categories. The distribution indicates
    an imbalance between the available classes.
    """)

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric("Total Samples", "100")

    with result_col2:
        st.metric("False / Misleading", "93")

    with result_col3:
        st.metric("True + Mixed", "7")

    st.markdown("### Class Distribution")

    result_distribution = pd.DataFrame({
        "Category": [
            "False / Misleading",
            "True",
            "Mixed / Qualified"
        ],
        "Samples": [93, 5, 2]
    })

    st.bar_chart(
        result_distribution.set_index("Category")
    )

    # OBSERVATIONS

    st.subheader("Dataset Observations")

    observation_col1, observation_col2 = st.columns(2)

    with observation_col1:

        st.markdown("""
        **Class Imbalance**

        The dataset contains a significantly larger
        number of false or misleading samples than
        true and mixed samples.
        """)

    with observation_col2:

        st.markdown("""
        **Evaluation Consideration**

        Accuracy alone may not adequately represent
        performance when class distributions are uneven.
        Precision, recall and F1-score should also
        be considered.
        """)

    st.divider()

    # SYSTEM ANALYSIS

    st.subheader("System Analysis")

    analysis_cards = [
        (
            "Evidence Retrieval",
            "Retrieves relevant information based on similarity "
            "between the claim and available evidence."
        ),
        (
            "Evidence Ranking",
            "Ranks retrieved evidence according to relevance "
            "using similarity scores."
        ),
        (
            "LLM Reasoning",
            "Uses retrieved contextual information to generate "
            "a verification explanation."
        ),
        (
            "Explainability",
            "Provides evidence and reasoning alongside the "
            "verification outcome."
        )
    ]

    for title, description in analysis_cards:

        st.markdown(f"""
        <div style="
            padding:16px;
            margin:8px 0;
            border-radius:12px;
            border:1px solid rgba(128,128,128,0.3);
        ">
            <h4>{title}</h4>
            <p>{description}</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # KEY OBSERVATIONS

    st.subheader("Key Observations")

    observations = [
        "Retrieval-Augmented Generation combines information retrieval with language-model reasoning.",
        "Evidence relevance plays an important role in the verification process.",
        "The system provides contextual explanations instead of relying only on a final classification.",
        "Dataset imbalance must be considered during model evaluation.",
        "The quality of retrieved evidence influences the reliability of generated explanations."
    ]

    for observation in observations:

        st.markdown(f"- {observation}")

    st.warning("""
    Experimental performance results such as accuracy,
    precision, recall, F1-score, and confusion matrix values
    must be populated from actual evaluation outputs.
    The observations presented here describe the system
    methodology and dataset, not experimentally verified
    performance claims.
    """)

    st.markdown("---")
        # ========================================================
    # 14. LIMITATIONS
    # ========================================================

    st.header("14. Limitations")

    st.markdown("""
    Although TraceVerify integrates retrieval-based evidence
    analysis and LLM-based reasoning, the system has several
    limitations that may affect its verification reliability.
    """)

    limitation_cards = [
        (
            "Dataset Imbalance",
            "The dataset contains an unequal distribution of "
            "verification categories, which may influence "
            "classification behavior and evaluation."
        ),
        (
            "Evidence Availability",
            "The verification process depends on the availability "
            "of relevant information in the evidence sources."
        ),
        (
            "Retrieval Quality",
            "Irrelevant or incomplete retrieved evidence may "
            "affect the quality of the generated explanation."
        ),
        (
            "LLM Hallucination",
            "Large language models may generate inaccurate "
            "or unsupported information."
        ),
        (
            "Source Reliability",
            "Retrieved information may originate from sources "
            "with varying reliability and credibility."
        ),
        (
            "Contextual Complexity",
            "Claims involving sarcasm, ambiguity, rapidly changing "
            "information, or missing context may be difficult "
            "to verify accurately."
        ),
        (
            "Multimodal Analysis",
            "Comprehensive image-text verification requires "
            "effective integration of visual and textual evidence."
        ),
        (
            "Automated Verification",
            "The system provides automated assistance and should "
            "not replace independent human fact-checking."
        )
    ]

    for i in range(0, len(limitation_cards), 2):

        limitation_col1, limitation_col2 = st.columns(2)

        with limitation_col1:

            title, description = limitation_cards[i]

            st.markdown(f"""
            <div style="
                padding:16px;
                margin:8px 0;
                border-radius:12px;
                border:1px solid rgba(128,128,128,0.3);
                min-height:150px;
            ">
                <h4>{title}</h4>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)

        with limitation_col2:

            title, description = limitation_cards[i + 1]

            st.markdown(f"""
            <div style="
                padding:16px;
                margin:8px 0;
                border-radius:12px;
                border:1px solid rgba(128,128,128,0.3);
                min-height:150px;
            ">
                <h4>{title}</h4>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    st.subheader("Limitation Summary")

    st.info("""
    TraceVerify aims to support misinformation verification
    through evidence retrieval and AI-generated reasoning.
    However, its output depends on dataset quality,
    evidence availability, retrieval relevance, source
    credibility, and the limitations of language models.
    """)

    st.markdown("---")
        # ========================================================
    # 15. FUTURE ENHANCEMENTS
    # ========================================================

    st.header("15. Future Enhancements")

    st.markdown("""
    Future development of TraceVerify can focus on improving
    verification accuracy, evidence quality, multimodal
    understanding, scalability, and explainability.
    """)

    future_cards = [
        (
            "Advanced Multimodal Models",
            "Integrate advanced vision-language models to jointly "
            "analyze textual claims and associated images."
        ),
        (
            "Improved Evidence Retrieval",
            "Incorporate additional reliable information sources "
            "and advanced retrieval techniques."
        ),
        (
            "Real-Time Verification",
            "Develop support for real-time misinformation "
            "verification across online information sources."
        ),
        (
            "Enhanced Source Credibility",
            "Introduce more comprehensive source reliability "
            "assessment mechanisms."
        ),
        (
            "Advanced Explainability",
            "Provide more detailed evidence-grounded explanations "
            "and transparent reasoning."
        ),
        (
            "Larger and Balanced Datasets",
            "Expand the dataset and improve class distribution "
            "to support more comprehensive evaluation."
        ),
        (
            "Model Optimization",
            "Explore improved models and optimization techniques "
            "for efficient verification."
        ),
        (
            "Scalable Deployment",
            "Improve deployment infrastructure to support "
            "larger workloads and multiple users."
        )
    ]

    for i in range(0, len(future_cards), 2):

        future_col1, future_col2 = st.columns(2)

        with future_col1:

            title, description = future_cards[i]

            st.markdown(f"""
            <div style="
                padding:16px;
                margin:8px 0;
                border-radius:12px;
                border:1px solid rgba(128,128,128,0.3);
                min-height:150px;
            ">
                <h4>{title}</h4>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)

        with future_col2:

            title, description = future_cards[i + 1]

            st.markdown(f"""
            <div style="
                padding:16px;
                margin:8px 0;
                border-radius:12px;
                border:1px solid rgba(128,128,128,0.3);
                min-height:150px;
            ">
                <h4>{title}</h4>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    st.subheader("Future Research Direction")

    st.markdown("""
    The future direction of TraceVerify is to develop a more
    robust, scalable, and explainable misinformation
    verification framework by combining multimodal analysis,
    reliable evidence retrieval, and advanced AI reasoning.
    """)

    st.markdown("---")
        # ========================================================
    # 16. CONCLUSION
    # ========================================================

    st.header("16. Conclusion")

    st.markdown("""
    TraceVerify is an AI-driven misinformation verification
    framework designed to combine evidence retrieval,
    similarity-based ranking, and Large Language Model
    reasoning to support claim verification.

    The system integrates Retrieval-Augmented Generation
    (RAG) with evidence-based reasoning to retrieve relevant
    information and generate contextual explanations for
    verification outcomes.

    By combining information retrieval with AI-generated
    reasoning, TraceVerify aims to improve transparency
    and provide users with more informative verification
    results than classification-only approaches.

    The project highlights the importance of evidence
    relevance, explainability, source credibility, and
    responsible use of AI in misinformation detection.

    Although the system has limitations related to dataset
    imbalance, evidence availability, retrieval quality,
    and language-model reliability, it provides a foundation
    for further research and development in explainable
    misinformation verification.
    """)

    st.divider()

    # FINAL PROJECT SUMMARY

    st.subheader("TraceVerify — Research Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric("Approach", "RAG + AI")

    with summary_col2:
        st.metric("Core Technology", "Evidence Retrieval")

    with summary_col3:
        st.metric("Goal", "Explainable Verification")

    st.markdown("""
    <div style="
        padding:22px;
        margin-top:15px;
        border-radius:15px;
        text-align:center;
        border:1px solid rgba(128,128,128,0.3);
    ">
        <h3>Evidence-Based AI Misinformation Verification</h3>
        <p>
            Retrieve relevant evidence • Analyze context •
            Generate explanations • Support informed decisions
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.caption(
        "TraceVerify | Research & Methodology | "
        "M.Tech Artificial Intelligence | "
        "Amrita Vishwa Vidyapeetham"
    )
   
# PART 7 - TEAM PAGE
# ============================================================

elif st.session_state.tv_page == "Team":

    st.title("Our Team")

    st.caption(
        "From research and analysis to implementation and deployment"
    )

    st.markdown("---")

    st.header("Project Team")

    st.markdown("""
    **TraceVerify** was developed collaboratively through research,
    system design, implementation, testing and documentation.
    """)

    st.markdown("---")

    # TEAM MEMBERS SIDE BY SIDE
    col1, col2 = st.columns(2, gap="large")

    # ========================================================
    # MEMBER 1 - TEJASWINI
    # ========================================================

    with col1:

        st.markdown("## Tejaswini")

        st.markdown("**Primary Focus: System Implementation & Development**")

        st.markdown("---")

        st.markdown("### Contributions")

        st.markdown("**Project Development & System Design**")

        st.markdown("""
        - Project idea and problem identification
        - Defining the system architecture
        - Designing the verification workflow
        - Planning the RAG-based system
        """)

        st.markdown("**Technical Implementation**")

        st.markdown("""
        - Streamlit website and user interface development
        - Backend implementation and integration
        - Retrieval-Augmented Generation (RAG) implementation
        - LLM integration
        - Evidence retrieval pipeline implementation
        - Verification and verdict generation
        - Integration of system components
        """)

        st.markdown("**Testing & Deployment**")

        st.markdown("""
        - Application implementation and debugging
        - System testing and troubleshooting
        - Integration testing
        - Deployment and application configuration
        """)

        st.markdown("**Technical Documentation**")

        st.markdown("""
        - System implementation documentation
        - Architecture and workflow documentation
        - Technical project documentation
        """)

        st.markdown("---")

        st.markdown("### Profiles")

        st.markdown(
            "[GitHub](https://github.com/Tejaswini-2207)"
        )

        st.markdown(
            "[LinkedIn](https://in.linkedin.com/in/"
            "tejaswini-s-4294b9252)"
        )

    # ========================================================
    # MEMBER 2 - MANJARI GUPTA
    # ========================================================

    with col2:

        st.markdown("## Manjari Gupta")

        st.markdown("**Primary Focus: Research, Analysis & Documentation**")

        st.markdown("---")

        st.markdown("### Contributions")

        st.markdown("**Research & Literature Review**")

        st.markdown("""
        - Literature review and research analysis
        - Studying existing misinformation verification approaches
        - Supporting problem formulation
        - Research methodology analysis
        - Reviewing relevant research work
        """)

        st.markdown("**Dataset & Methodology**")

        st.markdown("""
        - Dataset analysis and understanding
        - Supporting data preparation and analysis
        - Research methodology discussions
        - Analysis of verification approaches
        - Identifying relevant research directions
        """)

        st.markdown("**Technical Support & Validation**")

        st.markdown("""
        - Supporting system development
        - Implementation assistance
        - Testing project functionality
        - Reviewing system outputs
        - Identifying issues and supporting improvements
        """)

        st.markdown("**Research Documentation**")

        st.markdown("""
        - Research documentation
        - Literature and methodology documentation
        - Project report preparation
        - Supporting technical documentation
        """)

        st.markdown("---")

        st.markdown("### Profiles")

        st.markdown(
            "[GitHub](https://github.com/manjaariiii)"
        )

        st.markdown(
            "[LinkedIn](https://www.linkedin.com/in/"
            "manjari-gupta-439932298/)"
        )

    # ========================================================
    # COLLABORATIVE WORK
    # ========================================================

    st.markdown("---")

    st.header("Collaborative Work")

    st.markdown("""
    Both team members collaborated throughout the project on research,
    discussions, system planning, reviewing progress, testing and
    project completion.

    The project combines research-oriented contributions with
    technical implementation to develop the TraceVerify system.
    """)

    st.markdown("---")

    st.header("Project Journey")

    journey = [
        "Research & Problem Identification",
        "Literature Review",
        "System Design",
        "Implementation",
        "Integration",
        "Testing & Validation",
        "Documentation",
        "Deployment"
    ]

    for i, step in enumerate(journey, start=1):
        st.markdown(f"**{i}.** {step}")

    st.markdown("---")

    st.markdown(
        """
        <div style="text-align:center; padding:15px;">
            <h3>TraceVerify</h3>
            <p> M.Tech Artificial Intelligence </p>
            <p>Amrita Vishwa Vidyapeetham</p>
        </div>
        """,
        unsafe_allow_html=True
    )