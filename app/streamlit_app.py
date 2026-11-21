import sys
import json
import uuid
import inspect
from pathlib import Path
from datetime import datetime

import streamlit as st

# ============================================================
# TRACEVERIFY
# AI-Driven Explainable Multimodal Misinformation Verification
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# BACKEND CONNECTION
# ============================================================

try:
    from llm_verifier import verify_claim
    BACKEND_AVAILABLE = True
    BACKEND_ERROR = None

except Exception as error:
    BACKEND_AVAILABLE = False
    BACKEND_ERROR = str(error)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TraceVerify | Evidence Intelligence",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #0B1120;
    color: #E6EDF7;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background-color: #101827;
    border-right: 1px solid #26364B;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

.brand {
    font-size: 30px;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -1px;
}

.brand-subtitle {
    font-size: 10px;
    letter-spacing: 2px;
    color: #8395AD;
    margin-top: 5px;
}

.sidebar-caption {
    color: #8192A8;
    font-size: 12px;
}

/* HERO */

.hero {
    background:
        radial-gradient(
            circle at 85% 15%,
            rgba(34, 160, 160, 0.18),
            transparent 38%
        ),
        linear-gradient(135deg, #14243A, #101827);

    border: 1px solid #2B4058;
    border-radius: 22px;
    padding: 42px;
    margin-bottom: 25px;
}

.hero-label {
    color: #5CE0B7;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
}

.hero-title {
    color: #FFFFFF;
    font-size: 44px;
    font-weight: 800;
    line-height: 1.2;
    margin-top: 15px;
}

.hero-description {
    color: #A8B8CC;
    font-size: 15px;
    line-height: 1.8;
    margin-top: 15px;
    max-width: 750px;
}

/* SECTION */

.section-title {
    font-size: 25px;
    font-weight: 750;
    color: #FFFFFF;
    margin-top: 25px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #91A3BA;
    font-size: 14px;
    margin-bottom: 20px;
}

/* FEATURE CARDS */

.feature-card {
    background: #111C2E;
    border: 1px solid #27384E;
    border-radius: 17px;
    padding: 23px;
    min-height: 190px;
}

.feature-icon {
    font-size: 28px;
    margin-bottom: 12px;
}

.feature-name {
    color: #FFFFFF;
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 9px;
}

.feature-description {
    color: #9AABC1;
    font-size: 13px;
    line-height: 1.7;
}

/* METRICS */

.metric-card {
    background: #111C2E;
    border: 1px solid #27384E;
    border-radius: 15px;
    padding: 20px;
}

.metric-label {
    color: #91A3BA;
    font-size: 12px;
    font-weight: 600;
}

.metric-value {
    color: #FFFFFF;
    font-size: 28px;
    font-weight: 800;
    margin-top: 8px;
}

/* RESULT */

.result-card {
    background: #132238;
    border: 1px solid #304966;
    border-radius: 16px;
    padding: 25px;
}

/* INPUTS */

textarea {
    border-radius: 12px !important;
}

div[data-testid="stFileUploader"] {
    background: #111C2E;
    border: 1px solid #27384E;
    border-radius: 12px;
    padding: 12px;
}

/* BUTTONS */

.stButton > button {
    border-radius: 10px;
    min-height: 45px;
    font-weight: 650;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #64758B;
    font-size: 12px;
    padding-top: 40px;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_claim" not in st.session_state:
    st.session_state.last_claim = ""

if "navigation" not in st.session_state:
    st.session_state.navigation = "🏠 Overview"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def section(title, subtitle=""):

    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )

    if subtitle:
        st.markdown(
            f'<div class="section-subtitle">{subtitle}</div>',
            unsafe_allow_html=True
        )


def feature(icon, title, description):

    st.markdown(f"""
    <div class="feature-card">
        <div class="feature-icon">{icon}</div>
        <div class="feature-name">{title}</div>
        <div class="feature-description">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def metric(label, value):

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def safe_json(data):

    return json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
        default=str
    )


# ============================================================
# VERIFICATION FUNCTION
# ============================================================

def execute_verification(claim, image_path=None):

    if not BACKEND_AVAILABLE:
        raise RuntimeError(
            "Backend unavailable: " + str(BACKEND_ERROR)
        )

    signature = inspect.signature(verify_claim)
    parameters = signature.parameters

    kwargs = {}

    if "sample_id" in parameters:
        kwargs["sample_id"] = (
            "CUSTOM-" + uuid.uuid4().hex[:8].upper()
        )

    if "claim_override" in parameters:
        kwargs["claim_override"] = claim

    if "image_path" in parameters:
        kwargs["image_path"] = image_path

    if "return_details" in parameters:
        kwargs["return_details"] = True

    if "claim" in parameters:
        kwargs["claim"] = claim

    if "text" in parameters:
        kwargs["text"] = claim

    return verify_claim(**kwargs)


# ============================================================
# RESULT DISPLAY
# ============================================================

def display_result(result):

    section("Investigation Results")

    if isinstance(result, dict):

        verdict = None

        possible_verdict_keys = [
            "verdict",
            "prediction",
            "label",
            "classification",
            "final_verdict",
            "result"
        ]

        for key in possible_verdict_keys:

            if key in result:
                verdict = result[key]
                break

        if verdict is not None:

            st.markdown(f"""
            <div class="result-card">

                <div style="
                    color:#91A3BA;
                    font-size:12px;
                    font-weight:700;
                    letter-spacing:1px;
                ">
                FINAL ASSESSMENT
                </div>

                <div style="
                    color:white;
                    font-size:32px;
                    font-weight:800;
                    margin-top:12px;
                ">
                {verdict}
                </div>

            </div>
            """, unsafe_allow_html=True)

        explanation = None

        explanation_keys = [
            "explanation",
            "reasoning",
            "analysis",
            "report",
            "summary",
            "llm_response"
        ]

        for key in explanation_keys:

            if key in result:
                explanation = result[key]
                break

        if explanation is not None:

            section("🧠 Explainable Analysis")

            if isinstance(explanation, (dict, list)):
                st.json(explanation)
            else:
                st.markdown(str(explanation))

        section("Evidence & Technical Details")

        st.json(result)

    else:

        st.write(result)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="brand">🔎 TraceVerify</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-subtitle">'
        'EVIDENCE INTELLIGENCE PLATFORM'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    pages = [
        "🏠 Overview",
        "🔬 Verify a Claim",
        "📚 Investigation History",
        "📊 Research Dashboard",
        "👥 Team",
        "ℹ️ About"
    ]

    page = st.radio(
        "NAVIGATION",
        pages,
        key="navigation",
        label_visibility="collapsed"
    )

    st.divider()

    if BACKEND_AVAILABLE:

        st.success("Verification engine connected")

    else:

        st.error("Backend unavailable")

        with st.expander("Backend details"):
            st.code(BACKEND_ERROR)

    st.markdown("""
    <div class="sidebar-caption" style="margin-top:25px;">
    Research Prototype<br>
    Multimodal AI Investigation
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "🏠 Overview":

    st.markdown("""
    <div class="hero">

        <div class="hero-label">
        AI-POWERED EVIDENCE INTELLIGENCE
        </div>

        <div class="hero-title">
        Verify the claim.<br>
        Trace the evidence.<br>
        Understand the truth.
        </div>

        <div class="hero-description">
        TraceVerify investigates potentially misleading information
        through multimodal AI, retrieval-augmented generation,
        source assessment and explainable reasoning.
        </div>

    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:

        feature(
            "🧠",
            "Explainable AI",
            "Understand how the system evaluates information "
            "and reaches its conclusion."
        )

    with col2:

        feature(
            "🔎",
            "Evidence Retrieval",
            "Retrieve potentially relevant evidence from "
            "local knowledge and external sources."
        )

    with col3:

        feature(
            "🖼️",
            "Multimodal Analysis",
            "Analyze textual claims together with "
            "associated visual content."
        )

    section(
        "Start an Investigation",
        "Investigate a claim, image or potentially misleading information."
    )

    if st.button(
        "🔬 Start New Investigation",
        type="primary",
        use_container_width=True
    ):

        st.session_state.navigation = "🔬 Verify a Claim"
        st.rerun()

    section(
        "How TraceVerify Works",
        "An evidence-centered verification pipeline"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        feature(
            "01",
            "Analyze",
            "Understand the submitted claim and media."
        )

    with col2:

        feature(
            "02",
            "Retrieve",
            "Search for potentially relevant evidence."
        )

    with col3:

        feature(
            "03",
            "Evaluate",
            "Assess source credibility and evidence."
        )

    with col4:

        feature(
            "04",
            "Explain",
            "Generate an evidence-grounded conclusion."
        )


# ============================================================
# VERIFY CLAIM PAGE
# ============================================================

elif page == "🔬 Verify a Claim":

    section(
        "Claim Investigation Workspace",
        "Submit text, an image, or both for multimodal analysis."
    )

    st.markdown("""
    <div class="result-card">

    <b>Investigation Tip</b><br><br>

    Submit the complete claim and any associated image.
    Context such as the date, location or original source
    may help the investigation.

    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    claim = st.text_area(
        "Claim or statement",
        placeholder=(
            "Enter the claim you want TraceVerify to investigate..."
        ),
        height=160
    )

    uploaded_image = st.file_uploader(
        "Upload related image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_image:

        st.markdown("### Submitted Image")

        st.image(
            uploaded_image,
            use_container_width=True
        )

    verify_button = st.button(
        "🔎 Investigate Claim",
        type="primary",
        use_container_width=True
    )

    if verify_button:

        if not claim.strip() and not uploaded_image:

            st.warning(
                "Please enter a claim or upload an image."
            )

        elif not BACKEND_AVAILABLE:

            st.error("Verification backend is unavailable.")

            st.code(BACKEND_ERROR)

        else:

            image_path = None

            if uploaded_image:

                upload_directory = (
                    ROOT_DIR
                    / "outputs"
                    / "streamlit_uploads"
                )

                upload_directory.mkdir(
                    parents=True,
                    exist_ok=True
                )

                image_path = (
                    upload_directory
                    / (
                        uuid.uuid4().hex
                        + "_"
                        + uploaded_image.name
                    )
                )

                with open(image_path, "wb") as file:

                    file.write(
                        uploaded_image.getbuffer()
                    )

            section("Investigation in Progress")

            try:

                with st.spinner(
                    "TraceVerify is analyzing your submission..."
                ):

                    result = execute_verification(
                        claim=claim,
                        image_path=image_path
                    )

                st.session_state.last_result = result
                st.session_state.last_claim = claim

                st.session_state.history.append({
                    "id": uuid.uuid4().hex,
                    "claim": claim,
                    "result": result,
                    "timestamp": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                })

                st.success("Investigation completed.")

                display_result(result)

            except Exception as error:

                st.error("Verification encountered an error.")

                st.exception(error)

                st.info(
                    "The frontend is operational. The backend "
                    "function interface may need adjustment."
                )


# ============================================================
# INVESTIGATION HISTORY
# ============================================================

elif page == "📚 Investigation History":

    section(
        "Investigation History",
        "Review previous investigations performed in this session."
    )

    if not st.session_state.history:

        st.info(
            "No investigations yet. Perform your first "
            "investigation to see it here."
        )

    else:

        metric(
            "Total Investigations",
            len(st.session_state.history)
        )

        st.markdown("")

        for investigation in reversed(
            st.session_state.history
        ):

            with st.expander(
                investigation["claim"][:100]
                + " | "
                + investigation["timestamp"]
            ):

                st.markdown("### Submitted Claim")

                st.write(investigation["claim"])

                st.markdown("### Investigation Result")

                st.json(investigation["result"])


# ============================================================
# RESEARCH DASHBOARD
# ============================================================

elif page == "📊 Research Dashboard":

    section(
        "Research Dashboard",
        "System activity and research evaluation."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        metric(
            "Session Investigations",
            len(st.session_state.history)
        )

    with col2:

        metric(
            "Backend Status",
            "Connected" if BACKEND_AVAILABLE else "Unavailable"
        )

    with col3:

        metric(
            "System Type",
            "Multimodal AI"
        )

    section("Research Pipeline")

    st.markdown("""
    **Dataset Collection**

    ↓

    **Data Preprocessing**

    ↓

    **Feature Extraction**

    ↓

    **Multimodal Analysis**

    ↓

    **RAG Evidence Retrieval**

    ↓

    **Source & Evidence Assessment**

    ↓

    **LLM Reasoning**

    ↓

    **Explainable Verification**
    """)

    st.info("""
    Future research dashboard components will use actual project
    data: dataset statistics, classification evaluation,
    retrieval evaluation, coverage, abstention rate and
    confusion matrix.
    """)


# ============================================================
# TEAM PAGE
# ============================================================

elif page == "👥 Team":

    section(
        "Research Team",
        "Contributions to the TraceVerify project."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="feature-card">

        <div class="feature-icon">👩‍💻</div>

        <div class="feature-name">Tejaswini</div>

        <div style="color:#5CE0B7;font-weight:600;">
        Frontend & Multimodal AI Integration
        </div>

        <br>

        <b>Joint Contributions</b>

        <ul>
            <li>Dataset collection</li>
            <li>Data preprocessing</li>
            <li>Feature extraction</li>
        </ul>

        <b>Individual Contributions</b>

        <ul>
            <li>Streamlit frontend development</li>
            <li>RAG integration</li>
            <li>Image analysis</li>
            <li>AI and LLM integration</li>
            <li>Multimodal pipeline integration</li>
        </ul>

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="feature-card">

        <div class="feature-icon">🧠</div>

        <div class="feature-name">Manjari Gupta</div>

        <div style="color:#5CE0B7;font-weight:600;">
        Model Backend & Source Credibility
        </div>

        <br>

        <b>Joint Contributions</b>

        <ul>
            <li>Dataset collection</li>
            <li>Data preprocessing</li>
            <li>Feature extraction</li>
        </ul>

        <b>Individual Contributions</b>

        <ul>
            <li>Backend verification implementation</li>
            <li>Model implementation</li>
            <li>Source credibility assessment</li>
            <li>Source validation</li>
            <li>Evidence quality assessment</li>
        </ul>

        </div>
        """, unsafe_allow_html=True)


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "ℹ️ About":

    section(
        "About TraceVerify",
        "Explainable multimodal misinformation verification."
    )

    st.markdown("""
    <div class="hero">

        <div class="hero-label">
        RESEARCH PROJECT
        </div>

        <div class="hero-title">
        TraceVerify
        </div>

        <div class="hero-description">
        An AI-driven explainable multimodal misinformation
        verification system combining retrieval-augmented
        generation, image analysis, source assessment and
        language-model reasoning.
        </div>

    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        feature(
            "📚",
            "Retrieval-Augmented Generation",
            "Retrieves potentially relevant evidence "
            "to support grounded analysis."
        )

        feature(
            "🖼️",
            "Multimodal Analysis",
            "Analyzes textual claims and visual information."
        )

    with col2:

        feature(
            "🌐",
            "Source Assessment",
            "Evaluates source categories and evidence quality."
        )

        feature(
            "🧠",
            "Explainable Reasoning",
            "Produces conclusions with explanations "
            "and limitations."
        )

    st.warning("""
    TraceVerify is a research prototype. Its assessments are
    not guaranteed to be correct. Source availability,
    retrieval quality, model limitations and missing evidence
    may affect results.
    """)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

TraceVerify · Evidence Intelligence Platform

<br>

AI-Driven Explainable Multimodal Misinformation Verification

</div>
""", unsafe_allow_html=True)