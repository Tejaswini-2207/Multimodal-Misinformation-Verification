
from urllib.parse import urlparse
from collections import Counter


# ==================================================
# 1. HEURISTIC SOURCE WEIGHTS
# ==================================================

SOURCE_WEIGHTS = {
    "OFFICIAL": 1.00,
    "FACT_CHECK": 0.90,
    "ESTABLISHED_NEWS": 0.80,
    "OTHER_ORGANIZATION": 0.65,
    "UNKNOWN": 0.40,
    "INVALID": 0.00
}


# ==================================================
# 2. ARTICLE EXTRACTION WEIGHTS
# ==================================================

EXTRACTION_WEIGHTS = {
    "SUCCESS": 1.00,
    "FAILED": 0.35
}


# ==================================================
# 3. DOMAIN NORMALIZATION
# ==================================================

def get_domain(url):

    try:
        domain = urlparse(str(url)).hostname

        if not domain:
            return ""

        domain = domain.lower()

        if domain.startswith("www."):
            domain = domain[4:]

        return domain

    except Exception:
        return ""


# ==================================================
# 4. DUPLICATE DOMAIN CHECK
# ==================================================

def count_unique_sources(evidence):

    domains = set()

    for item in evidence:

        domain = (
            item.get("domain")
            or item.get("Domain")
            or get_domain(
                item.get("url") or item.get("URL", "")
            )
        )

        if domain:
            domains.add(domain.lower())

    return len(domains)


# ==================================================
# 5. CALCULATE SOURCE QUALITY
# ==================================================

def calculate_source_quality(evidence):

    if not evidence:
        return 0.0

    scores = []

    for item in evidence:

        source_type = item.get(
            "source_type",
            item.get("Source_Type", "UNKNOWN")
        )

        extraction_status = item.get(
            "extraction_status",
            item.get("Extraction_Status", "FAILED")
        )

        source_score = SOURCE_WEIGHTS.get(
            source_type,
            SOURCE_WEIGHTS["UNKNOWN"]
        )

        extraction_score = EXTRACTION_WEIGHTS.get(
            extraction_status,
            0.35
        )

        combined = (
            0.60 * source_score
            + 0.40 * extraction_score
        )

        scores.append(combined)

    return sum(scores) / len(scores)


# ==================================================
# 6. CALCULATE EVIDENCE AGREEMENT
# ==================================================

def calculate_evidence_agreement(evidence):

    if not evidence:
        return 0.0

    # Use unique domains to avoid counting
    # repeated results from the same publisher.
    domain_positions = {}

    for item in evidence:

        domain = (
            item.get("domain")
            or item.get("Domain")
            or get_domain(
                item.get("url") or item.get("URL", "")
            )
        )

        if not domain:
            continue

        domain_positions.setdefault(
            domain.lower(),
            item
        )

    unique_items = list(domain_positions.values())

    if not unique_items:
        return 0.0

    # Agreement requires explicit, manually or
    # independently assigned support/refute labels.
    positions = Counter(
        item.get("evidence_position", "unknown").lower()
        for item in unique_items
    )

    known = (
        positions.get("supports", 0)
        + positions.get("refutes", 0)
    )

    if known == 0:
        return 0.0

    # Proportion of the known evidence taking
    # the majority position.
    majority = max(
        positions.get("supports", 0),
        positions.get("refutes", 0)
    )

    return majority / known


# ==================================================
# 7. FINAL HEURISTIC EVIDENCE SCORE
# ==================================================

def calculate_confidence_score(evidence):

    if not evidence:

        return {
            "score": 0.0,
            "percentage": 0.0,
            "source_quality": 0.0,
            "evidence_agreement": 0.0,
            "unique_sources": 0,
            "interpretation": "NO EVIDENCE"
        }

    source_quality = calculate_source_quality(
        evidence
    )

    agreement = calculate_evidence_agreement(
        evidence
    )

    unique_sources = count_unique_sources(
        evidence
    )

    # Conservative score: agreement contributes
    # only when evidence positions are supplied.
    score = (
        0.70 * source_quality
        + 0.30 * agreement
    )

    percentage = round(score * 100, 2)

    if percentage >= 80:
        interpretation = "STRONG EVIDENCE QUALITY"

    elif percentage >= 60:
        interpretation = "MODERATE EVIDENCE QUALITY"

    elif percentage >= 40:
        interpretation = "LIMITED EVIDENCE QUALITY"

    else:
        interpretation = "WEAK EVIDENCE QUALITY"

    return {

        "score": round(score, 4),

        "percentage": percentage,

        "source_quality": round(source_quality, 4),

        "evidence_agreement": round(agreement, 4),

        "unique_sources": unique_sources,

        "interpretation": interpretation

    }


# ==================================================
# 8. DISPLAY SCORE
# ==================================================

def display_confidence_score(result):

    print("\n========== EVIDENCE QUALITY SCORE ==========")

    print(
        "Heuristic Score:",
        result["percentage"],
        "%"
    )

    print(
        "Source Quality:",
        result["source_quality"]
    )

    print(
        "Evidence Agreement:",
        result["evidence_agreement"]
    )

    print(
        "Unique Sources:",
        result["unique_sources"]
    )

    print(
        "Interpretation:",
        result["interpretation"]
    )

    print("\n============================================")


# ==================================================
# 9. TEST MODULE
# ==================================================

if __name__ == "__main__":

    test_evidence = [

        {
            "domain": "pib.gov.in",
            "source_type": "OFFICIAL",
            "extraction_status": "SUCCESS",
            "evidence_position": "refutes"
        },

        {
            "domain": "newsmobile.in",
            "source_type": "FACT_CHECK",
            "extraction_status": "SUCCESS",
            "evidence_position": "refutes"
        },

        {
            "domain": "thehindu.com",
            "source_type": "ESTABLISHED_NEWS",
            "extraction_status": "SUCCESS",
            "evidence_position": "refutes"
        },

        {
            "domain": "example.com",
            "source_type": "UNKNOWN",
            "extraction_status": "FAILED",
            "evidence_position": "unknown"
        }

    ]

    result = calculate_confidence_score(
        test_evidence
    )

    display_confidence_score(result)
