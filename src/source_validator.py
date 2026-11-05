
from urllib.parse import urlparse


# These are configured domain lists, not independent
# certifications that every article on these sites is accurate.

OFFICIAL_DOMAINS = {
    "pib.gov.in",
    "factcheck.pib.gov.in",
    "gov.in",
    "nic.in",
    "who.int",
    "un.org"
}

FACT_CHECK_DOMAINS = {
    "altnews.in",
    "boomlive.in",
    "factly.in",
    "vishvasnews.com",
    "newsmobile.in",
    "thequint.com",
    "ghanafact.com"
}

ESTABLISHED_NEWS_DOMAINS = {
    "thehindu.com",
    "hindustantimes.com",
    "indianexpress.com",
    "timesofindia.indiatimes.com",
    "economictimes.indiatimes.com",
    "bbc.com",
    "bbc.co.uk",
    "reuters.com",
    "apnews.com",
    "indiatoday.in",
    "livemint.com"
}

OTHER_RESEARCH_ORGANIZATIONS = {
    "cyberpeace.org"
}


# --------------------------------------------------
# 1. DOMAIN NORMALIZATION
# --------------------------------------------------

def normalize_domain(url):
    """
    Extracts a normalized hostname from a URL.

    Supports URLs with or without http/https.
    """

    if not url:
        return ""

    try:
        url = str(url).strip()

        if not url:
            return ""

        if "://" not in url:
            url = "https://" + url

        parsed = urlparse(url)
        domain = parsed.hostname

        if not domain:
            return ""

        domain = domain.lower().strip(".")

        if domain.startswith("www."):
            domain = domain[4:]

        return domain

    except (ValueError, TypeError):
        return ""


# --------------------------------------------------
# 2. DOMAIN MATCHING
# --------------------------------------------------

def domain_matches(domain, trusted_domain):
    """
    Matches the exact domain or a genuine subdomain.

    Example:
    www.boomlive.in -> boomlive.in
    article.boomlive.in -> boomlive.in
    fakeboomlive.in -> does not match boomlive.in
    """

    return (
        domain == trusted_domain
        or domain.endswith("." + trusted_domain)
    )


# --------------------------------------------------
# 3. SOURCE CLASSIFICATION
# --------------------------------------------------

def classify_source(url):
    domain = normalize_domain(url)

    if not domain:
        return {
            "domain": "",
            "source_type": "INVALID",
            "credibility_category": "UNCLASSIFIED",
            "classification_basis": "Invalid or missing URL"
        }

    if any(
        domain_matches(domain, item)
        for item in OFFICIAL_DOMAINS
    ):
        return {
            "domain": domain,
            "source_type": "OFFICIAL",
            "credibility_category": "OFFICIAL_SOURCE",
            "classification_basis": (
                "Domain matches the configured official-source list"
            )
        }

    if any(
        domain_matches(domain, item)
        for item in FACT_CHECK_DOMAINS
    ):
        return {
            "domain": domain,
            "source_type": "FACT_CHECK",
            "credibility_category": "SPECIALIST_FACT_CHECKER",
            "classification_basis": (
                "Domain matches the configured fact-check publisher list"
            )
        }

    if any(
        domain_matches(domain, item)
        for item in ESTABLISHED_NEWS_DOMAINS
    ):
        return {
            "domain": domain,
            "source_type": "ESTABLISHED_NEWS",
            "credibility_category": "NEWS_PUBLISHER",
            "classification_basis": (
                "Domain matches the configured news publisher list"
            )
        }

    if any(
        domain_matches(domain, item)
        for item in OTHER_RESEARCH_ORGANIZATIONS
    ):
        return {
            "domain": domain,
            "source_type": "OTHER_ORGANIZATION",
            "credibility_category": "ORGANIZATION",
            "classification_basis": (
                "Domain matches the configured organization list"
            )
        }

    return {
        "domain": domain,
        "source_type": "UNKNOWN",
        "credibility_category": "UNCLASSIFIED",
        "classification_basis": (
            "Domain is not present in the configured source lists"
        )
    }


# --------------------------------------------------
# 4. VALIDATE SOURCE COLLECTION
# --------------------------------------------------

def validate_sources(evidence_list):
    """
    Adds source classification fields without changing
    the original evidence fields.
    """

    validated_sources = []

    for item in evidence_list or []:
        if not isinstance(item, dict):
            continue

        url = (
            item.get("URL")
            or item.get("url")
            or ""
        )

        classification = classify_source(url)

        validated_item = {
            **item,
            "Domain": classification["domain"],
            "Source_Type": classification["source_type"],
            "Credibility_Category": (
                classification["credibility_category"]
            ),
            "Classification_Basis": (
                classification["classification_basis"]
            )
        }

        validated_sources.append(validated_item)

    return validated_sources


# --------------------------------------------------
# 5. DISPLAY VALIDATION RESULTS
# --------------------------------------------------

def display_source_validation(sources):
    print("\n========== SOURCE VALIDATION ==========")

    if not sources:
        print("No sources available for validation.")
        return

    for index, source in enumerate(sources, start=1):
        print(f"\nSOURCE {index}")

        print("Title:", source.get("Title", ""))
        print("Domain:", source.get("Domain", ""))
        print("URL:", source.get("URL", ""))
        print("Source Type:", source.get("Source_Type", ""))
        print(
            "Credibility Category:",
            source.get("Credibility_Category", "")
        )
        print(
            "Classification Basis:",
            source.get("Classification_Basis", "")
        )

    print("\nNote: Domain classification does not verify article accuracy.")
    print("=======================================")


# --------------------------------------------------
# 6. STANDALONE TEST
# --------------------------------------------------

if __name__ == "__main__":
    test_sources = [
        {
            "Title": "PIB Fact Check",
            "URL": "https://pib.gov.in/"
        },
        {
            "Title": "Alt News",
            "URL": "https://www.altnews.in/"
        },
        {
            "Title": "The Hindu",
            "URL": "https://www.thehindu.com/"
        },
        {
            "Title": "India Today",
            "URL": "https://www.indiatoday.in/"
        },
        {
            "Title": "Unknown Website",
            "URL": "https://example.com/article"
        },
        {
            "Title": "Domain without scheme",
            "URL": "boomlive.in"
        },
        {
            "Title": "Lookalike domain",
            "URL": "https://fakeboomlive.in/"
        }
    ]

    validated = validate_sources(test_sources)
    display_source_validation(validated)
