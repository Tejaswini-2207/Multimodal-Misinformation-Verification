
import re
from itertools import combinations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# CONFIGURATION
# ==========================================

DUPLICATE_THRESHOLD = 0.75


# ==========================================
# TEXT CLEANING
# ==========================================

def clean_article_text(text):

    if not text:
        return ""

    text = str(text).lower()

    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    return text.strip()


# ==========================================
# SOURCE FIELD HANDLING
# ==========================================

def get_source_value(source, *keys, default=""):

    for key in keys:

        value = source.get(key)

        if value is not None and str(value).strip():
            return value

    return default


# ==========================================
# SOURCE SIMILARITY ANALYSIS
# ==========================================

def analyze_source_similarity(sources):

    if not sources:

        return {
            "unique_sources": 0,
            "duplicate_groups": [],
            "source_relationships": [],
            "independent_source_count": 0
        }

    documents = []

    for source in sources:

        article_text = get_source_value(
            source,
            "Article_Text",
            "article_text"
        )

        snippet = get_source_value(
            source,
            "Snippet",
            "snippet"
        )

        content = article_text or snippet

        documents.append(
            clean_article_text(content)
        )

    valid_indices = [
        index
        for index, document in enumerate(documents)
        if document
    ]

    if len(valid_indices) <= 1:

        return {
            "unique_sources": len(sources),
            "duplicate_groups": [],
            "source_relationships": [],
            "independent_source_count": len(valid_indices)
        }

    valid_documents = [
        documents[index]
        for index in valid_indices
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(
        valid_documents
    )

    similarity_matrix = cosine_similarity(matrix)

    relationships = []
    duplicate_pairs = []

    for i, j in combinations(
        range(len(valid_indices)),
        2
    ):

        score = float(similarity_matrix[i, j])

        original_i = valid_indices[i]
        original_j = valid_indices[j]

        source_a = sources[original_i]
        source_b = sources[original_j]

        domain_a = get_source_value(
            source_a,
            "Domain",
            "domain",
            default="Unknown"
        )

        domain_b = get_source_value(
            source_b,
            "Domain",
            "domain",
            default="Unknown"
        )

        relationship = {

            "source_1": domain_a,
            "source_2": domain_b,
            "similarity": round(score, 4),

            "likely_related": (
                score >= DUPLICATE_THRESHOLD
            )
        }

        relationships.append(relationship)

        if score >= DUPLICATE_THRESHOLD:

            duplicate_pairs.append(
                (original_i, original_j)
            )

    # Build similarity clusters.
    parent = list(range(len(sources)))

    def find(x):

        while parent[x] != x:

            parent[x] = parent[parent[x]]
            x = parent[x]

        return x

    def union(a, b):

        root_a = find(a)
        root_b = find(b)

        if root_a != root_b:

            parent[root_b] = root_a

    for i, j in duplicate_pairs:

        union(i, j)

    groups = {}

    for index in valid_indices:

        root = find(index)

        groups.setdefault(
            root,
            []
        ).append(index)

    duplicate_groups = []

    for indices in groups.values():

        if len(indices) < 2:
            continue

        group = []

        for index in indices:

            source = sources[index]

            group.append({

                "domain": get_source_value(
                    source,
                    "Domain",
                    "domain",
                    default="Unknown"
                ),

                "url": get_source_value(
                    source,
                    "URL",
                    "url"
                )
            })

        duplicate_groups.append(group)

    independent_source_count = len({
        find(index)
        for index in valid_indices
    })

    return {

        "unique_sources": len(sources),

        "duplicate_groups": duplicate_groups,

        "source_relationships": relationships,

        "independent_source_count": (
            independent_source_count
        )
    }


# ==========================================
# DISPLAY RESULTS
# ==========================================

def display_source_similarity(report):

    print("\n========== SOURCE SIMILARITY ==========")

    print(
        "Sources received:",
        report["unique_sources"]
    )

    print(
        "Estimated source clusters:",
        report["independent_source_count"]
    )

    print("\nPairwise comparisons:")

    for item in report["source_relationships"]:

        print(
            f"{item['source_1']} <-> "
            f"{item['source_2']}: "
            f"{item['similarity']}"
        )

        if item["likely_related"]:

            print(
                "WARNING: High textual similarity. "
                "Review these sources."
            )

    if report["duplicate_groups"]:

        print("\nLikely related source groups:")

        for number, group in enumerate(
            report["duplicate_groups"],
            start=1
        ):

            print(f"\nGroup {number}:")

            for source in group:

                print(
                    source["domain"],
                    "|",
                    source["url"]
                )

    else:

        print(
            "\nNo highly similar article pairs detected."
        )

    print(
        "\nNote: Similarity is a textual heuristic. "
        "It does not prove source independence."
    )


# ==========================================
# STANDALONE TEST
# ==========================================

if __name__ == "__main__":

    demo_sources = [

        {
            "Title": "Example Article A",
            "Domain": "example-a.com",
            "URL": "https://example-a.com/article",
            "Snippet": "",
            "Article_Text": (
                "The government fact check unit confirmed "
                "that the viral resignation video was "
                "artificially generated."
            )
        },

        {
            "Title": "Example Article B",
            "Domain": "example-b.com",
            "URL": "https://example-b.com/article",
            "Snippet": "",
            "Article_Text": (
                "The government fact check unit confirmed "
                "that the viral resignation video was "
                "artificially generated."
            )
        },

        {
            "Title": "Example Article C",
            "Domain": "example-c.com",
            "URL": "https://example-c.com/article",
            "Snippet": "",
            "Article_Text": (
                "The weather department issued a heavy "
                "rainfall warning for the region."
            )
        }
    ]

    report = analyze_source_similarity(
        demo_sources
    )

    display_source_similarity(report)
