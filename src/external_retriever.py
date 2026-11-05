
import re
import time
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

from ddgs import DDGS
from trafilatura import fetch_url, extract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ExternalEvidenceRetriever:

    def __init__(self):
        self.RELEVANCE_THRESHOLD = 0.12
        self.MAX_ARTICLE_CHARS = 10000
        self.MAX_QUERIES = 5
        self.REQUEST_DELAY = 0.5

        self.stop_words = {
            "the", "is", "are", "was", "were", "a", "an",
            "and", "or", "of", "to", "in", "on", "for",
            "with", "by", "under", "from", "that", "this",
            "it", "as", "at", "be", "has", "have", "had",
            "allegedly", "claim", "claims", "claimed",
            "according", "report", "reports", "shows",
            "show", "social", "media", "circulating",
            "image", "viral", "photo", "picture", "video",
            "post", "says", "said", "fact", "check",
            "misinformation", "debunked", "together"
        }

    # --------------------------------------------------
    # TEXT CLEANING
    # --------------------------------------------------

    def clean_text(self, text):

        if text is None:
            return ""

        text = str(text).lower()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def extract_keywords(self, text):

        words = self.clean_text(text).split()

        return {
            word for word in words
            if len(word) > 2 and word not in self.stop_words
        }

    # --------------------------------------------------
    # SEARCH QUERY GENERATION
    # --------------------------------------------------

    def generate_search_queries(self, claim):

        claim = str(claim or "").strip()

        if not claim:
            return []

        clean_claim = self.clean_text(claim)

        words = [
            word for word in clean_claim.split()
            if len(word) > 2 and word not in self.stop_words
        ]

        if not words:
            return []

        # Find meaningful capitalized names and phrases
        named_phrases = re.findall(
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b",
            claim
        )

        named_phrases = list(dict.fromkeys(named_phrases))

        # Keep the most useful keywords
        core_keywords = words[:10]
        core = " ".join(core_keywords)

        queries = []

        # Query using names/entities
        if named_phrases:

            names = " ".join(named_phrases[:4])

            queries.append(
                f"{names} fact check"
            )

            queries.append(
                f"{names} viral image false claim"
            )

        # General fact-check query
        queries.append(
            f"{core} fact check"
        )

        # Search for debunking or manipulation
        queries.append(
            f"{core} fake misleading claim debunked"
        )

        # Search for official confirmation or denial
        queries.append(
            f"{core} official statement verification"
        )

        # Remove duplicate queries
        unique_queries = []
        seen = set()

        for query in queries:

            query = re.sub(r"\s+", " ", query).strip()
            key = query.lower()

            if key not in seen and query:
                seen.add(key)
                unique_queries.append(query)

        return unique_queries[:self.MAX_QUERIES]

    # --------------------------------------------------
    # URL NORMALIZATION
    # --------------------------------------------------

    def normalize_url(self, url):

        try:

            if not url:
                return ""

            url = str(url).strip()

            if "://" not in url:
                url = "https://" + url

            parsed = urlparse(url)

            if not parsed.hostname:
                return ""

            if parsed.scheme not in ("http", "https"):
                return ""

            host = parsed.hostname.lower()

            if host.startswith("www."):
                host = host[4:]

            tracking_parameters = {
                "fbclid", "gclid", "igshid",
                "ref", "ref_src"
            }

            query_items = []

            for key, value in parse_qsl(
                parsed.query,
                keep_blank_values=True
            ):

                if key.lower() in tracking_parameters:
                    continue

                if key.lower().startswith("utm_"):
                    continue

                query_items.append((key, value))

            normalized = urlunparse((
                parsed.scheme.lower(),
                host,
                parsed.path.rstrip("/") or "/",
                "",
                urlencode(query_items, doseq=True),
                ""
            ))

            return normalized

        except Exception:
            return ""

    # --------------------------------------------------
    # DOMAIN EXTRACTION
    # --------------------------------------------------

    def get_domain(self, url):

        try:

            domain = urlparse(url).hostname or ""

            domain = domain.lower()

            if domain.startswith("www."):
                domain = domain[4:]

            return domain

        except Exception:
            return ""

    # --------------------------------------------------
    # EXTERNAL WEB SEARCH
    # --------------------------------------------------

    def search_web(self, query, max_results=8):

        try:

            print(f"\nSearching: {query}")

            with DDGS() as ddgs:

                raw_results = list(
                    ddgs.text(
                        query,
                        max_results=max_results
                    )
                )

            results = []

            for item in raw_results or []:

                if not isinstance(item, dict):
                    continue

                title = str(
                    item.get("title") or ""
                ).strip()

                url = self.normalize_url(
                    item.get("href") or ""
                )

                snippet = str(
                    item.get("body") or ""
                ).strip()

                if not title or not url:
                    continue

                # Remove search/category pages
                excluded_paths = (
                    "/search",
                    "/tag/",
                    "/category/",
                    "/author/"
                )

                if any(
                    path in url.lower()
                    for path in excluded_paths
                ):
                    continue

                results.append({
                    "title": title,
                    "href": url,
                    "body": snippet
                })

            print(
                "Valid search results:",
                len(results)
            )

            return results

        except Exception as error:

            print(
                f"SEARCH ERROR ({type(error).__name__}): {error}"
            )

            return []

    # --------------------------------------------------
    # ARTICLE EXTRACTION
    # --------------------------------------------------

    def extract_article(self, url):

        try:

            downloaded = fetch_url(url)

            if not downloaded:
                return "", "FAILED"

            article_text = extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                favor_precision=True,
                deduplicate=True
            )

            if not article_text:
                return "", "FAILED"

            article_text = re.sub(
                r"\n{3,}",
                "\n\n",
                article_text
            ).strip()

            if len(article_text) < 120:
                return "", "FAILED"

            return (
                article_text[:self.MAX_ARTICLE_CHARS],
                "SUCCESS"
            )

        except Exception as error:

            print(
                f"Article extraction failed: {error}"
            )

            return "", "FAILED"

    # --------------------------------------------------
    # RELEVANCE CALCULATION
    # --------------------------------------------------

    def calculate_relevance(
        self,
        claim,
        title,
        snippet,
        article_text
    ):

        claim_text = self.clean_text(claim)

        title_snippet = self.clean_text(
            f"{title} {snippet}"
        )

        article = self.clean_text(
            article_text or ""
        )

        if not claim_text:
            return 0.0, 0.0

        candidates = []

        if title_snippet:
            candidates.append(title_snippet)

        if article:
            candidates.append(
                article[:self.MAX_ARTICLE_CHARS]
            )

        if not candidates:
            return 0.0, 0.0

        try:

            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True
            )

            matrix = vectorizer.fit_transform(
                [claim_text] + candidates
            )

            similarities = cosine_similarity(
                matrix[0:1],
                matrix[1:]
            ).flatten()

            title_score = float(
                similarities[0]
            )

            body_score = (
                float(similarities[1])
                if len(similarities) > 1
                else title_score
            )

            if article:
                textual_score = (
                    0.35 * title_score
                    + 0.65 * body_score
                )
            else:
                textual_score = title_score

            claim_words = self.extract_keywords(
                claim
            )

            candidate_words = self.extract_keywords(
                f"{title} {snippet} {article}"
            )

            if claim_words:

                overlap = (
                    len(claim_words & candidate_words)
                    / len(claim_words)
                )

            else:
                overlap = 0.0

            final_score = (
                0.75 * textual_score
                + 0.25 * overlap
            )

            return (
                round(final_score, 4),
                round(overlap, 4)
            )

        except (ValueError, TypeError) as error:

            print(
                "Relevance calculation error:",
                error
            )

            return 0.0, 0.0

    # --------------------------------------------------
    # MAIN EVIDENCE RETRIEVAL
    # --------------------------------------------------

    def retrieve(self, claim, max_results=5):

        print(
            "\n========== EXTERNAL EVIDENCE RETRIEVAL =========="
        )

        claim = str(claim or "").strip()

        if not claim:
            print("Empty claim. Search skipped.")
            return []

        queries = self.generate_search_queries(
            claim
        )

        print("\nGenerated search queries:")

        for query in queries:
            print("-", query)

        # Collect unique URLs
        collected = {}

        for query in queries:

            search_results = self.search_web(
                query,
                max_results=8
            )

            for result in search_results:

                url = result["href"]

                if url not in collected:

                    collected[url] = {
                        "Title": result["title"],
                        "URL": url,
                        "Snippet": result["body"],
                        "Search_Queries": [query]
                    }

                else:

                    current = collected[url]

                    if len(result["body"]) > len(
                        current["Snippet"]
                    ):
                        current["Snippet"] = result["body"]

                    if query not in current["Search_Queries"]:
                        current["Search_Queries"].append(query)

            time.sleep(self.REQUEST_DELAY)

        if not collected:

            print(
                "\nNo valid external search results returned."
            )

            return []

        print(
            "\nUnique URLs collected:",
            len(collected)
        )

        results = []

        for item in collected.values():

            domain = self.get_domain(
                item["URL"]
            )

            print(
                f"\nProcessing article: {item['Title']}"
            )

            print("Domain:", domain)

            article_text, extraction_status = (
                self.extract_article(item["URL"])
            )

            print(
                "Article extraction:",
                extraction_status
            )

            # Always retain snippet as fallback
            evidence_text = (
                article_text
                if article_text
                else item["Snippet"]
            )

            score, overlap = self.calculate_relevance(
                claim,
                item["Title"],
                item["Snippet"],
                evidence_text
            )

            results.append({

                "Title": item["Title"],

                "URL": item["URL"],

                "Domain": domain,

                "Snippet": item["Snippet"],

                "Article_Text": article_text,

                "Extraction_Status": extraction_status,

                "Claim_Relevance_Score": score,

                "Keyword_Overlap": overlap,

                "Search_Queries": item["Search_Queries"]

            })

        # Sort by relevance
        results.sort(
            key=lambda item: (
                item["Claim_Relevance_Score"],
                item["Keyword_Overlap"]
            ),
            reverse=True
        )

        relevant = [
            item
            for item in results
            if item["Claim_Relevance_Score"]
            >= self.RELEVANCE_THRESHOLD
        ][:max_results]

        print(
            "\n========== RETRIEVAL SUMMARY =========="
        )

        print(
            "Unique articles collected:",
            len(results)
        )

        print(
            "Relevant articles retained:",
            len(relevant)
        )

        print(
            "Filtered articles:",
            len(results) - len(relevant)
        )

        for item in results:

            status = (
                "RELEVANT"
                if item["Claim_Relevance_Score"]
                >= self.RELEVANCE_THRESHOLD
                else "FILTERED"
            )

            print(
                f"{status} | {item['Domain']} | "
                f"Score: {item['Claim_Relevance_Score']}"
            )

        print("========================================")

        return relevant

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    def display_results(self, results):

        if not results:

            print(
                "\nNo sufficiently relevant external articles found."
            )

            return

        for number, item in enumerate(
            results,
            start=1
        ):

            print(f"\nSOURCE {number}")

            print("Title:", item.get("Title", ""))

            print("Domain:", item.get("Domain", ""))

            print("URL:", item.get("URL", ""))

            print(
                "Relevance:",
                item.get("Claim_Relevance_Score", 0)
            )

            print(
                "Keyword overlap:",
                item.get("Keyword_Overlap", 0)
            )

            print(
                "Extraction status:",
                item.get("Extraction_Status", "")
            )

            evidence_text = (
                item.get("Article_Text")
                or item.get("Snippet")
                or ""
            )

            print(
                "Evidence text:",
                evidence_text[:1500]
            )

            print(
                "Search queries:",
                item.get("Search_Queries", [])
            )


# --------------------------------------------------
# STANDALONE TEST
# --------------------------------------------------

if __name__ == "__main__":

    retriever = ExternalEvidenceRetriever()

    test_claim = (
        "Free laptops are being offered under the "
        "Students Laptop Scheme 2026, and recipients "
        "are asked to share personal details."
    )

    evidence = retriever.retrieve(
        test_claim,
        max_results=5
    )

    retriever.display_results(evidence)
