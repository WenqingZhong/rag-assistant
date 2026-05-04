import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PaperQueryBuilder:
    """
    Translates high-level search intent into an OpenSearch Query DSL dict.

    WHY a builder class instead of building the query dict inline?
    ─────────────────────────────────────────────────────────────
    Even a "simple" BM25 search over multiple fields with filters, sorting,
    and highlighting is 40-60 lines of nested dicts. Inlining that inside
    every search function mixes two concerns:
      - WHAT to search for   (the caller's job: query="transformers", size=5)
      - HOW to express that  (the builder's job: multi_match, bool, filter, ...)

    Separating them means:
    1. You can unit-test query construction without a live OpenSearch cluster.
    2. Different callers (API endpoint, Airflow task, notebook) all produce
       exactly the same query shape for the same parameters.
    3. Adding a new option (e.g. a date filter) only requires changing one place.

    Usage:
        body = PaperQueryBuilder(query="attention", size=5).build()
        response = opensearch_client.search(index=index_name, body=body)
    """

    # Default fields to search and their relevance boost multipliers.
    # "title^3" means a title match scores 3× higher than an authors match.
    # This reflects real intuition: if "transformer" appears in the title,
    # that paper is almost certainly relevant; if it only appears in the
    # authors field (a name coincidence), much less so.
    DEFAULT_FIELDS = ["title^3", "abstract^2", "authors^1"]

    def __init__(
        self,
        query: str,
        size: int = 10,
        from_: int = 0,
        fields: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        source_filter: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        track_total_hits: bool = True,
        latest_papers: bool = False,
    ):
        """
        :param query: Free-text search string entered by the user.
        :param size: Max results to return (SQL equivalent: LIMIT).
        :param from_: Result offset for pagination (SQL equivalent: OFFSET).
                      Page 2 of 10 results per page → from_=10.
        :param fields: Which fields to search, with optional boost multipliers
                       (e.g. ["title^3", "abstract^2"]). Defaults to DEFAULT_FIELDS.
        :param categories: arXiv category filter (e.g. ["cs.AI", "cs.LG"]).
                           Applies as an OR-match: any paper in any listed category passes.
        :param source_filter: Filter by document source field (e.g. "arxiv", "legal").
                              Exact keyword match — not scored.
        :param date_from: ISO date lower bound for published_date (e.g. "2024-01-01").
        :param date_to: ISO date upper bound for published_date (e.g. "2024-12-31").
        :param track_total_hits: If True, OpenSearch reports the exact total match count.
                                 Slightly slower but required for "Showing 1–10 of 4,532".
                                 OpenSearch defaults to stopping at 10,000 if False.
        :param latest_papers: Sort by published_date DESC instead of BM25 relevance.
                              Useful for "show me the newest papers" mode.
        """
        self.query = query
        self.size = size
        self.from_ = from_
        self.fields = fields or self.DEFAULT_FIELDS
        self.categories = categories
        self.source_filter = source_filter
        self.date_from = date_from
        self.date_to = date_to
        self.track_total_hits = track_total_hits
        self.latest_papers = latest_papers

    def build(self) -> Dict[str, Any]:
        """
        Assemble the complete OpenSearch request body dict.

        The returned dict is passed directly as the `body` argument to
        opensearch_client.search(index=..., body=<this>).

        Structure overview:
            {
              "query":            { "bool": { "must": [...], "filter": [...] } },
              "size":             10,
              "from":             0,
              "track_total_hits": True,
              "_source":          ["id", "title", ...],   # fields to return
              "highlight":        { "fields": {...} },    # snippet highlighting
              "sort":             [...]                   # omitted for relevance order
            }
        """
        body: Dict[str, Any] = {
            "query": self._build_query(),
            "size": self.size,
            "from": self.from_,
            "track_total_hits": self.track_total_hits,
            "_source": self._build_source_fields(),
            "highlight": self._build_highlight(),
        }

        sort = self._build_sort()
        if sort:
            body["sort"] = sort

        return body

    def _build_query(self) -> Dict[str, Any]:
        """
        Build the bool query — the outermost query wrapper.

        OpenSearch bool query has four clause types:
          must     → clauses MUST match AND affect the relevance score
          filter   → clauses MUST match but do NOT affect the score
          should   → clauses SHOULD match (boost score if they do, not required)
          must_not → clauses MUST NOT match

        We use:
          "must"   for the full-text search  → affects BM25 score
          "filter" for source/category/date  → binary yes/no, not scored

        WHY separate must vs filter?
        Filters are cached by OpenSearch and execute much faster than scored
        clauses. A date range filter in "filter" context: OpenSearch caches
        the matching document set and reuses it across queries. If it were in
        "must", it would be re-evaluated and scored every time.
        """
        must_clauses = []

        if self.query.strip():
            must_clauses.append(self._build_text_query())

        filter_clauses = self._build_filters()

        bool_query: Dict[str, Any] = {}

        if must_clauses:
            bool_query["must"] = must_clauses
        else:
            # No text query → match every document.
            # Used for "browse all papers" mode (empty search bar).
            bool_query["must"] = [{"match_all": {}}]

        if filter_clauses:
            bool_query["filter"] = filter_clauses

        return {"bool": bool_query}

    def _build_text_query(self) -> Dict[str, Any]:
        """
        Build the multi_match query for BM25 full-text scoring.

        multi_match searches multiple fields in one clause, applying
        the boost multipliers from self.fields.

        "type": "best_fields"
            Score by the single best-matching field. Good when documents have
            one "primary" field (title) and the rest are supplementary.
            Alternative "cross_fields" scores as if all fields were one big field.

        "operator": "or"
            A document qualifies if ANY search term matches (more recall).
            "and" would require ALL terms to match (more precision, fewer results).

        "fuzziness": "AUTO"
            Tolerates typos automatically based on word length:
              ≤ 2 chars: no fuzziness (exact match required)
              3–5 chars: 1 character edit allowed
              > 5 chars: 2 character edits allowed

        "prefix_length": 2
            The first 2 characters must always match exactly.
            Without this, "cat" could fuzzy-match "bat" — too aggressive.
        """
        return {
            "multi_match": {
                "query": self.query,
                "fields": self.fields,
                "type": "best_fields",
                "operator": "or",
                "fuzziness": "AUTO",
                "prefix_length": 2,
            }
        }

    def _build_filters(self) -> List[Dict[str, Any]]:
        """
        Build filter clauses — hard constraints that don't influence scoring.

        "term":  Exact keyword match. Use for enum-like fields (source, status).
                 Works on "keyword" type fields — not analyzed/tokenized.
        "terms": Like "term" but matches any value in a list (OR logic).
                 e.g. {"terms": {"categories": ["cs.AI", "cs.LG"]}}
        "range": Numeric or date range with gte/lte/gt/lt operators.

        WHY filters don't affect score:
        Score reflects how well a document matches the user's QUERY INTENT.
        A date filter is just a boundary — a paper from 2024 isn't inherently
        more relevant than one from 2023 just because we're filtering for 2024.
        Keeping filters out of scoring keeps BM25 meaningful.
        """
        filters: List[Dict[str, Any]] = []

        # arXiv category filter: matches any paper in any of the listed categories.
        # "terms" = OR logic across the list.
        if self.categories:
            filters.append({"terms": {"categories": self.categories}})

        # Source filter: exact keyword match on the "source" field.
        # In rag-assistant: "arxiv", "legal", "upload", etc.
        if self.source_filter:
            filters.append({"term": {"source": self.source_filter}})

        # Date range filter on the published_date field.
        # "gte" = ≥ from_date,  "lte" = ≤ to_date
        # Either bound can be omitted for open-ended ranges.
        if self.date_from or self.date_to:
            date_range: Dict[str, str] = {}
            if self.date_from:
                date_range["gte"] = self.date_from
            if self.date_to:
                date_range["lte"] = self.date_to
            filters.append({"range": {"published_date": date_range}})

        return filters

    def _build_source_fields(self) -> List[str]:
        """
        Specify which fields to include in each search result hit.

        We explicitly EXCLUDE "full_text" here. A single paper's full_text
        can be 100k+ characters. Returning it in every search result would:
          1. Make search responses huge (slow network transfer)
          2. Make Airflow XCom serialization expensive
          3. Make UI response times noticeably slow

        Callers fetch full_text separately when the user clicks a specific paper.
        """
        return ["id", "title", "authors", "abstract", "published_date", "source"]

    def _build_highlight(self) -> Dict[str, Any]:
        """
        Configure snippet highlighting — wraps matched terms in HTML tags.

        Without highlighting, search results show raw text and users must
        mentally locate why their query matched.

        With highlighting:
          "...the <mark>transformer</mark> attention mechanism reduces..."

        fragment_size:       max characters per highlighted snippet
        number_of_fragments: how many non-contiguous snippets to return
        pre_tags/post_tags:  HTML to wrap matched terms with

        For title: fragment_size=0, number_of_fragments=0 means return the
        ENTIRE title field un-fragmented (titles are short, always show all).

        For abstract: up to 3 fragments of 150 chars each — shows context
        around each match without returning the full abstract every time.

        "require_field_match": False
            If True: only highlight fields you explicitly searched.
            If False: highlight matching terms in ALL configured fields,
            even if the query matched a different field.
            We use False so abstract highlights appear even when the query
            matched primarily in the title.
        """
        return {
            "fields": {
                "title": {
                    "fragment_size": 0,
                    "number_of_fragments": 0,
                },
                "abstract": {
                    "fragment_size": 150,
                    "number_of_fragments": 3,
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"],
                },
                "authors": {
                    "fragment_size": 0,
                    "number_of_fragments": 0,
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"],
                },
            },
            "require_field_match": False,
        }

    def _build_sort(self) -> Optional[List[Dict[str, Any]]]:
        """
        Build sorting configuration.

        Default (return None): no explicit sort → OpenSearch sorts by _score
        (BM25 relevance). The most relevant document comes first.

        latest_papers=True: sort by published_date DESC. The newest paper comes
        first regardless of BM25 score. "_score" as secondary sort breaks ties.

        Empty query + latest_papers=False: no text to score, so sort by date.
        This is the "browse all papers" case — newest first makes sense.

        NOTE: When you specify an explicit sort, _score is NOT computed by
        default (optimization). Adding "_score" as a secondary sort key forces
        OpenSearch to compute scores anyway so they're available in results.
        """
        if self.latest_papers:
            return [{"published_date": {"order": "desc"}}, "_score"]

        # Text query present → relevance order (let OpenSearch score and sort)
        if self.query.strip():
            return None

        # Empty query (browse mode) → newest first
        return [{"published_date": {"order": "desc"}}, "_score"]


def build_search_query(
    query: str,
    size: int = 10,
    from_: int = 0,
    categories: Optional[List[str]] = None,
    source_filter: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function: build a query dict without instantiating the class directly.
    Useful in scripts, notebooks, or tests where you want a one-liner.
    """
    builder = PaperQueryBuilder(
        query=query,
        size=size,
        from_=from_,
        categories=categories,
        source_filter=source_filter,
        date_from=date_from,
        date_to=date_to,
    )
    return builder.build()
