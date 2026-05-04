import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError, RequestError

from src.config import get_settings
from .index_config import INDEX_NAME, INDEX_SETTINGS
from .query_builder import PaperQueryBuilder

logger = logging.getLogger(__name__)


class OpenSearchClient:
    """
    Service layer for all OpenSearch operations.

    WHY a class instead of module-level functions?
    ──────────────────────────────────────────────
    1. STATE: The client holds configuration (host, index name, max_text_size)
       that would otherwise be threaded through every function as arguments.

    2. LIFECYCLE: One instance is created at startup and shared across requests,
       avoiding connection overhead on every call. The raw opensearchpy.OpenSearch
       connection pool lives inside self.client and is reused.

    3. TESTABILITY: You can inject a mock OpenSearchClient in tests without
       monkey-patching module-level functions — just swap the object.

    4. DISCOVERABILITY: All OpenSearch operations live in one class. When a new
       developer asks "how do I check if OpenSearch is healthy?", they look here:
       client.health_check(). Compare to grepping across 10 function files.

    The underlying raw opensearchpy.OpenSearch object (self.client) is kept as
    an implementation detail. External code should call methods on THIS class,
    not reach into self.client directly.
    """

    def __init__(self, host: Optional[str] = None):
        """
        :param host: OpenSearch endpoint URL (e.g. "http://localhost:9200").
                     Reads from config if not provided.
        """
        settings = get_settings()
        self.host = host or settings.opensearch.host
        self.index_name = settings.opensearch.index_name
        self.max_text_size = settings.opensearch.max_text_size

        # The low-level opensearchpy connection.
        # http_compress=True:        gzip-compress request bodies → less network I/O
        # use_ssl=False:             local/dev OpenSearch has no TLS
        # ssl_assert_hostname=False: skip hostname verification (dev only)
        # ssl_show_warn=False:       suppress SSL warning logs in dev
        self.client = OpenSearch(
            hosts=[self.host],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )
        logger.info(f"OpenSearch client initialized: {self.host} (index: {self.index_name})")

    # ──────────────────────────────────────────────────────────────────────────
    # Index management
    # ──────────────────────────────────────────────────────────────────────────

    def create_index(self, force: bool = False) -> bool:
        """
        Create the index with the mapping defined in index_config.py.

        The mapping tells OpenSearch:
          - which fields exist and their types (text, keyword, date, knn_vector)
          - how to analyze/tokenize text fields (which analyzer to use)
          - BM25 tuning parameters (k1, b)

        This method is IDEMPOTENT when force=False: calling it multiple times
        on an already-created index is safe — it just returns False.

        :param force: If True, DELETE the existing index first, then recreate.
                      DANGEROUS in production — you lose all indexed documents.
                      Useful in development when you change the mapping schema.
                      In production, use OpenSearch's reindex API instead.
        :returns: True if the index was just created, False if it already existed.
        """
        try:
            exists = self.client.indices.exists(index=self.index_name)

            if exists:
                if force:
                    logger.warning(f"force=True: deleting existing index '{self.index_name}'")
                    self.client.indices.delete(index=self.index_name)
                else:
                    logger.info(f"Index '{self.index_name}' already exists — skipping creation")
                    return False

            response = self.client.indices.create(index=self.index_name, body=INDEX_SETTINGS)

            if response.get("acknowledged"):
                logger.info(f"Created index '{self.index_name}' successfully")
                return True
            else:
                logger.error(f"Index creation not acknowledged by cluster: {response}")
                return False

        except RequestError as e:
            # RequestError covers bad mapping syntax, illegal argument, etc.
            logger.error(f"RequestError creating index: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error creating index: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # Document indexing
    # ──────────────────────────────────────────────────────────────────────────

    def index_paper(self, paper_data: Dict[str, Any]) -> bool:
        """
        Index (upsert) a single document into OpenSearch.

        "Upsert" = insert if new, update if already exists.
        OpenSearch's index() operation always uses the document ID to
        decide: if a doc with that ID already exists, replace it entirely.
        This means re-running the pipeline never creates duplicate documents.

        Document ID resolution:
        - Production Paper model uses "arxiv_id" as the identifier field
        - rag-assistant Document model uses "id" (which IS the arXiv ID)
        We support both so this class works with either model.

        :param paper_data: Dict with document fields. Must contain "id" or "arxiv_id".
        :returns: True if indexed successfully, False otherwise.
        """
        try:
            # Support both field naming conventions
            doc_id = paper_data.get("arxiv_id") or paper_data.get("id")
            if not doc_id:
                logger.error("Cannot index: paper_data missing both 'id' and 'arxiv_id'")
                return False

            # Inject timestamps if the caller didn't provide them.
            # setdefault() only sets the key if it doesn't already exist.
            now = datetime.now(timezone.utc).isoformat()
            paper_data.setdefault("created_at", now)
            paper_data.setdefault("updated_at", now)

            # NOTE: we intentionally do NOT join authors into a single string.
            # OpenSearch text fields natively support arrays — ["Vaswani", "Shazeer"]
            # is indexed identically to "Vaswani Shazeer" for BM25 purposes.
            # Keeping the list means search results return authors as a list,
            # which matches the SearchResult.authors: list schema in the API.

            # Truncate full_text to the configured limit.
            # Docling can produce 300k+ character strings for long papers.
            # OpenSearch has a default 10MB per document limit, but storing
            # huge strings also inflates index size and slows search.
            if paper_data.get("full_text"):
                paper_data["full_text"] = paper_data["full_text"][: self.max_text_size]

            response = self.client.index(
                index=self.index_name,
                id=doc_id,
                body=paper_data,
                # refresh=True: make this document searchable IMMEDIATELY after indexing.
                # The cost: a small extra I/O per index call (flushes the write buffer).
                # For our volumes (tens of papers/day), the cost is negligible.
                # High-throughput systems use refresh=False and rely on OpenSearch's
                # default 1-second auto-refresh interval instead.
                refresh=True,
            )

            if response.get("result") in ["created", "updated"]:
                logger.debug(f"Indexed document '{doc_id}'")
                return True
            else:
                logger.error(f"Unexpected index response for '{doc_id}': {response}")
                return False

        except Exception as e:
            doc_id_str = paper_data.get("arxiv_id") or paper_data.get("id", "?")
            logger.error(f"Error indexing '{doc_id_str}': {e}")
            return False

    def bulk_index_papers(self, papers: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Index a list of papers, collecting per-document success/failure counts.

        WHY not use OpenSearch's native bulk API (helpers.bulk)?
        The native bulk API requires a specific action/document envelope format:
            [{"index": {"_id": "..."}}, {<doc>}, ...]
        For our volumes (tens to hundreds of papers per day), the overhead of
        calling index() per document is negligible. We gain simpler code and
        per-document error isolation: one bad document doesn't abort the whole batch.

        If you were indexing millions of documents, switch to helpers.bulk() from
        the opensearch-py helpers module for 10x+ throughput improvement.

        :returns: {"success": int, "failed": int}
        """
        results = {"success": 0, "failed": 0}

        for paper in papers:
            if self.index_paper(paper):
                results["success"] += 1
            else:
                results["failed"] += 1

        logger.info(
            f"Bulk indexing complete: {results['success']} indexed, "
            f"{results['failed']} failed"
        )
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Search
    # ──────────────────────────────────────────────────────────────────────────

    def search_papers(
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
    ) -> Dict[str, Any]:
        """
        Execute a BM25 search using PaperQueryBuilder to construct the query.

        This method bridges two layers:
          - Caller thinks in business terms: "search for 'attention', limit 5"
          - OpenSearch needs a 50-line Query DSL JSON body

        PaperQueryBuilder handles the translation. This class handles the
        execution and response normalization.

        :returns: {
            "total":   total matching documents (for pagination),
            "results": [ {<source fields>, "score": float, "highlight": {...}} ],
            "query":   the original query string (for UI display)
          }
        """
        try:
            query_body = PaperQueryBuilder(
                query=query,
                size=size,
                from_=from_,
                fields=fields,
                categories=categories,
                source_filter=source_filter,
                date_from=date_from,
                date_to=date_to,
                track_total_hits=track_total_hits,
                latest_papers=latest_papers,
            ).build()

            response = self.client.search(index=self.index_name, body=query_body)

            total = response["hits"]["total"]["value"]
            results = []
            for hit in response["hits"]["hits"]:
                result = {
                    **hit["_source"],                  # all stored fields
                    "score": hit["_score"],            # BM25 relevance score
                    "highlight": hit.get("highlight", {}),  # matched snippets
                }
                results.append(result)

            logger.info(f"Search '{query}' → {total} total, {len(results)} returned")
            return {"total": total, "results": results, "query": query}

        except NotFoundError:
            # Index doesn't exist yet — return empty results rather than 500 error.
            # This happens on a fresh cluster before the first ingestion run.
            logger.warning(f"Index '{self.index_name}' not found — run ingestion first")
            return {"total": 0, "results": [], "query": query}
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {"total": 0, "results": [], "query": query, "error": str(e)}

    # ──────────────────────────────────────────────────────────────────────────
    # Cluster & index introspection
    # ──────────────────────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """
        Return True if OpenSearch is accepting requests.

        Cluster status meanings:
          green:  all primary AND replica shards assigned → fully healthy
          yellow: all primary shards assigned, some replicas unassigned →
                  acceptable for a single-node dev cluster (we set 0 replicas,
                  so the cluster will never reach green with one node)
          red:    some primary shards unassigned → data loss or cluster down

        We accept both green and yellow because our dev cluster will
        always be yellow (single node, no replicas to assign).
        """
        try:
            health = self.client.cluster.health()
            is_healthy = health["status"] in ["green", "yellow"]
            logger.debug(f"Cluster health: {health['status']} → {'OK' if is_healthy else 'UNHEALTHY'}")
            return is_healthy
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Return basic statistics about the index.

        Used by:
          - The Airflow report task: "how many total documents are indexed?"
          - The /search/health API endpoint: display doc count to operators
          - Startup logging: confirm the index was created successfully

        :returns: {
            "index_name":      str,
            "document_count":  int,
            "size_in_bytes":   int,
            "health":          "green" | "yellow" | "red"
          }
          OR {"error": str} if the call failed.
        """
        try:
            stats = self.client.indices.stats(index=self.index_name)
            count = self.client.count(index=self.index_name)
            health = self.client.cluster.health(index=self.index_name)

            return {
                "index_name": self.index_name,
                "document_count": count["count"],
                "size_in_bytes": stats["indices"][self.index_name]["total"]["store"]["size_in_bytes"],
                "health": health["status"],
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"error": str(e)}

    def get_cluster_health(self) -> Optional[Dict[str, Any]]:
        """Return the raw cluster health response — useful for debugging."""
        try:
            return self.client.cluster.health()
        except Exception as e:
            logger.error(f"Error fetching cluster health: {e}")
            return None
