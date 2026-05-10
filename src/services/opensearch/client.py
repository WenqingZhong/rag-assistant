import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError, RequestError

from src.config import get_settings
from .index_config import INDEX_NAME, INDEX_SETTINGS
from .index_config_hybrid import (
    ARXIV_PAPERS_CHUNKS_INDEX,
    ARXIV_PAPERS_CHUNKS_MAPPING,
    HYBRID_RRF_PIPELINE,
)
from .query_builder import PaperQueryBuilder, QueryBuilder

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
        # Separate index for chunk-level hybrid search (arxiv-papers-chunks).
        # Kept distinct from self.index_name (arxiv-papers) because chunk documents
        # and full-paper documents must not share the same BM25 term statistics.
        self.chunks_index_name = ARXIV_PAPERS_CHUNKS_INDEX

        logger.info(f"OpenSearch client initialized: {self.host} (index: {self.index_name}, chunks: {self.chunks_index_name})")

    # ──────────────────────────────────────────────────────────────────────────
    # Index management
    # ──────────────────────────────────────────────────────────────────────────

    def create_papers_index(self, force: bool = False) -> bool:
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

    def upsert_paper(self, paper_data: Dict[str, Any]) -> bool:
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

    def bulk_upsert_papers(self, papers: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Index a list of papers, collecting per-document success/failure counts.

        WHY not use OpenSearch's native bulk API (helpers.bulk)?
        The native bulk API requires a specific action/document envelope format:
            [{"index": {"_id": "..."}}, {<doc>}, ...]
        For our volumes (tens to hundreds of papers per day), the overhead of
        calling index() per document is negligible. We gain simpler code and
        per-document error isolation: one bad document doesn't abort the whole batch.

        If we were indexing millions of documents, switch to helpers.bulk() from
        the opensearch-py helpers module for 10x+ throughput improvement.

        :returns: {"success": int, "failed": int}
        """
        results = {"success": 0, "failed": 0}

        for paper in papers:
            if self.upsert_paper(paper):
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

    # ──────────────────────────────────────────────────────────────────────────
    # Hybrid search setup (chunks index + RRF pipeline)
    # ──────────────────────────────────────────────────────────────────────────

    def setup_hybrid_indices(self, force: bool = False) -> Dict[str, bool]:
        """
        One-shot setup: create the chunks index and register the RRF pipeline.

        Call this once on application startup (or from an Airflow setup task)
        before any chunks are indexed. Idempotent when force=False — safe to
        call repeatedly without losing data.

        :param force: If True, delete and recreate both the index and pipeline.
                      DESTRUCTIVE — you lose all indexed chunks. Dev-only.
        :returns: {"hybrid_index": True/False, "rrf_pipeline": True/False}
                  True = just created, False = already existed.
        """
        return {
            "hybrid_index": self._create_chunks_index(force),
            "rrf_pipeline": self._create_rrf_pipeline(force),
        }

    def _create_chunks_index(self, force: bool = False) -> bool:
        """
        Create the arxiv-papers-chunks KNN index using ARXIV_PAPERS_CHUNKS_MAPPING.

        WHY not reuse create_index()?
        create_index() is wired to self.index_name (arxiv-papers) and uses
        INDEX_SETTINGS (the paper mapping). The chunks index lives at
        self.chunks_index_name and needs a completely different mapping
        (knn_vector field, chunk_text instead of full_text, etc.).

        :returns: True if created, False if already existed.
        """
        try:
            exists = self.client.indices.exists(index=self.chunks_index_name)
            if exists:
                if force:
                    logger.warning(f"force=True: deleting chunks index '{self.chunks_index_name}'")
                    self.client.indices.delete(index=self.chunks_index_name)
                else:
                    logger.info(f"Chunks index '{self.chunks_index_name}' already exists — skipping")
                    return False

            self.client.indices.create(index=self.chunks_index_name, body=ARXIV_PAPERS_CHUNKS_MAPPING)
            logger.info(f"Created chunks index '{self.chunks_index_name}'")
            return True

        except Exception as e:
            logger.error(f"Error creating chunks index: {e}")
            raise

    def _create_rrf_pipeline(self, force: bool = False) -> bool:
        """
        Register the RRF search pipeline with OpenSearch.

        Search pipelines are NOT the same as ingest pipelines — they run
        at query time, not at index time. The opensearch-py client doesn't
        have a native method for search pipelines, so we use the raw
        transport.perform_request() to hit the /_search/pipeline/ endpoint.

        Once registered, any search request that passes
        params={"search_pipeline": "hybrid-rrf-pipeline"} will have its
        BM25 + KNN result lists merged using RRF automatically.

        :returns: True if created, False if already existed.
        """
        pipeline_id = HYBRID_RRF_PIPELINE["id"]

        try:
            # Check existence via GET — 404 means the pipeline doesn't exist yet.
            self.client.transport.perform_request("GET", f"/_search/pipeline/{pipeline_id}")
            if force:
                self.client.transport.perform_request("DELETE", f"/_search/pipeline/{pipeline_id}")
                logger.info(f"Deleted RRF pipeline '{pipeline_id}' for recreation")
            else:
                logger.info(f"RRF pipeline '{pipeline_id}' already exists — skipping")
                return False
        except Exception:
            pass  # 404 → pipeline doesn't exist yet, proceed to create

        try:
            pipeline_body = {
                "description": HYBRID_RRF_PIPELINE["description"],
                "phase_results_processors": HYBRID_RRF_PIPELINE["phase_results_processors"],
            }
            self.client.transport.perform_request(
                "PUT", f"/_search/pipeline/{pipeline_id}", body=pipeline_body
            )
            logger.info(f"Created RRF search pipeline '{pipeline_id}'")
            return True

        except Exception as e:
            logger.error(f"Error creating RRF pipeline: {e}")
            raise

    # ──────────────────────────────────────────────────────────────────────────
    # Chunk indexing
    # ──────────────────────────────────────────────────────────────────────────

    def bulk_upsert_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Bulk index chunk documents (each with an embedding vector) into the
        arxiv-papers-chunks index using OpenSearch's native bulk API.

        Each item in `chunks` must be a dict with two keys:
            "chunk_data": dict of fields matching the ARXIV_PAPERS_CHUNKS_MAPPING
            "embedding":  List[float] of length 1024 (Jina v3 output)

        WHY use helpers.bulk() here (unlike bulk_index_papers which loops)?
        Chunks are produced in large batches — a single 10,000-word paper can
        generate 15-20 chunks, each with a 1024-float embedding. For 15 papers
        that's ~200-300 HTTP requests if done one-by-one. helpers.bulk() batches
        them into a single request body using the bulk NDJSON format, reducing
        HTTP overhead by ~100x.

        Example input:
            chunks = [
                {
                    "chunk_data": {
                        "chunk_id": "2301.07041_chunk_0",
                        "arxiv_id": "2301.07041",
                        "chunk_text": "We propose a new...",
                        ...
                    },
                    "embedding": [0.12, -0.04, 0.88, ...]  # 1024 floats
                },
                ...
            ]

        :returns: {"success": int, "failed": int}
        """
        from opensearchpy import helpers

        actions = []
        for item in chunks:
            doc = item["chunk_data"].copy()
            doc["embedding"] = item["embedding"]
            actions.append({"_index": self.chunks_index_name, "_source": doc})

        try:
            success, failed = helpers.bulk(self.client, actions, refresh=True)
            logger.info(f"Bulk indexed {success} chunks, {len(failed)} failed")
            return {"success": success, "failed": len(failed)}
        except Exception as e:
            logger.error(f"Bulk chunk indexing error: {e}")
            raise

    def delete_paper_chunks(self, arxiv_id: str) -> int:
        """
        Delete all chunks belonging to a paper, identified by arxiv_id.

        Called before re-indexing a paper (e.g. if its PDF was re-parsed)
        to avoid accumulating stale duplicate chunks.

        delete_by_query runs a term filter on the keyword field arxiv_id —
        this is an exact match, not a full-text search. All matching chunks
        are deleted in a single atomic operation. refresh=True makes the
        deletion visible to subsequent searches immediately.

        :returns: Number of chunks deleted.
        """
        try:
            response = self.client.delete_by_query(
                index=self.chunks_index_name,
                body={"query": {"term": {"arxiv_id": arxiv_id}}},
                refresh=True,
            )
            deleted = response.get("deleted", 0)
            logger.info(f"Deleted {deleted} chunks for paper '{arxiv_id}'")
            return deleted
        except Exception as e:
            logger.error(f"Error deleting chunks for '{arxiv_id}': {e}")
            return 0

    # ──────────────────────────────────────────────────────────────────────────
    # Unified search (BM25 / hybrid)
    # ──────────────────────────────────────────────────────────────────────────

    def search_unified(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        size: int = 10,
        from_: int = 0,
        categories: Optional[List[str]] = None,
        use_hybrid: bool = True,
        min_score: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Main search entry point for the hybrid search endpoint.

        Routing logic:
            query_embedding provided AND use_hybrid=True  → hybrid (BM25 + KNN + RRF)
            otherwise                                     → BM25 only on chunks

        WHY expose use_hybrid as a flag?
        During development or when Jina is unavailable, you can disable vector
        search and fall back to pure BM25 without changing the call site.
        The response shape is identical in both modes.

        :param query:           User's text query.
        :param query_embedding: 1024-dim Jina vector for the query. None → BM25 only.
        :param size:            Number of results to return.
        :param from_:           Pagination offset.
        :param categories:      arXiv category filter (OR logic).
        :param use_hybrid:      If False, force BM25-only even if embedding is provided.
        :param min_score:       Drop results below this RRF score (hybrid mode only).
        :returns: {"total": int, "hits": [{"chunk_text": ..., "score": ..., ...}]}
        """
        if query_embedding and use_hybrid:
            return self._search_hybrid_native(query, query_embedding, size, categories, min_score)
        return self._search_bm25_chunks(query, size, from_, categories)

    def _search_bm25_chunks(
        self,
        query: str,
        size: int,
        from_: int,
        categories: Optional[List[str]],
    ) -> Dict[str, Any]:
        """
        Pure BM25 search on the arxiv-papers-chunks index.

        Uses QueryBuilder(search_chunks=True) which searches chunk_text^3,
        title^2, abstract^1 and excludes the embedding vector from results.
        """
        search_body = QueryBuilder(
            query=query,
            size=size,
            from_=from_,
            categories=categories,
            search_chunks=True,
        ).build()

        try:
            response = self.client.search(index=self.chunks_index_name, body=search_body)
            hits = []
            for hit in response["hits"]["hits"]:
                doc = hit["_source"]
                doc["score"] = hit["_score"]
                doc["chunk_id"] = hit["_id"]
                if "highlight" in hit:
                    doc["highlights"] = hit["highlight"]
                hits.append(doc)

            total = response["hits"]["total"]["value"]
            logger.info(f"BM25 chunk search '{query[:50]}' → {total} total, {len(hits)} returned")
            return {"total": total, "hits": hits}

        except Exception as e:
            logger.error(f"BM25 chunk search error: {e}")
            return {"total": 0, "hits": []}

    def _search_hybrid_native(
        self,
        query: str,
        query_embedding: List[float],
        size: int,
        categories: Optional[List[str]],
        min_score: float,
    ) -> Dict[str, Any]:
        """
        Native OpenSearch hybrid search: BM25 + KNN merged by RRF pipeline.

        How the query is constructed:
            1. Build a normal BM25 body via QueryBuilder to get the bool query
               and highlight/source config (avoids duplicating that logic).
            2. Extract just the "query" clause from it.
            3. Wrap it alongside a knn query inside {"hybrid": {"queries": [...]}}.
            4. Pass search_pipeline param so OpenSearch applies RRF post-processing.

        WHY size * 2 for the candidate pool?
        Each sub-query (BM25 and KNN) independently retrieves `size * 2` candidates.
        After RRF merges and re-ranks them, we truncate to `size`. The larger
        candidate pool gives RRF more material to work with — a document that ranks
        #12 in BM25 but #1 in KNN can still surface in the top-10 final results.
        With a pool of only `size`, such documents would be cut before RRF runs.
        """
        # Build BM25 body just to borrow its query clause, _source, and highlight
        bm25_body = QueryBuilder(
            query=query,
            size=size * 2,
            from_=0,
            categories=categories,
            search_chunks=True,
        ).build()

        hybrid_query = {
            "hybrid": {
                "queries": [
                    bm25_body["query"],  # BM25 sub-query
                    {"knn": {"embedding": {"vector": query_embedding, "k": size * 2}}},  # KNN sub-query
                ]
            }
        }

        search_body = {
            "size": size,
            "query": hybrid_query,
            "_source": bm25_body["_source"],
            "highlight": bm25_body["highlight"],
        }

        try:
            response = self.client.search(
                index=self.chunks_index_name,
                body=search_body,
                params={"search_pipeline": HYBRID_RRF_PIPELINE["id"]},
            )

            hits = []
            for hit in response["hits"]["hits"]:
                if hit["_score"] < min_score:
                    continue
                doc = hit["_source"]
                doc["score"] = hit["_score"]
                doc["chunk_id"] = hit["_id"]
                if "highlight" in hit:
                    doc["highlights"] = hit["highlight"]
                hits.append(doc)

            logger.info(f"Hybrid search '{query[:50]}' → {len(hits)} results (min_score={min_score})")
            return {"total": len(hits), "hits": hits}

        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            return {"total": 0, "hits": []}
