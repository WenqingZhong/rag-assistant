"""
OpenSearch index configuration for hybrid search (BM25 + KNN vector).

Two things are defined here:

    ARXIV_PAPERS_CHUNKS_MAPPING  — the index mapping (schema + settings)
    HYBRID_RRF_PIPELINE          — the search pipeline that combines BM25 + KNN scores

WHY a separate chunks index (arxiv-papers-chunks) instead of adding vectors to arxiv-papers?
The existing arxiv-papers index stores ONE document per paper (full_text, title, abstract).
Hybrid search operates at the CHUNK level — we embed each 600-word passage separately and
store each as its own document. Mixing chunk documents and paper documents in the same index
would confuse BM25 (term statistics would be polluted by tiny chunk texts vs. full papers)
and complicate search (you'd have to filter by document type on every query).
Keeping them separate is cleaner and lets you tune each index independently.
"""

ARXIV_PAPERS_CHUNKS_INDEX = "arxiv-papers-chunks"

ARXIV_PAPERS_CHUNKS_MAPPING = {
    "settings": {
        # ── Shards & replicas ───────────────────────────────────────────────────
        # number_of_shards=1: For our scale (thousands of chunks, single node),
        # one shard is optimal. Multiple shards would split the BM25 term
        # statistics across shards, causing slight score inconsistencies.
        # Only increase shards when a single shard exceeds ~30-50 GB.
        "number_of_shards": 1,

        # number_of_replicas=0: Replicas are copies of a shard on OTHER nodes.
        # We have a single-node dev cluster — there are no other nodes to
        # place a replica on. Setting this to 0 avoids "yellow" cluster health
        # that would otherwise appear because replicas can't be assigned.
        "number_of_replicas": 0,

        # ── KNN (vector search) settings ────────────────────────────────────────
        # index.knn=True: enables the KNN plugin for this index.
        # Without this, "knn_vector" fields in the mapping are ignored and
        # approximate nearest-neighbour search will fail at query time.
        "index.knn": True,

        # index.knn.space_type: global default similarity metric.
        # cosinesimil: score = cos(angle between query vector and doc vector).
        # Range: [-1, 1], where 1 = identical direction, 0 = orthogonal.
        # WHY cosine instead of l2 (Euclidean)? Cosine is scale-invariant —
        # a 1024-dim Jina vector is scored the same regardless of its magnitude,
        # which is desirable since Jina normalizes its output vectors anyway.
        "index.knn.space_type": "cosinesimil",

        # ── Text analyzers ──────────────────────────────────────────────────────
        "analysis": {
            "analyzer": {
                # standard_analyzer: tokenize → lowercase → remove English stopwords.
                # Used for the "authors" field — author names shouldn't be stemmed
                # ("Manning" should not become "mann") but stopwords in names are rare
                # enough that the setting is harmless.
                "standard_analyzer": {
                    "type": "standard",
                    "stopwords": "_english_",
                },

                # text_analyzer: tokenize → lowercase → remove stopwords → stem.
                # Stemming: "running" → "run", "methods" → "method".
                # WHY stem? A search for "attention mechanisms" should match a chunk
                # that says "attention mechanism" (singular). Without stemming, the
                # BM25 scorer treats "mechanism" and "mechanisms" as different terms.
                # The snowball filter is a standard English stemmer — conservative
                # enough that it doesn't mangle technical vocabulary too badly.
                "text_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "stop", "snowball"],
                },
            }
        },
    },

    "mappings": {
        # dynamic=strict: any field NOT listed below will be rejected at index time
        # rather than silently auto-mapped. This prevents typos in field names from
        # creating phantom unmapped fields that waste space and confuse queries.
        "dynamic": "strict",

        "properties": {
            # ── Chunk identity ──────────────────────────────────────────────────
            # keyword fields: stored as exact strings, not tokenized.
            # Used for equality filters (WHERE chunk_id = "..."), aggregations,
            # and sort. Never used for full-text BM25 search.
            "chunk_id": {"type": "keyword"},   # "<arxiv_id>_chunk_<N>"
            "arxiv_id": {"type": "keyword"},   # e.g. "2301.07041"
            "paper_id": {"type": "keyword"},   # PostgreSQL UUID / row ID
            "chunk_index": {"type": "integer"},  # 0-based position within the paper

            # ── The searchable chunk text ────────────────────────────────────────
            # type=text: tokenised and BM25-indexed — this is the field BM25 runs on.
            # text_analyzer: lowercase + stop + snowball (see above).
            # .keyword sub-field: raw, untokenised copy. Lets you do exact phrase
            # matching or aggregations on the chunk text without a separate field.
            # ignore_above=256: don't index keyword values longer than 256 chars
            # (chunk texts can be thousands of chars — the keyword sub-field is only
            # useful for short texts anyway).
            "chunk_text": {
                "type": "text",
                "analyzer": "text_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },

            "chunk_word_count": {"type": "integer"},
            "start_char": {"type": "integer"},  # character offset in original doc
            "end_char": {"type": "integer"},

            # ── Vector embedding ────────────────────────────────────────────────
            # type=knn_vector: stored as a 1024-dimensional float array AND indexed
            # into an HNSW graph for approximate nearest-neighbour (ANN) search.
            # Without the knn_vector type, you'd have to do exact k-NN (comparing
            # the query vector against ALL stored vectors) — O(N*D) instead of O(log N).
            "embedding": {
                "type": "knn_vector",
                "dimension": 1024,  # must match Jina v3 output dimensions exactly
                "method": {
                    # HNSW = Hierarchical Navigable Small World graph.
                    # Organises vectors into a layered graph where each node
                    # (chunk) is connected to its M nearest neighbours.
                    # Search traverses the graph from a random entry point,
                    # greedily moving toward the query vector — very fast.
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",  # nmslib: faster indexing; faiss: faster search
                    "parameters": {
                        # ef_construction: size of the candidate list during index build.
                        # Higher → each node considers more neighbours → better graph
                        # quality → better recall at search time, but slower indexing.
                        # 512 is a good recall/speed tradeoff for our scale.
                        # (Default is 512; we set it explicitly to make the intent clear.)
                        "ef_construction": 512,

                        # m: number of bi-directional links per node in the graph.
                        # Higher m → more connections → better recall, more memory.
                        # 16 is the standard default. Increase to 32 or 64 for
                        # very high-recall requirements (at ~2x memory cost).
                        "m": 16,
                    },
                },
            },

            # ── Paper metadata (stored on every chunk for zero-join retrieval) ──
            # WHY duplicate title/abstract on every chunk?
            # Search results are returned as individual chunks, not whole papers.
            # The UI needs title/authors/abstract to show a useful result card.
            # If these were only in arxiv-papers, every hit would require a JOIN —
            # an extra DB round-trip per result. Storing them here trades index size
            # (cheap) for latency (expensive).
            "title": {
                "type": "text",
                "analyzer": "text_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "authors": {
                "type": "text",
                "analyzer": "standard_analyzer",  # no stemming for names
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "abstract": {"type": "text", "analyzer": "text_analyzer"},
            "categories": {"type": "keyword"},       # e.g. ["cs.CL", "cs.LG"]
            "published_date": {"type": "date"},
            "section_title": {"type": "keyword"},    # e.g. "Methods (part 2)"
            "embedding_model": {"type": "keyword"},  # e.g. "jina-embeddings-v3"
            "created_at": {"type": "date"},
            "updated_at": {"type": "date"},
        },
    },
}

# ── Search pipeline: Reciprocal Rank Fusion ──────────────────────────────────
#
# A search pipeline is a POST-PROCESSOR that runs AFTER both BM25 and KNN
# searches complete but BEFORE the merged results are returned to the caller.
#
# WHY RRF instead of a weighted average (normalization-processor)?
#
# Option A — Weighted average: score = 0.3 * bm25_score + 0.7 * knn_score
#   Problem: BM25 scores and cosine similarity scores live on completely different
#   scales. BM25 scores for a 10-document corpus look nothing like BM25 scores
#   for a 100,000-document corpus. You'd need to re-tune the weights every time
#   the corpus size changes. Also, if one modality dominates (e.g. a very specific
#   keyword match), the weighted average gets skewed.
#
# Option B — RRF: score = 1/(k + rank_in_bm25_list) + 1/(k + rank_in_knn_list)
#   Key insight: RRF operates on RANKS, not raw scores. Ranks are always
#   comparable — rank 1 in BM25 and rank 1 in KNN are both "the top result"
#   regardless of what the underlying scores are.
#   k=60: dampens the score difference between high and low ranks. With k=60:
#     rank 1  → 1/61  ≈ 0.0164
#     rank 10 → 1/70  ≈ 0.0143   (only 13% lower than rank 1)
#     rank 60 → 1/120 ≈ 0.0083   (50% lower than rank 1)
#   A document ranked #2 in BOTH lists scores higher than one ranked #1 in one
#   list and #50 in the other — exactly the right behavior for hybrid search.
#
# Example:
#   Query: "transformer attention mechanism"
#   BM25 result list: [chunk_A(rank1), chunk_B(rank2), chunk_C(rank3), ...]
#   KNN result list:  [chunk_B(rank1), chunk_D(rank2), chunk_A(rank3), ...]
#
#   chunk_A: 1/(60+1) + 1/(60+3) = 0.0164 + 0.0156 = 0.0320
#   chunk_B: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325  ← wins (strong in both)
#   chunk_D: 0        + 1/(60+2) = 0      + 0.0161 = 0.0161   (only in KNN list)

HYBRID_RRF_PIPELINE = {
    "id": "hybrid-rrf-pipeline",
    "description": "Post processor for hybrid RRF search",
    "phase_results_processors": [
        {
            "score-ranker-processor": {
                "combination": {
                    "technique": "rrf",
                    "rank_constant": 60,  # k in 1/(k+rank). 60 is OpenSearch's default.
                }
            }
        }
    ],
}
