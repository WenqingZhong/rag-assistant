# The index mapping tells OpenSearch:
# - which fields exist
# - how to treat each field (full-text search vs exact match vs date)
# - BM25 tuning parameters
INDEX_NAME = "documents"

# BM25 settings:
# k1 controls term frequency saturation — how much does repeating a word help?
#    1.2 = moderate boost (default). Higher = more reward for repetition.
# b controls document length normalization — does length matter?
#    0.75 = moderate penalty for long docs (default). 0 = length ignored.
BM25_SETTINGS = {
    "similarity": {
        "custom_bm25": {
            "type": "BM25",
            "k1": 1.2,
            "b": 0.75,
        }
    }
}

INDEX_SETTINGS = {
   "settings": {
        **BM25_SETTINGS,
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "index.knn": True,        # ← enables the KNN plugin for vector search (Week 4)
        "analysis": {
            "analyzer": {
                "text_analyzer": {
                    "type": "standard",
                    "stopwords": "_english_"
                }
            }
        }
    },
    "mappings": {
        "properties": {
            # --- Identity fields ---
            "id": {
                "type": "keyword"   # exact match only, not tokenized
                                    # "keyword" = filter/sort field in OpenSearch
            },
            "source": {
                "type": "keyword"   # "arxiv", "legal", etc. — used for filtering
            },

            # --- Full-text search fields (BM25 applies here) ---
            "title": {
                "type": "text",
                "analyzer": "text_analyzer",
                "similarity": "custom_bm25",
                "boost": 3.0,       # ← this is what the comment was referring to
            },
            "abstract": {
                "type": "text",
                "analyzer": "text_analyzer",
                "similarity": "custom_bm25",
                "boost": 2.0,       # ← abstract matches count 2x
            },
            "full_text": {
                "type": "text",
                "analyzer": "text_analyzer",
                "similarity": "custom_bm25",
                # no boost — counts normally (1x)
            },
            "authors": {
                "type": "text",
                "analyzer": "text_analyzer",
                # also store as keyword for exact author filtering
                "fields": {
                    "keyword": {"type": "keyword"}
                }
            },

            # --- Filter/sort fields (exact match, no BM25) ---
            "published_date": {
                "type": "date"      # enables date range filtering
            },
            "pdf_parsed": {
                "type": "keyword"   # "success", "failed", "pending"
            },

            # --- Reserved for Week 4 (vector search) ---
            # We define the vector field now so we don't need to rebuild
            # the index in Week 4. Empty for now.
            "embedding": {
                "type": "knn_vector",
                "dimension": 1024,  # Jina AI embedding size (Week 4)
                "method": {
                    "name": "hnsw",
                    "engine": "lucene",
                }
            }
        }
    }
}