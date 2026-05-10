import hashlib
import json
import logging
from datetime import timedelta
from typing import Optional

import redis

from src.config import RedisSettings
from src.schemas.api.ask import AskRequest, AskResponse

logger = logging.getLogger(__name__)


class CacheClient:
    """
    Redis-based exact-match cache for RAG responses.

    WHY cache RAG responses at all?
    A RAG request is expensive:
      1. Jina API call  (~100-200ms)
      2. OpenSearch query (~50ms)
      3. Ollama generation (~15-20 seconds for llama3.2:1b)

    If the same question is asked again (e.g. a user refreshes, or two users
    ask "What is self-attention?"), we can skip all three steps and return the
    cached answer instantly.

    WHY exact-match (SHA-256 hash) instead of semantic similarity?
    Semantic caching would find "What is attention in transformers?" as a near-
    match for "What is self-attention?" — but that requires another embedding
    call and a vector similarity search. Exact-match is O(1): one Redis GET.
    For RAG responses that are slow to generate, even exact-match hits save
    significant time on repeated queries.

    CACHE KEY:
    We hash the full request parameters — query, model, top_k, use_hybrid,
    categories — so different parameter combinations never collide:
        {"query": "...", "model": "llama3.2:1b", "top_k": 3, ...}
        → SHA-256 → first 16 hex chars → "exact_cache:a3f9d1b2c8e4f701"

    WHY only 16 chars of the hash?
    SHA-256 produces 64 hex chars. 16 chars = 64 bits of entropy → collision
    probability is 1 in 2^64 ≈ zero for our scale. Shorter keys = less Redis
    memory per entry.

    TTL: 6 hours by default. Stale answers are worse than no cache — if new
    papers are indexed, cached answers referencing old chunks stay wrong until
    they expire.
    """

    def __init__(self, redis_client: redis.Redis, settings: RedisSettings):
        self.redis = redis_client
        self.ttl = timedelta(hours=settings.ttl_hours)

    def _cache_key(self, request: AskRequest) -> str:
        """
        Deterministic cache key from all request parameters.

        sort_keys=True ensures {"a":1,"b":2} and {"b":2,"a":1} produce
        the same JSON string (and therefore the same hash).
        categories is sorted for the same reason — list order shouldn't matter.
        """
        key_data = {
            "query": request.query,
            "model": request.model,
            "top_k": request.top_k,
            "use_hybrid": request.use_hybrid,
            "categories": sorted(request.categories) if request.categories else [],
        }
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        return f"exact_cache:{key_hash}"

    async def find_cached_response(self, request: AskRequest) -> Optional[AskResponse]:
        """
        Look up a cached response. Returns None on cache miss or any error.

        This is a single Redis GET — O(1), typically < 1ms.
        The stored value is a JSON string produced by AskResponse.model_dump_json().
        """
        try:
            key = self._cache_key(request)
            cached = self.redis.get(key)
            if cached:
                logger.info(f"Cache HIT for query: '{request.query[:60]}'")
                return AskResponse(**json.loads(cached))
            logger.debug(f"Cache MISS for query: '{request.query[:60]}'")
            return None
        except Exception as e:
            logger.warning(f"Cache read failed (proceeding without cache): {e}")
            return None

    async def store_response(self, request: AskRequest, response: AskResponse) -> bool:
        """
        Store a response in Redis with the configured TTL.

        Uses model_dump_json() (Pydantic v2) so the stored string can be
        deserialised directly back into AskResponse(**json.loads(cached)).
        """
        try:
            key = self._cache_key(request)
            self.redis.set(key, response.model_dump_json(), ex=self.ttl)
            logger.info(f"Cache STORE for query: '{request.query[:60]}' (TTL: {self.ttl})")
            return True
        except Exception as e:
            logger.warning(f"Cache write failed (response still returned to user): {e}")
            return False
