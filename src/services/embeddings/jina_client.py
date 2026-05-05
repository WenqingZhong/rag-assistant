import logging
from typing import List

import httpx

from src.schemas.embeddings.jina import JinaEmbeddingRequest, JinaEmbeddingResponse

logger = logging.getLogger(__name__)


class JinaEmbeddingsClient:
    """
    Client for the Jina AI embeddings API.

    WHY Jina instead of OpenAI embeddings?
    Jina's jina-embeddings-v3 model is purpose-built for retrieval tasks and
    supports *task-aware* encoding — the same model produces different vector
    spaces depending on whether you declare the input as a passage (document
    chunk being indexed) or a query (user search input). This asymmetry is
    important: a good retrieval model should map queries and their relevant
    passages close together even though they're written differently.

    WHY async (httpx.AsyncClient)?
    Embedding a batch of 50 chunks means 50 texts in one HTTP request, waiting
    on Jina's API. During that wait the FastAPI event loop can handle other
    requests. If this were sync (requests library), the worker thread would
    block entirely. For batch operations especially, async pays off.

    WHY a persistent client (self.client) instead of creating one per call?
    httpx.AsyncClient reuses the underlying TCP connection pool across calls.
    Creating a new client per embed_passages() call would do a fresh TLS
    handshake every time — unnecessary latency at indexing time.
    Call close() (or use as async context manager) when done.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.jina.ai/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        # Persistent connection pool — reused across all embed calls.
        # 30s timeout: Jina is fast (<1s for small batches) but large batches
        # of 100 texts can take a few seconds.
        self.client = httpx.AsyncClient(timeout=30.0)
        logger.info("Jina embeddings client initialised")

    async def embed_passages(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Embed text passages for indexing (document chunks).

        Uses task="retrieval.passage" — the model optimises these vectors
        to be *retrieved* by a query, not to retrieve things themselves.

        WHY batch_size=100?
        Jina's API accepts up to ~2000 tokens per text and up to 2048 texts
        per request. 100 chunks at ~600 words each stays comfortably within
        rate limits and keeps response times predictable.

        Args:
            texts:      List of chunk texts (output of TextChunker)
            batch_size: Max texts per API call

        Returns:
            List of 1024-dim float vectors, one per input text, same order.
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            request_data = JinaEmbeddingRequest(
                model="jina-embeddings-v3",
                task="retrieval.passage",
                dimensions=1024,
                input=batch,
            )

            try:
                response = await self.client.post(
                    f"{self.base_url}/embeddings",
                    headers=self.headers,
                    json=request_data.model_dump(),
                )
                response.raise_for_status()

                result = JinaEmbeddingResponse(**response.json())
                # result.data is ordered by index — safe to extend in order
                batch_embeddings = [item["embedding"] for item in result.data]
                embeddings.extend(batch_embeddings)

                logger.debug(f"Embedded batch of {len(batch)} passages")

            except httpx.HTTPStatusError as e:
                logger.error(f"Jina API error {e.response.status_code}: {e.response.text}")
                raise
            except httpx.HTTPError as e:
                logger.error(f"HTTP error embedding passages: {e}")
                raise

        logger.info(f"Embedded {len(texts)} passages → {len(embeddings)} vectors")
        return embeddings

    async def embed_query(self, query: str) -> List[float]:
        """
        Embed a single search query.

        Uses task="retrieval.query" — a different vector space than passages.
        Jina v3 trains both spaces jointly so that query vectors and passage
        vectors are directly comparable via cosine similarity, even though
        they were encoded with different task heads.

        Args:
            query: The user's search string

        Returns:
            A single 1024-dim float vector.
        """
        request_data = JinaEmbeddingRequest(
            model="jina-embeddings-v3",
            task="retrieval.query",
            dimensions=1024,
            input=[query],
        )

        try:
            response = await self.client.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json=request_data.model_dump(),
            )
            response.raise_for_status()

            result = JinaEmbeddingResponse(**response.json())
            embedding = result.data[0]["embedding"]

            logger.debug(f"Embedded query: '{query[:60]}...' → {len(embedding)}-dim vector")
            return embedding

        except httpx.HTTPStatusError as e:
            logger.error(f"Jina API error {e.response.status_code}: {e.response.text}")
            raise
        except httpx.HTTPError as e:
            logger.error(f"HTTP error embedding query: {e}")
            raise

    async def close(self):
        """Close the underlying HTTP connection pool. Call on app shutdown."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
