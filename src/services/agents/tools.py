import logging

from langchain_core.documents import Document
from langchain_core.tools import tool

from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.opensearch.client import OpenSearchClient

logger = logging.getLogger(__name__)


def create_retriever_tool(
    opensearch_client: OpenSearchClient,
    embeddings_client: JinaEmbeddingsClient,
    top_k: int = 3,
    use_hybrid: bool = True,
):
    """
    Build the `retrieve_papers` LangChain tool used by the retrieve node.

    WHY a factory function?
    The @tool decorator creates a LangGraph-compatible tool at definition time.
    We need the tool to close over opensearch_client and embeddings_client —
    but those aren't available until the AgenticRAGService is instantiated.
    The factory captures them via closure so the tool is stateless from
    LangGraph's perspective (just a plain async function).

    HOW retrieval works inside the tool:
    1. Embed the query with Jina (1536-dim vector)
    2. Run search_unified() — BM25 + kNN hybrid or BM25-only
    3. Convert each OpenSearch hit to a LangChain Document with metadata
    4. LangGraph's ToolNode serialises the Document list as the ToolMessage content
    """

    @tool
    async def retrieve_papers(query: str) -> list[Document]:
        """Search and return relevant arXiv research papers.

        Use this tool when the user asks about:
        - Machine learning concepts or techniques
        - Deep learning architectures (transformers, CNNs, RNNs, etc.)
        - Natural language processing
        - Computer vision methods
        - AI research topics
        - Specific algorithms or models

        :param query: The search query describing what papers to find
        :returns: List of relevant paper excerpts with metadata
        """
        logger.info(f"Tool: retrieve_papers — query: {query[:80]}...")

        query_embedding = await embeddings_client.embed_query(query)

        search_results = opensearch_client.search_unified(
            query=query,
            query_embedding=query_embedding,
            size=top_k,
            use_hybrid=use_hybrid,
        )

        documents = []
        for hit in search_results.get("hits", []):
            doc = Document(
                page_content=hit["chunk_text"],
                metadata={
                    "arxiv_id": hit["arxiv_id"],
                    "title": hit.get("title", ""),
                    "authors": hit.get("authors", ""),
                    "score": hit.get("score", 0.0),
                    "source": f"https://arxiv.org/pdf/{hit['arxiv_id']}.pdf",
                    "section": hit.get("section_name", ""),
                    "search_mode": "hybrid" if use_hybrid else "bm25",
                },
            )
            documents.append(doc)

        logger.info(f"Tool: retrieve_papers — returned {len(documents)} documents")
        return documents

    return retrieve_papers
