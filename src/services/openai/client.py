import logging
from typing import Any, AsyncGenerator, Dict, List

from openai import AsyncOpenAI

from src.services.ollama.prompts import RAGPromptBuilder

logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    Drop-in replacement for OllamaClient backed by the OpenAI API.

    Method signatures and return shapes deliberately match OllamaClient so
    callers (ask.py, stream router, agent nodes) require no changes other than
    swapping which client is injected at startup.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self._client = AsyncOpenAI(api_key=api_key)
        self.prompt_builder = RAGPromptBuilder()

    # ── RAG answer (non-streaming) ────────────────────────────────────────────

    async def generate_rag_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = "",
        use_structured_output: bool = True,
    ) -> Dict[str, Any]:
        """
        Build a RAG prompt, call OpenAI, return the same dict shape as OllamaClient.

        `model` arg is accepted for API compatibility but ignored — the model
        is fixed at construction time via OpenAISettings.
        """
        prompt = self.prompt_builder.create_rag_prompt(query, chunks)

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant. Answer based only on the provided context."},
                    {"role": "user", "content": prompt},
                ],
            )
            answer = response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise

        sources = self._extract_sources(chunks)
        citations = list({c.get("arxiv_id") for c in chunks if c.get("arxiv_id")})[:5]

        return {
            "answer": answer,
            "sources": sources,
            "confidence": None,
            "citations": citations,
        }

    # ── RAG answer (streaming) ────────────────────────────────────────────────

    async def generate_rag_answer_stream(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = "",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream tokens from OpenAI, yielding dicts in the same format as
        OllamaClient.generate_stream() so the /stream router needs no changes:

            {"response": "Trans", "done": false}
            {"response": "formers", "done": false}
            ...
            {"response": "", "done": true, "total_duration": 0}
        """
        prompt = self.prompt_builder.create_rag_prompt(query, chunks)

        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                stream=True,
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant. Answer based only on the provided context."},
                    {"role": "user", "content": prompt},
                ],
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                finish = chunk.choices[0].finish_reason

                if delta:
                    yield {"response": delta, "done": False}

                if finish == "stop":
                    yield {"response": "", "done": True, "total_duration": 0}

        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise

    # ── LangChain model (used by agent nodes) ────────────────────────────────

    def get_langchain_model(self, model: str = "", temperature: float = 0.0):
        """
        Return a LangChain ChatOpenAI instance compatible with agent nodes.

        Agent nodes call get_langchain_model() then use .with_structured_output()
        and .ainvoke() — both supported by ChatOpenAI out of the box.
        """
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=self._client.api_key,
            model=self.model,
            temperature=temperature,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[str]:
        seen: set = set()
        sources = []
        for chunk in chunks:
            arxiv_id = chunk.get("arxiv_id", "")
            if arxiv_id:
                clean_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                url = f"https://arxiv.org/pdf/{clean_id}.pdf"
                if url not in seen:
                    sources.append(url)
                    seen.add(url)
        return sources
