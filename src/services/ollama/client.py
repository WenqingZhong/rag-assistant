import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from src.exceptions import OllamaConnectionError, OllamaException, OllamaTimeoutError
from src.schemas.ollama import RAGResponse
from src.services.ollama.prompts import RAGPromptBuilder, ResponseParser

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with the Ollama local LLM service."""

    def __init__(self, base_url: str, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(float(timeout))
        # Loaded once at construction — avoids re-reading the .txt file per request
        self.prompt_builder = RAGPromptBuilder()
        self.response_parser = ResponseParser()

    async def health_check(self) -> Dict[str, Any]:
        """
        Check if Ollama is healthy and return version info.

        Returns {"status": "healthy", "message": ..., "version": ...}
        Raises OllamaConnectionError / OllamaTimeoutError on failure.
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/version")
                if response.status_code == 200:
                    version_data = response.json()
                    return {
                        "status": "healthy",
                        "message": "Ollama service is running",
                        "version": version_data.get("version", "unknown"),
                    }
                raise OllamaException(f"Ollama returned status {response.status_code}")
        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}")
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Ollama health check failed: {e}")

    async def list_models(self) -> List[Dict[str, Any]]:
        """Return list of locally available models from /api/tags."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    return response.json().get("models", [])
                raise OllamaException(f"Failed to list models: {response.status_code}")
        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}")
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Error listing models: {e}")

    async def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate text using the specified model.

        Args:
            model:  Model name (e.g. "llama3.2:1b")
            prompt: Input prompt
            stream: Streaming not yet implemented — keep False
            **kwargs: Extra Ollama generation parameters (temperature, etc.)
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {"model": model, "prompt": prompt, "stream": stream, **kwargs}
                response = await client.post(f"{self.base_url}/api/generate", json=payload)
                if response.status_code == 200:
                    return response.json()
                raise OllamaException(f"Generation failed: {response.status_code}")
        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}")
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Error generating with Ollama: {e}")

    async def generate_stream(
        self,
        model: str,
        prompt: str,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream tokens from Ollama one chunk at a time.

        HOW streaming works:
        Ollama's /api/generate endpoint with stream=True sends a sequence of
        newline-delimited JSON objects, one per generated token:

            {"model": "llama3.2:1b", "response": "Trans", "done": false}
            {"model": "llama3.2:1b", "response": "formers", "done": false}
            {"model": "llama3.2:1b", "response": " are", "done": false}
            ...
            {"model": "llama3.2:1b", "response": "", "done": true, "total_duration": 14823000000}

        We use httpx's client.stream() context manager + response.aiter_lines()
        to read these one line at a time as they arrive, parse each as JSON,
        and yield the dict to the caller. The caller (ask.py) formats these
        into Server-Sent Events for the browser.

        WHY NOT use structured output (format=schema) for streaming?
        Structured output constrains the model to produce valid JSON as a
        complete object. JSON only makes sense when complete — you can't
        display partial JSON to the user token-by-token. Streaming uses
        plain text prompts so each token fragment is immediately readable.

        :yields: Dicts like {"response": "token", "done": False} per token,
                 then {"done": True, ...stats...} at the end.
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {"model": model, "prompt": prompt, "stream": True, **kwargs}
                async with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
                    if response.status_code != 200:
                        raise OllamaException(f"Streaming failed: {response.status_code}")

                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                chunk = json.loads(line)
                                # Strip context (full token ID history for multi-turn
                                # conversations) — not needed for single-turn RAG and
                                # makes the final SSE event ~10x larger than necessary.
                                chunk.pop("context", None)
                                yield chunk
                            except json.JSONDecodeError:
                                logger.warning(f"Could not parse streaming chunk: {line[:100]}")
                                continue

        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}")
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Error in streaming generation: {e}")

    async def generate_rag_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = "llama3.2:1b",
        use_structured_output: bool = True,
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline: build prompt → call Ollama → parse → fill sources.

        This is the non-streaming path used by POST /api/v1/ask.

        Step 1 — Build prompt:
            use_structured_output=True:  create_structured_prompt() adds
                                         "format": RAGResponse.model_json_schema()
                                         to the request, forcing Ollama to emit JSON.
            use_structured_output=False: create_rag_prompt() sends plain text.
                                         Ollama still asked to respond in JSON via
                                         format="json" (less strict than schema).

        Step 2 — Generate:
            Calls self.generate() which sends POST /api/generate and waits for
            the full response (stream=False). Ollama's response["response"] is
            the raw string the model produced.

        Step 3 — Parse:
            ResponseParser.parse_structured_response() tries three levels:
            clean JSON → regex-extracted JSON → plain text fallback.

        Step 4 — Fill sources:
            The LLM is NOT asked to generate PDF URLs (it hallucinates them).
            Instead, we build URLs deterministically from the arxiv_ids of the
            chunks that were actually retrieved:
                arxiv_id = "2301.07041v1"
                clean_id = "2301.07041"   (strip version suffix)
                url      = "https://arxiv.org/pdf/2301.07041.pdf"
            Deduplication via seen_urls set — multiple chunks from the same
            paper produce only one source URL.

        :param query:                The user's question.
        :param chunks:               OpenSearch hits from search_unified().
        :param model:                Ollama model name.
        :param use_structured_output: Whether to use Ollama's JSON schema feature.
        :returns: Dict with keys: answer, sources, confidence, citations.
        """
        try:
            if use_structured_output:
                prompt_data = self.prompt_builder.create_structured_prompt(query, chunks)
                response = await self.generate(
                    model=model,
                    prompt=prompt_data["prompt"],
                    temperature=0.7,
                    top_p=0.9,
                    format=prompt_data["format"],
                )
            else:
                prompt = self.prompt_builder.create_rag_prompt(query, chunks)
                response = await self.generate(
                    model=model,
                    prompt=prompt,
                    temperature=0.7,
                    top_p=0.9,
                    format="json",
                )

            if not response or "response" not in response:
                raise OllamaException("No response generated from Ollama")

            logger.debug(f"Raw Ollama response: {response['response'][:300]}")
            parsed = self.response_parser.parse_structured_response(response["response"])

            # Fill sources from retrieved chunk arxiv_ids (never from LLM output)
            if not parsed.get("sources"):
                seen_urls: set = set()
                sources = []
                for chunk in chunks:
                    arxiv_id = chunk.get("arxiv_id", "")
                    if arxiv_id:
                        # Strip version suffix: "2301.07041v1" → "2301.07041"
                        clean_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                        url = f"https://arxiv.org/pdf/{clean_id}.pdf"
                        if url not in seen_urls:
                            sources.append(url)
                            seen_urls.add(url)
                parsed["sources"] = sources

            # Fill citations from chunk arxiv_ids if LLM didn't produce them
            if not parsed.get("citations"):
                parsed["citations"] = list(
                    {chunk.get("arxiv_id") for chunk in chunks if chunk.get("arxiv_id")}
                )[:5]

            logger.info(f"RAG answer generated: {len(parsed['sources'])} sources, confidence={parsed.get('confidence')}")
            return parsed

        except OllamaException:
            raise
        except Exception as e:
            logger.error(f"Error generating RAG answer: {e}")
            raise OllamaException(f"Failed to generate RAG answer: {e}")

    async def generate_rag_answer_stream(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = "llama3.2:1b",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming RAG: build prompt → stream tokens from Ollama one by one.

        Used by POST /api/v1/stream. Each yielded dict is one token fragment
        from Ollama, which the router wraps in a Server-Sent Event (SSE) and
        forwards to the browser immediately.

        WHY plain text prompt (not structured)?
        Structured output only works for complete JSON objects — you can't
        meaningfully stream partial JSON. The browser would receive:
            '{"ans'   ← not displayable
            'wer": '  ← not displayable
            '"Trans'  ← not displayable
        With plain text, each chunk is a readable word fragment:
            "Trans"   ← display immediately
            "formers" ← append to display
            " use"    ← append to display

        Sources are NOT yielded here — they're generated by the router after
        streaming completes, using the same arxiv_id → PDF URL logic as
        generate_rag_answer().

        :yields: {"response": "token_fragment", "done": False} per token,
                 then {"done": True} when generation is complete.
        """
        try:
            prompt = self.prompt_builder.create_rag_prompt(query, chunks)
            async for chunk in self.generate_stream(
                model=model,
                prompt=prompt,
                temperature=0.7,
                top_p=0.9,
            ):
                yield chunk

        except OllamaException:
            raise
        except Exception as e:
            logger.error(f"Error generating streaming RAG answer: {e}")
            raise OllamaException(f"Failed to generate streaming RAG answer: {e}")

    def get_langchain_model(self, model: str, temperature: float = 0.0):
        """
        Return a LangChain-compatible ChatOllama instance.

        WHY this method exists:
        The agent nodes (guardrail, grade_documents, rewrite_query,
        generate_answer) need to call Ollama with structured output
        (llm.with_structured_output(SomeModel)) and async invocation
        (await llm.ainvoke(prompt)). The plain httpx-based generate()
        method doesn't support either. ChatOllama from langchain-ollama
        wraps the same Ollama API but exposes the full LangChain interface.

        WHY not store ChatOllama at init time?
        Each node may request a different temperature (0.0 for grading,
        0.3 for query rewriting). Creating the object here is cheap —
        it's just a config object, no HTTP connection is made until invoke.

        :param model: Ollama model name, e.g. "llama3.2:1b"
        :param temperature: Sampling temperature (0.0 = deterministic)
        :returns: ChatOllama instance ready for .ainvoke() or .with_structured_output()
        """
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url=self.base_url,
            model=model,
            temperature=temperature,
        )
