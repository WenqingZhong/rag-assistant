import json
import re
from pathlib import Path
from typing import Any, Dict, List

from pydantic import ValidationError

from src.schemas.ollama import RAGResponse


class RAGPromptBuilder:
    """
    Assembles the text prompt that gets sent to Ollama.

    WHY a class instead of a plain function?
    The system prompt is loaded from a .txt file on disk. Loading it once at
    construction time (rather than on every call) avoids a file read per request.
    The class also makes it easy to swap prompt files in tests by patching
    self.prompts_dir without touching the file system.

    WHY a separate .txt file instead of a hardcoded string?
    Prompt engineering is iterative — you change wording, add/remove instructions,
    tune the word limit. A .txt file lets you edit the prompt without touching
    Python code. Non-engineers can also edit it without fear of breaking syntax.
    A hardcoded Python string buries the prompt inside code and makes diffs noisy.

    TWO prompt methods:
        create_rag_prompt()        → plain text prompt, used for streaming
        create_structured_prompt() → adds a JSON schema, used for non-streaming
                                     so Ollama returns a parseable structured response
    """

    def __init__(self):
        # Path is relative to THIS file's location, not the working directory.
        # Path(__file__).parent resolves to src/services/ollama/
        # so "prompts/rag_system.txt" always resolves correctly regardless of
        # where the process is launched from.
        self.prompts_dir = Path(__file__).parent / "prompts"
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """
        Load system prompt from rag_system.txt.

        Falls back to a minimal inline prompt if the file is missing — this
        prevents a hard crash on startup if the file was accidentally deleted,
        at the cost of slightly worse answer quality.
        """
        prompt_file = self.prompts_dir / "rag_system.txt"
        if not prompt_file.exists():
            return (
                "You are an AI assistant specialized in answering questions about "
                "academic papers from arXiv. Base your answer STRICTLY on the provided "
                "paper excerpts."
            )
        return prompt_file.read_text().strip()

    def create_rag_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Build a plain-text prompt from the system instructions + retrieved chunks + question.

        The prompt structure:

            <system instructions>

            ### Context from Papers:

            [1. arXiv:2301.07041]
            We propose a new method that...

            [2. arXiv:1706.03762]
            The attention mechanism allows...

            ### Question:
            What is self-attention?

            ### Answer (cite sources using [arXiv:id] format):

        WHY only include chunk_text and arxiv_id — not title, authors, section?
        Fewer tokens = faster generation and less chance the model gets distracted
        by metadata. The model only needs the text content to answer the question.
        The arxiv_id is included so the model can form citations ([arXiv:2301.07041])
        that we can later match back to real papers.

        WHY not include the full paper or abstract?
        Token budget. A 1B-parameter model like llama3.2:1b handles ~4k tokens well.
        3 chunks × 600 words ≈ 2400 tokens for context, leaving ~1600 for the answer.
        Adding titles/abstracts/authors would push past that limit and degrade quality.

        :param query:  The user's question.
        :param chunks: List of OpenSearch hit dicts — each must have 'chunk_text'
                       and 'arxiv_id'.
        :returns: A single string ready to send to Ollama's /api/generate endpoint.
        """
        prompt = f"{self.system_prompt}\n\n"
        prompt += "### Context from Papers:\n\n"

        for i, chunk in enumerate(chunks, 1):
            chunk_text = chunk.get("chunk_text", chunk.get("content", ""))
            arxiv_id = chunk.get("arxiv_id", "")
            prompt += f"[{i}. arXiv:{arxiv_id}]\n"
            prompt += f"{chunk_text}\n\n"

        prompt += f"### Question:\n{query}\n\n"
        prompt += "### Answer (cite sources using [arXiv:id] format):\n"

        return prompt

    def create_structured_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build a prompt dict that tells Ollama to respond in structured JSON.

        Returns a dict with two keys:
            "prompt": the same text as create_rag_prompt()
            "format": the JSON schema of RAGResponse

        The "format" key is an Ollama-specific feature (available from Ollama 0.3+).
        When present, Ollama constrains its token sampling so the output is always
        valid JSON matching the schema. This replaces the need to regex-parse free-form
        text — the model is literally forced to produce {"answer": "...", "sources": [...], ...}.

        Example of what gets sent to Ollama:
            {
                "model": "llama3.2:1b",
                "prompt": "You are an AI assistant...",
                "format": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "sources": {"type": "array", "items": {"type": "string"}},
                        ...
                    },
                    "required": ["answer"]
                }
            }

        Used by OllamaClient.generate_rag_answer() (non-streaming).
        Not used for streaming because structured output can't be emitted token-by-token
        — the JSON only makes sense as a complete object.
        """
        return {
            "prompt": self.create_rag_prompt(query, chunks),
            "format": RAGResponse.model_json_schema(),
        }


class ResponseParser:
    """
    Parses and validates the raw string returned by Ollama into a RAGResponse.

    WHY three levels of parsing?

    Level 1 — direct JSON parse + Pydantic validation:
        Ollama's structured output should produce clean JSON. This is the
        happy path — works ~90% of the time with structured format enabled.

    Level 2 — regex JSON extraction:
        Sometimes the model wraps JSON in markdown fences or adds extra text
        before/after. e.g.: "Here is my answer:\n```json\n{...}\n```"
        The regex r"\{.*\}" with DOTALL finds the JSON object within the noise.

    Level 3 — plain text fallback:
        If no valid JSON is found at all, treat the entire response as the
        answer text. The user still gets a readable response, just without
        structured sources/confidence/citations. confidence="low" signals
        to the caller that the structured parsing failed.

    WHY not just crash if JSON is malformed?
    A 503 error because Ollama returned slightly malformed JSON is a terrible
    user experience. A plain-text answer with confidence="low" is far better.
    """

    @staticmethod
    def parse_structured_response(response: str) -> Dict[str, Any]:
        """
        Parse Ollama's response into a validated dict.

        :param response: Raw string from Ollama (should be JSON if structured output was used).
        :returns: Dict matching RAGResponse fields.

        parse_structured_response handles the case where the ENTIRE response is
        clean JSON — nothing before it, nothing after it:

            response = '{"answer": "Transformers use self-attention...", "sources": [], "confidence": "high", "citations": ["1706.03762"]}'

            json.loads(response)  ← succeeds immediately
            RAGResponse(**parsed) ← Pydantic validates the fields
            → {"answer": "Transformers use self-attention...", "sources": [], ...}

        If json.loads fails (response is not pure JSON), it calls _extract_json_fallback().
        """
        try:
            parsed_json = json.loads(response)
            validated = RAGResponse(**parsed_json)
            return validated.model_dump()
        except (json.JSONDecodeError, ValidationError):
            return ResponseParser._extract_json_fallback(response)

    @staticmethod
    def _extract_json_fallback(response: str) -> Dict[str, Any]:
        """
        Try to find a JSON object embedded in noisy text, then fall back to plain text.

        "Noisy text" means the model added prose AROUND the JSON instead of
        returning pure JSON. This happens when the model partially ignores the
        structured output instruction. For example:

        Example — JSON wrapped in markdown code fence:
            response = '''
            Here is my answer in JSON format:
            ```json
            {"answer": "Transformers use self-attention...", "confidence": "high"}
            ```
            '''
            json.loads(response)       ← FAILS (the whole string is not valid JSON)
            regex finds {"answer": ...} ← succeeds, extracts just the JSON object
            → {"answer": "Transformers use self-attention...", ...}

        The regex r"\{.*\}" with re.DOTALL:
            \{   = literal opening brace
            .*   = any characters (including newlines, because of re.DOTALL)
            \}   = literal closing brace
            This greedily matches from the FIRST { to the LAST } in the string,
            capturing the entire JSON object even if it spans multiple lines.
        """
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                validated = RAGResponse(**parsed)
                return validated.model_dump()
            except (json.JSONDecodeError, ValidationError):
                pass

        # Final fallback: return the whole response as plain answer text
        return {
            "answer": response,
            "sources": [],
            "confidence": "low",
            "citations": [],
        }
