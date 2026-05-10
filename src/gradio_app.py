import json
import logging
from typing import AsyncIterator

import gradio as gr
import httpx

logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8000/api/v1"
DEFAULT_MODEL = "llama3.2:1b"


async def stream_response(
    query: str,
    top_k: int = 3,
    use_hybrid: bool = True,
    model: str = DEFAULT_MODEL,
    categories: str = "",
) -> AsyncIterator[str]:
    """
    Stream tokens from POST /api/v1/stream and yield progressive Markdown.

    Our API sends two SSE event types:
        Token events:  {"response": "Trans", "done": false}
                       {"response": "form",  "done": false}
                       ...
                       {"response": "",      "done": true, "total_duration": ...}

        Sources event: {"type": "sources", "sources": [...], "chunks_used": 3, "search_mode": "hybrid"}

    We accumulate token["response"] fragments into current_answer and yield
    the growing answer string after each token so Gradio updates the UI
    in real time. After the sources event arrives we append the source links.
    """
    if not query.strip():
        yield "Please enter a question."
        return

    category_list = [c.strip() for c in categories.split(",") if c.strip()] if categories else None

    payload = {
        "query": query,
        "top_k": top_k,
        "use_hybrid": use_hybrid,
        "model": model,
        "categories": category_list,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", f"{API_BASE_URL}/stream", json=payload) as response:
                if response.status_code != 200:
                    yield f"Error: API returned status {response.status_code}"
                    return

                current_answer = ""
                sources: list = []
                chunks_used = 0
                search_mode = ""

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    try:
                        data = json.loads(line[6:])  # strip "data: " prefix
                    except json.JSONDecodeError:
                        continue

                    if "error" in data:
                        yield f"**Error:** {data['error']}"
                        return

                    # Sources event — save metadata, don't update text yet
                    if data.get("type") == "sources":
                        sources = data.get("sources", [])
                        chunks_used = data.get("chunks_used", 0)
                        search_mode = data.get("search_mode", "")
                        # Yield the final answer with sources appended
                        yield _format(current_answer, sources, chunks_used, search_mode)
                        continue

                    # Token event — accumulate and stream
                    fragment = data.get("response", "")
                    if fragment:
                        current_answer += fragment
                        yield _format(current_answer, sources, chunks_used, search_mode)

    except httpx.RequestError as e:
        yield f"**Connection error:** {e}\n\nMake sure the API server is running at `{API_BASE_URL}`."
    except Exception as e:
        yield f"**Unexpected error:** {e}"


def _format(answer: str, sources: list, chunks_used: int, search_mode: str) -> str:
    """Render answer + optional source footer as Markdown."""
    if not (sources or chunks_used):
        return answer

    footer = f"\n\n---\n**Search info:** mode={search_mode}, chunks={chunks_used}"
    if sources:
        links = "\n".join(
            f"{i}. [{s.split('/')[-1]}]({s})" for i, s in enumerate(sources[:5], 1)
        )
        footer += f"\n\n**Sources:**\n{links}"
        if len(sources) > 5:
            footer += f"\n... and {len(sources) - 5} more"

    return answer + footer


async def agentic_response(query: str) -> tuple[str, str, str]:
    """
    Call POST /api/v1/ask-agentic and return (answer, reasoning, sources).

    Returns three strings so Gradio can populate three separate output boxes:
    - answer: the final LLM answer
    - reasoning: bullet list of what the agent decided at each node
    - sources: formatted list of retrieved papers with metadata
    """
    if not query.strip():
        return "Please enter a question.", "", ""

    payload = {"query": query, "top_k": 3, "use_hybrid": True, "model": DEFAULT_MODEL}

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{API_BASE_URL}/ask-agentic", json=payload)

        if response.status_code != 200:
            return f"Error: API returned status {response.status_code}\n{response.text}", "", ""

        data = response.json()

        answer = data.get("answer", "No answer generated.")

        steps = data.get("reasoning_steps", [])
        guardrail = data.get("guardrail_score")
        attempts = data.get("retrieval_attempts", 0)
        rewritten = data.get("rewritten_query")

        reasoning_parts = [f"- {s}" for s in steps]
        if guardrail is not None:
            reasoning_parts.append(f"\n**Guardrail score:** {guardrail}/100")
        reasoning_parts.append(f"**Retrieval attempts:** {attempts}")
        if rewritten:
            reasoning_parts.append(f"**Rewritten query:** {rewritten}")
        reasoning = "\n".join(reasoning_parts)

        raw_sources = data.get("sources", [])
        if raw_sources and isinstance(raw_sources[0], dict):
            source_lines = []
            for i, s in enumerate(raw_sources[:5], 1):
                title = s.get("title", "Unknown")
                arxiv_id = s.get("arxiv_id", "")
                url = s.get("url", f"https://arxiv.org/abs/{arxiv_id}")
                score = s.get("relevance_score", 0.0)
                source_lines.append(f"{i}. [{title}]({url}) — score: {score:.3f}")
            sources = "\n".join(source_lines) or "No sources returned."
        elif raw_sources:
            sources = "\n".join(f"{i}. {s}" for i, s in enumerate(raw_sources[:5], 1))
        else:
            sources = "No sources returned."

        return answer, reasoning, sources

    except httpx.RequestError as e:
        return f"**Connection error:** {e}\n\nMake sure the API is running at `{API_BASE_URL}`.", "", ""
    except Exception as e:
        return f"**Unexpected error:** {e}", "", ""


def create_interface() -> gr.Blocks:
    with gr.Blocks(title="arXiv RAG Chat", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# arXiv Paper Curator — RAG Chat")

        with gr.Tabs():

            # ── Tab 1: Classic streaming RAG ──────────────────────────────────
            with gr.TabItem("Streaming RAG"):
                gr.Markdown("Stream tokens live from `/api/v1/stream` — classic retrieve → generate pipeline.")

                with gr.Row():
                    with gr.Column(scale=3):
                        query_input = gr.Textbox(
                            label="Your question",
                            placeholder="What is self-attention?",
                            lines=2,
                            max_lines=5,
                        )
                    with gr.Column(scale=1):
                        submit_btn = gr.Button("Ask", variant="primary", size="lg")

                with gr.Accordion("Advanced options", open=False):
                    top_k = gr.Slider(
                        minimum=1, maximum=10, value=3, step=1,
                        label="Chunks to retrieve",
                    )
                    use_hybrid = gr.Checkbox(value=True, label="Hybrid search (BM25 + vector)")
                    model_choice = gr.Dropdown(
                        choices=["llama3.2:1b", "llama3.2:3b", "llama3.1:8b"],
                        value=DEFAULT_MODEL,
                        label="Ollama model",
                    )
                    categories = gr.Textbox(
                        label="arXiv categories (optional)",
                        placeholder="cs.AI, cs.LG",
                    )

                response_output = gr.Markdown(
                    value="Ask a question to get started.",
                    label="Answer",
                    height=400,
                )

                gr.Examples(
                    examples=[
                        ["What is self-attention?", 3, True, "llama3.2:1b", "cs.AI, cs.LG"],
                        ["How do convolutional neural networks work?", 5, True, "llama3.2:1b", "cs.CV"],
                        ["What is reinforcement learning?", 3, True, "llama3.2:1b", "cs.LG, cs.AI"],
                        ["Explain large language model pre-training", 4, True, "llama3.2:1b", "cs.CL"],
                    ],
                    inputs=[query_input, top_k, use_hybrid, model_choice, categories],
                )

                submit_btn.click(
                    fn=stream_response,
                    inputs=[query_input, top_k, use_hybrid, model_choice, categories],
                    outputs=[response_output],
                )
                query_input.submit(
                    fn=stream_response,
                    inputs=[query_input, top_k, use_hybrid, model_choice, categories],
                    outputs=[response_output],
                )

            # ── Tab 2: Agentic RAG ────────────────────────────────────────────
            with gr.TabItem("Agentic RAG"):
                gr.Markdown(
                    "Calls `/api/v1/ask-agentic` — the agent decides whether to retrieve, "
                    "grades document relevance, and rewrites the query if needed."
                )

                with gr.Row():
                    with gr.Column(scale=3):
                        agentic_query = gr.Textbox(
                            label="Your question",
                            placeholder="What is self-attention?",
                            lines=2,
                            max_lines=5,
                        )
                    with gr.Column(scale=1):
                        agentic_btn = gr.Button("Ask Agent", variant="primary", size="lg")

                agentic_answer = gr.Markdown(
                    value="Ask a question to get started.",
                    label="Answer",
                    height=300,
                )

                with gr.Row():
                    with gr.Column():
                        agentic_reasoning = gr.Markdown(
                            value="",
                            label="Agent reasoning",
                            height=150,
                        )
                    with gr.Column():
                        agentic_sources = gr.Markdown(
                            value="",
                            label="Sources",
                            height=150,
                        )

                gr.Examples(
                    examples=[
                        ["What is self-attention in transformers?"],
                        ["How does BERT work?"],
                        ["What is 2+2?"],          # should be rejected by guardrail
                        ["What is a dog?"],         # should be rejected by guardrail
                    ],
                    inputs=[agentic_query],
                )

                agentic_btn.click(
                    fn=agentic_response,
                    inputs=[agentic_query],
                    outputs=[agentic_answer, agentic_reasoning, agentic_sources],
                )
                agentic_query.submit(
                    fn=agentic_response,
                    inputs=[agentic_query],
                    outputs=[agentic_answer, agentic_reasoning, agentic_sources],
                )

        gr.Markdown(
            "**API:** `http://localhost:8000` must be running. "
            "**Categories:** cs.AI · cs.LG · cs.CL · cs.CV · cs.NE · stat.ML"
        )

    return interface


def main():
    print(f"Starting Gradio UI — connecting to {API_BASE_URL}")
    create_interface().launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
