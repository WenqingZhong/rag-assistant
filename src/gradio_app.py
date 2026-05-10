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


def create_interface() -> gr.Blocks:
    with gr.Blocks(title="arXiv RAG Chat", theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            """
            # arXiv Paper Curator — RAG Chat

            Ask questions about machine learning and AI research papers.
            The system retrieves relevant chunks from indexed papers and generates an answer.
            """
        )

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
                info="More chunks = more context but slower generation",
            )
            use_hybrid = gr.Checkbox(
                value=True,
                label="Hybrid search (BM25 + vector)",
                info="Better recall than keyword-only; falls back to BM25 if Jina is unavailable",
            )
            model_choice = gr.Dropdown(
                choices=["llama3.2:1b", "llama3.2:3b", "llama3.1:8b"],
                value=DEFAULT_MODEL,
                label="Ollama model",
                info="Larger models give better answers but are slower",
            )
            categories = gr.Textbox(
                label="arXiv categories (optional)",
                placeholder="cs.AI, cs.LG",
                info="Comma-separated. Leave empty to search all categories.",
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

        gr.Markdown(
            """
            ---
            **API:** `http://localhost:8000` must be running before using this UI.

            **Common categories:** cs.AI · cs.LG · cs.CL · cs.CV · cs.NE · stat.ML
            """
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
