import logging
import time
from typing import List, Optional

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.langfuse.client import LangfuseTracer
from src.services.ollama.client import OllamaClient
from src.services.opensearch.client import OpenSearchClient

from .config import GraphConfig
from .context import Context
from .nodes import (
    ainvoke_generate_answer_step,
    ainvoke_grade_documents_step,
    ainvoke_guardrail_step,
    ainvoke_out_of_scope_step,
    ainvoke_retrieve_step,
    ainvoke_rewrite_query_step,
    continue_after_guardrail,
)
from .state import AgentState
from .tools import create_retriever_tool

logger = logging.getLogger(__name__)


class AgenticRAGService:
    """
    Orchestrates the agentic RAG workflow via a compiled LangGraph graph.

    WHY a graph instead of a simple pipeline?
    The classic RAG pipeline always runs: retrieve → generate.
    That's fine when retrieval always works — but it silently generates
    wrong answers when retrieved docs are irrelevant.

    The graph adds three capabilities:
    1. Guardrail  — rejects out-of-scope queries before touching the index
    2. Grading    — checks whether retrieved docs actually answer the question
    3. Rewriting  — if grading fails, reformulates the query and retries

    WHY compile once, invoke many times?
    Compiling the graph (StateGraph → CompiledGraph) validates all edges,
    builds routing tables, and JIT-optimises the async execution plan.
    This takes ~50ms. Since the graph structure never changes between requests,
    we compile once at startup and call graph.ainvoke() per request, passing
    a fresh Context with the per-request clients.
    """

    def __init__(
        self,
        opensearch_client: OpenSearchClient,
        ollama_client: OllamaClient,
        embeddings_client: JinaEmbeddingsClient,
        langfuse_tracer: Optional[LangfuseTracer] = None,
        graph_config: Optional[GraphConfig] = None,
    ):
        self.opensearch = opensearch_client
        self.ollama = ollama_client
        self.embeddings = embeddings_client
        self.langfuse_tracer = langfuse_tracer
        self.graph_config = graph_config or GraphConfig()

        self.graph = self._build_graph()
        logger.info("AgenticRAGService ready")

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self):
        """
        Define nodes and edges, then compile to an executable graph.

        Node order is encoded entirely in add_edge / add_conditional_edges —
        the node functions themselves are stateless and know nothing about
        what comes before or after them.

        context_schema=Context tells LangGraph to pass a Context instance
        to every node via Runtime[Context]. This is how nodes access clients
        without closures or global state.
        """
        workflow = StateGraph(AgentState, context_schema=Context)

        # The retriever tool is the only thing that needs to be created upfront
        # (ToolNode needs the actual tool object to know its schema at compile time)
        retriever_tool = create_retriever_tool(
            opensearch_client=self.opensearch,
            embeddings_client=self.embeddings,
            top_k=self.graph_config.top_k,
            use_hybrid=self.graph_config.use_hybrid,
        )

        # ── Nodes ─────────────────────────────────────────────────────────────
        workflow.add_node("guardrail", ainvoke_guardrail_step)
        workflow.add_node("out_of_scope", ainvoke_out_of_scope_step)
        workflow.add_node("retrieve", ainvoke_retrieve_step)
        workflow.add_node("tool_retrieve", ToolNode([retriever_tool]))
        workflow.add_node("grade_documents", ainvoke_grade_documents_step)
        workflow.add_node("rewrite_query", ainvoke_rewrite_query_step)
        workflow.add_node("generate_answer", ainvoke_generate_answer_step)

        # ── Edges ─────────────────────────────────────────────────────────────
        workflow.add_edge(START, "guardrail")

        # Score >= threshold → retrieve; score < threshold → reject
        workflow.add_conditional_edges(
            "guardrail",
            continue_after_guardrail,
            {"continue": "retrieve", "out_of_scope": "out_of_scope"},
        )

        workflow.add_edge("out_of_scope", END)

        # If retrieve node emitted tool_calls → execute them; else → END
        workflow.add_conditional_edges(
            "retrieve",
            tools_condition,
            {"tools": "tool_retrieve", END: END},
        )

        workflow.add_edge("tool_retrieve", "grade_documents")

        # grade_documents writes routing_decision into state; edge reads it back
        workflow.add_conditional_edges(
            "grade_documents",
            lambda state: state.get("routing_decision", "generate_answer"),
            {"generate_answer": "generate_answer", "rewrite_query": "rewrite_query"},
        )

        workflow.add_edge("rewrite_query", "retrieve")   # retry loop
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    # ── Public API ────────────────────────────────────────────────────────────

    async def ask(
        self,
        query: str,
        user_id: str = "api_user",
        model: Optional[str] = None,
    ) -> dict:
        """
        Run the agentic RAG workflow for one query.

        Returns a dict with:
            query, answer, sources, reasoning_steps,
            retrieval_attempts, rewritten_query, execution_time, guardrail_score
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        model_to_use = model or self.graph_config.model
        logger.info(f"Agentic RAG — query: '{query[:80]}', model: {model_to_use}")

        # ── Trace (optional) ──────────────────────────────────────────────────
        trace = None
        if self.langfuse_tracer and self.langfuse_tracer.client:
            try:
                trace = self.langfuse_tracer.client.start_as_current_span(
                    name="agentic_rag_request"
                )
            except Exception as e:
                logger.warning(f"Failed to create Langfuse trace: {e}")

        async def _run(active_trace):
            return await self._execute_graph(query, model_to_use, user_id, active_trace)

        try:
            if trace is not None:
                with trace as trace_obj:
                    trace_obj.update(
                        input={"query": query},
                        user_id=user_id,
                        session_id=f"session_{user_id}",
                    )
                    return await _run(trace_obj)
            else:
                return await _run(None)
        except Exception as e:
            logger.error(f"Agentic RAG failed: {e}")
            raise

    async def _execute_graph(
        self,
        query: str,
        model_to_use: str,
        user_id: str,
        trace,
    ) -> dict:
        """Invoke the compiled graph and extract structured results."""
        start = time.time()

        state_input = {
            "messages": [HumanMessage(content=query)],
            "retrieval_attempts": 0,
            "guardrail_result": None,
            "routing_decision": None,
            "sources": None,
            "relevant_sources": [],
            "relevant_tool_artefacts": None,
            "grading_results": [],
            "metadata": {},
            "original_query": None,
            "rewritten_query": None,
        }

        runtime_context = Context(
            ollama_client=self.ollama,
            opensearch_client=self.opensearch,
            embeddings_client=self.embeddings,
            langfuse_tracer=self.langfuse_tracer,
            trace=trace,
            langfuse_enabled=(
                self.langfuse_tracer is not None
                and self.langfuse_tracer.client is not None
            ),
            model_name=model_to_use,
            temperature=self.graph_config.temperature,
            top_k=self.graph_config.top_k,
            max_retrieval_attempts=self.graph_config.max_retrieval_attempts,
            guardrail_threshold=self.graph_config.guardrail_threshold,
        )

        config = {"thread_id": f"user_{user_id}_{int(start)}"}

        # CallbackHandler auto-traces every LangChain LLM call inside the graph
        if self.langfuse_tracer and trace:
            try:
                handler = self.langfuse_tracer.get_callback_handler()
                if handler:
                    config["callbacks"] = [handler]
            except Exception as e:
                logger.warning(f"CallbackHandler unavailable: {e}")

        result = await self.graph.ainvoke(
            state_input,
            config=config,
            context=runtime_context,
        )

        elapsed = time.time() - start

        answer = self._extract_answer(result)
        sources = self._extract_sources(result)
        reasoning_steps = self._extract_reasoning_steps(result)

        if trace:
            try:
                trace.update(output={
                    "answer": answer,
                    "sources_count": len(sources),
                    "retrieval_attempts": result.get("retrieval_attempts", 0),
                    "execution_time": elapsed,
                })
                trace.end()
                self.langfuse_tracer.flush()
            except Exception:
                pass

        logger.info(
            f"Agentic RAG done — {elapsed:.2f}s, "
            f"{result.get('retrieval_attempts', 0)} retrieval attempt(s), "
            f"{len(sources)} sources"
        )

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "reasoning_steps": reasoning_steps,
            "retrieval_attempts": result.get("retrieval_attempts", 0),
            "rewritten_query": result.get("rewritten_query"),
            "execution_time": elapsed,
            "guardrail_score": (
                result.get("guardrail_result").score
                if result.get("guardrail_result") else None
            ),
        }

    # ── Result extraction helpers ─────────────────────────────────────────────

    def _extract_answer(self, result: dict) -> str:
        messages = result.get("messages", [])
        if not messages:
            return "No answer generated."
        final = messages[-1]
        return final.content if hasattr(final, "content") else str(final)

    def _extract_sources(self, result: dict) -> List[dict]:
        return [
            s.to_dict() if hasattr(s, "to_dict") else s
            for s in result.get("relevant_sources", [])
        ]

    def _extract_reasoning_steps(self, result: dict) -> List[str]:
        steps = []
        if gr := result.get("guardrail_result"):
            steps.append(f"Validated query scope (score: {gr.score}/100)")
        if (attempts := result.get("retrieval_attempts", 0)) > 0:
            steps.append(f"Retrieved documents ({attempts} attempt(s))")
        grading = result.get("grading_results", [])
        if grading:
            relevant = sum(1 for g in grading if g.is_relevant)
            steps.append(f"Graded documents ({relevant} relevant)")
        if result.get("rewritten_query"):
            steps.append("Rewrote query for better retrieval")
        steps.append("Generated answer from context")
        return steps
