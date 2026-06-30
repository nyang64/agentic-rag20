"""
scraper/llamaindex_agent.py - LlamaIndex ReActAgent for Agentic RAG (LlamaIndex 0.14+)

LlamaIndex 0.14 moved to a workflow-based agent API:
  agent.run(user_msg="...") → WorkflowHandler
  await handler             → AgentOutput (final response)
  handler.stream_events()   → async generator of events (ToolCall, ToolCallResult, AgentOutput...)

Three tools mirror the LangGraph implementation:
  - web_search              : DuckDuckGo web search
  - fetch_webpage           : Download and parse a URL
  - search_local_knowledge  : Query the pgvector knowledge base
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent, AgentOutput, ToolCall, ToolCallResult
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.llms.openai.utils import ALL_AVAILABLE_MODELS

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _setup_tracing() -> None:
    """Configure OTel → Phoenix tracing if PHOENIX_COLLECTOR_ENDPOINT is set.

    Mirrors the LangSmith pattern: env var present = tracing on, absent = zero overhead.
    Safe to call multiple times (OTel SDK ignores duplicate provider registration).
    """
    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    if not endpoint:
        return

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    LlamaIndexInstrumentor().instrument(tracer_provider=provider)


_setup_tracing()


# Register OpenRouter model names so LlamaIndex can resolve their context window.
# LlamaIndex's OpenAI class validates model names against a hardcoded dict;
# OpenRouter uses namespaced names like "provider/model:tier" that aren't in that list.
_OPENROUTER_MODELS = {
    "openai/gpt-oss-20b:free": 128000,
    "openai/gpt-4o-mini": 128000,
    "openai/gpt-4o": 128000,
    "nvidia/nemotron-3-super-120b-a12b:free": 128000,
    "deepseek/deepseek-chat-v3.1:free": 64000,
    "meta-llama/llama-3.3-70b-instruct:free": 128000,
    "google/gemma-3-27b-it:free": 8192,
}
ALL_AVAILABLE_MODELS.update(_OPENROUTER_MODELS)


SYSTEM_PROMPT = """You are an intelligent research assistant with access to three tools:

1. web_search              - Search the web for current events, news, and real-time facts
2. fetch_webpage           - Retrieve full text content from a specific URL
3. search_local_knowledge  - Query the local pgvector knowledge base of scraped content

Instructions:
- For current events, weather, or recent facts: use web_search first
- To get full details from a URL: use fetch_webpage
- For domain-specific queries about scraped content: use search_local_knowledge
- Always cite sources with URLs when available
- If information is unavailable, say so clearly
- Synthesize information from multiple sources when helpful

After your main answer, add a Sources section listing URLs used."""


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _web_search(query: str) -> str:
    """Search the web for current information using DuckDuckGo."""
    try:
        import requests
        from bs4 import BeautifulSoup

        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        results = []
        for result in soup.find_all("div", class_="result__body")[:5]:
            title_elem = result.find("a", class_="result__a")
            snippet_elem = result.find("a", class_="result__snippet")
            if title_elem and snippet_elem:
                results.append(
                    f"Title: {title_elem.get_text(strip=True)}\n"
                    f"Snippet: {snippet_elem.get_text(strip=True)}\n"
                    f"URL: {title_elem.get('href', '')}\n"
                )
        return "\n".join(results) if results else "No search results found."
    except Exception as e:
        return f"Error searching web: {str(e)}"


def _fetch_webpage(url: str) -> str:
    """Fetch and return the text content of a webpage."""
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        return soup.get_text(" ", strip=True)[:5000]
    except Exception as e:
        return f"Error fetching {url}: {str(e)}"


def _search_local_knowledge(query: str) -> str:
    """Search the local pgvector knowledge base for domain-specific content."""
    try:
        from scraper.raq_query import retrieve_top3
        docs = retrieve_top3(query)

        if not docs:
            return "No relevant documents found in local knowledge base."

        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(
                f"Document {i}:\n"
                f"Source: {doc.metadata.get('url', 'Unknown')}\n"
                f"Title: {doc.metadata.get('title', 'Untitled')}\n"
                f"Content: {doc.page_content[:500]}\n"
            )
        return "\n".join(parts)
    except Exception as e:
        return f"Error searching local knowledge: {str(e)}"


# ---------------------------------------------------------------------------
# LlamaIndex FunctionTools
# ---------------------------------------------------------------------------

web_search_tool = FunctionTool.from_defaults(
    fn=_web_search,
    name="web_search",
    description=(
        "Search the web for current information using DuckDuckGo. "
        "Use this for recent events, news, weather, or any real-time facts."
    ),
)

fetch_webpage_tool = FunctionTool.from_defaults(
    fn=_fetch_webpage,
    name="fetch_webpage",
    description=(
        "Fetch the full text content of a specific webpage given its URL. "
        "Use this to get detailed information from a URL found in search results."
    ),
)

search_local_knowledge_tool = FunctionTool.from_defaults(
    fn=_search_local_knowledge,
    name="search_local_knowledge",
    description=(
        "Search the local pgvector knowledge base of previously scraped content. "
        "Use this for domain-specific queries about content in the local database."
    ),
)

_TOOLS = [web_search_tool, fetch_webpage_tool, search_local_knowledge_tool]


# ---------------------------------------------------------------------------
# LLM (OpenRouter — OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

def _build_llm() -> LlamaOpenAI:
    model = os.getenv("OPENAI_FREE_MODEL", "openai/gpt-oss-20b:free")
    # Register the model if not in the known list so context_window lookup succeeds.
    if model not in ALL_AVAILABLE_MODELS:
        ALL_AVAILABLE_MODELS[model] = 128000
    return LlamaOpenAI(
        model=model,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        api_base="https://openrouter.ai/api/v1",
        temperature=0.1,
        # max_tokens must be set explicitly: LlamaIndex otherwise calls tiktoken to
        # infer the budget, and tiktoken doesn't know OpenRouter model names.
        max_tokens=4096,
    )


# ---------------------------------------------------------------------------
# Agent factory
# LlamaIndex 0.14: ReActAgent is a Workflow; instantiate directly (no from_tools).
# agent.run(user_msg=...) → WorkflowHandler → await for AgentOutput
# ---------------------------------------------------------------------------

def get_agent() -> ReActAgent:
    """Create a fresh ReActAgent for a single request (stateless)."""
    return ReActAgent(
        tools=_TOOLS,
        llm=_build_llm(),
        system_prompt=SYSTEM_PROMPT,
        timeout=120,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Public query interface
# ---------------------------------------------------------------------------

async def aquery_agent(question: str) -> Dict[str, Any]:
    """Run the agent and return the final answer."""
    agent = get_agent()
    handler = agent.run(user_msg=question)
    result: AgentOutput = await handler
    return {
        "answer": result.response.content if result.response else "",
        "tools_used": [tc.tool_name for tc in result.tool_calls] if result.tool_calls else [],
    }


async def astream_agent_events(question: str):
    """
    Async generator that yields human-readable strings as the agent runs.
    Yields tool call notifications followed by the final answer.
    """
    agent = get_agent()
    handler = agent.run(user_msg=question)

    async for event in handler.stream_events():
        if isinstance(event, ToolCall):
            tool_input = str(event.tool_kwargs or "")[:200]
            yield f"[Tool: {event.tool_name}] {tool_input}\n"
        elif isinstance(event, ToolCallResult):
            preview = str(event.tool_output or "")[:300]
            yield f"[Result]: {preview}...\n\n"
        elif isinstance(event, AgentOutput):
            if event.response and event.response.content:
                yield f"\n{event.response.content}"


__all__ = [
    "get_agent",
    "aquery_agent",
    "astream_agent_events",
    "web_search_tool",
    "fetch_webpage_tool",
    "search_local_knowledge_tool",
]
