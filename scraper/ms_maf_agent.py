"""
scraper/ms_maf_agent.py - Microsoft Agent Framework (MAF) agent for Agentic RAG

MAF concepts used here:
  Agent                      - LLM-backed agent; handles tool-calling loop internally
  OpenAIChatCompletionClient - OpenAI Chat Completions client (works with OpenRouter)
  @tool                      - decorator that registers a Python function as an agent tool
  AgentResponse              - result of agent.run(); use .text for the final answer
  ResponseStream             - async iterable of AgentResponseUpdate for streaming

Why OpenAIChatCompletionClient (not OpenAIChatClient):
  OpenAIChatClient uses the OpenAI Responses API which is not yet supported by
  OpenRouter. OpenAIChatCompletionClient uses the Chat Completions API which is
  OpenAI-compatible and works with any OpenRouter model.

MAF package: agent-framework-core + agent-framework-openai
"""

import os
import asyncio
from typing import Dict, Any
from dotenv import load_dotenv

from agent_framework import Agent, tool
from agent_framework_openai import OpenAIChatCompletionClient

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _setup_tracing() -> None:
    """Wire MAF's native OTel instrumentation to a Phoenix collector.

    Reuses the same PHOENIX_COLLECTOR_ENDPOINT env var as the llamaindex branch.
    MAF expects OTEL_EXPORTER_OTLP_ENDPOINT as the base URL (without /v1/traces),
    so we strip the path before handing it to configure_otel_providers().
    No-op when the env var is absent.
    """
    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    if not endpoint:
        return
    from urllib.parse import urlparse
    parsed = urlparse(endpoint)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", base_url)
    os.environ.setdefault("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    from agent_framework.observability import configure_otel_providers
    configure_otel_providers(enable_sensitive_data=True)


_setup_tracing()


SYSTEM_PROMPT = """You are an intelligent research assistant. You have EXACTLY three tools available — no others:

1. web_search(query)             - Search the web; returns snippet + page content from top result
2. fetch_webpage(url)            - Retrieve full text of a specific URL
3. search_local_knowledge(query) - Query the local pgvector knowledge base of scraped website content

TOOL SELECTION RULES (follow strictly):
- User asks about "local knowledge", "knowledge base", or content from a specific website → call search_local_knowledge FIRST
- User asks about current events, weather, news, or real-time data → call web_search
- User provides a URL to read → call fetch_webpage
- Never call a tool that is not in this list (web_search, fetch_webpage, search_local_knowledge)
- Never say you are "ready to use" tools — always call the right tool immediately

After using a tool, synthesize its output into a direct, informative answer.
Always cite sources with URLs when available."""


# ---------------------------------------------------------------------------
# Tools — decorated with @tool so MAF infers the JSON schema from the
# function signature and docstring automatically.
# ---------------------------------------------------------------------------

@tool
async def web_search(query: str) -> str:
    """Search the web for current information using DuckDuckGo.

    Automatically fetches the top result's page content so the agent
    receives actual data, not just a list of URLs.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        from ddgs import DDGS

        def _search():
            with DDGS() as ddgs:
                hits = list(ddgs.text(query, max_results=5))
            if not hits:
                return "No search results found."

            snippets = []
            for r in hits:
                snippets.append(
                    f"Title: {r.get('title', '')}\n"
                    f"Snippet: {r.get('body', '')}\n"
                    f"URL: {r.get('href', '')}\n"
                )

            # Auto-fetch top result for actual page content
            top_url = hits[0].get("href", "")
            fetched = ""
            if top_url:
                try:
                    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
                    resp = requests.get(top_url, headers=headers, timeout=8)
                    soup = BeautifulSoup(resp.text, "html.parser")
                    for el in soup(["script", "style", "nav", "footer", "header"]):
                        el.decompose()
                    fetched = soup.get_text(" ", strip=True)[:2000]
                except Exception:
                    pass

            result = "Search Results:\n" + "\n".join(snippets)
            if fetched:
                result += f"\n\nPage content from {top_url}:\n{fetched}"
            return result

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _search)
    except Exception as e:
        return f"Error searching web: {str(e)}"


@tool
async def fetch_webpage(url: str) -> str:
    """Fetch the full text content of a specific webpage given its URL."""
    try:
        import requests
        from bs4 import BeautifulSoup

        def _fetch():
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()
            return soup.get_text(" ", strip=True)[:5000]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _fetch)
    except Exception as e:
        return f"Error fetching {url}: {str(e)}"


@tool
async def search_local_knowledge(query: str) -> str:
    """Search the local pgvector knowledge base for domain-specific content."""
    try:
        from scraper.raq_query import retrieve_top3

        def _search():
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

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _search)
    except Exception as e:
        return f"Error searching local knowledge: {str(e)}"


# ---------------------------------------------------------------------------
# Client and agent factory — fresh agent per request (stateless)
# ---------------------------------------------------------------------------

def _build_client() -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(
        model=os.getenv("OPENAI_FREE_MODEL", "openai/gpt-oss-20b:free"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )


def _build_agent() -> Agent:
    return Agent(
        client=_build_client(),
        instructions=SYSTEM_PROMPT,
        tools=[web_search, fetch_webpage, search_local_knowledge],
    )


# ---------------------------------------------------------------------------
# Public query interface
# ---------------------------------------------------------------------------

_LOCAL_KEYWORDS = ("local knowledge", "knowledge base", "local kb", "scraped")


def _needs_local_search(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _LOCAL_KEYWORDS)


def _augment_with_local(question: str) -> str:
    """Pre-fetch local KB results and embed them in the prompt so free models always see them."""
    try:
        from scraper.raq_query import retrieve_top3, format_docs
        docs = retrieve_top3(question)
        if not docs:
            return question
        context = format_docs(docs)
        sources = "\n".join(
            f"- {d.metadata.get('url', '')} ({d.metadata.get('title', '')})"
            for d in docs
        )
        return (
            f"{question}\n\n"
            f"[Local knowledge base results for your reference:]\n{context}\n\n"
            f"[Sources:]\n{sources}"
        )
    except Exception:
        return question


async def aquery_agent(question: str) -> Dict[str, Any]:
    """Run the MAF agent to completion and return the final answer."""
    augmented = _augment_with_local(question) if _needs_local_search(question) else question
    agent = _build_agent()
    response = await agent.run(augmented)
    return {"answer": response.text or "", "tools_used": []}


async def astream_agent_events(question: str):
    """Async generator yielding text chunks as the MAF agent runs."""
    augmented = _augment_with_local(question) if _needs_local_search(question) else question
    agent = _build_agent()
    # stream=True returns ResponseStream directly (not Awaitable) — no await here
    stream = agent.run(augmented, stream=True)
    async for update in stream:
        text = update.text
        if text:
            yield text


def query_agent(question: str) -> Dict[str, Any]:
    """Synchronous wrapper — used by integration tests."""
    return asyncio.run(aquery_agent(question))


__all__ = [
    "query_agent",
    "aquery_agent",
    "astream_agent_events",
]
