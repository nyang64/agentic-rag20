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


SYSTEM_PROMPT = """You are an intelligent research assistant with access to three tools:

1. web_search              - Search the web and returns page content from the top result
2. fetch_webpage           - Retrieve full text content from a specific URL
3. search_local_knowledge  - Query the local pgvector knowledge base of scraped content

Instructions:
- For current events, weather, or real-time facts: use web_search (it fetches page content automatically)
- To dig deeper into a specific URL: use fetch_webpage
- For domain-specific queries about scraped content: use search_local_knowledge
- Always cite sources with URLs when available
- Never return just a URL — always extract and present the actual information

After your main answer, add a Sources section listing URLs used."""


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

async def aquery_agent(question: str) -> Dict[str, Any]:
    """Run the MAF agent to completion and return the final answer."""
    agent = _build_agent()
    response = await agent.run(question)
    return {"answer": response.text or "", "tools_used": []}


async def astream_agent_events(question: str):
    """
    Async generator that yields text chunks as the MAF agent runs.

    MAF's ResponseStream is a plain AsyncIterable[AgentResponseUpdate].
    Each update's .text property returns the text content of that chunk.
    """
    agent = _build_agent()
    # stream=True returns ResponseStream directly (not Awaitable) — no await here
    stream = agent.run(question, stream=True)
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
