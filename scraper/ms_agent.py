"""
scraper/ms_agent.py - Microsoft AutoGen ReAct agent for Agentic RAG

AutoGen concepts used here:
  AssistantAgent          - LLM-backed agent; runs a ReAct loop internally
  OpenAIChatCompletionClient - OpenAI-compatible model client (works with OpenRouter)
  Tools                   - plain async Python functions registered on the agent

The agent calls tools in a loop until it has enough information, then produces
a final response.  A fresh agent is created per request (stateless).

AutoGen version: autogen-agentchat 0.7.x
"""

import os
import asyncio
from typing import Dict, Any, List
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

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
# Tool implementations (async — AutoGen's executor calls these directly)
# ---------------------------------------------------------------------------

async def web_search(query: str) -> str:
    """Search the web for current information using DuckDuckGo.

    Automatically fetches the top result's page content so the caller gets
    real data, not just a list of URLs to follow up on.
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

            # Snippets for all results
            snippets = []
            for r in hits:
                snippets.append(
                    f"Title: {r.get('title', '')}\n"
                    f"Snippet: {r.get('body', '')}\n"
                    f"URL: {r.get('href', '')}\n"
                )

            # Auto-fetch top result so the LLM gets actual page content
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
                    pass  # fall back to snippets only

            result = "Search Results:\n" + "\n".join(snippets)
            if fetched:
                result += f"\n\nPage content from {top_url}:\n{fetched}"
            return result

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _search)
    except Exception as e:
        return f"Error searching web: {str(e)}"


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
# Model client and agent factory
# model_info is required for OpenRouter models — they aren't in AutoGen's
# built-in registry, so AutoGen can't infer capabilities without it.
# ---------------------------------------------------------------------------

def _build_model_client() -> OpenAIChatCompletionClient:
    model = os.getenv("OPENAI_FREE_MODEL", "openai/gpt-oss-20b:free")
    return OpenAIChatCompletionClient(
        model=model,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": "unknown",
        },
    )


def _build_agent() -> AssistantAgent:
    return AssistantAgent(
        name="research_assistant",
        model_client=_build_model_client(),
        tools=[web_search, fetch_webpage, search_local_knowledge],
        system_message=SYSTEM_PROMPT,
        reflect_on_tool_use=True,  # generate a coherent final response after tools finish
    )


# ---------------------------------------------------------------------------
# Public query interface
# ---------------------------------------------------------------------------

async def aquery_agent(question: str) -> Dict[str, Any]:
    """Run the agent to completion and return the final answer."""
    agent = _build_agent()
    result: TaskResult = await agent.run(task=question)

    answer = ""
    tools_used: List[str] = []

    for msg in result.messages:
        cls = type(msg).__name__
        # TextMessage = final answer when reflect_on_tool_use generates a new LLM response
        # ToolCallSummaryMessage = final answer when reflect_on_tool_use summarises tool output
        # Both carry the synthesised answer as a plain string in .content
        if cls in ("TextMessage", "ToolCallSummaryMessage") and getattr(msg, "source", "") not in ("user", ""):
            if hasattr(msg, "content") and isinstance(msg.content, str):
                answer = msg.content
        # Track tool calls
        if "ToolCallRequest" in cls and hasattr(msg, "content"):
            for call in msg.content:
                name = getattr(call, "name", "")
                if name and name not in tools_used:
                    tools_used.append(name)

    # Last-resort fallback: use the final message whatever its type
    if not answer and result.messages:
        last = result.messages[-1]
        if hasattr(last, "content") and isinstance(last.content, str):
            answer = last.content

    return {"answer": answer, "tools_used": tools_used}


async def astream_agent_events(question: str):
    """
    Async generator that yields human-readable strings as the agent runs.
    Yields tool call notifications followed by the final answer.

    Uses duck typing on class name so it stays robust across minor AutoGen
    version changes (0.4 → 0.7 renamed several event classes).
    """
    agent = _build_agent()

    async for event in agent.run_stream(task=question):
        cls = type(event).__name__

        if "ToolCallRequest" in cls and hasattr(event, "content"):
            for call in event.content:
                name = getattr(call, "name", "unknown")
                args = str(getattr(call, "arguments", ""))[:200]
                yield f"[Tool: {name}] {args}\n"

        elif "ToolCallExecution" in cls and hasattr(event, "content"):
            for result in event.content:
                preview = str(getattr(result, "content", result))[:300]
                yield f"[Result]: {preview}...\n\n"

        elif cls in ("TextMessage", "ToolCallSummaryMessage") and getattr(event, "source", "") not in ("user", ""):
            if hasattr(event, "content") and isinstance(event.content, str):
                yield f"\n{event.content}"

        elif cls == "TaskResult":
            break


def query_agent(question: str) -> Dict[str, Any]:
    """Synchronous wrapper — used by integration tests."""
    return asyncio.run(aquery_agent(question))


__all__ = [
    "query_agent",
    "aquery_agent",
    "astream_agent_events",
]
