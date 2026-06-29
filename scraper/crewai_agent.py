"""
scraper/crewai_agent.py - CrewAI agent for Agentic RAG

CrewAI concepts used here:
  Agent    - autonomous unit with a role, goal, backstory, and tools
  Task     - a specific job described in natural language, assigned to an agent
  Crew     - orchestrates one or more agents through one or more tasks
  BaseTool - base class for custom tools the agent can call
  LLM      - wraps LiteLLM; OpenRouter models use the "openrouter/<model>" prefix

For a single-user Q&A use case we create a fresh Task and Crew per request
so each query is self-contained with no shared state between requests.
"""

import os
from typing import Any, Dict
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# LLM — CrewAI uses LiteLLM under the hood.
# OpenRouter models require the "openrouter/" prefix so LiteLLM routes them
# to https://openrouter.ai/api/v1 automatically.
# ---------------------------------------------------------------------------

def _build_llm() -> LLM:
    model = os.getenv("OPENAI_FREE_MODEL", "openai/gpt-4o-mini")
    return LLM(
        model=f"openrouter/{model}",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.1,
        max_tokens=4096,
    )


# ---------------------------------------------------------------------------
# Custom Tools (BaseTool subclasses)
# ---------------------------------------------------------------------------

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Search the web for current information using DuckDuckGo. "
        "Use this for recent events, news, weather, or any real-time facts. "
        "Input: a search query string."
    )

    def _run(self, query: str) -> str:
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


class FetchWebpageTool(BaseTool):
    name: str = "fetch_webpage"
    description: str = (
        "Fetch the full text content of a specific webpage given its URL. "
        "Use this to get detailed information from a URL found in search results. "
        "Input: a URL string."
    )

    def _run(self, url: str) -> str:
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


class SearchLocalKnowledgeTool(BaseTool):
    name: str = "search_local_knowledge"
    description: str = (
        "Search the local pgvector knowledge base of previously scraped content. "
        "Use this for domain-specific queries about content stored in the local database. "
        "Input: a search query string."
    )

    def _run(self, query: str) -> str:
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
# Agent definition
# The agent's role/goal/backstory act as the system prompt in CrewAI.
# ---------------------------------------------------------------------------

AGENT_ROLE = "Research Assistant"

AGENT_GOAL = (
    "Find accurate, well-sourced answers to user questions using available tools. "
    "Always cite URLs as sources when using web search or fetch_webpage."
)

AGENT_BACKSTORY = (
    "You are an expert researcher with access to three tools: web search for "
    "real-time information, webpage fetching for reading specific URLs, and a "
    "local knowledge base of scraped domain content. You synthesize information "
    "from multiple sources and always provide clear, concise answers with sources cited."
)


def _build_agent() -> Agent:
    return Agent(
        role=AGENT_ROLE,
        goal=AGENT_GOAL,
        backstory=AGENT_BACKSTORY,
        tools=[WebSearchTool(), FetchWebpageTool(), SearchLocalKnowledgeTool()],
        llm=_build_llm(),
        verbose=False,
        max_iter=10,
        allow_delegation=False,
    )


# ---------------------------------------------------------------------------
# Public query interface
# A new Task and Crew are created per query so requests are stateless.
# crew.kickoff() is synchronous; use aquery_agent for async FastAPI contexts.
# ---------------------------------------------------------------------------

def query_agent(question: str) -> Dict[str, Any]:
    """Synchronously run the CrewAI agent and return the answer."""
    agent = _build_agent()

    task = Task(
        description=(
            f"Answer the following question thoroughly using your available tools:\n\n"
            f"{question}\n\n"
            "Use web_search for current information, fetch_webpage to read specific "
            "URLs, and search_local_knowledge for domain-specific content. "
            "Cite all sources with their URLs."
        ),
        expected_output=(
            "A comprehensive, accurate answer followed by a Sources section "
            "listing all URLs referenced."
        ),
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    return {
        "answer": result.raw if hasattr(result, "raw") else str(result),
    }


async def aquery_agent(question: str) -> Dict[str, Any]:
    """Async wrapper — runs crew.kickoff() in a thread pool."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, query_agent, question)


__all__ = [
    "query_agent",
    "aquery_agent",
    "WebSearchTool",
    "FetchWebpageTool",
    "SearchLocalKnowledgeTool",
]
