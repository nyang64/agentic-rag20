import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import create_agent
from langsmith import Client  # LangSmith client for checkpointing
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# Optional Brave Search
try:
    from langchain_community.utilities import BraveSearchWrapper
    BRAVE_AVAILABLE = True
except ImportError:
    BRAVE_AVAILABLE = False
    import requests
    from bs4 import BeautifulSoup


load_dotenv()

# -------------------------------------------------------------------
# Initialize LLM
# -------------------------------------------------------------------
llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_FREE_MODEL", "tngtech/deepseek-r1t2-chimera:free"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.1,
)

# -------------------------------------------------------------------
# Tools
# -------------------------------------------------------------------

if BRAVE_AVAILABLE and os.getenv("BRAVE_API_KEY"):
    @tool
    def web_search(query: str) -> str:
        """Search the web for current information using Brave Search."""
        try:
            search = BraveSearchWrapper(
                api_key=os.getenv("BRAVE_API_KEY"),
                search_kwargs={"count": 5}
            )
            return search.run(query)
        except Exception as e:
            return f"Error searching web: {str(e)}"
else:
    @tool
    def web_search(query: str) -> str:
        """Search the web using DuckDuckGo HTML scraping."""
        try:
            url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")

            results = []
            for result in soup.find_all("div", class_="result__body")[:5]:
                title = result.find("a", class_="result__a")
                snippet = result.find("a", class_="result__snippet")
                if title and snippet:
                    results.append(
                        f"Title: {title.get_text(strip=True)}\n"
                        f"Snippet: {snippet.get_text(strip=True)}\n"
                        f"URL: {title.get('href', '')}\n"
                    )

            return "\n".join(results) if results else "No search results found."

        except Exception as e:
            return f"Error searching web: {str(e)}"


@tool
def fetch_webpage(url: str) -> str:
    """Fetch the full content of a specific webpage."""
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        text = soup.get_text(" ", strip=True)
        return text[:5000]

    except Exception as e:
        return f"Error fetching {url}: {str(e)}"


@tool
def search_local_knowledge(query: str) -> str:
    """Search local scraped knowledge."""
    try:
        from scraper.raq_query import retrieve_top3
        docs = retrieve_top3(query)

        if not docs:
            return "No relevant documents found."

        result = []
        for i, doc in enumerate(docs, 1):
            result.append(
                f"Document {i}:\n"
                f"Source: {doc.metadata.get('url', 'Unknown')}\n"
                f"Title: {doc.metadata.get('title', 'Untitled')}\n"
                f"Content: {doc.page_content[:500]}...\n"
            )
        return "\n".join(result)

    except Exception as e:
        return f"Error searching local knowledge: {str(e)}"


# -------------------------------------------------------------------
# Agent Creation (LangGraph)
# -------------------------------------------------------------------

SYSTEM_MESSAGE = f"""
You are an intelligent research assistant with access to multiple tools.

Tools:
1. web_search              - search current web results
2. fetch_webpage           - retrieve full content from a URL
3. search_local_knowledge  - search domain-specific scraped content

Guidelines:
- For recent events or news → use web_search
- For details from specific URLs → use fetch_webpage
- For domain-specific topics → use search_local_knowledge
- Cite URLs when possible.
- If you cannot find information, say so clearly.
- Synthesize results from multiple tools when needed.

Current date: 2025-12-01
"""


def create_agentic_rag():
    """Create a LangGraph ReAct-style agent using new API."""

    tools = [
        web_search,
        fetch_webpage,
        search_local_knowledge,
    ]

    memory = MemorySaver()
    
    langsmith_client = Client(
        api_key=os.getenv("LANGCHAIN_API_KEY")  # set your LangSmith API key
    )
  
    # Build prompt wrapper
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MESSAGE),
        ("user", "{input}"),
        MessagesPlaceholder("messages")
    ])

    # 👇 NEW API
    agent = create_agent(
        model=llm,            # <-- REQUIRED keyword for LangGraph 1.x
        tools=tools,
        checkpointer=memory, 
    )

    return agent


def test_agent():
    graph = create_agentic_rag()
    result = graph.invoke({"input": "What is the weather in New York today?"},
                          thread_id="test_thread_001")
    print(result["messages"][-1]["content"])
    return result


if __name__ == "__main__":
    print("Testing Agentic RAG (LangGraph)...")
    test_agent()

# Export
__all__ = ["create_agentic_rag", "web_search", "fetch_webpage", "search_local_knowledge"]