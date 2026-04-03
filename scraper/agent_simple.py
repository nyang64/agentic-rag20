# scraper/agent_simple.py - Simplified version without external search APIs
# Uses basic requests or can be adapted to use MCP brave-search

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
llm = ChatOpenAI(
    model=os.getenv("NEMOTRON_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.1,
)

# Define Tools using @tool decorator

@tool
def web_search(query: str) -> str:
    """Search the web for current information using DuckDuckGo. Use this for recent events, news, facts, or any current information."""
    try:
        # Simple DuckDuckGo search (no API key required)
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract search results
        results = []
        for result in soup.find_all('div', class_='result__body')[:5]:
            title_elem = result.find('a', class_='result__a')
            snippet_elem = result.find('a', class_='result__snippet')
            if title_elem and snippet_elem:
                title = title_elem.get_text(strip=True)
                snippet = snippet_elem.get_text(strip=True)
                link = title_elem.get('href', '')
                results.append(f"Title: {title}\nSnippet: {snippet}\nURL: {link}\n")
        
        if not results:
            return "No search results found."
        
        return "\n".join(results)
    except Exception as e:
        return f"Error searching web: {str(e)}. Try rephrasing your query."

@tool
def fetch_webpage(url: str) -> str:
    """Fetch the full content of a specific webpage. Use this to get detailed information from a URL found in search results."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Return first 5000 characters
        return text[:5000]
    except Exception as e:
        return f"Error fetching {url}: {str(e)}"

@tool
def search_local_knowledge(query: str) -> str:
    """Search the local knowledge base (previously scraped content). Use this for domain-specific information from scraped websites."""
    try:
        from scraper.raq_query import retrieve_top3
        
        docs = retrieve_top3(query)
        if not docs:
            return "No relevant documents found in local knowledge base."
        
        results = []
        for i, doc in enumerate(docs, 1):
            results.append(
                f"Document {i}:\n"
                f"Source: {doc.metadata['url']}\n"
                f"Title: {doc.metadata.get('title', 'Untitled')}\n"
                f"Content: {doc.page_content[:500]}...\n"
            )
        return "\n".join(results)
    except Exception as e:
        return f"Error searching local knowledge: {str(e)}"


_TOOLS_BY_NAME = {}  # populated in create_agentic_rag


class _AgentWrapper:
    """Simple tool-call loop: call LLM → execute tools → repeat until text answer."""

    def __init__(self, llm_with_tools, system_prompt: str, max_rounds: int = 5):
        self._llm = llm_with_tools
        self._system = system_prompt
        self._max_rounds = max_rounds

    def invoke(self, inputs: dict) -> dict:
        query = inputs.get("input", "")
        messages = [SystemMessage(self._system), HumanMessage(query)]

        for _ in range(self._max_rounds):
            response = self._llm.invoke(messages)
            messages.append(response)

            # No tool calls → final text answer
            if not response.tool_calls:
                return {"output": response.content, "messages": messages}

            # Execute every requested tool
            for tc in response.tool_calls:
                tool_fn = _TOOLS_BY_NAME.get(tc["name"])
                if tool_fn:
                    result = tool_fn.invoke(tc["args"])
                else:
                    result = f"Unknown tool: {tc['name']}"
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        # Fallback: return whatever the last message says
        return {"output": messages[-1].content, "messages": messages}


def create_agentic_rag():
    """Create the Agentic RAG 2.0 system"""

    tools = [
        web_search,
        fetch_webpage,
        search_local_knowledge,
    ]
    _TOOLS_BY_NAME.update({t.name: t for t in tools})

    system_prompt = """You are an intelligent research assistant with access to multiple tools.

Your capabilities:
1. Search the web for current information (web_search) - uses DuckDuckGo
2. Fetch full content from specific webpages (fetch_webpage)
3. Search local knowledge base for domain-specific information (search_local_knowledge)

Guidelines:
- Use each tool at most once per query — do not repeat the same search
- Use search_local_knowledge for domain-specific queries about previously scraped content
- Use web_search for current events, news, or facts not in the local knowledge base
- After receiving tool results, synthesize and respond with a final answer
- Cite your sources with URLs

Current date: 2026-04-03"""

    llm_with_tools = llm.bind_tools(tools)
    return _AgentWrapper(llm_with_tools, system_prompt)


# Export
__all__ = ["create_agentic_rag", "web_search", "fetch_webpage", "search_local_knowledge"]
