# scraper/agent_simple.py - Simplified version without external search APIs
# Uses basic requests or can be adapted to use MCP brave-search

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
llm = ChatOpenAI(
    model=os.getenv("OPENAI_FREE_MODEL", "openai/gpt-oss-20b:free"),
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


def create_agentic_rag():
    """Create the Agentic RAG 2.0 system"""
    
    # Define tools list
    tools = [
        web_search,
        fetch_webpage,
        search_local_knowledge,
    ]
    
    # Define system prompt
    system_prompt = """You are an intelligent research assistant with access to multiple tools.

Your capabilities:
1. Search the web for current information (web_search) - uses DuckDuckGo
2. Fetch full content from specific webpages (fetch_webpage)
3. Search local knowledge base for domain-specific information (search_local_knowledge)

Guidelines:
- ALWAYS use web_search for current events, recent information, weather, news, or facts
- Use fetch_webpage to get detailed content from promising URLs in search results
- Use search_local_knowledge for domain-specific queries related to previously scraped content
- Synthesize information from multiple sources when needed
- Cite your sources with URLs
- If information is not found, say so clearly

Thought Process:
1. Analyze the query to determine what information is needed
2. Choose appropriate tools (may use multiple tools in sequence)
3. Gather information from tools
4. Synthesize and provide a comprehensive answer with sources

Current date: 2025-MM-DD (use web search for real-time information)
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create agent using create_tool_calling_agent (LangChain 1.0+)
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
    
    return agent_executor


# Export
__all__ = ["create_agentic_rag", "web_search", "fetch_webpage", "search_local_knowledge"]
