# scraper/agent.py - Fixed for LangChain 1.0+ with proper imports

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Correct imports for LangChain 1.0+
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

# Optional: Try to import Brave Search, fallback to ddgs
try:
    from langchain_community.utilities import BraveSearchWrapper
    BRAVE_AVAILABLE = True
except ImportError:
    BRAVE_AVAILABLE = False

load_dotenv()

# Helper function to format the output properly
def format_answer_with_sources(text: str) -> str:
    """Ensure Sources section is properly formatted on separate lines"""
    import re
    
    # Pattern to find "**Source**" or "**Sources:**" that might not be on its own line
    # This regex looks for the pattern with potential content before it
    patterns = [
        r'([^\n])\s*\*\*Sources?:?\*\*',  # **Source** or **Sources:** not after newline
        r'([^\n])\s*\bSources?:?\s*-',      # Sources: - without proper line break
    ]
    
    result = text
    
    # Add proper line breaks before Sources section
    for pattern in patterns:
        result = re.sub(pattern, r'\1\n\n**Sources:**\n', result, flags=re.IGNORECASE)
    
    # Ensure Sources: is standardized
    result = re.sub(r'\*\*Source\*\*', '**Sources:**', result)
    result = re.sub(r'\*\*Sources\*\*', '**Sources:**', result)
    
    # Ensure each source URL starts on a new line
    # Look for URLs that aren't on their own line after Sources:
    lines = result.split('\n')
    formatted_lines = []
    in_sources = False
    
    for line in lines:
        if '**Sources:**' in line:
            in_sources = True
            formatted_lines.append(line)
        elif in_sources and ('http://' in line or 'https://' in line):
            # Ensure source lines start with dash
            line = line.strip()
            if not line.startswith('-'):
                line = '- ' + line
            formatted_lines.append(line)
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

# Initialize LLM ## model=os.getenv("DEEPSEEK_FREE_MODEL", "tngtech/deepseek-r1t2-chimera:free"),
llm = ChatOpenAI(
    model=os.getenv("OPENAI_FREE_MODEL", "openai/gpt-oss-20b:free"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.1,
)

# Define Tools

if BRAVE_AVAILABLE and os.getenv("BRAVE_API_KEY"):
    @tool
    def web_search(query: str) -> str:
        """Search the web for current information using Brave Search. Use this for recent events, news, facts, or current information."""
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
        """Search the web for current information using DuckDuckGo. Use this for recent events, news, facts, or current information."""
        try:
            from ddgs import DDGS
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=5):
                    results.append(
                        f"Title: {r.get('title', '')}\n"
                        f"Snippet: {r.get('body', '')}\n"
                        f"URL: {r.get('href', '')}\n"
                    )
            return "\n".join(results) if results else "No search results found."
        except Exception as e:
            return f"Error searching web: {str(e)}"

@tool
def fetch_webpage(url: str) -> str:
    """Fetch the full content of a specific webpage. Use this to get detailed information from a URL."""
    try:
        import requests
        from bs4 import BeautifulSoup
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        
        text = soup.get_text(separator=' ', strip=True)
        return text[:5000]  # Limit to 5000 chars
    except Exception as e:
        return f"Error fetching {url}: {str(e)}"

@tool
def search_local_knowledge(query: str) -> str:
    """Search the local knowledge base (previously scraped content). Use this for domain-specific information."""
    try:
        from scraper.raq_query import retrieve_top3
        
        docs = retrieve_top3(query)
        if not docs:
            return "No relevant documents found in local knowledge base."
        
        results = []
        for i, doc in enumerate(docs, 1):
            results.append(
                f"Document {i}:\n"
                f"Source: {doc.metadata.get('url', 'Unknown')}\n"
                f"Title: {doc.metadata.get('title', 'Untitled')}\n"
                f"Content: {doc.page_content[:500]}...\n"
            )
        return "\n".join(results)
    except Exception as e:
        return f"Error searching local knowledge: {str(e)}"


def create_agentic_rag() -> AgentExecutor:
    """Create the Agentic RAG 2.0 system"""
    
    # Define tools list (order matters: local knowledge checked first)
    tools = [
        search_local_knowledge,
        web_search,
        fetch_webpage,
    ]
    
    # System prompt
    system_message = """You are an intelligent research assistant with access to multiple tools:

1. search_local_knowledge - Search the local knowledge base (scraped documents, internal docs)
2. web_search - Search current web information (news, events, facts)
3. fetch_webpage - Get full content from specific URLs

**Decision rules — follow in order:**
1. Does the query mention a specific company, product, project, or technical term that might be in internal docs? → call search_local_knowledge FIRST, then decide if web_search is also needed.
2. Does the query ask for current events, live data (weather, stock prices, news), or recent facts? → call web_search.
3. Do you have a specific URL you need to read in full? → call fetch_webpage.
4. If search_local_knowledge returns useful results, prefer those over web results.
5. Do NOT call web_search for company-specific or product-specific queries before trying search_local_knowledge.

**Formatting Requirements:**
- After your main answer, ALWAYS add TWO blank lines
- Then add a "**Sources:**" header on its own line
- List each source URL on a separate line with a dash prefix (e.g., "- URL")
- Example format:
  [Your answer here]


  **Sources:**
  - https://example.com/page1
  - https://example.com/page2

**Current date:** 2026-05-12 (use web search for real-time information)"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=8,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
    
    return agent_executor


# Simple test function
def test_agent():
    """Test the agent with a simple query"""
    agent = create_agentic_rag()
    result = agent.invoke({"input": "What is the weather like in New York today?"})
    formatted_output = format_answer_with_sources(result["output"])
    print(formatted_output)
    return result


if __name__ == "__main__":
    print("Testing Agentic RAG 2.0...")
    test_agent()


# Export
__all__ = ["create_agentic_rag", "web_search", "fetch_webpage", "search_local_knowledge", "format_answer_with_sources"]
