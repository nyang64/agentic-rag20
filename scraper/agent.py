# scraper/agent.py - Agentic RAG 2.0 using langgraph.prebuilt.create_react_agent
#
# NOTE: This implementation uses langgraph.prebuilt.create_react_agent which is
# DEPRECATED as of LangGraph v1.0. The recommended approach is to use
# langchain.agents.create_agent instead (see 'new-agent' branch).
#
# This branch exists to demonstrate the langgraph.prebuilt API for educational
# purposes and comparison.

import os
import re
import warnings
from typing import List, Dict, Any
from dotenv import load_dotenv

# Suppress the deprecation warning for demonstration purposes
warnings.filterwarnings("ignore", message=".*create_react_agent has been moved.*")

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
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


# -------------------------------------------------------------------
# Helper function to format the output properly
# -------------------------------------------------------------------
def format_answer_with_sources(text: str) -> str:
    """Ensure Sources section is properly formatted on separate lines"""
    patterns = [
        r'([^\n])\s*\*\*Sources?:?\*\*',
        r'([^\n])\s*\bSources?:?\s*-',
    ]

    result = text

    for pattern in patterns:
        result = re.sub(pattern, r'\1\n\n**Sources:**\n', result, flags=re.IGNORECASE)

    result = re.sub(r'\*\*Source\*\*', '**Sources:**', result)
    result = re.sub(r'\*\*Sources\*\*', '**Sources:**', result)

    lines = result.split('\n')
    formatted_lines = []
    in_sources = False

    for line in lines:
        if '**Sources:**' in line:
            in_sources = True
            formatted_lines.append(line)
        elif in_sources and ('http://' in line or 'https://' in line):
            line = line.strip()
            if not line.startswith('-'):
                line = '- ' + line
            formatted_lines.append(line)
        else:
            formatted_lines.append(line)

    return '\n'.join(formatted_lines)


# -------------------------------------------------------------------
# Initialize LLM
# -------------------------------------------------------------------
llm = ChatOpenAI(
    model=os.getenv("OPENAI_FREE_MODEL", "openai/gpt-oss-20b:free"),
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
            import requests
            from bs4 import BeautifulSoup

            url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }
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


@tool
def fetch_webpage(url: str) -> str:
    """Fetch the full content of a specific webpage. Use this to get detailed information from a URL."""
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
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
    """Search the local knowledge base (previously scraped content). Use this for domain-specific information."""
    try:
        from scraper.raq_query import retrieve_top3
        docs = retrieve_top3(query)

        if not docs:
            return "No relevant documents found in local knowledge base."

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
# System Prompt
# -------------------------------------------------------------------

SYSTEM_PROMPT = """You are an intelligent research assistant with access to multiple tools:

1. web_search - Search current web information (news, events, facts)
2. fetch_webpage - Get full content from specific URLs
3. search_local_knowledge - Search previously scraped domain-specific content

**Instructions:**
- For current events, weather, news, recent facts: use web_search first
- To get details from a specific URL: use fetch_webpage
- For domain-specific queries about scraped content: use search_local_knowledge
- Always cite sources with URLs when available
- If you can't find information, say so clearly
- Synthesize information from multiple sources when needed

**Formatting Requirements:**
- After your main answer, ALWAYS add TWO blank lines
- Then add a "**Sources:**" header on its own line
- List each source URL on a separate line with a dash prefix (e.g., "- URL")
- Example format:
  [Your answer here]


  **Sources:**
  - https://example.com/page1
  - https://example.com/page2

**Current date:** Use web search for real-time information."""


# -------------------------------------------------------------------
# Agent Creation using langgraph.prebuilt.create_react_agent
# -------------------------------------------------------------------

def create_agentic_rag(checkpointer=None):
    """Create an Agentic RAG 2.0 agent using langgraph.prebuilt.create_react_agent.

    NOTE: This API is DEPRECATED as of LangGraph v1.0.
    The recommended approach is: from langchain.agents import create_agent

    This function uses the langgraph.prebuilt module directly, which provides
    a pre-built ReAct agent implementation. The create_react_agent function:
    - Creates a state graph with 'agent' and 'tools' nodes
    - Implements the ReAct (Reasoning + Acting) loop
    - Handles tool binding and execution automatically

    Args:
        checkpointer: Optional checkpointer for conversation persistence.

    Returns:
        CompiledStateGraph: A LangGraph agent that can be invoked with
            {"messages": [{"role": "user", "content": "..."}]}
    """
    tools = [
        web_search,
        fetch_webpage,
        search_local_knowledge,
    ]

    # Create agent using langgraph.prebuilt.create_react_agent
    # This is the low-level LangGraph prebuilt function
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    return agent


# Global agent instance
_agent = None


def get_agent():
    """Get or create the global agent instance."""
    global _agent
    if _agent is None:
        _agent = create_agentic_rag()
    return _agent


def query_agent(question: str, thread_id: str = None) -> Dict[str, Any]:
    """Query the agent with a question.

    Args:
        question: The user's question
        thread_id: Optional thread ID for conversation memory

    Returns:
        Dict with "answer", "messages", and "tool_calls" keys
    """
    agent = get_agent()

    inputs = {
        "messages": [{"role": "user", "content": question}]
    }

    config = {}
    if thread_id:
        config = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke(inputs, config)

    messages = result.get("messages", [])
    answer = ""
    tool_calls_made = []

    for msg in messages:
        if isinstance(msg, AIMessage):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls_made.extend(msg.tool_calls)
            else:
                answer = msg.content
        elif isinstance(msg, ToolMessage):
            pass

    return {
        "answer": answer,
        "messages": messages,
        "tool_calls": tool_calls_made,
    }


# -------------------------------------------------------------------
# Wrapper for web_app.py compatibility
# -------------------------------------------------------------------

class AgentExecutorWrapper:
    """Wrapper that provides AgentExecutor-like interface for the LangGraph agent.

    This allows the LangGraph prebuilt agent to be used as a drop-in replacement
    in web_app.py which expects the classic AgentExecutor interface.
    """

    def __init__(self, use_memory: bool = False):
        checkpointer = MemorySaver() if use_memory else None
        self.agent = create_agentic_rag(checkpointer=checkpointer)
        self.use_memory = use_memory
        self._thread_counter = 0

    def invoke(self, inputs: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Invoke the agent with AgentExecutor-compatible interface.

        Args:
            inputs: Dict with "input" key containing the user query
            config: Optional config dict

        Returns:
            Dict with "output" key (answer) and "intermediate_steps" (tool usage)
        """
        query = inputs.get("input", "")

        agent_inputs = {
            "messages": [{"role": "user", "content": query}]
        }

        invoke_config = config or {}
        if self.use_memory and "configurable" not in invoke_config:
            self._thread_counter += 1
            invoke_config = {"configurable": {"thread_id": f"thread_{self._thread_counter}"}}

        result = self.agent.invoke(agent_inputs, invoke_config)

        messages = result.get("messages", [])
        output = ""
        intermediate_steps = []

        for msg in messages:
            if isinstance(msg, AIMessage):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        class Action:
                            def __init__(self, tool, tool_input):
                                self.tool = tool
                                self.tool_input = tool_input
                        intermediate_steps.append((
                            Action(tc.get("name", ""), tc.get("args", {})),
                            ""
                        ))
                else:
                    output = msg.content
            elif isinstance(msg, ToolMessage):
                if intermediate_steps:
                    action, _ = intermediate_steps[-1]
                    intermediate_steps[-1] = (action, msg.content)

        return {
            "output": output,
            "intermediate_steps": intermediate_steps,
        }

    async def astream(self, inputs: Dict[str, Any], config: Dict[str, Any] = None):
        """Async stream the agent's execution for real-time updates."""
        query = inputs.get("input", "")

        agent_inputs = {
            "messages": [{"role": "user", "content": query}]
        }

        invoke_config = config or {}
        if self.use_memory and "configurable" not in invoke_config:
            self._thread_counter += 1
            invoke_config = {"configurable": {"thread_id": f"thread_{self._thread_counter}"}}

        async for event in self.agent.astream_events(agent_inputs, invoke_config, version="v2"):
            kind = event.get("event", "")

            if kind == "on_tool_start":
                tool_name = event.get("name", "")
                tool_input = event.get("data", {}).get("input", {})

                class Action:
                    def __init__(self, tool, tool_input):
                        self.tool = tool
                        self.tool_input = tool_input

                yield {"actions": [Action(tool_name, tool_input)]}

            elif kind == "on_tool_end":
                output = event.get("data", {}).get("output", "")

                class Step:
                    def __init__(self, observation):
                        self.observation = observation

                yield {"steps": [Step(str(output))]}

            elif kind == "on_chat_model_end":
                output = event.get("data", {}).get("output", None)
                if output and hasattr(output, "content"):
                    if not (hasattr(output, "tool_calls") and output.tool_calls):
                        yield {"output": output.content}


def create_agentic_rag_executor(use_memory: bool = False) -> AgentExecutorWrapper:
    """Create an AgentExecutor-compatible wrapper for web_app.py.

    This is the main entry point for backward compatibility with web_app.py.
    """
    return AgentExecutorWrapper(use_memory=use_memory)


# -------------------------------------------------------------------
# Test function
# -------------------------------------------------------------------

def test_agent():
    """Test the agent with a simple query."""
    print("=" * 60)
    print("Testing langgraph.prebuilt.create_react_agent")
    print("NOTE: This API is deprecated in favor of langchain.agents.create_agent")
    print("=" * 60)

    print("\nCreating agent...")
    agent = create_agentic_rag()
    print(f"Agent type: {type(agent)}")

    print("\nTesting with simple query...")
    result = agent.invoke({
        "messages": [{"role": "user", "content": "What is 2+2?"}]
    })

    messages = result.get("messages", [])
    print(f"\nTotal messages: {len(messages)}")

    for i, msg in enumerate(messages):
        print(f"\n[{i}] {type(msg).__name__}:")
        if hasattr(msg, "content"):
            content = msg.content[:200] if len(msg.content) > 200 else msg.content
            print(f"    Content: {content}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"    Tool calls: {[tc.get('name') for tc in msg.tool_calls]}")

    final_answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not (hasattr(msg, "tool_calls") and msg.tool_calls):
            final_answer = msg.content
            break

    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(format_answer_with_sources(final_answer))

    return result


if __name__ == "__main__":
    test_agent()


# Export
__all__ = [
    "create_agentic_rag",
    "create_agentic_rag_executor",
    "AgentExecutorWrapper",
    "query_agent",
    "get_agent",
    "web_search",
    "fetch_webpage",
    "search_local_knowledge",
    "format_answer_with_sources",
    "SYSTEM_PROMPT",
]
