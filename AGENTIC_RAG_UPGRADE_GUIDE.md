# Upgrading to Agentic RAG 2.0: From Vector DB to Web Search

## Overview

This guide explains how to transform your RAG 1.0 application (vector database retrieval) into an Agentic RAG 2.0 system that can dynamically search the web, reason about information needs, and provide answers based on real-time web content.

## RAG 1.0 vs Agentic RAG 2.0

### RAG 1.0 (Current Implementation)
```
User Query → Vector Search → Retrieve Top-K Documents → LLM Response
```
**Limitations**:
- Fixed knowledge base (only scraped content)
- No access to recent information
- Cannot handle queries outside scraped domain
- Passive retrieval (always searches same way)

### Agentic RAG 2.0 (Target Implementation)
```
User Query → Agent Planning → Multi-Tool Execution → Self-Correction → Response
                    ↓
              [Web Search, Web Fetch, Vector DB (optional), Calculator, etc.]
```
**Advantages**:
- Dynamic information gathering from web
- Access to real-time information
- Multi-step reasoning and planning
- Tool selection based on query type
- Self-correction and verification

## Architecture Changes

### New Component: Agent System

```python
# New file: scraper/agent.py

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import BraveSearch
from langchain_community.utilities import BraveSearchWrapper
from langchain_core.tools import Tool
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
## we cannot use deepseek free model for agentic RAG via open router;
## my tests showed that deepseek free model did not have a function calling
## endpoint, either by the model itself, or via openrouter routing. My testing
## also showed that tool use and function calling on openai free model works, 
## even via open router routing so an openai free model is used.
llm = ChatOpenAI(
    model=os.getenv("OPENAI_FREE_MODEL", "openai/gpt-oss-20b:free"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.1,
)

# Define Tools
def create_web_search_tool():
    """Create web search tool using Brave Search API"""
    search = BraveSearchWrapper(
        api_key=os.getenv("BRAVE_API_KEY"),
        search_kwargs={"count": 5}
    )
    return Tool(
        name="web_search",
        description="Useful for searching current information on the web. "
                    "Use this when you need recent events, facts, or information "
                    "not in your training data. Input should be a search query.",
        func=search.run,
    )

def create_web_fetch_tool():
    """Fetch full content from a specific URL"""
    from langchain_community.document_loaders import WebBaseLoader
    
    def fetch_url(url: str) -> str:
        """Fetch and return text content from URL"""
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            return docs[0].page_content[:5000]  # Limit to 5000 chars
        except Exception as e:
            return f"Error fetching {url}: {str(e)}"
    
    return Tool(
        name="fetch_webpage",
        description="Fetch the full content of a specific webpage URL. "
                    "Use this after web_search to get detailed information "
                    "from a specific source. Input should be a URL.",
        func=fetch_url,
    )

def create_vector_db_tool():
    """Optional: Keep vector DB as a tool for domain-specific knowledge"""
    from scraper.raq_query import retrieve_top3
    
    def search_local_kb(query: str) -> str:
        """Search local knowledge base"""
        docs = retrieve_top3(query)
        if not docs:
            return "No relevant documents found in local knowledge base."
        
        results = []
        for i, doc in enumerate(docs, 1):
            results.append(
                f"Document {i}:\n"
                f"Source: {doc.metadata['url']}\n"
                f"Content: {doc.page_content[:500]}...\n"
            )
        return "\n".join(results)
    
    return Tool(
        name="search_local_knowledge",
        description="Search the local knowledge base (previously scraped content). "
                    "Use this for domain-specific information that was scraped. "
                    "Input should be a search query.",
        func=search_local_kb,
    )

# Create agent with tools
def create_agentic_rag():
    """Create the Agentic RAG 2.0 system"""
    
    # Define tools
    tools = [
        create_web_search_tool(),
        create_web_fetch_tool(),
        create_vector_db_tool(),  # Optional: hybrid approach
    ]
    
    # Define system prompt
    system_prompt = """You are an intelligent research assistant with access to multiple tools.

Your capabilities:
1. Search the web for current information
2. Fetch full content from specific webpages
3. Search local knowledge base for domain-specific information

Guidelines:
- ALWAYS use web_search for current events, recent information, or facts
- Use fetch_webpage to get detailed content from promising search results
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
    
    # Create agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
    )
    
    return agent_executor

# Export
__all__ = ["create_agentic_rag"]
```

## Updated Web Application

### Modified `web_app.py`

```python
# web_app.py - Agentic RAG 2.0 version
import os
import asyncio
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# NEW: Import agent instead of simple RAG chain
from scraper.agent import create_agentic_rag

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize agent
agent_executor = create_agentic_rag()

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/ask")
async def ask(query: str = Form(...)):
    async def stream_answer():
        try:
            # NEW: Use agent executor instead of simple RAG chain
            # Agent will decide which tools to use
            result = agent_executor.invoke({"input": query})
            
            answer = result["output"]
            
            # Extract sources from intermediate steps
            sources = []
            if "intermediate_steps" in result:
                for action, observation in result["intermediate_steps"]:
                    if hasattr(action, "tool") and action.tool == "web_search":
                        # Extract URLs from web search results
                        # (depends on search API response format)
                        sources.append(f"Web Search: {action.tool_input}")
                    elif hasattr(action, "tool") and action.tool == "fetch_webpage":
                        sources.append(f'<a href="{action.tool_input}" target="_blank">{action.tool_input}</a>')
            
            sources_html = "<br>".join(sources) if sources else "Multiple web sources"
            full = f"{answer}<br><br>Sources:<br>{sources_html}"
            
            # Stream response
            for char in full:
                yield char
                await asyncio.sleep(0.01)
                
        except Exception as e:
            yield f"Error: {str(e)}"
    
    return StreamingResponse(stream_answer(), media_type="text/plain")

# NEW: Add endpoint to see agent reasoning (optional)
@app.post("/ask_verbose")
async def ask_verbose(query: str = Form(...)):
    """Return full agent reasoning trace for debugging"""
    result = agent_executor.invoke({"input": query})
    return {
        "answer": result["output"],
        "steps": [
            {
                "tool": action.tool,
                "input": action.tool_input,
                "output": observation[:500]  # Truncate
            }
            for action, observation in result.get("intermediate_steps", [])
        ]
    }
```

## Alternative: Using LangGraph for Advanced Agent Control

For more sophisticated agent behavior with custom logic and state management:

```python
# scraper/langgraph_agent.py

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain_openai import ChatOpenAI
import operator
import os

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_action: str
    sources: list

# Initialize components
llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_FREE_MODEL"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.1,
)

# Create tools (same as before)
from scraper.agent import create_web_search_tool, create_web_fetch_tool, create_vector_db_tool

tools = [
    create_web_search_tool(),
    create_web_fetch_tool(),
    create_vector_db_tool(),
]

tool_executor = ToolExecutor(tools)

# Define agent logic nodes
def should_continue(state: AgentState) -> str:
    """Decide whether to continue or end"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If LLM decided to use a tool, continue
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    # Otherwise end
    return "end"

def call_model(state: AgentState):
    """Call LLM to decide next action"""
    messages = state["messages"]
    response = llm.bind_tools(tools).invoke(messages)
    return {"messages": [response]}

def call_tool(state: AgentState):
    """Execute the tool chosen by LLM"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Execute each tool call
    tool_calls = last_message.tool_calls
    tool_messages = []
    sources = state.get("sources", [])
    
    for tool_call in tool_calls:
        tool_result = tool_executor.invoke(
            ToolInvocation(
                tool=tool_call["name"],
                tool_input=tool_call["args"],
            )
        )
        
        # Track sources
        if tool_call["name"] in ["web_search", "fetch_webpage"]:
            sources.append({
                "tool": tool_call["name"],
                "input": tool_call["args"],
            })
        
        tool_messages.append({
            "role": "tool",
            "content": tool_result,
            "tool_call_id": tool_call["id"],
        })
    
    return {
        "messages": tool_messages,
        "sources": sources,
    }

# Build graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

# Set entry point
workflow.set_entry_point("agent")

# Add conditional edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    }
)

# Add edge from action back to agent
workflow.add_edge("action", "agent")

# Compile
app = workflow.compile()

def query_langgraph_agent(question: str) -> dict:
    """Query the LangGraph agent"""
    inputs = {
        "messages": [HumanMessage(content=question)],
        "sources": [],
    }
    
    result = app.invoke(inputs)
    
    return {
        "answer": result["messages"][-1].content,
        "sources": result.get("sources", []),
        "steps": len(result["messages"]) - 1,  # Number of iterations
    }
```

## Configuration Changes

### Updated `.env`

```bash
# Existing
PGVECTOR_DB_URL=postgresql://myuser:mypassword@localhost:5432/myprojdb
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxx
DEEPSEEK_FREE_MODEL=deepseek/deepseek-chat-v3.1:free
TOKENIZERS_PARALLELISM=false

# NEW: Add web search API key
BRAVE_API_KEY=BSA_xxxxxxxxxxxxxxxx  # Get from https://brave.com/search/api/

# Alternative search APIs:
# SERPAPI_API_KEY=xxx  # https://serpapi.com/
# GOOGLE_API_KEY=xxx   # Google Custom Search
# GOOGLE_CSE_ID=xxx
```

### Updated `requirements.txt`

Add these new dependencies:

```txt
# Existing dependencies...

# NEW: Agent and web search dependencies
langchain-community>=0.2.0
brave-search-python>=1.0.0
beautifulsoup4>=4.12.0
playwright>=1.40.0  # For fetching JavaScript-rendered pages
langgraph>=0.0.40  # Optional: for advanced agent control
```

## Implementation Options

### Option 1: Pure Web Search (No Vector DB)

**Best for**: General knowledge queries, current events

```python
# Simplified agent with only web tools
tools = [
    create_web_search_tool(),
    create_web_fetch_tool(),
]
```

### Option 2: Hybrid Approach (Web + Vector DB)

**Best for**: Combining domain-specific knowledge with current information

```python
# Keep all tools including vector DB
tools = [
    create_web_search_tool(),
    create_web_fetch_tool(),
    create_vector_db_tool(),
]

# Agent will intelligently choose based on query
```

### Option 3: Router Pattern

**Best for**: Explicitly routing different query types

```python
# scraper/query_router.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class RouteQuery(BaseModel):
    """Route query to appropriate data source"""
    datasource: str = Field(
        description="Choose 'web_search' for current events, "
                    "'vector_db' for domain-specific knowledge, "
                    "'both' if you need information from both sources"
    )
    reasoning: str = Field(description="Explain why you chose this route")

llm = ChatOpenAI(...)

router_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at routing questions to the right data source."),
    ("user", "{question}"),
])

router_chain = router_prompt | llm.with_structured_output(RouteQuery)

def route_query(question: str):
    """Route question to appropriate retrieval method"""
    route = router_chain.invoke({"question": question})
    
    if route.datasource == "web_search":
        return use_web_agent(question)
    elif route.datasource == "vector_db":
        return use_vector_rag(question)
    else:  # both
        return use_hybrid_approach(question)
```

## Advanced Features

### 1. Multi-Step Reasoning

```python
# The agent can chain multiple searches:
# Query: "Compare the GDP of countries that hosted Olympics in last 10 years"

# Agent reasoning:
# Step 1: Use web_search("countries hosted Olympics 2014-2024")
# Step 2: Extract country list from results
# Step 3: For each country, use web_search("{country} GDP 2024")
# Step 4: Synthesize comparison
```

### 2. Self-Correction

```python
def create_self_correcting_agent():
    """Agent that verifies its own answers"""
    
    verification_prompt = """
    Review your previous answer for accuracy.
    Search for additional sources if needed to verify facts.
    Correct any errors found.
    """
    
    # Agent will automatically re-search if uncertain
```

### 3. Source Validation

```python
def validate_sources(sources: list) -> list:
    """Filter and rank sources by credibility"""
    
    trusted_domains = [
        "wikipedia.org",
        ".gov",
        ".edu",
        "reuters.com",
        "apnews.com",
    ]
    
    validated = []
    for source in sources:
        credibility_score = 0
        if any(domain in source["url"] for domain in trusted_domains):
            credibility_score += 10
        
        source["credibility"] = credibility_score
        validated.append(source)
    
    return sorted(validated, key=lambda x: x["credibility"], reverse=True)
```

### 4. Streaming Agent Thoughts

```python
# web_app.py - Stream agent reasoning in real-time

@app.post("/ask_streaming")
async def ask_streaming(query: str = Form(...)):
    async def stream_with_thoughts():
        # Stream agent's thought process
        yield "🤔 Thinking about your question...\n\n"
        
        async for chunk in agent_executor.astream({"input": query}):
            if "actions" in chunk:
                for action in chunk["actions"]:
                    yield f"🔧 Using tool: {action.tool}\n"
                    yield f"   Input: {action.tool_input}\n\n"
            
            if "steps" in chunk:
                for step in chunk["steps"]:
                    yield f"💡 Found: {step.observation[:200]}...\n\n"
            
            if "output" in chunk:
                yield f"✅ Answer:\n{chunk['output']}\n"
        
    return StreamingResponse(stream_with_thoughts(), media_type="text/plain")
```

## Updated Frontend

### Enhanced UI with Agent Visualization

```html
<!-- static/index.html - Enhanced version -->
<!DOCTYPE html>
<html>
<head>
  <title>Agentic RAG 2.0 - Web Search</title>
  <style>
    /* Existing styles... */
    
    .thinking-step {
      background: #f0f8ff;
      border-left: 3px solid #007bff;
      padding: 10px;
      margin: 10px 0;
      font-size: 0.9em;
      font-family: monospace;
    }
    
    .tool-use {
      color: #28a745;
      font-weight: bold;
    }
    
    .source-card {
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 12px;
      margin: 8px 0;
    }
    
    .source-card a {
      color: #007bff;
      text-decoration: none;
      font-weight: bold;
    }
    
    .source-card .snippet {
      color: #666;
      font-size: 0.9em;
      margin-top: 5px;
    }
  </style>
</head>
<body>
  <h1>🤖 Agentic RAG 2.0 - Web-Powered AI</h1>
  <p style="color: #666;">Ask anything - I'll search the web and provide current information!</p>
  
  <div class="chat" id="chat">
    <div class="message bot">
      Hi! I'm an AI agent with web search capabilities. I can:
      <ul>
        <li>🌐 Search the web for current information</li>
        <li>📄 Fetch and analyze specific webpages</li>
        <li>💾 Search local knowledge base (if available)</li>
        <li>🔗 Cite sources with links</li>
      </ul>
      Try asking about recent events, current facts, or specific topics!
    </div>
  </div>
  
  <form id="form">
    <input type="text" id="query" 
           placeholder="e.g., What are the latest developments in AI?" 
           required autocomplete="off">
    <button type="submit">Ask</button>
  </form>
  
  <div style="margin-top: 20px; text-align: center; color: #999; font-size: 0.9em;">
    <label>
      <input type="checkbox" id="showThinking" checked>
      Show agent reasoning process
    </label>
  </div>

  <script>
    const form = document.getElementById('form');
    const chat = document.getElementById('chat');
    const queryInput = document.getElementById('query');
    const showThinking = document.getElementById('showThinking');

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const query = queryInput.value.trim();
      if (!query) return;

      // User message
      const userMsg = document.createElement('div');
      userMsg.className = 'message user';
      userMsg.textContent = query;
      chat.appendChild(userMsg);

      // Bot response container
      const botMsg = document.createElement('div');
      botMsg.className = 'message bot';
      chat.appendChild(botMsg);
      
      // Thinking container
      const thinkingDiv = document.createElement('div');
      thinkingDiv.className = 'thinking-step';
      thinkingDiv.textContent = '🤔 Analyzing query...';
      if (showThinking.checked) {
        botMsg.appendChild(thinkingDiv);
      }
      
      chat.scrollTop = chat.scrollHeight;
      queryInput.value = '';

      try {
        const response = await fetch('/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: new URLSearchParams({ query })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          const chunk = decoder.decode(value);
          fullText += chunk;
          
          // Parse and display with formatting
          botMsg.innerHTML = formatResponse(fullText, showThinking.checked);
          chat.scrollTop = chat.scrollHeight;
        }

      } catch (err) {
        botMsg.textContent = 'Error: ' + err.message;
      }
    });
    
    function formatResponse(text, showSteps) {
      // Parse thinking steps vs final answer
      // Format sources nicely
      // (Implementation depends on response format)
      return text.replace(/Sources:/g, '<br><strong>Sources:</strong><br>');
    }
  </script>
</body>
</html>
```

## Testing Your Agentic RAG 2.0

### Test Cases

```python
# test_agent.py

from scraper.agent_classic import create_agentic_rag

agent = create_agentic_rag()

# Test 1: Current events
result = agent.invoke({
    "input": "What are the latest developments in AI regulation?"
})
print(result["output"])

# Test 2: Hybrid query (web + local knowledge)
result = agent.invoke({
    "input": "How do current gas prices compare to information about driving costs in China?"
})
print(result["output"])

# Test 3: Multi-step reasoning
result = agent.invoke({
    "input": "Find the top 3 fastest trains in the world and compare their speeds"
})
print(result["output"])

# Test 4: Verification
result = agent.invoke({
    "input": "Is the claim that China has the world's fastest train true? Verify with multiple sources."
})
print(result["output"])
```

## Migration Steps

### Step-by-Step Migration from RAG 1.0 to 2.0

1. **Install new dependencies**:
   ```bash
   pip install langchain-community brave-search-python langgraph
   ```

2. **Get Brave Search API key**:
   - Visit https://brave.com/search/api/
   - Sign up for free tier (2000 queries/month)
   - Add to `.env`

3. **Create agent module**:
   ```bash
   cp scraper/raq_query.py scraper/raq_query_v1_backup.py
   touch scraper/agent.py
   # Add agent code from above
   ```

4. **Update web_app.py**:
   ```bash
   cp web_app.py web_app_v1_backup.py
   # Update with new agent code
   ```

5. **Test incrementally**:
   ```bash
   # Test agent directly
   python -c "from scraper.agent_classic import create_agentic_rag; \
              agent = create_agentic_rag(); \
              print(agent.invoke({'input': 'how is the weather like in New York today?'}))"
   
   # Test web app
   uvicorn web_app:app --reload --port=8080
   ```

6. **Optional: Keep hybrid mode**:
   - Keep vector DB for domain-specific queries
   - Use router to choose between web and vector search
   - Best of both worlds!

## Performance Optimization

### Caching Web Search Results

```python
# scraper/cache.py

from functools import lru_cache
import hashlib
import json
from datetime import datetime, timedelta

class SearchCache:
    def __init__(self, ttl_hours=24):
        self.cache = {}
        self.ttl = timedelta(hours=ttl_hours)
    
    def get(self, query: str):
        key = hashlib.md5(query.encode()).hexdigest()
        if key in self.cache:
            result, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                return result
        return None
    
    def set(self, query: str, result):
        key = hashlib.md5(query.encode()).hexdigest()
        self.cache[key] = (result, datetime.now())

# Use in agent
cache = SearchCache(ttl_hours=24)

def cached_web_search(query: str):
    cached_result = cache.get(query)
    if cached_result:
        return cached_result
    
    result = web_search_tool.run(query)
    cache.set(query, result)
    return result
```

### Parallel Tool Execution

```python
# Execute multiple searches in parallel
import asyncio
from langchain_core.runnables import RunnableParallel

parallel_search = RunnableParallel(
    web=create_web_search_tool(),
    local=create_vector_db_tool(),
)

# Both searches run simultaneously
results = parallel_search.invoke({"query": "your question"})
```

## Cost Considerations

### API Usage

| Component | Free Tier | Paid Tier |
|-----------|-----------|-----------|
| Brave Search | 2000 queries/month | $5/1000 queries |
| OpenRouter (DeepSeek) | Free with limits | ~$0.14/1M tokens |
| Vector DB (PostgreSQL) | Free (self-hosted) | N/A |

### Optimization Tips

1. **Implement caching** (24-hour TTL for search results)
2. **Rate limiting** (max queries per user per minute)
3. **Smart routing** (use vector DB when possible, web when needed)
4. **Batch processing** (combine related queries)

## Comparison: RAG 1.0 vs RAG 2.0

| Feature | RAG 1.0 (Vector DB) | RAG 2.0 (Agentic) |
|---------|---------------------|-------------------|
| Data Source | Static (scraped content) | Dynamic (web search) |
| Information Freshness | Outdated after scraping | Real-time |
| Query Scope | Limited to scraped domain | Unlimited |
| Setup Complexity | Medium (scraping + DB) | Medium (API keys) |
| Response Time | Fast (~1-2s) | Slower (~3-10s) |
| Accuracy | High for domain | Variable (depends on web results) |
| Cost | Low (storage only) | Higher (API calls) |
| Best For | Domain-specific, controlled | General knowledge, current events |

## Conclusion

### When to Use Each Approach

**Use RAG 1.0 (Vector DB)** when:
- You have a specific, controlled knowledge domain
- Information doesn't change frequently
- You need consistent, fast responses
- Privacy is critical (all data local)

**Use Agentic RAG 2.0 (Web Search)** when:
- You need current, real-time information
- Query scope is broad and unpredictable
- Source diversity is important
- You can accept variable latency

**Use Hybrid Approach** when:
- You want best of both worlds
- Some queries are domain-specific, others general
- You have budget for API calls but want fallback
- You need both speed and breadth

## Next Steps

1. Choose your approach (pure web, hybrid, or router)
2. Set up API keys (Brave Search, OpenRouter)
3. Implement agent system
4. Test with real queries
5. Monitor performance and costs
6. Iterate and improve!

## Resources

- [LangChain Agents Documentation](https://python.langchain.com/docs/modules/agents/)
- [LangGraph Tutorial](https://langchain-ai.github.io/langgraph/)
- [Brave Search API](https://brave.com/search/api/)
- [OpenRouter Documentation](https://openrouter.ai/docs)
- [Agentic RAG Research Paper](https://arxiv.org/abs/2310.06825)

## Support

For questions or issues, contact: nyang63@gmail.com
