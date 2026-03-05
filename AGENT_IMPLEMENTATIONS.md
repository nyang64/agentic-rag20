# Agent Implementations Comparison

This document explains the differences between the three agent implementations in this project:

- `scraper/agent_classic.py` - Legacy implementation using `langchain_classic`
- `scraper/agent.py` - New implementation using `langchain.agents.create_agent`
- `scraper/langgraph_agent.py` - Custom implementation using `langgraph.graph.StateGraph`

---

## Table of Contents

1. [Overview](#overview)
2. [Library & Import Differences](#library--import-differences)
3. [Architecture Comparison](#architecture-comparison)
4. [Code Comparison](#code-comparison)
5. [Key Differences Summary](#key-differences-summary)
6. [When to Use Each](#when-to-use-each)
7. [Evolution Path](#evolution-path)
8. [Relationship Between agent.py and langgraph_agent.py](#relationship-between-agentpy-and-langgraph_agentpy)

---

## Overview

All three implementations achieve the same goal: an Agentic RAG system that can:
- Search the web for current information
- Fetch content from specific URLs
- Search a local knowledge base (vector database)

The difference lies in **how** they implement the agent loop and **what level of control** they provide.

---

## Library & Import Differences

| File | Library | Key Import |
|------|---------|------------|
| `agent_classic.py` | `langchain_classic` (legacy) | `from langchain_classic.agents import AgentExecutor, create_tool_calling_agent` |
| `agent.py` | `langchain` (new) | `from langchain.agents import create_agent` |
| `langgraph_agent.py` | `langgraph` (low-level) | `from langgraph.graph import StateGraph, END` |

---

## Architecture Comparison

### agent_classic.py - Legacy Architecture

```
┌─────────────────────────────────────────────────────┐
│              langchain_classic.agents                │
├─────────────────────────────────────────────────────┤
│                                                      │
│   create_tool_calling_agent(llm, tools, prompt)     │
│              │                                       │
│              ▼                                       │
│   AgentExecutor(                                    │
│       agent=agent,                                  │
│       tools=tools,                                  │
│       max_iterations=5,          ◄── Manual config  │
│       handle_parsing_errors=True,                   │
│       return_intermediate_steps=True,               │
│   )                                                 │
│              │                                       │
│              ▼                                       │
│   Returns: AgentExecutor                            │
│                                                      │
└─────────────────────────────────────────────────────┘

Input:  {"input": "question"}
Output: {"output": "answer", "intermediate_steps": [...]}
```

**Characteristics:**
- Uses the legacy `langchain_classic` package
- Requires manual `ChatPromptTemplate` with `MessagesPlaceholder`
- Returns an `AgentExecutor` object
- Configuration is explicit (max_iterations, handle_parsing_errors, etc.)

---

### agent.py - New High-Level Architecture

```
┌─────────────────────────────────────────────────────┐
│                 langchain.agents                     │
│              (wraps LangGraph internally)            │
├─────────────────────────────────────────────────────┤
│                                                      │
│   create_agent(                                     │
│       model=llm,                                    │
│       tools=tools,                                  │
│       system_prompt=SYSTEM_PROMPT,                  │
│       checkpointer=checkpointer,  ◄── Built-in     │
│   )                                                 │
│              │                                       │
│              ▼                                       │
│   Returns: CompiledStateGraph (LangGraph)           │
│                                                      │
└─────────────────────────────────────────────────────┘

Input:  {"messages": [{"role": "user", "content": "question"}]}
Output: {"messages": [HumanMessage, AIMessage, ToolMessage, ...]}
```

**Characteristics:**
- Uses the new `langchain.agents` package
- Simple API: just pass model, tools, and system prompt
- Internally builds a LangGraph workflow
- Returns a `CompiledStateGraph` (LangGraph runtime)
- Built-in support for memory via `checkpointer` parameter

---

### langgraph_agent.py - Custom Low-Level Architecture

```
┌─────────────────────────────────────────────────────┐
│                   langgraph.graph                    │
│               (full manual control)                  │
├─────────────────────────────────────────────────────┤
│                                                      │
│   CustomAgentState = TypedDict(                     │
│       messages: [...],                              │
│       sources: [...],        ◄── Custom state      │
│       iteration_count: int,  ◄── Custom state      │
│       tools_used: [...],     ◄── Custom state      │
│   )                                                 │
│                                                      │
│   workflow = StateGraph(CustomAgentState)           │
│   workflow.add_node("agent", call_model)            │
│   workflow.add_node("tools", call_tool)             │
│   workflow.add_conditional_edges(...)  ◄── Custom  │
│   workflow.compile()                                │
│              │                                       │
│              ▼                                       │
│   Returns: CompiledStateGraph                       │
│                                                      │
└─────────────────────────────────────────────────────┘

Input:  {"messages": [...], "sources": [], "iteration_count": 0, ...}
Output: {"messages": [...], "sources": [...], "iteration_count": N, ...}
```

**Characteristics:**
- Uses `langgraph` directly for full control
- Define custom state (track anything you need)
- Define custom nodes (call_model, call_tool, etc.)
- Define custom routing logic (should_continue)
- Returns a `CompiledStateGraph`

---

## Code Comparison

### Creating the Agent

**agent_classic.py:**
```python
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),  # Required!
])

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)
```

**agent.py:**
```python
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,  # Just a string, no template needed
    checkpointer=checkpointer,
)
```

**langgraph_agent.py:**
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
import operator

class CustomAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sources: List[Dict[str, Any]]
    iteration_count: int
    tools_used: List[str]

def should_continue(state):
    # Custom routing logic
    if state["iteration_count"] > 10:
        return "end"
    if hasattr(state["messages"][-1], "tool_calls"):
        return "continue"
    return "end"

def call_model(state):
    # Call LLM with tools bound
    response = llm.bind_tools(tools).invoke(state["messages"])
    return {"messages": [response], "iteration_count": state["iteration_count"] + 1}

def call_tool(state):
    # Execute tools and return results
    # ... tool execution logic ...
    return {"messages": tool_messages, "sources": sources}

workflow = StateGraph(CustomAgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tool)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
workflow.add_edge("tools", "agent")
app = workflow.compile()
```

---

### Invoking the Agent

**agent_classic.py:**
```python
result = agent_executor.invoke({"input": "What is the weather in NYC?"})
answer = result["output"]
steps = result["intermediate_steps"]
```

**agent.py:**
```python
result = agent.invoke({
    "messages": [{"role": "user", "content": "What is the weather in NYC?"}]
})
# Extract final answer from messages
for msg in reversed(result["messages"]):
    if isinstance(msg, AIMessage) and not msg.tool_calls:
        answer = msg.content
        break
```

**langgraph_agent.py:**
```python
result = workflow.invoke({
    "messages": [HumanMessage(content="What is the weather in NYC?")],
    "sources": [],
    "iteration_count": 0,
    "tools_used": [],
})
# Access custom state
answer = result["messages"][-1].content
sources = result["sources"]
iterations = result["iteration_count"]
tools_used = result["tools_used"]
```

---

## Key Differences Summary

| Feature | agent_classic.py | agent.py | langgraph_agent.py |
|---------|-----------------|----------|-------------------|
| **Package** | `langchain_classic` | `langchain` | `langgraph` |
| **Abstraction Level** | Medium | High | Low |
| **Returns** | `AgentExecutor` | `CompiledStateGraph` | `CompiledStateGraph` |
| **Prompt Format** | `ChatPromptTemplate` with `MessagesPlaceholder` | Simple string | Manual `SystemMessage` |
| **State** | Fixed (input/output) | Fixed (messages) | **Custom** (anything) |
| **Memory** | Manual setup | `checkpointer` param | `checkpointer` param |
| **Routing Logic** | Fixed ReAct loop | Fixed ReAct loop | **Custom** logic |
| **Input format** | `{"input": "..."}` | `{"messages": [...]}` | Custom state dict |
| **Output format** | `{"output": "...", "intermediate_steps": [...]}` | `{"messages": [...]}` | Custom state dict |
| **Iteration Control** | `max_iterations` param | Internal | Custom in `should_continue` |
| **Error Handling** | `handle_parsing_errors` param | Internal | Custom |

---

## When to Use Each

| Use Case | Recommended |
|----------|-------------|
| Legacy codebase, proven stability | `agent_classic.py` |
| New projects, simple needs | `agent.py` |
| Quick prototyping | `agent.py` |
| Custom state tracking (sources, metadata) | `langgraph_agent.py` |
| Human-in-the-loop workflows | `langgraph_agent.py` |
| Multi-agent systems | `langgraph_agent.py` |
| Custom routing logic | `langgraph_agent.py` |
| Debugging/learning LangGraph internals | `langgraph_agent.py` |

---

## Evolution Path

```
langchain_classic (legacy)     langchain (new)        langgraph (low-level)
         │                          │                        │
         ▼                          ▼                        ▼
  AgentExecutor              create_agent()            StateGraph()
  (monolithic)               (uses LangGraph           (full control)
                              internally)
         │                          │                        │
         └──────────────────────────┴────────────────────────┘
                                    │
                                    ▼
                         All return CompiledStateGraph
                         (except classic returns AgentExecutor)
```

The new `langchain.agents.create_agent` is essentially a high-level wrapper that builds a LangGraph workflow for you. If you need more control, you build the workflow yourself with `langgraph.graph.StateGraph`.

---

## Relationship Between agent.py and langgraph_agent.py

Both files use LangGraph as the runtime, but at different abstraction levels:

### agent.py (High-Level)

Uses `langchain.agents.create_agent` which:
- Automatically creates the state graph
- Automatically handles the ReAct loop
- Automatically manages tool execution
- Returns a ready-to-use `CompiledStateGraph`

```python
from langchain.agents import create_agent

agent = create_agent(model=llm, tools=tools, system_prompt="...")
# That's it - the graph is built for you
```

### langgraph_agent.py (Low-Level)

Builds the state graph manually which:
- Gives you control over state structure
- Gives you control over node behavior
- Gives you control over routing logic
- Requires more code but offers more flexibility

```python
from langgraph.graph import StateGraph, END

# You define everything yourself
workflow = StateGraph(CustomAgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tool)
workflow.add_conditional_edges("agent", should_continue, {...})
workflow.add_edge("tools", "agent")
app = workflow.compile()
```

### Visual Comparison

```
agent.py (high-level):
┌─────────────────────────────────────┐
│  create_agent(llm, tools)           │  ◄── One function call
│  (everything handled internally)    │
└─────────────────────────────────────┘

langgraph_agent.py (low-level):
┌─────────────────────────────────────┐
│  [agent node] ──► should_continue?  │
│       ▲              │              │
│       │         yes  │  no          │
│       │              ▼              │
│       └──── [tools node]     [END]  │
│                                     │
│  (you control every piece)          │
└─────────────────────────────────────┘
```

### When to Use Which?

- **agent.py**: For most use cases. It's simpler and the LangGraph runtime handles everything.
- **langgraph_agent.py**: When you need custom state, custom routing, human-in-the-loop, or multi-agent coordination.

---

## File Locations

- `scraper/agent_classic.py` - Legacy implementation (classic branch)
- `scraper/agent.py` - New high-level implementation (new-agent branch)
- `scraper/langgraph_agent.py` - Custom low-level implementation (new-agent branch)
- `web_app.py` - FastAPI application that uses the agent

---

## Testing

Each implementation can be tested independently:

```bash
# Test agent_classic.py
python -m scraper.agent_classic

# Test agent.py
python -m scraper.agent

# Test langgraph_agent.py
python -m scraper.langgraph_agent
```

Or via the web interface:
```bash
uvicorn web_app:app --reload
# Visit http://localhost:8000
```
