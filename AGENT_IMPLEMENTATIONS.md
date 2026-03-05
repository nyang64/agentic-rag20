# Agent Implementations Comparison

This document explains the differences between the agent implementations across different branches:

| Branch | File | Library | Status |
|--------|------|---------|--------|
| `classic` | `agent_classic.py` | `langchain_classic` | Legacy, stable |
| `new-agent` | `agent.py` | `langchain.agents` | **Recommended** |
| `new-agent` | `langgraph_agent.py` | `langgraph.graph` | Custom workflows |
| `prebuilt` | `agent.py` | `langgraph.prebuilt` | **Deprecated** |

---

## Table of Contents

1. [Overview](#overview)
2. [Branch Summary](#branch-summary)
3. [Library & Import Differences](#library--import-differences)
4. [Architecture Comparison](#architecture-comparison)
5. [Code Comparison](#code-comparison)
6. [Key Differences Summary](#key-differences-summary)
7. [When to Use Each](#when-to-use-each)
8. [Evolution Path](#evolution-path)
9. [Deprecation Notice](#deprecation-notice)

---

## Overview

All implementations achieve the same goal: an Agentic RAG system that can:
- Search the web for current information
- Fetch content from specific URLs
- Search a local knowledge base (vector database)

The difference lies in **which library/API** they use and **what level of control** they provide.

---

## Branch Summary

### `classic` branch
- Uses `langchain_classic.agents.AgentExecutor`
- The original, proven implementation
- Best for: Legacy compatibility, stability

### `new-agent` branch (Recommended)
- Uses `langchain.agents.create_agent` (high-level)
- Also includes `langgraph_agent.py` for custom workflows
- Best for: New projects, production use

### `prebuilt` branch
- Uses `langgraph.prebuilt.create_react_agent`
- **DEPRECATED** as of LangGraph v1.0
- Best for: Educational purposes, understanding LangGraph internals

---

## Library & Import Differences

| Branch | File | Library | Key Import |
|--------|------|---------|------------|
| `classic` | `agent_classic.py` | `langchain_classic` | `from langchain_classic.agents import AgentExecutor, create_tool_calling_agent` |
| `new-agent` | `agent.py` | `langchain` | `from langchain.agents import create_agent` |
| `new-agent` | `langgraph_agent.py` | `langgraph` | `from langgraph.graph import StateGraph, END` |
| `prebuilt` | `agent.py` | `langgraph.prebuilt` | `from langgraph.prebuilt import create_react_agent` |

---

## Architecture Comparison

### agent_classic.py - Legacy Architecture (classic branch)

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

### agent.py - New High-Level Architecture (new-agent branch)

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

### agent.py - Prebuilt Architecture (prebuilt branch) - DEPRECATED

```
┌─────────────────────────────────────────────────────┐
│               langgraph.prebuilt                     │
│        (DEPRECATED - moved to langchain.agents)      │
├─────────────────────────────────────────────────────┤
│                                                      │
│   create_react_agent(                               │
│       model=llm,                                    │
│       tools=tools,                                  │
│       prompt=SYSTEM_PROMPT,                         │
│       checkpointer=checkpointer,                    │
│   )                                                 │
│              │                                       │
│              ▼                                       │
│   Returns: CompiledStateGraph                       │
│                                                      │
└─────────────────────────────────────────────────────┘

Input:  {"messages": [{"role": "user", "content": "question"}]}
Output: {"messages": [HumanMessage, AIMessage, ToolMessage, ...]}
```

**Characteristics:**
- Uses `langgraph.prebuilt` package directly
- **DEPRECATED** as of LangGraph v1.0
- Same functionality as `langchain.agents.create_agent`
- Exists for educational purposes and backward compatibility

---

### langgraph_agent.py - Custom Low-Level Architecture (new-agent branch)

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

**agent_classic.py (classic branch):**
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

**agent.py (new-agent branch) - RECOMMENDED:**
```python
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,  # Just a string, no template needed
    checkpointer=checkpointer,
)
```

**agent.py (prebuilt branch) - DEPRECATED:**
```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)
```

**langgraph_agent.py (new-agent branch):**
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
    if state["iteration_count"] > 10:
        return "end"
    if hasattr(state["messages"][-1], "tool_calls"):
        return "continue"
    return "end"

def call_model(state):
    response = llm.bind_tools(tools).invoke(state["messages"])
    return {"messages": [response], "iteration_count": state["iteration_count"] + 1}

def call_tool(state):
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

## Key Differences Summary

| Feature | agent_classic.py | agent.py (new-agent) | agent.py (prebuilt) | langgraph_agent.py |
|---------|-----------------|---------------------|--------------------|--------------------|
| **Branch** | `classic` | `new-agent` | `prebuilt` | `new-agent` |
| **Package** | `langchain_classic` | `langchain.agents` | `langgraph.prebuilt` | `langgraph.graph` |
| **Status** | Legacy | **Recommended** | **Deprecated** | For custom needs |
| **Abstraction** | Medium | High | High | Low |
| **Returns** | `AgentExecutor` | `CompiledStateGraph` | `CompiledStateGraph` | `CompiledStateGraph` |
| **State** | Fixed | Fixed | Fixed | **Custom** |
| **Routing** | Fixed | Fixed | Fixed | **Custom** |
| **Input** | `{"input": "..."}` | `{"messages": [...]}` | `{"messages": [...]}` | Custom dict |

---

## When to Use Each

| Use Case | Recommended Branch |
|----------|-------------------|
| New projects | `new-agent` with `langchain.agents.create_agent` |
| Legacy codebase compatibility | `classic` |
| Custom state tracking | `new-agent` with `langgraph_agent.py` |
| Human-in-the-loop workflows | `new-agent` with `langgraph_agent.py` |
| Multi-agent systems | `new-agent` with `langgraph_agent.py` |
| Learning LangGraph internals | `prebuilt` (educational only) |

---

## Evolution Path

```
                                    DEPRECATED
                                        │
langchain_classic        langgraph.prebuilt        langchain.agents
(legacy)                 (v0.x)                    (v1.0+, recommended)
     │                        │                          │
     ▼                        ▼                          ▼
AgentExecutor         create_react_agent ──────►  create_agent
     │                        │                          │
     │                        └──────────────────────────┤
     │                                                   │
     │                   langgraph.graph                 │
     │                   (low-level)                     │
     │                        │                          │
     │                        ▼                          │
     │                   StateGraph()                    │
     │                   (full control)                  │
     │                        │                          │
     └────────────────────────┴──────────────────────────┘
                              │
                              ▼
                    CompiledStateGraph
                    (LangGraph runtime)
```

**Migration path:**
1. `langchain_classic.agents.AgentExecutor` → Legacy, still works
2. `langgraph.prebuilt.create_react_agent` → **Deprecated**, migrate to #3
3. `langchain.agents.create_agent` → **Recommended** for new code
4. `langgraph.graph.StateGraph` → For custom workflows

---

## Deprecation Notice

### langgraph.prebuilt.create_react_agent

As of **LangGraph v1.0**, `create_react_agent` has been moved from `langgraph.prebuilt` to `langchain.agents` and renamed to `create_agent`.

**Old (deprecated):**
```python
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(model=llm, tools=tools, prompt="...")
```

**New (recommended):**
```python
from langchain.agents import create_agent
agent = create_agent(model=llm, tools=tools, system_prompt="...")
```

The `prebuilt` branch exists for educational purposes to demonstrate the original LangGraph prebuilt API.

---

## File Locations by Branch

### classic branch
- `scraper/agent_classic.py` - Main agent implementation

### new-agent branch
- `scraper/agent.py` - Uses `langchain.agents.create_agent`
- `scraper/langgraph_agent.py` - Custom `StateGraph` implementation

### prebuilt branch
- `scraper/agent.py` - Uses `langgraph.prebuilt.create_react_agent`

### All branches
- `web_app.py` - FastAPI application that uses the agent

---

## Testing

Each implementation can be tested independently:

```bash
# Test agent_classic.py (classic branch)
git checkout classic
python -m scraper.agent_classic

# Test agent.py with langchain.agents (new-agent branch)
git checkout new-agent
python -m scraper.agent

# Test langgraph_agent.py (new-agent branch)
python -m scraper.langgraph_agent

# Test agent.py with langgraph.prebuilt (prebuilt branch)
git checkout prebuilt
python -m scraper.agent
```

Or via the web interface:
```bash
uvicorn web_app:app --reload
# Visit http://localhost:8000
```
