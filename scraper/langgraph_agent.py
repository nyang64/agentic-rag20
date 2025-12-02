from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
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
    model=os.getenv("DEEPSEEK_FREE_MODEL", "tngtech/deepseek-r1t2-chimera:free"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.1,
)

# Create tools (same as before)
from scraper.agent import *

tools = [
    web_search(),
    fetch_webpage(),
    search_local_knowledge(),
]

#tool_executor = ToolExecutor(tools)

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

    tool_calls = getattr(last_message, "tool_calls", [])
    tool_messages = []
    sources = state.get("sources", [])

    for tool_call in tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]

        # Execute tool directly
        if name == "web_search":
            result = web_search(args)
        elif name == "fetch_webpage":
            result = fetch_webpage(args)
        elif name == "search_local_knowledge":
            result = search_local_knowledge(args)
        else:
            result = f"Unknown tool: {name}"

        # Track sources for web-related tools
        if name in ["web_search", "fetch_webpage"]:
            sources.append({"tool": name, "input": args})

        tool_messages.append({
            "role": "tool",
            "content": result,
            "tool_call_id": tool_call.get("id"),
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
    
def test_langgraph_agent():
    question = "What is the weather in New York today?"
    inputs = {
        "messages": [HumanMessage(content=question)],
        "sources": [],
    }

    result = app.invoke(inputs)

    # Print last message content (agent answer)
    answer = result["messages"][-1].content
    print("Answer:", answer)

    # Print sources used by agent
    sources = result.get("sources", [])
    print("Sources:", sources)

    # Print number of steps taken
    steps = len(result["messages"]) - 1
    print("Steps:", steps)

    return result

if __name__ == "__main__":
    test_langgraph_agent()
    
    
__all__ = [
    "query_langgraph_agent",
    "app",
    "workflow",
    "tools",
]
    
    