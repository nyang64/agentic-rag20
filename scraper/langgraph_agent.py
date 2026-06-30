# scraper/langgraph_agent.py - Custom LangGraph Workflow for Agentic RAG 2.0
#
# This module demonstrates how to build a CUSTOM LangGraph workflow when you need
# more control than what langchain.agents.create_agent provides.
#
# Relationship to agent.py:
# - agent.py uses `langchain.agents.create_agent` (high-level, recommended for most cases)
# - langgraph_agent.py builds the workflow manually using `langgraph.graph.StateGraph`
#
# Use this approach when you need:
# - Custom state beyond messages (e.g., tracking sources, metadata)
# - Custom routing logic between nodes
# - Human-in-the-loop interrupts at specific points
# - Multi-agent coordination
# - Custom middleware or hooks

import os
import operator
from typing import TypedDict, Annotated, Sequence, List, Dict, Any
from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Import tools from agent.py to avoid duplication
from scraper.agent import (
    web_search,
    fetch_webpage,
    search_local_knowledge,
    format_answer_with_sources,
    SYSTEM_PROMPT,
)

load_dotenv()


def _setup_tracing() -> None:
    """Wire LangChain/LangGraph OTel instrumentation to a local Phoenix collector.

    Uses the same PHOENIX_COLLECTOR_ENDPOINT env var as the llamaindex branch.
    LangChainInstrumentor patches LangChain's global callback system so every
    chain, LLM call, and tool invocation is captured automatically — no agent
    code changes needed.  No-op when the env var is absent.
    """
    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    if not endpoint:
        return
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from openinference.instrumentation.langchain import LangChainInstrumentor
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    LangChainInstrumentor().instrument(tracer_provider=provider)


_setup_tracing()


# -------------------------------------------------------------------
# Custom Agent State (extends beyond just messages)
# -------------------------------------------------------------------

class CustomAgentState(TypedDict):
    """Custom state that tracks additional information beyond messages.

    This is the advantage of building a custom workflow - you can track
    whatever state you need throughout the agent's execution.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sources: List[Dict[str, Any]]  # Track sources used
    iteration_count: int  # Track number of iterations
    tools_used: List[str]  # Track which tools were used


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
# Tools list and mapping
# -------------------------------------------------------------------

tools = [web_search, fetch_webpage, search_local_knowledge]

tool_map = {
    "web_search": web_search,
    "fetch_webpage": fetch_webpage,
    "search_local_knowledge": search_local_knowledge,
}


# -------------------------------------------------------------------
# Graph Node Functions
# -------------------------------------------------------------------

def should_continue(state: CustomAgentState) -> str:
    """Decide whether to continue executing tools or end the workflow.

    This is where you can add custom routing logic, e.g.:
    - Limit iterations: if state["iteration_count"] > 5: return "end"
    - Require certain tools: if "search_local_knowledge" not in state["tools_used"]: ...
    """
    messages = state["messages"]
    last_message = messages[-1]

    # Check iteration limit (custom logic example)
    if state.get("iteration_count", 0) > 10:
        return "end"

    # If LLM decided to use a tool, continue
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"

    return "end"


def call_model(state: CustomAgentState) -> Dict[str, Any]:
    """Call the LLM to decide the next action or generate final response."""
    messages = list(state["messages"])

    # Ensure system message is at the beginning
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    # Bind tools and invoke
    model_with_tools = llm.bind_tools(tools)
    response = model_with_tools.invoke(messages)

    # Increment iteration count
    new_iteration = state.get("iteration_count", 0) + 1

    return {
        "messages": [response],
        "iteration_count": new_iteration,
    }


def call_tool(state: CustomAgentState) -> Dict[str, Any]:
    """Execute the tool(s) chosen by the LLM."""
    messages = state["messages"]
    last_message = messages[-1]

    tool_calls = getattr(last_message, "tool_calls", [])
    tool_messages = []
    sources = list(state.get("sources", []))
    tools_used = list(state.get("tools_used", []))

    for tool_call in tool_calls:
        name = tool_call.get("name", "")
        args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id", "")

        tool_fn = tool_map.get(name)

        if tool_fn:
            try:
                result = tool_fn.invoke(args)
            except Exception as e:
                result = f"Error executing {name}: {str(e)}"

            # Track source info (custom state tracking)
            if name in ["web_search", "fetch_webpage"]:
                sources.append({
                    "tool": name,
                    "input": args,
                    "result_preview": str(result)[:200]
                })

            # Track tool usage
            if name not in tools_used:
                tools_used.append(name)
        else:
            result = f"Unknown tool: {name}"

        tool_messages.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call_id,
        ))

    return {
        "messages": tool_messages,
        "sources": sources,
        "tools_used": tools_used,
    }


# -------------------------------------------------------------------
# Build the Graph Workflow
# -------------------------------------------------------------------

def build_custom_workflow(checkpointer=None):
    """Build and compile the custom LangGraph workflow.

    This gives you full control over the agent's execution flow.

    Args:
        checkpointer: Optional memory checkpointer for conversation persistence.

    Returns:
        Compiled LangGraph workflow.
    """
    workflow = StateGraph(CustomAgentState)

    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tool)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional routing
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        }
    )

    # Loop back from tools to agent
    workflow.add_edge("tools", "agent")

    # Compile
    return workflow.compile(checkpointer=checkpointer)


# -------------------------------------------------------------------
# Query Functions
# -------------------------------------------------------------------

def query_custom_agent(question: str, thread_id: str = None) -> Dict[str, Any]:
    """Query the custom LangGraph agent.

    Args:
        question: The user's question
        thread_id: Optional thread ID for conversation memory

    Returns:
        Dict with answer, sources, tools_used, and iteration_count
    """
    workflow = build_custom_workflow()

    inputs = {
        "messages": [HumanMessage(content=question)],
        "sources": [],
        "iteration_count": 0,
        "tools_used": [],
    }

    config = {}
    if thread_id:
        config = {"configurable": {"thread_id": thread_id}}

    result = workflow.invoke(inputs, config)

    # Extract final answer
    answer = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not (hasattr(msg, "tool_calls") and msg.tool_calls):
            answer = msg.content
            break

    return {
        "answer": answer,
        "sources": result.get("sources", []),
        "tools_used": result.get("tools_used", []),
        "iteration_count": result.get("iteration_count", 0),
    }


# -------------------------------------------------------------------
# Wrapper for web_app.py compatibility
# -------------------------------------------------------------------

class CustomWorkflowWrapper:
    """Wrapper that provides AgentExecutor-like interface for the custom workflow."""

    def __init__(self, use_memory: bool = False):
        checkpointer = MemorySaver() if use_memory else None
        self.workflow = build_custom_workflow(checkpointer=checkpointer)
        self.use_memory = use_memory
        self._thread_counter = 0

    def invoke(self, inputs: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Invoke with AgentExecutor-compatible interface."""
        query = inputs.get("input", "")

        workflow_inputs = {
            "messages": [HumanMessage(content=query)],
            "sources": [],
            "iteration_count": 0,
            "tools_used": [],
        }

        invoke_config = config or {}
        if self.use_memory and "configurable" not in invoke_config:
            self._thread_counter += 1
            invoke_config = {"configurable": {"thread_id": f"thread_{self._thread_counter}"}}

        result = self.workflow.invoke(workflow_inputs, invoke_config)

        # Extract output and intermediate steps
        output = ""
        intermediate_steps = []

        for msg in result["messages"]:
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
            "sources": result.get("sources", []),
            "tools_used": result.get("tools_used", []),
            "iteration_count": result.get("iteration_count", 0),
        }

    async def astream(self, inputs: Dict[str, Any], config: Dict[str, Any] = None):
        """Async stream the workflow's execution."""
        query = inputs.get("input", "")

        workflow_inputs = {
            "messages": [HumanMessage(content=query)],
            "sources": [],
            "iteration_count": 0,
            "tools_used": [],
        }

        invoke_config = config or {}
        if self.use_memory and "configurable" not in invoke_config:
            self._thread_counter += 1
            invoke_config = {"configurable": {"thread_id": f"thread_{self._thread_counter}"}}

        async for event in self.workflow.astream_events(workflow_inputs, invoke_config, version="v2"):
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


def create_custom_workflow_executor(use_memory: bool = False) -> CustomWorkflowWrapper:
    """Create an AgentExecutor-compatible wrapper for the custom workflow."""
    return CustomWorkflowWrapper(use_memory=use_memory)


# -------------------------------------------------------------------
# Test Function
# -------------------------------------------------------------------

def test_custom_workflow():
    """Test the custom LangGraph workflow."""
    print("Testing custom LangGraph workflow...")
    print("This workflow tracks: sources, tools_used, iteration_count")

    result = query_custom_agent("What is the weather in New York today?")

    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(format_answer_with_sources(result["answer"]))

    print("\n" + "=" * 60)
    print("CUSTOM STATE INFO:")
    print("=" * 60)
    print(f"Tools used: {result['tools_used']}")
    print(f"Iterations: {result['iteration_count']}")
    print(f"Sources tracked: {len(result['sources'])}")
    for src in result["sources"]:
        print(f"  - {src['tool']}: {src['input']}")

    return result


if __name__ == "__main__":
    test_custom_workflow()


# Export
__all__ = [
    "build_custom_workflow",
    "query_custom_agent",
    "create_custom_workflow_executor",
    "CustomWorkflowWrapper",
    "CustomAgentState",
    "tools",
]
