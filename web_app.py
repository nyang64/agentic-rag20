import os
import asyncio
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Import from the LangChain agent implementation (agent.py)
from scraper.agent import create_agentic_rag_executor, format_answer_with_sources

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize agent using langchain.agents.create_agent (agent.py)
agent_executor = create_agentic_rag_executor()


@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/ask")
async def ask(query: str = Form(...)):
    async def stream_answer():
        try:
            # Use the LangGraph agent executor
            result = agent_executor.invoke({"input": query})

            # Format the answer to ensure Sources section is properly formatted
            answer = format_answer_with_sources(result["output"])

            # Extract sources from intermediate steps
            sources = []
            if "intermediate_steps" in result:
                for action, observation in result["intermediate_steps"]:
                    if hasattr(action, "tool") and action.tool == "web_search":
                        # Extract URLs from web search results
                        sources.append(f"Web Search: {action.tool_input}")
                    elif hasattr(action, "tool") and action.tool == "fetch_webpage":
                        tool_input = action.tool_input
                        if isinstance(tool_input, dict):
                            tool_input = tool_input.get("url", str(tool_input))
                        sources.append(f'<a href="{tool_input}" target="_blank">{tool_input}</a>')

            sources_html = "<br>".join(sources) if sources else "Multiple web sources"
            full = f"{answer}<br><br>Sources:<br>{sources_html}"

            # Stream response
            for char in full:
                yield char
                await asyncio.sleep(0.01)

        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(stream_answer(), media_type="text/plain")


@app.post("/ask_verbose")
async def ask_verbose(query: str = Form(...)):
    """Return full agent reasoning trace for debugging"""
    result = agent_executor.invoke({"input": query})

    steps = []
    for action, observation in result.get("intermediate_steps", []):
        tool_input = action.tool_input
        if isinstance(tool_input, dict):
            tool_input = str(tool_input)
        steps.append({
            "tool": action.tool,
            "input": tool_input,
            "output": observation[:500] if observation else ""  # Truncate
        })

    return {
        "answer": result["output"],
        "steps": steps
    }


@app.post("/ask_streaming")
async def ask_streaming(query: str = Form(...)):
    async def stream_with_thoughts():
        # Stream agent's thought process
        yield "Thinking about your question...<br>"

        try:
            async for chunk in agent_executor.astream({"input": query}):
                if "actions" in chunk:
                    for action in chunk["actions"]:
                        tool_input = action.tool_input
                        if isinstance(tool_input, dict):
                            tool_input = str(tool_input)
                        yield f"Using tool: {action.tool}<br>"
                        yield f"   Input: {tool_input}<br>"

                if "steps" in chunk:
                    for step in chunk["steps"]:
                        yield f"Found: {step.observation[:500]}...<br>"

                if "output" in chunk:
                    yield f"<br>Answer:<br>{chunk['output']}"
        except Exception as e:
            yield f"<br>Error during streaming: {str(e)}"

    return StreamingResponse(stream_with_thoughts(), media_type="text/plain")
