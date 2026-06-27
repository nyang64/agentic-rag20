import os
import asyncio
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage

from scraper.agent import create_agentic_rag, format_answer_with_sources

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

agent = create_agentic_rag()


def _extract_answer(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not (hasattr(msg, "tool_calls") and msg.tool_calls):
            return msg.content
    return ""


@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/ask")
async def ask(query: str = Form(...)):
    async def stream_answer():
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": query}]})
            messages = result.get("messages", [])
            answer = format_answer_with_sources(_extract_answer(messages))

            sources = []
            for msg in messages:
                if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc.get("name") == "fetch_webpage":
                            url = tc.get("args", {}).get("url", "")
                            if url:
                                sources.append(f'<a href="{url}" target="_blank">{url}</a>')
                        elif tc.get("name") == "web_search":
                            sources.append(f"Web Search: {tc.get('args', {}).get('query', '')}")

            sources_html = "<br>".join(sources) if sources else "Multiple web sources"
            full = f"{answer}<br><br>Sources:<br>{sources_html}"

            for char in full:
                yield char
                await asyncio.sleep(0.01)

        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(stream_answer(), media_type="text/plain")


@app.post("/ask_verbose")
async def ask_verbose(query: str = Form(...)):
    """Return full agent reasoning trace for debugging"""
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = result.get("messages", [])

    steps = []
    pending: dict = {}

    for msg in messages:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                pending[tc.get("id", "")] = {
                    "tool": tc.get("name", ""),
                    "input": str(tc.get("args", {})),
                    "output": "",
                }
        elif isinstance(msg, ToolMessage):
            tool_id = getattr(msg, "tool_call_id", "")
            if tool_id in pending:
                pending[tool_id]["output"] = (msg.content or "")[:500]
                steps.append(pending.pop(tool_id))

    return {
        "answer": _extract_answer(messages),
        "steps": steps,
    }


@app.post("/ask_streaming")
async def ask_streaming(query: str = Form(...)):
    async def stream_with_thoughts():
        yield "Thinking about your question...<br>"
        try:
            async for event in agent.astream_events(
                {"messages": [{"role": "user", "content": query}]},
                version="v2",
            ):
                kind = event.get("event", "")

                if kind == "on_tool_start":
                    tool_name = event.get("name", "")
                    tool_input = event.get("data", {}).get("input", {})
                    if isinstance(tool_input, dict):
                        tool_input = str(tool_input)
                    yield f"Using tool: {tool_name}<br>"
                    yield f"   Input: {tool_input}<br>"

                elif kind == "on_tool_end":
                    output = str(event.get("data", {}).get("output", ""))
                    yield f"Found: {output[:500]}...<br>"

                elif kind == "on_chat_model_end":
                    output = event.get("data", {}).get("output", None)
                    if output and hasattr(output, "content"):
                        if not (hasattr(output, "tool_calls") and output.tool_calls):
                            yield f"<br>Answer:<br>{output.content}"

        except Exception as e:
            yield f"<br>Error during streaming: {str(e)}"

    return StreamingResponse(stream_with_thoughts(), media_type="text/plain")
