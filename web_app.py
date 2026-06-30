"""
web_app.py - FastAPI web app backed by the LlamaIndex ReActAgent (0.14+ workflow API).

Endpoints:
  GET  /               Serve the chat UI (static/index.html)
  POST /ask            Stream final answer character-by-character
  POST /ask_streaming  Stream agent events (tool calls + final answer) in real-time
  POST /ask_verbose    Return full answer + tools used as JSON (for debugging)
"""

import os
import asyncio
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from scraper.ms_agent import aquery_agent, astream_agent_events

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(title="Agentic RAG — AutoGen")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/ask")
async def ask(query: str = Form(...)):
    """Run the agent and stream the final answer character-by-character."""
    async def stream_answer():
        try:
            result = await aquery_agent(query)
            for char in result["answer"]:
                yield char
                await asyncio.sleep(0.005)
        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(stream_answer(), media_type="text/plain")


@app.post("/ask_streaming")
async def ask_streaming(query: str = Form(...)):
    """Stream tool-call notifications and the final answer as they are produced."""
    async def stream_events():
        yield "Thinking...<br>"
        try:
            async for chunk in astream_agent_events(query):
                yield chunk
        except Exception as e:
            yield f"<br>Error: {str(e)}"

    return StreamingResponse(stream_events(), media_type="text/plain")


@app.post("/ask_verbose")
async def ask_verbose(query: str = Form(...)):
    """Return full answer and tools used as JSON."""
    result = await aquery_agent(query)
    return result
