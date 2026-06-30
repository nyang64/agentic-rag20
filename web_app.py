"""
web_app.py - FastAPI web app backed by the Microsoft Agent Framework (MAF) agent.

Endpoints:
  GET  /               Serve the chat UI (static/index.html)
  POST /ask            Stream final answer character-by-character
  POST /ask_streaming  Stream agent events (tool calls + final answer) in real-time
  POST /ask_verbose    Return full answer + tools used as JSON (for debugging)
"""

import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from scraper.ms_maf_agent import aquery_agent, astream_agent_events, create_sessions_table

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_url = os.getenv("PGVECTOR_DB_URL", "")
    if db_url:
        try:
            create_sessions_table(db_url)
            print("maf_chat_sessions table ready")
        except Exception as e:
            print(f"Warning: could not create sessions table ({e})")
    yield


app = FastAPI(title="Agentic RAG — Microsoft Agent Framework", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/ask")
async def ask(query: str = Form(...), session_id: str = Form(default="")):
    """Run the agent and stream the final answer character-by-character."""
    async def stream_answer():
        try:
            result = await aquery_agent(query, session_id=session_id)
            for char in result["answer"]:
                yield char
                await asyncio.sleep(0.005)
        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(stream_answer(), media_type="text/plain")


@app.post("/ask_streaming")
async def ask_streaming(query: str = Form(...), session_id: str = Form(default="")):
    """Stream tool-call notifications and the final answer as they are produced."""
    async def stream_events():
        yield "Thinking...<br>"
        try:
            async for chunk in astream_agent_events(query, session_id=session_id):
                yield chunk
        except Exception as e:
            yield f"<br>Error: {str(e)}"

    return StreamingResponse(stream_events(), media_type="text/plain")


@app.post("/ask_verbose")
async def ask_verbose(query: str = Form(...), session_id: str = Form(default="")):
    """Return full answer and tools used as JSON."""
    result = await aquery_agent(query, session_id=session_id)
    return result
