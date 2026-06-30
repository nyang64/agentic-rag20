import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from scraper.langgraph_agent import create_custom_workflow_executor
from scraper.agent import format_answer_with_sources

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start up: create async DB pool + AsyncPostgresSaver checkpointer."""
    db_url = os.getenv("PGVECTOR_DB_URL", "")
    checkpointer = None
    pool = None

    if db_url:
        try:
            from psycopg_pool import AsyncConnectionPool
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            # setup() runs CREATE INDEX CONCURRENTLY which requires autocommit;
            # from_conn_string opens with autocommit=True, so we run setup there,
            # then switch to a pool for all subsequent requests.
            async with AsyncPostgresSaver.from_conn_string(db_url) as tmp:
                await tmp.setup()
            pool = AsyncConnectionPool(conninfo=db_url, open=False)
            await pool.open()
            checkpointer = AsyncPostgresSaver(pool)
            print("Conversation memory: PostgreSQL checkpointer ready")
        except Exception as e:
            print(f"Warning: could not init PostgreSQL checkpointer ({e}); running without memory")

    app.state.agent = create_custom_workflow_executor(checkpointer=checkpointer)
    yield

    if pool:
        await pool.close()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/ask")
async def ask(request: Request, query: str = Form(...), session_id: str = Form(default="")):
    agent = request.app.state.agent

    async def stream_answer():
        try:
            result = await agent.ainvoke({"input": query}, session_id=session_id)
            answer = format_answer_with_sources(result["output"])

            sources = []
            for action, observation in result.get("intermediate_steps", []):
                if hasattr(action, "tool") and action.tool == "web_search":
                    sources.append(f"Web Search: {action.tool_input}")
                elif hasattr(action, "tool") and action.tool == "fetch_webpage":
                    tool_input = action.tool_input
                    if isinstance(tool_input, dict):
                        tool_input = tool_input.get("url", str(tool_input))
                    sources.append(f'<a href="{tool_input}" target="_blank">{tool_input}</a>')

            sources_html = "<br>".join(sources) if sources else "Multiple web sources"
            full = f"{answer}<br><br>Sources:<br>{sources_html}"

            for char in full:
                yield char
                await asyncio.sleep(0.01)

        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(stream_answer(), media_type="text/plain")


@app.post("/ask_verbose")
async def ask_verbose(request: Request, query: str = Form(...), session_id: str = Form(default="")):
    """Return full agent reasoning trace for debugging."""
    agent = request.app.state.agent
    result = await agent.ainvoke({"input": query}, session_id=session_id)

    steps = []
    for action, observation in result.get("intermediate_steps", []):
        tool_input = action.tool_input
        if isinstance(tool_input, dict):
            tool_input = str(tool_input)
        steps.append({
            "tool": action.tool,
            "input": tool_input,
            "output": observation[:500] if observation else "",
        })

    return {
        "answer": result["output"],
        "steps": steps,
    }


@app.post("/ask_streaming")
async def ask_streaming(request: Request, query: str = Form(...), session_id: str = Form(default="")):
    agent = request.app.state.agent

    async def stream_with_thoughts():
        yield "Thinking about your question...<br>"
        try:
            async for chunk in agent.astream({"input": query}, session_id=session_id):
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
