"""
scraper/llamaindex_agent.py - LlamaIndex ReActAgent for Agentic RAG (LlamaIndex 0.14+)

LlamaIndex 0.14 moved to a workflow-based agent API:
  agent.run(user_msg="...") → WorkflowHandler
  await handler             → AgentOutput (final response)
  handler.stream_events()   → async generator of events (ToolCall, ToolCallResult, AgentOutput...)

Three tools mirror the LangGraph implementation:
  - web_search              : DuckDuckGo web search
  - fetch_webpage           : Download and parse a URL
  - search_local_knowledge  : Query the pgvector knowledge base
"""

import os
import json
import psycopg2
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent, AgentOutput, ToolCall, ToolCallResult
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.storage.chat_store import BaseChatStore
from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.llms.openai.utils import ALL_AVAILABLE_MODELS

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _setup_tracing() -> None:
    """Configure OTel → Phoenix tracing if PHOENIX_COLLECTOR_ENDPOINT is set.

    Mirrors the LangSmith pattern: env var present = tracing on, absent = zero overhead.
    Safe to call multiple times (OTel SDK ignores duplicate provider registration).
    """
    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    if not endpoint:
        return

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    LlamaIndexInstrumentor().instrument(tracer_provider=provider)


_setup_tracing()


# Register OpenRouter model names so LlamaIndex can resolve their context window.
# LlamaIndex's OpenAI class validates model names against a hardcoded dict;
# OpenRouter uses namespaced names like "provider/model:tier" that aren't in that list.
_OPENROUTER_MODELS = {
    "openai/gpt-oss-20b:free": 128000,
    "openai/gpt-4o-mini": 128000,
    "openai/gpt-4o": 128000,
    "nvidia/nemotron-3-super-120b-a12b:free": 128000,
    "deepseek/deepseek-chat-v3.1:free": 64000,
    "meta-llama/llama-3.3-70b-instruct:free": 128000,
    "google/gemma-3-27b-it:free": 8192,
}
ALL_AVAILABLE_MODELS.update(_OPENROUTER_MODELS)


SYSTEM_PROMPT = """You are an intelligent research assistant with access to three tools:

1. web_search              - Search the web for current events, news, and real-time facts
2. fetch_webpage           - Retrieve full text content from a specific URL
3. search_local_knowledge  - Query the local pgvector knowledge base of scraped content

Instructions:
- For current events, weather, or recent facts: use web_search first
- To get full details from a URL: use fetch_webpage
- For domain-specific queries about scraped content: use search_local_knowledge
- Always cite sources with URLs when available
- If information is unavailable, say so clearly
- Synthesize information from multiple sources when helpful

After your main answer, add a Sources section listing URLs used."""


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _web_search(query: str) -> str:
    """Search the web for current information using DuckDuckGo."""
    try:
        import requests
        from bs4 import BeautifulSoup

        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        results = []
        for result in soup.find_all("div", class_="result__body")[:5]:
            title_elem = result.find("a", class_="result__a")
            snippet_elem = result.find("a", class_="result__snippet")
            if title_elem and snippet_elem:
                results.append(
                    f"Title: {title_elem.get_text(strip=True)}\n"
                    f"Snippet: {snippet_elem.get_text(strip=True)}\n"
                    f"URL: {title_elem.get('href', '')}\n"
                )
        return "\n".join(results) if results else "No search results found."
    except Exception as e:
        return f"Error searching web: {str(e)}"


def _fetch_webpage(url: str) -> str:
    """Fetch and return the text content of a webpage."""
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        return soup.get_text(" ", strip=True)[:5000]
    except Exception as e:
        return f"Error fetching {url}: {str(e)}"


def _search_local_knowledge(query: str) -> str:
    """Search the local pgvector knowledge base for domain-specific content."""
    try:
        from scraper.raq_query import retrieve_top3
        docs = retrieve_top3(query)

        if not docs:
            return "No relevant documents found in local knowledge base."

        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(
                f"Document {i}:\n"
                f"Source: {doc.metadata.get('url', 'Unknown')}\n"
                f"Title: {doc.metadata.get('title', 'Untitled')}\n"
                f"Content: {doc.page_content[:500]}\n"
            )
        return "\n".join(parts)
    except Exception as e:
        return f"Error searching local knowledge: {str(e)}"


# ---------------------------------------------------------------------------
# LlamaIndex FunctionTools
# ---------------------------------------------------------------------------

web_search_tool = FunctionTool.from_defaults(
    fn=_web_search,
    name="web_search",
    description=(
        "Search the web for current information using DuckDuckGo. "
        "Use this for recent events, news, weather, or any real-time facts."
    ),
)

fetch_webpage_tool = FunctionTool.from_defaults(
    fn=_fetch_webpage,
    name="fetch_webpage",
    description=(
        "Fetch the full text content of a specific webpage given its URL. "
        "Use this to get detailed information from a URL found in search results."
    ),
)

search_local_knowledge_tool = FunctionTool.from_defaults(
    fn=_search_local_knowledge,
    name="search_local_knowledge",
    description=(
        "Search the local pgvector knowledge base of previously scraped content. "
        "Use this for domain-specific queries about content in the local database."
    ),
)

_TOOLS = [web_search_tool, fetch_webpage_tool, search_local_knowledge_tool]


# ---------------------------------------------------------------------------
# PostgreSQL-backed chat store for persistent conversation memory
# ---------------------------------------------------------------------------

_TABLE = "llamaindex_chat_sessions"


class PostgresChatStore(BaseChatStore):
    """Persists LlamaIndex chat history in a PostgreSQL JSONB column."""

    db_url: str

    @classmethod
    def class_name(cls) -> str:
        return "PostgresChatStore"

    def _conn(self):
        return psycopg2.connect(self.db_url)

    @staticmethod
    def _serialize(messages: List[ChatMessage]) -> str:
        return json.dumps([m.model_dump(mode="json") for m in messages])

    @staticmethod
    def _text_from_blocks(blocks) -> str:
        """Extract plain text from LlamaIndex 0.14 blocks format."""
        parts = []
        for b in blocks or []:
            text = b.get("text") if isinstance(b, dict) else getattr(b, "text", None)
            if text:
                parts.append(text)
        return " ".join(parts)

    @classmethod
    def _deserialize(cls, rows) -> List[ChatMessage]:
        messages = [ChatMessage.model_validate(m) for m in rows]
        # LlamaIndex 0.14 stores content in `blocks`; ChatSummaryMemoryBuffer
        # counts tokens via m.content, so backfill it here.
        for msg in messages:
            if not msg.content:
                raw = msg.model_dump(mode="json").get("blocks") or []
                msg.content = cls._text_from_blocks(raw)
        return messages

    def set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        data = self._serialize(messages)
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""INSERT INTO {_TABLE} (session_id, messages, updated_at)
                        VALUES (%s, %s::jsonb, NOW())
                        ON CONFLICT (session_id)
                        DO UPDATE SET messages=%s::jsonb, updated_at=NOW()""",
                    (key, data, data),
                )
            conn.commit()

    def get_messages(self, key: str) -> List[ChatMessage]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT messages FROM {_TABLE} WHERE session_id=%s", (key,))
                row = cur.fetchone()
        return self._deserialize(row[0]) if row else []

    def add_message(self, key: str, message: ChatMessage, idx: Optional[int] = None) -> None:
        msgs = self.get_messages(key)
        if idx is None:
            msgs.append(message)
        else:
            msgs.insert(idx, message)
        self.set_messages(key, msgs)

    def delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        msgs = self.get_messages(key)
        self.set_messages(key, [])
        return msgs

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        msgs = self.get_messages(key)
        if idx >= len(msgs):
            return None
        removed = msgs.pop(idx)
        self.set_messages(key, msgs)
        return removed

    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        msgs = self.get_messages(key)
        if not msgs:
            return None
        last = msgs.pop()
        self.set_messages(key, msgs)
        return last

    def get_keys(self) -> List[str]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT session_id FROM {_TABLE}")
                return [r[0] for r in cur.fetchall()]


def create_sessions_table(db_url: str) -> None:
    """Create the chat sessions table if it doesn't exist (called once at startup)."""
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {_TABLE} (
                    session_id TEXT PRIMARY KEY,
                    messages   JSONB NOT NULL DEFAULT '[]',
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
        conn.commit()


def get_memory(session_id: str, db_url: str) -> ChatSummaryMemoryBuffer:
    """Return a ChatSummaryMemoryBuffer backed by PostgreSQL for this session."""
    token_limit = int(os.getenv("SUMMARIZE_TOKEN_LIMIT", "2000"))
    store = PostgresChatStore(db_url=db_url)
    return ChatSummaryMemoryBuffer.from_defaults(
        llm=_build_llm(),
        token_limit=token_limit,
        chat_store=store,
        chat_store_key=session_id,
    )


# ---------------------------------------------------------------------------
# LLM (OpenRouter — OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

def _build_llm() -> LlamaOpenAI:
    model = os.getenv("OPENAI_FREE_MODEL", "openai/gpt-oss-20b:free")
    # Register the model if not in the known list so context_window lookup succeeds.
    if model not in ALL_AVAILABLE_MODELS:
        ALL_AVAILABLE_MODELS[model] = 128000
    return LlamaOpenAI(
        model=model,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        api_base="https://openrouter.ai/api/v1",
        temperature=0.1,
        # max_tokens must be set explicitly: LlamaIndex otherwise calls tiktoken to
        # infer the budget, and tiktoken doesn't know OpenRouter model names.
        max_tokens=4096,
    )


# ---------------------------------------------------------------------------
# Agent factory
# LlamaIndex 0.14: ReActAgent is a Workflow; instantiate directly (no from_tools).
# agent.run(user_msg=...) → WorkflowHandler → await for AgentOutput
# ---------------------------------------------------------------------------

def get_agent() -> ReActAgent:
    """Create a fresh ReActAgent for a single request (stateless)."""
    return ReActAgent(
        tools=_TOOLS,
        llm=_build_llm(),
        system_prompt=SYSTEM_PROMPT,
        timeout=120,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Public query interface
# ---------------------------------------------------------------------------

async def aquery_agent(question: str, session_id: str = "") -> Dict[str, Any]:
    """Run the agent and return the final answer."""
    agent = get_agent()
    db_url = os.getenv("PGVECTOR_DB_URL", "")
    memory = get_memory(session_id, db_url) if session_id and db_url else None
    handler = agent.run(user_msg=question, memory=memory)
    result: AgentOutput = await handler
    return {
        "answer": result.response.content if result.response else "",
        "tools_used": [tc.tool_name for tc in result.tool_calls] if result.tool_calls else [],
    }


async def astream_agent_events(question: str, session_id: str = ""):
    """Async generator that yields human-readable strings as the agent runs."""
    agent = get_agent()
    db_url = os.getenv("PGVECTOR_DB_URL", "")
    memory = get_memory(session_id, db_url) if session_id and db_url else None
    handler = agent.run(user_msg=question, memory=memory)

    async for event in handler.stream_events():
        if isinstance(event, ToolCall):
            tool_input = str(event.tool_kwargs or "")[:200]
            yield f"[Tool: {event.tool_name}] {tool_input}\n"
        elif isinstance(event, ToolCallResult):
            preview = str(event.tool_output or "")[:300]
            yield f"[Result]: {preview}...\n\n"
        elif isinstance(event, AgentOutput):
            if event.response and event.response.content:
                yield f"\n{event.response.content}"


__all__ = [
    "get_agent",
    "aquery_agent",
    "astream_agent_events",
    "create_sessions_table",
    "get_memory",
    "web_search_tool",
    "fetch_webpage_tool",
    "search_local_knowledge_tool",
]
