# scraper/api.py
#
# FastAPI wrapper around the RAG knowledge base.
# Exposes a search endpoint for any agent/client to query pgvector.
#
# Run:
#   source env/bin/activate
#   uvicorn scraper.api:app --host 127.0.0.1 --port 8000
#
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(title="Nyklabs RAG API", version="1.0")

# Lazy-load the embedding model + DB on first request (avoids slow startup)
_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        from scraper.raq_query import retrieve_top3
        _retriever = retrieve_top3
    return _retriever


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


class SearchResult(BaseModel):
    content: str
    url: str
    title: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    retrieve = get_retriever()
    docs = retrieve(req.query)
    results = [
        SearchResult(
            content=doc.page_content,
            url=doc.metadata.get("url", ""),
            title=doc.metadata.get("title", "Untitled"),
        )
        for doc in docs[: req.top_k]
    ]
    return SearchResponse(query=req.query, results=results)
