# raq_query.py  — ms-maf branch: LangChain-free pgvector retrieval
import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------------------------------
# Plain document container — no LangChain dependency
# -------------------------------------------------
@dataclass
class Document:
    page_content: str
    metadata: dict = field(default_factory=dict)


# -------------------------------------------------
# Embeddings — must match the model used in pipelines.py
# Data was stored with nomic-embed-text-v1.5 truncated to 256 dims.
# Queries must use the "search_query: " prefix.
# -------------------------------------------------
_embed_model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1.5",
    trust_remote_code=True,
    truncate_dim=256,
)

CONNECTION_STRING = os.getenv(
    "PGVECTOR_DB_URL",
    "postgresql://myuser:mypassword@localhost:5432/myprojdb"
)


def retrieve_top3(query: str) -> List[Document]:
    """Run a pgvector similarity search and return the top-3 matching documents."""
    query_vec = _embed_model.encode(
        "search_query: " + query,
        normalize_embeddings=True,
    ).tolist()

    conn = psycopg2.connect(CONNECTION_STRING)
    register_vector(conn)
    cur = conn.cursor()
    cur.execute("SET search_path TO scraper, public;")
    cur.execute(
        """
        SELECT id, url, title, content
        FROM pages
        ORDER BY embedding <=> %s::vector
        LIMIT 3;
        """,
        (query_vec,),
    )
    rows = cur.fetchall()

    docs = []
    for _id, url, title, content in rows:
        docs.append(Document(
            page_content=content or "",
            metadata={"url": url, "title": title or "Untitled"},
        ))

    cur.close()
    conn.close()
    return docs


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


__all__ = ["Document", "retrieve_top3", "format_docs"]
