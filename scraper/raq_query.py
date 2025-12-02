# raq_query.py
import os
import time
from typing import List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
import psycopg2
from pgvector.psycopg2 import register_vector
from openai import RateLimitError

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------------------------------
# 1. Embeddings
# -------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------------------------
# 2. LLM – OpenRouter (DeepSeek V3)
# -------------------------------------------------
llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_FREE_MODEL", "deepseek/deepseek-chat-v3.1:free"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.1,
)

# -------------------------------------------------
# 3. DB
# -------------------------------------------------
CONNECTION_STRING = os.getenv(
    "PGVECTOR_DB_URL",
    "postgresql://myuser:mypassword@localhost:5432/myprojdb"
)

# -------------------------------------------------
# 4. Custom Retriever – **NO BaseRetriever import**
# -------------------------------------------------
def retrieve_top3(query: str) -> List[Document]:
    """Run vector search → fetch only top-3 rows → return LangChain Documents."""
    # 1. Embed the query
    query_vec = embeddings.embed_query(query)

    # 2. Connect + search
    conn = psycopg2.connect(CONNECTION_STRING)
    register_vector(conn)
    cur = conn.cursor()
    cur.execute("SET search_path TO scraper, public;")

    # 3. Top-3 nearest neighbors
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

    # 4. Build Documents
    docs = []
    for _id, url, title, content in rows:
        docs.append(
            Document(
                page_content=content or "",
                metadata={"url": url, "title": title or "Untitled"},
            )
        )

    cur.close()
    conn.close()
    return docs

# -------------------------------------------------
# 5. RAG Chain (uses the function directly)
# -------------------------------------------------
prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Use the following context to answer the question.

Context: {context}

Question: {question}

Answer:"""
)

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# Build the chain – `retrieve_top3` is called on every query
rag_chain = (
    {
        "context": lambda x: format_docs(retrieve_top3(x)),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# -------------------------------------------------
# 6. Retry helper
# -------------------------------------------------
def invoke_with_retry(chain, query: str, max_retries: int = 5) -> str:
    retry = 0
    while True:
        try:
            return chain.invoke(query)
        except RateLimitError:
            if retry >= max_retries:
                raise
            wait = 2 ** retry
            print(f"Rate limited. Retrying in {wait}s... ({retry+1}/{max_retries})")
            time.sleep(wait)
            retry += 1

# -------------------------------------------------
# 7. Export for web_app.py
# -------------------------------------------------
__all__ = ["rag_chain", "invoke_with_retry", "retrieve_top3"]
