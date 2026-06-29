"""
Shared pytest fixtures and LLM configuration for DeepEval and Ragas tests.

Both libraries need an LLM for scoring. We point them at the same
OpenRouter endpoint the application already uses so no extra API keys
are required.
"""

import os
import asyncio
import pytest
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Shared sample dataset
# ---------------------------------------------------------------------------

SAMPLE_QA = [
    {
        "question": "What is pgvector and how is it used in PostgreSQL?",
        "contexts": [
            (
                "pgvector is an open-source PostgreSQL extension that adds support for "
                "vector similarity search. It allows you to store vector embeddings "
                "directly in PostgreSQL tables and perform nearest-neighbor searches "
                "using operators like <=> (cosine distance), <-> (Euclidean distance), "
                "and <#> (inner product). It is widely used for semantic search, "
                "recommendation systems, and RAG pipelines."
            ),
            (
                "To install pgvector, run CREATE EXTENSION vector; in your PostgreSQL "
                "database. You can then create a column of type vector(N) where N is "
                "the number of dimensions. Indexes such as HNSW or IVFFlat speed up "
                "approximate nearest-neighbor queries at scale."
            ),
        ],
        "ground_truth": (
            "pgvector is a PostgreSQL extension that enables vector similarity search. "
            "It stores embeddings as vector columns and supports cosine, Euclidean, "
            "and inner-product distance operators. It is commonly used for semantic "
            "search and RAG applications."
        ),
        "answer": (
            "pgvector is an open-source extension for PostgreSQL that adds vector "
            "similarity search capabilities. You can store high-dimensional embeddings "
            "in a vector(N) column and query the nearest neighbors using operators "
            "like <=> for cosine distance. It is often paired with embedding models "
            "to build semantic-search and retrieval-augmented generation (RAG) systems."
        ),
    },
    {
        "question": "How does the RAG pipeline retrieve relevant documents?",
        "contexts": [
            (
                "The RAG pipeline first encodes the user query into a vector embedding "
                "using the nomic-embed-text-v1.5 model (truncated to 256 dimensions). "
                "It then performs a cosine-similarity search against all stored page "
                "embeddings in PostgreSQL using pgvector's <=> operator, returning the "
                "top-3 most relevant documents."
            ),
            (
                "Embedding queries with nomic-embed-text-v1.5 requires a "
                "'search_query: ' prefix to be prepended to the query text before "
                "encoding. The resulting 256-dimensional vector is then compared with "
                "stored document embeddings to rank relevance."
            ),
        ],
        "ground_truth": (
            "The pipeline encodes the user query with the nomic-embed-text-v1.5 model "
            "and performs a cosine-similarity search in PostgreSQL via pgvector, "
            "returning the top-3 closest documents."
        ),
        "answer": (
            "To retrieve documents, the pipeline embeds the user query using "
            "nomic-embed-text-v1.5 (truncated to 256 dimensions) with a "
            "'search_query: ' prefix, then performs a cosine-similarity search "
            "against stored page embeddings in PostgreSQL using pgvector's <=> "
            "operator, returning the top-3 most relevant documents."
        ),
    },
    {
        "question": "What tools does the LangGraph agent have access to?",
        "contexts": [
            (
                "The custom LangGraph agent has three tools: web_search for fetching "
                "current information from the internet via DuckDuckGo or Brave Search, "
                "fetch_webpage for downloading and parsing the HTML content of a "
                "specific URL, and search_local_knowledge for querying the local "
                "pgvector knowledge base with previously scraped content."
            ),
        ],
        "ground_truth": (
            "The LangGraph agent has three tools: web_search, fetch_webpage, and "
            "search_local_knowledge."
        ),
        "answer": (
            "The LangGraph agent can use three tools: (1) web_search – searches "
            "DuckDuckGo or Brave for real-time information, (2) fetch_webpage – "
            "retrieves and parses the text of a given URL, and (3) "
            "search_local_knowledge – queries the pgvector knowledge base for "
            "domain-specific scraped content."
        ),
    },
    {
        "question": "What embedding model is used and why was it chosen?",
        "contexts": [
            (
                "The project uses nomic-ai/nomic-embed-text-v1.5 truncated to 256 "
                "dimensions for both storing and querying embeddings. This model was "
                "chosen because it supports Matryoshka embedding truncation, which "
                "allows reducing dimensions while preserving most of the semantic "
                "quality. The 256-dimension setting offers a good balance between "
                "storage efficiency and retrieval accuracy."
            ),
        ],
        "ground_truth": (
            "nomic-embed-text-v1.5 truncated to 256 dimensions is used because it "
            "supports Matryoshka truncation, balancing storage efficiency with "
            "retrieval quality."
        ),
        "answer": (
            "The project uses nomic-ai/nomic-embed-text-v1.5 truncated to 256 "
            "dimensions. It was chosen because it supports Matryoshka embedding "
            "truncation, which allows reducing dimensions while preserving most of "
            "the semantic quality. The 256-dimension setting offers a good balance "
            "between storage efficiency and retrieval accuracy."
        ),
    },
]

# A case with a deliberately bad (unfaithful) answer – useful for negative tests.
# The answer is completely fabricated and contradicts the context.
BAD_ANSWER_CASE = {
    "question": "What is pgvector and how is it used in PostgreSQL?",
    "contexts": [
        (
            "pgvector is an open-source PostgreSQL extension that adds support for "
            "vector similarity search. It stores embeddings in a vector(N) column."
        )
    ],
    "ground_truth": (
        "pgvector is a PostgreSQL extension for vector similarity search."
    ),
    # Faithfulness negative: claims contradict the context entirely.
    # Note: answer_relevancy measures question-answer coherence, not factual accuracy,
    # so use faithfulness (not answer_relevancy) to detect this kind of hallucination.
    "answer": (
        "pgvector is a Redis module that caches SQL query results in memory. "
        "It speeds up relational JOIN operations and has no relation to machine "
        "learning, neural networks, or embedding vectors."
    ),
}


@pytest.fixture(scope="session")
def event_loop():
    """Provide a shared asyncio event loop for the whole test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Remove this if you see "ScopeMismatch" warnings — the loop fixture above is
# only needed by async test functions that explicitly request it.

