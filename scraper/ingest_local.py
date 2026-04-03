# scraper/ingest_local.py
#
# Ingest a local directory into the RAG knowledge base (same `scraper.pages` table).
#
# Usage:
#   python -m scraper.ingest_local /path/to/docs
#   python -m scraper.ingest_local /path/to/docs --glob "**/*.md"
#
import argparse
import os
import sys
from pathlib import Path

import psycopg2
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Optional PDF support
# ---------------------------------------------------------------------------
try:
    import pypdf
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ---------------------------------------------------------------------------
# File types handled as plain text
# ---------------------------------------------------------------------------
PLAIN_TEXT_EXTS = {
    ".txt", ".md", ".rst", ".csv",
    ".py", ".js", ".ts", ".java", ".go", ".rb", ".c", ".cpp", ".h",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env",
    ".xml", ".sql",
}

# ---------------------------------------------------------------------------
# Dirs / files to skip
# ---------------------------------------------------------------------------
SKIP_DIRS = {
    ".git", ".hg", ".svn",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules", ".venv", "venv", "env",
    "dist", "build", ".tox",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_text(path: Path) -> str | None:
    """Return plain text for a file, or None if unsupported / unreadable."""
    ext = path.suffix.lower()

    if ext in PLAIN_TEXT_EXTS:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"  [warn] could not read {path}: {e}")
            return None

    if ext in {".html", ".htm"}:
        try:
            soup = BeautifulSoup(path.read_bytes(), "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            return soup.get_text(separator=" ", strip=True)
        except Exception as e:
            print(f"  [warn] could not parse HTML {path}: {e}")
            return None

    if ext == ".pdf":
        if not PDF_AVAILABLE:
            print(f"  [skip] {path.name}: install pypdf for PDF support")
            return None
        try:
            reader = pypdf.PdfReader(str(path))
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(pages)
        except Exception as e:
            print(f"  [warn] could not read PDF {path}: {e}")
            return None

    return None  # unsupported extension


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks (same logic as PgVectorPipeline)."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if end >= len(text):
            break
    return chunks


# ---------------------------------------------------------------------------
# Main ingestion routine
# ---------------------------------------------------------------------------

def ingest_directory(root: Path, glob_pattern: str = "**/*") -> None:
    db_url = os.getenv(
        "PGVECTOR_DB_URL",
        "postgresql://myuser:mypassword@localhost:5432/myprojdb",
    )

    print(f"Loading embedding model …")
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        truncate_dim=256,
    )

    conn = psycopg2.connect(db_url)
    register_vector(conn)
    cur = conn.cursor()
    cur.execute("SET search_path TO scraper, public;")

    files = sorted(
        p for p in root.rglob(glob_pattern.lstrip("**/") if "/" not in glob_pattern else glob_pattern.replace("**/", ""))
        if p.is_file()
        and not any(part.startswith(".") or part in SKIP_DIRS for part in p.parts)
    )

    # Use rglob properly
    files = [
        p for p in root.rglob("*")
        if p.is_file()
        and not any(part in SKIP_DIRS or part.startswith(".") for part in p.relative_to(root).parts)
    ]
    # Apply optional glob filter
    if glob_pattern != "**/*":
        import fnmatch
        files = [p for p in files if fnmatch.fnmatch(p.name, glob_pattern.split("/")[-1])]

    files = sorted(files)
    print(f"Found {len(files)} candidate files under {root}\n")

    total_chunks = 0
    for path in files:
        text = extract_text(path)
        if not text or not text.strip():
            continue

        rel = path.relative_to(root)
        url = f"file://{path.resolve()}"
        title = str(rel)
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            embedding = model.encode(
                "search_document: " + chunk,
                normalize_embeddings=True,
            )
            chunk_title = f"{title} [Chunk {i+1}/{len(chunks)}]"
            cur.execute(
                """
                INSERT INTO pages (url, title, content, embedding, chunk_id)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (url, chunk_id) DO UPDATE SET
                    title     = EXCLUDED.title,
                    content   = EXCLUDED.content,
                    embedding = EXCLUDED.embedding;
                """,
                (url, chunk_title, chunk, embedding.tolist(), i),
            )

        conn.commit()
        total_chunks += len(chunks)
        print(f"  {rel}  ({len(chunks)} chunk{'s' if len(chunks) != 1 else ''})")

    cur.close()
    conn.close()
    print(f"\nDone. {len(files)} files → {total_chunks} chunks stored in scraper.pages.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest a local directory into the RAG knowledge base."
    )
    parser.add_argument("directory", help="Root directory to ingest")
    parser.add_argument(
        "--glob",
        default="**/*",
        help='File glob filter, e.g. "**/*.md" (default: all files)',
    )
    args = parser.parse_args()

    root = Path(args.directory).resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory.", file=sys.stderr)
        sys.exit(1)

    ingest_directory(root, args.glob)
