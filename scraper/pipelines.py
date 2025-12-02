# scraper/pipelines.py
# scraper/pipelines.py
from itemadapter import ItemAdapter
from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector

class PgVectorPipeline:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.conn = psycopg2.connect(
            "postgresql://myuser:mypassword@localhost:5432/myprojdb"
        )
        register_vector(self.conn)
        self.cur = self.conn.cursor()
        self.cur.execute("SET search_path TO scraper, public;")

    def chunk_text(self, text: str, chunk_size: int = 1500, overlap: int = 50):
        """Split text into overlapping chunks (pure Python)."""
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
            if end >= len(text):
                break
        return chunks

    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        url = adapter["url"]
        title = adapter.get("title", "")
        full_text = adapter["text"]

        chunks = self.chunk_text(full_text, chunk_size=1500, overlap=50)

        for i, chunk in enumerate(chunks):
            embedding = self.model.encode(chunk, normalize_embeddings=True)
            chunk_title = f"{title} [Chunk {i+1}/{len(chunks)}]" if title else f"Chunk {i+1}"

            # UNIQUE KEY: (url, chunk_id) or (url, title)
            self.cur.execute(
                """
                INSERT INTO pages (url, title, content, embedding, chunk_id)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (url, chunk_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding;
                """,
                (url, chunk_title, chunk, embedding.tolist(), i)
            )

        self.conn.commit()
        return item

    def close_spider(self, spider):
        self.cur.close()
        self.conn.close()