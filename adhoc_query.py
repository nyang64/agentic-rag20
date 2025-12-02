from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector
import sys

if len(sys.argv) == 2:
    query = sys.argv[1]
else:
    query = input("\nEnter your query: ").strip()

model = SentenceTransformer("all-MiniLM-L6-v2")
conn = psycopg2.connect("postgresql://myuser:mypassword@localhost:5432/myprojdb")
register_vector(conn)
cur = conn.cursor()
cur.execute("SET search_path TO scraper, public;")

#query = "toll roads and expressways in China"
vec = model.encode(query, normalize_embeddings=True)

# vector operator:  <=> cosine distance; <-> Euclidean L2; <#> inner product
# %s is the place holder for the vec
cur.execute("""
    SELECT url, title, content, embedding <=> %s AS distance
    FROM pages
    ORDER BY distance
    LIMIT 3;
""", (vec,))

for row in cur.fetchall():
    print(f"{row[0]} | dist={row[3]:.4f} | {row[1][:60]}...")
