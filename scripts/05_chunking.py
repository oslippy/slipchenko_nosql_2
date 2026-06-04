import os
import re

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768
N_DOCS = 30
CHUNK_WORDS = 50
OVERLAP = 10
MAX_WORDS = 50
BATCH_SIZE = 100
TOP_K = 5

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")

df["n_words"] = df["abstract"].str.split().str.len()
top = df.nlargest(N_DOCS, "n_words").reset_index(drop=True)


def chunk_fixed(text, size=CHUNK_WORDS, overlap=OVERLAP):
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


def chunk_semantic(text, max_words=MAX_WORDS):
    text = " ".join(text.split())
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current, count = [], [], 0
    for sent in sentences:
        n = len(sent.split())
        if current and count + n > max_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sent)
        count += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def ensure_index(name):
    if not pc.has_index(name):
        pc.create_index(
            name=name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(name)


def load_chunks(index, chunker, label):
    items = []
    for _, row in top.iterrows():
        for n, chunk in enumerate(chunker(row["abstract"])):
            items.append((row, n, chunk))

    texts = [c for _, _, c in items]
    embs = model.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )

    vectors = []
    for (row, n, chunk), emb in zip(items, embs):
        vectors.append(
            {
                "id": f"{row['id']}_{n}",
                "values": emb.tolist(),
                "metadata": {
                    "arxiv_id": str(row["id"]),
                    "title": str(row["title"]),
                    "chunk": chunk,
                    "chunk_num": n,
                    "year": int(row["year"]),
                    "category": str(row["category"]),
                },
            }
        )

    for i in tqdm(range(0, len(vectors), BATCH_SIZE), desc=f"upload {label}"):
        index.upsert(vectors=vectors[i : i + BATCH_SIZE])
    return len(vectors)


fixed_index = ensure_index("arxiv-chunks-fixed")
semantic_index = ensure_index("arxiv-chunks-semantic")

n_fixed = load_chunks(fixed_index, chunk_fixed, "fixed")
n_semantic = load_chunks(semantic_index, chunk_semantic, "semantic")
print(f"\n{N_DOCS} статей -> fixed: {n_fixed} чанків, semantic: {n_semantic} чанків")


def search_chunks(index, query):
    qv = model.encode(query, normalize_embeddings=True).tolist()
    return index.query(vector=qv, top_k=TOP_K, include_metadata=True).matches


def show_chunks(matches):
    for i, m in enumerate(matches, 1):
        md = m.metadata
        chunk = " ".join(md["chunk"].split())[:140]
        print(f"{i}. {m.score:.3f}  {md['title'][:55]}  [чанк {int(md['chunk_num'])}]")
        print(f"   {chunk}")


TEST_QUERIES = [
    "dark matter distribution in galaxies",
    "gamma-ray burst afterglow",
]
for q in TEST_QUERIES:
    print("\nзапит:", q)
    print("fixed:")
    show_chunks(search_chunks(fixed_index, q))
    print("semantic:")
    show_chunks(search_chunks(semantic_index, q))
