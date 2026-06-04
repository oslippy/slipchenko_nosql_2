import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(INDEX_NAME)

df = pd.read_parquet(INPUT_PARQUET)
embeddings = np.load(INPUT_EMBEDDINGS)
assert len(df) == len(embeddings), (
    f"Записів у датасеті ({len(df)}) і ембеддингів ({len(embeddings)}) має бути порівну"
)

for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Завантаження в Pinecone"):
    batch_df = df.iloc[start:start + BATCH_SIZE]
    batch_emb = embeddings[start:start + BATCH_SIZE]

    vectors = []
    for offset, (_, row) in enumerate(batch_df.iterrows()):
        vectors.append({
            "id": f"paper_{start + offset}",
            "values": batch_emb[offset].tolist(),
            "metadata": {
                "arxiv_id": str(row["id"]),
                "title": str(row["title"]),
                "abstract": str(row["abstract"])[:500],
                "authors": str(row["authors"])[:200],
                "year": int(row["year"]),
                "category": str(row["category"]),
            },
        })

    index.upsert(vectors=vectors)

stats = index.describe_index_stats()
print(f"Завантаження завершено. Векторів в індексі '{INDEX_NAME}': {stats.total_vector_count}")
