import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5
EMBEDDINGS_FILE = "embeddings/embeddings.npy"

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")


def encode_query(text):
    return model.encode(text, normalize_embeddings=True)


def show(matches):
    if not matches:
        print("  нічого не знайдено")
        return
    for i, m in enumerate(matches, 1):
        md = m.metadata
        title = " ".join(md["title"].split())
        abstract = " ".join(md["abstract"].split())[:150]
        print(f"{i}. {m.score:.3f}  {md['category']} {int(md['year'])}  {title}")
        print(f"   {abstract}")


QUERY = "teaching machines to recognize objects in pictures"
print("\nкрок 3: семантичний пошук")
print("запит:", QUERY)
res = index.query(
    vector=encode_query(QUERY).tolist(), top_k=TOP_K, include_metadata=True
)
show(res.matches)


QUERY_RL = "reinforcement learning"
qvec_rl = encode_query(QUERY_RL).tolist()

print("\nкрок 4: пошук з фільтром, запит:", QUERY_RL)
print("A) cs.LG, останні 5 років (рік >= 2021):")
res_a = index.query(
    vector=qvec_rl,
    top_k=TOP_K,
    include_metadata=True,
    filter={"category": {"$eq": "cs.LG"}, "year": {"$gte": 2021}},
)
show(res_a.matches)

print("\nB) будь-яка категорія, рік < 2015:")
res_b = index.query(
    vector=qvec_rl, top_k=TOP_K, include_metadata=True, filter={"year": {"$lt": 2015}}
)
show(res_b.matches)

print("\nкатегорії в A:", sorted({m.metadata["category"] for m in res_a.matches}))
print("категорії в B:", sorted({m.metadata["category"] for m in res_b.matches}))


print("\nкрок 5: метрики для запиту:", QUERY)
embeddings = np.load(EMBEDDINGS_FILE)
q = encode_query(QUERY)

cosine = (embeddings @ q) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q))
dot = embeddings @ q
l2 = np.linalg.norm(embeddings - q, axis=1)


def top_ids(scores, largest=True):
    order = np.argsort(scores)
    return order[::-1][:TOP_K] if largest else order[:TOP_K]


def show_metric(name, ids, scores):
    print(name)
    for i in ids:
        t = " ".join(df.iloc[int(i)]["title"].split())[:55]
        print(f"   {scores[i]:.4f}  paper_{int(i)}  {t}")


ids_cos = top_ids(cosine, largest=True)
ids_dot = top_ids(dot, largest=True)
ids_l2 = top_ids(l2, largest=False)

show_metric("cosine:", ids_cos, cosine)
show_metric("dot:", ids_dot, dot)
show_metric("l2 (відстань, менша = ближча):", ids_l2, l2)

same = list(map(int, ids_cos)) == list(map(int, ids_dot)) == list(map(int, ids_l2))
print("\nтоп-5 однаковий для всіх трьох метрик:", same)
