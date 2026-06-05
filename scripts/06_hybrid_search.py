import os
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10
RRF_K = 60

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)


def tokenize(text):
    return re.findall(r"\w+", text.lower())


corpus = (df["title"] + " " + df["abstract"]).tolist()
bm25 = BM25Okapi([tokenize(doc) for doc in corpus])


def search_bm25(query, top_k=TOP_K):
    scores = bm25.get_scores(tokenize(query))
    return [int(i) for i in np.argsort(scores)[::-1][:top_k]]


def search_vector(query, top_k=TOP_K):
    qv = model.encode(query, normalize_embeddings=True).tolist()
    res = index.query(vector=qv, top_k=top_k, include_metadata=False)
    return [int(m.id.split("_")[1]) for m in res.matches]


def rrf(rank_lists, k=RRF_K, top=5):
    scores = {}
    for ranked in rank_lists:
        for rank, doc in enumerate(ranked, 1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])[:top]


def search_hybrid(query, k=RRF_K, top=5):
    return rrf([search_bm25(query), search_vector(query)], k=k, top=top)


def title_of(i):
    return " ".join(df.iloc[i]["title"].split())


QUERIES = [
    "BERT fine-tuning",
    "Yann LeCun convolutional networks",
    "making computers understand human emotions from text",
]

for q in QUERIES:
    print("\nзапит:", q)
    print("bm25:")
    for r, i in enumerate(search_bm25(q)[:5], 1):
        print(f"{r}. {title_of(i)[:65]}")
    print("вектор:")
    for r, i in enumerate(search_vector(q)[:5], 1):
        print(f"{r}. {title_of(i)[:65]}")
    print("гібрид (rrf):")
    for r, (i, s) in enumerate(search_hybrid(q), 1):
        print(f"{r}. {s:.4f}  {title_of(i)[:60]}")
