import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_FILE = "data/arxiv_subset.parquet"
OUTPUT_DIR = "embeddings"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "embeddings.npy")
MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 64

df = pd.read_parquet(INPUT_FILE)

texts = (df["title"] + " [SEP] " + df["abstract"]).tolist()

model = SentenceTransformer(MODEL_NAME)

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
)

print(f"Оброблено текстів:       {len(embeddings)}")
print(f"Розмірність ембеддингів: {embeddings.shape[1]}")
print(f"Норма першого вектора:   {np.linalg.norm(embeddings[0]):.4f}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(OUTPUT_FILE, embeddings)
print(f"Збережено у {OUTPUT_FILE} (shape={embeddings.shape})")
