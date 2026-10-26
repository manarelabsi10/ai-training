import json
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# read all pages
reader = PdfReader("FitStyle_AI_Documentation_final (1).pdf")
pages = [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]

# split text into small chunks
def chunk_text(text, size=800, overlap=50):
    return [text[i:i + size] for i in range(0, len(text), size - overlap)]

chunks = []
for page_num, text in pages:
    for piece in chunk_text(text):
        if piece.strip():
            chunks.append({"page": page_num, "text": piece})

print("chunks:", len(chunks))

# turn every chunk into a vector (locally, no API)
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(
    [c["text"] for c in chunks],
    normalize_embeddings=True,
    show_progress_bar=True,
)

# save page + text in one file
with open("embeddings.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

# save the vectors alone in their own file
np.save("vectors.npy", vectors)

print("saved embeddings.json and vectors.npy", vectors.shape)