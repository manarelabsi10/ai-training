import os
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

client = OpenAI(base_url="https://openrouter.ai/api/v1",api_key=os.environ["OPENROUTER_API_KEY"],)

EMBED_MODEL = "nvidia/nemotron-3.5-embed-1b:free"

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

# turn texts into vectors (in small batches to avoid rate limits)
def embed(texts, batch_size=32):
    vecs = []
    for i in range(0, len(texts), batch_size):
        res = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + batch_size])
        vecs.extend(d.embedding for d in res.data)
    vecs = np.array(vecs)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)  # length 1

vectors = embed([c["text"] for c in chunks])

# save chunk + vector together in one file
data = []
for chunk, vector in zip(chunks, vectors):
    data.append({"page": chunk["page"], "text": chunk["text"]})

with open("embeddings.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)



# also save the vectors alone in their own file
np.save("vectors.npy", vectors)

print("saved embeddings.json and vectors.npy")