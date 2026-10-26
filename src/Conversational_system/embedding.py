import json
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"   # put all your pdfs here

# split text into small chunks
def chunk_text(text, size=500, overlap=50):
    return [text[i:i + size] for i in range(0, len(text), size - overlap)]

pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
if not pdf_files:
    raise FileNotFoundError(f"No PDF files found in {DOCS_DIR}")

chunks = []
for pdf_path in pdf_files:
    reader = PdfReader(pdf_path)
    count = 0
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        for piece in chunk_text(text):
            if piece.strip():
                chunks.append({"source": pdf_path.name, "page": i + 1, "text": piece})
                count += 1
    print(pdf_path.name, "->", count, "chunks")
    if count == 0:
        print("  warning: no text found, this pdf may be a scan")

print("total chunks:", len(chunks))

# turn every chunk into a vector (locally)
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=True)

with open(BASE_DIR / "embeddings.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)
np.save(BASE_DIR / "vectors.npy", vectors)

print("saved", vectors.shape)