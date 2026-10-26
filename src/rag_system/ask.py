import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
CHAT_MODEL = "gemma3:4b"   # use the exact name from `ollama list`

SYSTEM_PROMPT = """You are an assistant that answers questions about the FitStyle AI project documentation.

Rules:
- Use only the context provided. Do not add facts, numbers, or claims that are not in it.
- If the context does not contain the answer, reply: "I couldn't find this in the document."
- Write one paragraph of 2 to 4 short sentences, in simple clear English.
- Start with a direct answer to the question, then add one or two key details.
- Do not copy long phrases from the context. Explain in your own words.
- Ignore table of contents, page numbers, and dotted lines in the context.
- If the answer is a list of items (models, steps, components), list all of them that appear in the context, and say what each one does. Never merge items or guess a role that is not written.
"""


@lru_cache(maxsize=1)
def load_index():
    """Load the models and saved files once, on the first question."""
    vectors_path = BASE_DIR / "vectors.npy"
    data_path = BASE_DIR / "embeddings.json"
    if not vectors_path.exists() or not data_path.exists():
        raise FileNotFoundError("The RAG index files are missing. Run embedding.py first to create them.")

    vectors = np.load(vectors_path)
    with data_path.open(encoding="utf-8") as f:
        data = json.load(f)

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    return vectors, data, embed_model, ollama_client


def answer_question(question: str) -> tuple[str, list[int]]:
    """Answer a question using the saved document embeddings."""
    question = question.strip()
    if not question:
        raise ValueError("Please enter a question.")

    vectors, data, embed_model, ollama_client = load_index()

    # turn the question into a vector (same model as the chunks)
    q_vec = embed_model.encode([question], normalize_embeddings=True)[0]

    # find the 6 closest chunks
    top = np.argsort(vectors @ q_vec)[::-1][:6]
    context = "\n\n".join(data[int(i)]["text"] for i in top)
    pages = sorted({data[int(i)]["page"] for i in top})

    # let the local model answer from those chunks
    response = ollama_client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )

    return response.choices[0].message.content or "I couldn't find this in the document.", pages