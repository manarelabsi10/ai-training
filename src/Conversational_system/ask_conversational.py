import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
CHAT_MODEL = "gemma3:1b"   # exact name from `ollama list`
MAX_TURNS = 3              # how many previous Q&A pairs we remember

ANSWER_PROMPT = """You are an assistant that answers questions about the provided documents.

Rules:
- Use only the context provided. Do not add facts that are not in it.
- If the context does not contain the answer, reply: "I couldn't find this in the document."
- Write one paragraph of 2 to 4 short sentences, in simple clear English.
- Ignore table of contents, page numbers, and dotted lines in the context.
- Each passage in the context starts with the document name and page in brackets.
- If the documents say different things, say which document says what.
- If the question asks for more detail, add new information from the context. Do not repeat the previous answer.
- Answer in your own words. Do not quote long sentences from the context.
"""

REWRITE_PROMPT = """Given the conversation and a follow-up question, rewrite the follow-up as one complete standalone question.
Replace words like "it", "that", "the third one" with what they refer to.
If the question is already standalone, return it unchanged.
Return only the question, nothing else."""


@lru_cache(maxsize=1)
def load_index():
    vectors = np.load(BASE_DIR / "vectors.npy")
    with (BASE_DIR / "embeddings.json").open(encoding="utf-8") as f:
        data = json.load(f)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    llm = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    return vectors, data, embed_model, llm


def list_sources():
    """Names of the indexed pdfs (useful for a dropdown in the UI)."""
    _, data, _, _ = load_index()
    return sorted({d["source"] for d in data})


def rewrite_question(llm, question, history):
    """Turn a follow-up question into a standalone one, using the recent chat."""
    if not history:
        return question
    recent = history[-MAX_TURNS * 2:]
    chat_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
    res = llm.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": f"Conversation:\n{chat_text}\n\nFollow-up question: {question}"},
        ],
    )
    return (res.choices[0].message.content or question).strip()


def answer_question(question, history=None, source=None):
    """history = list of {"role": "user" or "assistant", "content": "..."}
    source = optional pdf name, to search only inside that file."""
    question = question.strip()
    if not question:
        raise ValueError("Please enter a question.")
    history = history or []

    vectors, data, embed_model, llm = load_index()

    # 1. make the question standalone
    standalone = rewrite_question(llm, question, history)
    print("searching for:", standalone)

    # 2. search (optionally only inside one pdf)
    q_vec = embed_model.encode([standalone], normalize_embeddings=True)[0]
    scores = vectors @ q_vec
    if source:
        for i, d in enumerate(data):
            if d["source"] != source:
                scores[i] = -1

    # take the 5 best chunks, but drop the weak ones
    ranked = np.argsort(scores)[::-1][:5]
    best = scores[ranked[0]]
    top = [i for i in ranked if scores[i] >= best - 0.15]

    context = "\n\n".join(
        f"[{data[int(i)]['source']}, page {data[int(i)]['page']}] {data[int(i)]['text']}"
        for i in top
    )

    # group page numbers under each file name
    by_file = {}
    for i in top:
        d = data[int(i)]
        by_file.setdefault(d["source"], set()).add(d["page"])

    pages = [f"{name}: pages {', '.join(str(p) for p in sorted(nums))}"
             for name, nums in by_file.items()]

    # 3. answer, with the recent chat so the tone stays natural
    messages = [{"role": "system", "content": ANSWER_PROMPT}]
    messages += history[-MAX_TURNS * 2:]
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {standalone}"})

    res = llm.chat.completions.create(model=CHAT_MODEL, temperature=0, messages=messages)
    return res.choices[0].message.content or "I couldn't find this in the document.", pages


if __name__ == "__main__":
    # quick terminal test (the UI does not need this block)
    history = []
    while True:
        q = input("\nYou: ")
        if q.strip().lower() == "exit":
            break
        answer, pages = answer_question(q, history)
        print("\n" + answer)
        print("Sources:")
        for line in pages:
            print(" -", line)
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]