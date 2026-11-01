import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
# the index is shared with the old system (read only)
INDEX_DIR = Path(os.environ.get("RAG_INDEX_DIR", BASE_DIR.parent / "Conversational_system"))

CHAT_MODEL = "qwen2.5:1.5b"
MAX_TURNS = 3            # previous Q&A pairs remembered
MAX_SUBQUESTIONS = 3     # max parts a question is split into
MAX_RETRIES = 1          # extra search attempts per sub-question
TOP_K = 5
MIN_SCORE = 0.30         # below this, nothing relevant exists -> not found
STRONG_SCORE = 0.70      # above this, skip the LLM check and trust retrieval
REL_MARGIN = 0.15        # drop chunks much weaker than the best one
MAX_CHUNKS = 5           # total chunks sent to the final answer
DUP_THRESHOLD = 0.85     # sub-questions more similar than this are duplicates
NOT_FOUND = "I couldn't find this in the document."

GREETING_WORDS = {"hello", "hi", "hey", "thanks", "thank", "bye", "goodbye"}
FOLLOWUP_WORDS = {"he", "she", "it", "they", "them", "his", "her", "its", "their",
                  "this", "that", "these", "those", "one", "ones", "there"}

REWRITE_PROMPT = """Given the conversation and a follow-up question, rewrite the follow-up as one complete standalone question.
Replace words like "it", "that", "he", "the third one" with what they refer to.
Return only the question, nothing else."""

DECOMPOSE_PROMPT = f"""Split the question ONLY if it asks about two or more DIFFERENT things.
Never write the same question twice in different words.
Keep the names of people, places and things in every part.
If it is one question, return it unchanged on one line.
Return at most {MAX_SUBQUESTIONS} lines, no numbering, no explanation.

Examples:
Compare A and B.
What is A?
What is B?

Tell me about X and its price.
Tell me about X.
What is the price of X?

Who built the pyramid?
Who built the pyramid?"""

CHECK_PROMPT = """You check if a context contains the information needed to answer a question.
Reply YES if the answer is clearly in the context. Reply NO otherwise.
Reply with one word only: YES or NO."""

SEARCH_REWRITE_PROMPT = """The search for this question found nothing useful.
Write a different short search query (3 to 8 keywords) that could find the answer in a document.
Use different words than the first query. Return only the query."""

ANSWER_PROMPT = """You are an assistant that answers questions about the provided documents.

Rules:
- Use only the context provided. Do not add facts that are not in it.
- Write 2 to 5 short sentences in simple clear English. If the question had several parts, answer each part.
- Ignore table of contents, page numbers, and dotted lines in the context.
- Each passage starts with the document name and page in brackets.
- If the documents say different things, say which document says what.
- Answer in your own words. Do not quote long sentences from the context.
"""

CHAT_PROMPT = "You are a friendly assistant for a document question-answering app. Reply briefly. Do not invent facts."


@lru_cache(maxsize=1)
def load_index():
    vectors = np.load(INDEX_DIR / "vectors.npy")
    with (INDEX_DIR / "embeddings.json").open(encoding="utf-8") as f:
        data = json.load(f)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    llm = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    return vectors, data, embed_model, llm


def list_sources():
    _, data, _, _ = load_index()
    return sorted({d["source"] for d in data})


def ask(system, user, history=None):
    """One LLM call."""
    _, _, _, llm = load_index()
    messages = [{"role": "system", "content": system}]
    if history:
        messages += history
    messages.append({"role": "user", "content": user})
    res = llm.chat.completions.create(model=CHAT_MODEL, temperature=0, messages=messages)
    return (res.choices[0].message.content or "").strip()


# ---------- agent steps ----------

def route(question):
    """CHAT only for short greetings / thanks (decided by code, not by the small model).
    Everything else is searched."""
    words = re.findall(r"[a-z']+", question.lower())
    if words and len(words) <= 4 and words[0] in GREETING_WORDS:
        return "CHAT"
    return "SEARCH"


def looks_like_followup(question):
    """Short question that uses words like he / it / that -> probably needs the history."""
    q = question.lower().strip()
    words = re.findall(r"[a-z']+", q)
    return len(words) <= 8 and (
        any(w in FOLLOWUP_WORDS for w in words)
        or q.startswith(("and ", "what about", "how about"))
    )


def make_standalone(question, history):
    """Rewrite with the chat history, only when the question is a real follow-up."""
    if not history or not looks_like_followup(question):
        return question
    recent = history[-MAX_TURNS * 2:]
    chat_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
    out = ask(REWRITE_PROMPT, f"Conversation:\n{chat_text}\n\nFollow-up question: {question}")
    return out or question


def dedupe(subs):
    """Remove sub-questions that mean the same thing."""
    if len(subs) < 2:
        return subs
    _, _, embed_model, _ = load_index()
    vecs = embed_model.encode(subs, normalize_embeddings=True)
    keep = []
    for i in range(len(subs)):
        if all(float(vecs[i] @ vecs[j]) < DUP_THRESHOLD for j in keep):
            keep.append(i)
    return [subs[i] for i in keep]


def decompose(question):
    out = ask(DECOMPOSE_PROMPT, question)
    lines = [re.sub(r"^\s*(\d+[\.\)]|[-*•])\s*", "", l).strip() for l in out.splitlines()]
    lines = [l for l in lines if len(l) > 8 and "->" not in l]
    seen, subs = set(), []
    for l in lines:
        if l.lower() not in seen:
            seen.add(l.lower())
            subs.append(l)
    return dedupe(subs[:MAX_SUBQUESTIONS]) or [question]


def retrieve(query, source=None):
    """Return a list of (chunk_index, score). Empty list = nothing relevant."""
    vectors, data, embed_model, _ = load_index()
    q_vec = embed_model.encode([query], normalize_embeddings=True)[0]
    scores = vectors @ q_vec
    if source:
        for i, d in enumerate(data):
            if d["source"] != source:
                scores[i] = -1
    ranked = np.argsort(scores)[::-1][:TOP_K]
    best = scores[ranked[0]]
    if best < MIN_SCORE:
        return []
    return [(int(i), float(scores[i])) for i in ranked if scores[i] >= best - REL_MARGIN]


def format_context(idxs):
    _, data, _, _ = load_index()
    return "\n\n".join(
        f"[{data[i]['source']}, page {data[i]['page']}] {data[i]['text']}" for i in idxs
    )


def is_sufficient(question, idxs):
    out = ask(CHECK_PROMPT, f"Context:\n{format_context(idxs)}\n\nQuestion: {question}").upper()
    return out.startswith("YES")


def search_with_retry(sub, source, trace):
    """Search, check the result, and retry with a new query if it looks weak."""
    query = sub
    for attempt in range(MAX_RETRIES + 1):
        hits = retrieve(query, source)
        best = hits[0][1] if hits else 0.0
        trace.append(f"  search '{query}' -> {len(hits)} chunks, best score {best:.2f}")
        if hits:
            idxs = [i for i, _ in hits]
            if best >= STRONG_SCORE or is_sufficient(sub, idxs):
                return idxs
            trace.append("  context judged not enough")
        if attempt < MAX_RETRIES:
            query = ask(SEARCH_REWRITE_PROMPT, f"Question: {sub}\nFirst query: {query}") or sub
            trace.append(f"  retry with new query: '{query}'")
    return []


# ---------- main entry ----------

def answer_question(question, history=None, source=None, verbose=False):
    """Returns (answer, pages).
    history = list of {"role": "user"/"assistant", "content": "..."}"""
    question = question.strip()
    if not question:
        raise ValueError("Please enter a question.")
    history = history or []
    trace = []

    # 1. greeting or real question?
    if route(question) == "CHAT":
        trace.append("route: CHAT (no search)")
        if verbose:
            print("\n".join(trace))
        return ask(CHAT_PROMPT, question, history[-MAX_TURNS * 2:]), []
    trace.append("route: SEARCH")

    # 2. standalone question using memory (only for real follow-ups)
    standalone = make_standalone(question, history)
    trace.append(f"standalone: {standalone}")

    # 3. split into sub-questions
    subs = decompose(standalone)
    trace.append(f"sub-questions: {subs}")

    # 4. search each sub-question (with check + retry)
    used, missing = [], []
    for sub in subs:
        trace.append(f"* {sub}")
        idxs = search_with_retry(sub, source, trace)
        if idxs:
            used += [i for i in idxs if i not in used]
        else:
            missing.append(sub)

    if verbose:
        print("\n".join(trace))

    if not used:
        return NOT_FOUND, []

    # 5. answer from the collected context
    used = used[:MAX_CHUNKS]
    answer = ask(
        ANSWER_PROMPT,
        f"Context:\n{format_context(used)}\n\nQuestion: {standalone}",
        history[-MAX_TURNS * 2:],
    ) or NOT_FOUND

    if missing:  # added by code, so the small model cannot forget it
        answer += "\n\nNote: I couldn't find information about: " + "; ".join(missing)

    _, data, _, _ = load_index()
    by_file = {}
    for i in used:
        by_file.setdefault(data[i]["source"], set()).add(data[i]["page"])
    pages = [f"{name}: pages {', '.join(str(p) for p in sorted(nums))}"
             for name, nums in by_file.items()]
    return answer, pages


if __name__ == "__main__":
    print("Commands: 'new' = clear memory, 'sources' = list pdfs, 'exit' = quit")
    history = []
    while True:
        q = input("\nYou: ").strip()
        if not q:
            continue
        cmd = q.lower()
        if cmd == "exit":
            break
        if cmd == "new":
            history = []
            print("History cleared.")
            continue
        if cmd == "sources":
            for name in list_sources():
                print(" -", name)
            continue
        answer, pages = answer_question(q, history, verbose=True)
        print("\n" + answer)
        for line in pages:
            print(" -", line)
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]