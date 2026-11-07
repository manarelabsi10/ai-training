import os
import re
import json
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
import requests
import chromadb
import pymupdf                      
import ollama
import threading
import sys
import unicodedata
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sklearn.cluster import KMeans
from collections import Counter  

import logging

# the shared logging_config.py lives in src/, one level above this folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from logging_config import setup_logging  

setup_logging(reset=(__name__ == "__main__"))
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration (all values come from .env)
# --------------------------------------------------------------------------- #
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBED_MODEL = os.getenv("OPENROUTER_MODEL")      
OLLAMA_HOST = os.getenv("OLLAMA_HOST")           
DRAFT_MODEL = os.getenv("LOCAL_MODEL")            
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VERIFY_MODEL = os.getenv("GEMINI_MODEL")                 

REQUIRED = {
    "OPENROUTER_API_KEY": OPENROUTER_API_KEY,
    "OPENROUTER_MODEL": EMBED_MODEL,
    "OLLAMA_HOST": OLLAMA_HOST,
    "LOCAL_MODEL": DRAFT_MODEL,
    "api-key": GEMINI_API_KEY,
    "model": VERIFY_MODEL,
}
missing = [name for name, value in REQUIRED.items() if not value]
if missing:
    logger.critical("Missing values in .env: %s", ", ".join(missing))
    raise RuntimeError(f"Missing values in .env: {', '.join(missing)}")

ollama_client = ollama.Client(host=OLLAMA_HOST, timeout=300)
gemini_client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        timeout=45_000,
        retry_options=types.HttpRetryOptions(attempts=1),
    ),
)

CHUNK_SIZE = 1000
OVERLAP = 200
EMBED_BATCH_SIZE = 20
INDEX_VERSION = "clean2"

BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = str(BASE_DIR / "chroma_db")

# Every PDF lives here: PDFs you drop in manually and PDFs uploaded from the UI.
DOCS_DIR = BASE_DIR.parent / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

if not any(DOCS_DIR.glob("*.pdf")):
    logger.warning("No PDFs found in %s yet. Upload one from the UI or drop it in the folder", DOCS_DIR)

print(f"Verifier model: {VERIFY_MODEL!r}")
logger.info("Verifier model: %r", VERIFY_MODEL)

# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
def get_chroma_collection():
    """Return the persistent ChromaDB collection (created if missing)."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name="docs",
        metadata={"hnsw:space": "cosine"},
    )


def extract_pdf(pdf_path):
    """Extract text from a PDF, dropping page numbers and table-of-contents lines."""
    lines = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            for line in page.get_text().split("\n"):
                line = line.strip()
                if not line or line.isdigit():
                    continue
                if re.search(r"\.{4,}", line):  # table of contents / list of figures
                    continue
                lines.append(line)
    return "\n".join(lines)

def clean_text(text):
    # turn ligatures like "ﬁ" into normal letters ("fi")
    text = unicodedata.normalize("NFKC", text)
    # join words that were split at the end of a line: "specifi-\ncations" -> "specifications"
    text = re.sub(r"-\n(\w)", r"\1", text)
    # collapse extra spaces and new lines
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def looks_like_toc(chunk):
    """Return True if a chunk looks like a table of contents or a list of titles."""
    words = chunk.split()
    if not words:
        return True
    # mostly page numbers
    if sum(w.isdigit() for w in words) / len(words) > 0.25:
        return True
    # mostly Title Case words (chapter titles) instead of normal prose
    alpha = [w for w in words if w[0].isalpha()]
    if alpha and sum(w[0].isupper() for w in alpha) / len(alpha) > 0.6:
        return True
    return False

MAX_CHUNK_CHARS = 2000

def fix_chunk_sizes(chunks):
    """Drop empty chunks and split chunks that are too long for the embedding model."""
    fixed = []
    for c in chunks:
        for i in range(0, len(c), MAX_CHUNK_CHARS):
            piece = c[i:i + MAX_CHUNK_CHARS].strip()
            if piece:
                fixed.append(piece)
    return fixed


def split_text_to_chunks(text):
    """Split text into whole-sentence chunks with overlap."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    chunks, current, length = [], [], 0

    for sentence in sentences:
        if length + len(sentence) > CHUNK_SIZE and current:
            chunks.append(" ".join(current))
            # keep only the tail that fits in the overlap window
            while current and length > OVERLAP:
                length -= len(current[0]) + 1
                current.pop(0)
        current.append(sentence)
        length += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


def get_embeddings(texts):
    """Embed texts in batches through the OpenRouter embeddings API."""
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        response = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={"model": EMBED_MODEL, "input": batch},
            timeout=60,
        )
        if not response.ok:
            print(f"Embeddings error: {response.text[:500]}")
            logger.error("Embeddings request failed: %s", response.text[:500])
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        vectors.extend(item["embedding"] for item in data)
    return vectors


def file_sha256(path):
    """Return the SHA-256 hash of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ingest_pdf(pdf_path, name=None, progress_callback=None):
    """Chunk, embed and store one PDF in ChromaDB.

    Returns (status, chunk_count) where status is "added", "unchanged" or "empty".
    - Identical file already indexed  -> skipped ("unchanged").
    - Same file name but new content  -> old chunks are replaced.
    progress_callback(done, total) is called after each embedded batch.
    """
    pdf_path = str(pdf_path)
    name = name or os.path.basename(pdf_path)
    file_hash = file_sha256(pdf_path) + "_" + INDEX_VERSION
    collection = get_chroma_collection()

    if collection.get(where={"file_hash": file_hash}, limit=1)["ids"]:
        logger.debug("%s is unchanged, skipping", name)
        return "unchanged", 0

    # same name, different content: drop the stale version first
    collection.delete(where={"source": name})

    chunks = split_text_to_chunks(clean_text(extract_pdf(pdf_path)))
    before = len(chunks)
    chunks = [c for c in chunks if not looks_like_toc(c)]
    logger.info("%s: skipped %d table-of-contents chunks", name, before - len(chunks))
    chunks = fix_chunk_sizes(chunks)
    if not chunks:
        logger.warning("No text could be extracted from %s", name)
        return "empty", 0
    logger.info("%s: longest chunk is %d chars", name, max(len(c) for c in chunks))

    total = len(chunks)
    try:
        for start in range(0, total, EMBED_BATCH_SIZE):
            batch = chunks[start:start + EMBED_BATCH_SIZE]
            collection.add(
                documents=batch,
                embeddings=get_embeddings(batch),
                metadatas=[{
                    "source": name,
                    "file_hash": file_hash,
                    "embed_model": EMBED_MODEL,
                }] * len(batch),
                ids=[f"{file_hash[:16]}_{start + j}" for j in range(len(batch))],
            )
            if progress_callback:
                progress_callback(min(start + len(batch), total), total)
    except Exception:
        # remove the partial book so the next run starts it from scratch
        collection.delete(where={"file_hash": file_hash})
        logger.exception("Indexing %s failed, partial chunks were removed", name)
        raise

    logger.info("Indexed %s: %d chunks", name, total)
    print(f"Indexed {name}: {total} chunks")
    return "added", total
def build_index():
    """Make sure every PDF in DOCS_DIR is indexed.

    Unchanged PDFs are skipped quickly. If the embedding model changed since the
    index was built, the whole collection is rebuilt.
    """
    collection = get_chroma_collection()
    if collection.count() > 0:
        sample = collection.get(limit=1, include=["metadatas"])["metadatas"][0]
        if sample.get("embed_model") != EMBED_MODEL:
            logger.warning("Embedding model changed (or old index format). Rebuilding the index")
            print("Embedding model changed (or old index format). Rebuilding the index.")
            chromadb.PersistentClient(path=CHROMA_PATH).delete_collection(name="docs")

    for pdf_path in sorted(DOCS_DIR.glob("*.pdf")):
        status, _ = ingest_pdf(pdf_path)
        if status == "unchanged":
            logger.debug("%s: already indexed", pdf_path.name)
            print(f"{pdf_path.name}: already indexed.")
        elif status == "empty":
            logger.warning("No text found in %s, skipping", pdf_path.name)
            print(f"No text found in {pdf_path.name}, skipping.")
    return get_chroma_collection()


def list_documents():
    """Return {file name: chunk count} for everything currently indexed."""
    collection = get_chroma_collection()
    if collection.count() == 0:
        return {}
    metas = collection.get(include=["metadatas"])["metadatas"]
    return dict(Counter(m["source"] for m in metas))


def delete_document(name):
    """Remove a document from the index (and from DOCS_DIR if it is there)."""
    logger.info("Deleting document %s from the index", name)
    get_chroma_collection().delete(where={"source": name})
    path = DOCS_DIR / Path(name).name
    if path.exists():
        path.unlink()


# --------------------------------------------------------------------------- #
# Retrieval, clustering, subset building
# --------------------------------------------------------------------------- #
def retrieve(question, k=10):
    """Return (documents, metadatas, embeddings) of the top-k chunks."""
    collection = get_chroma_collection()
    count = collection.count()
    if count == 0:
        logger.warning("Retrieval skipped: the index is empty")
        return [], [], np.array([])
    q_emb = get_embeddings([question])[0]
    result = collection.query(
        query_embeddings=[q_emb],
        n_results=min(k, count),
        include=["documents", "metadatas", "embeddings"],
    )
    logger.debug("Retrieved %d chunks from ChromaDB (k=%d)", len(result["documents"][0]), k)
    return (
        result["documents"][0],
        result["metadatas"][0],
        np.array(result["embeddings"][0]),
    )


def cluster(embeddings, n_clusters=3):
    """Cluster embeddings with KMeans and return the label of each one."""
    n_clusters = min(n_clusters, len(embeddings))
    logger.debug("Clustering %d embeddings into %d clusters", len(embeddings), n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    return kmeans.fit_predict(embeddings)


def build_subsets(docs, labels, n_subsets=3, shared_count=1):
    """Share the top chunks, then diversify by cluster."""
    shared_count = min(shared_count, len(docs))
    shared_docs = docs[:shared_count]
    groups = {}
    for doc, label in zip(docs[shared_count:], labels[shared_count:]):
        groups.setdefault(label, []).append(doc)

    subsets = []
    for i in range(n_subsets):
        # cycle through a group if it has fewer docs than n_subsets
        subsets.append(shared_docs + [group[i % len(group)] for group in groups.values()])
    logger.debug("Built %d subsets (%d shared chunks)", len(subsets), shared_count)
    return subsets

   
# --------------------------------------------------------------------------- #
# Drafting (local Ollama model)
# --------------------------------------------------------------------------- #
DRAFT_PROMPT = """
Answer the question using ONLY the documents below.
Do not use outside knowledge, memory, or guesses.
Return JSON with exactly two keys: "answer" and "rationale".
The answer must be 3 to 5 sentences, using only facts from the documents, and clear enough for someone who has not seen the documents to understand.
Prefer real names (a published model, method or product
name) over internal labels such as "Model 3". If the documents give both, include
both in one sentence, for example: "The payment component (Module 2) is
built on Stripe." Never answer with only an internal label. If the question
is broad, cover every relevant item in the documents.
The rationale must include an exact short quote from the documents that supports
the answer, then briefly say which document it came from. First locate the
sentence containing the requested fact; do not guess from a nearby heading or
internal label. If no document supports an answer, say that clearly.
If the documents do not contain the answer, say so in the answer.

Documents:
{docs}

Question: {question}
"""


def draft(question, subset):
    """Generate one draft answer + rationale from a subset of documents."""
    docs_text = "\n\n".join(f"[{i}] {d}" for i, d in enumerate(subset, start=1))
    prompt = DRAFT_PROMPT.format(docs=docs_text, question=question)

    logger.debug("Calling Ollama model %s with %d chunks", DRAFT_MODEL, len(subset))
    raw = None
    # try twice: the second time with a higher temperature to break a repeat loop
    for temperature in (0.2, 0.6):
        try:
            res = ollama_client.chat(
                model=DRAFT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": temperature, "num_predict": 768, "repeat_penalty": 1.15},
            )
            raw = res["message"]["content"]
            break
        except Exception as e:
            logger.warning("Ollama call failed (temperature=%s): %s", temperature, e)

    if raw is None:
        return None            # the caller will skip this draft

    try:
        data = json.loads(raw)
        return {
            "answer": str(data.get("answer", "")),
            "rationale": str(data.get("rationale", "")),
        }
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Ollama returned a response that is not valid JSON, using it as the raw answer")
        return {"answer": raw, "rationale": ""}


# --------------------------------------------------------------------------- #
# Verification (Gemini)
# --------------------------------------------------------------------------- #
VERIFY_PROMPT = """You are grading candidate answers to a question.
Use ONLY the source passages as evidence. Do not use outside knowledge.
The candidate drafts may be incomplete or wrong. Do not assume a draft is correct.

Rules:
- First identify the fact requested by the question from all source passages.
- Score each candidate from 0 to 10, in candidate order.
- A candidate scores high only if its answer is directly supported by the
  source passages and actually answers the question.
- If the question asks for a name, the answer must contain the real name
  given in the passages, not just an internal label.
- For a question asking for a model or product name, search the passages for the
  actual name associated with that component.
- A candidate that says the information is missing scores low if a passage
  contains it.


Question:
{question}

Source passages (each passage appears once):
{source_documents}

Candidate drafts and the source passage indexes each draft saw:
{candidates}

Scoring rules:
- Score each candidate using ONLY the source documents provided. Do not use your own knowledge.
- If the answer contains a claim that is not supported by the source documents, lower the score, even if the claim is true in general.
- If the answer is about a different subject than the question (for example, the question asks about one topic but the passage and answer describe a related but different topic), give it a low score.
- If the source documents do not answer the question, give all candidates a low score (3 or less).
- The answer must address every part of the question. If the question asks for a difference or a comparison between A and B, an answer that describes only one of them scores 5 or less.
- Check the rationale: if the main claim in the answer does not appear in the quoted text or in the source passages, score it 3 or less, even if it sounds reasonable.
- A long list of loosely related points is not better than a short, correct answer.
- Do not give 10 unless the answer is fully supported by the documents and answers the whole question.

Return JSON only in this shape:
{{"scores": [8, 6, 4]}}
"""


_gemini_lock = threading.Lock()
_quota_exhausted = threading.Event()
_last_call = 0.0
MIN_INTERVAL = 13  # free tier allows 5 requests per minute


def wait_for_slot():
    """Make sure Gemini calls are spaced out enough to stay under the rate limit."""
    global _last_call
    with _gemini_lock:
        if _quota_exhausted.is_set():
            raise RuntimeError("Gemini quota is exhausted; verification cannot continue.")
        gap = time.time() - _last_call
        if gap < MIN_INTERVAL:
            logger.debug("Waiting %.1fs to respect the Gemini rate limit", MIN_INTERVAL - gap)
            if _quota_exhausted.wait(MIN_INTERVAL - gap):
                raise RuntimeError("Gemini quota is exhausted; verification cannot continue.")
        _last_call = time.time()

def verify(question, drafts, subsets, retries=5):
    """Use all unique source passages to correct and select among drafts."""
    source_documents = []
    source_indexes = {}
    candidates = []
    for i, (draft_data, docs) in enumerate(zip(drafts, subsets)):
        indexes = []
        for doc in docs:
            if doc not in source_indexes:
                source_indexes[doc] = len(source_documents)
                source_documents.append(doc)
            indexes.append(source_indexes[doc])
        candidates.append({
            "index": i,
            "answer": draft_data["answer"],
            "rationale": draft_data["rationale"],
            "source_passage_indexes": indexes,
        })
    prompt = VERIFY_PROMPT.format(
        question=question,
        source_documents=json.dumps(source_documents, ensure_ascii=False),
        candidates=json.dumps(candidates, ensure_ascii=False),
    )

    last_error = None
    for attempt in range(retries):
        wait_for_slot()
        try:
            res = gemini_client.models.generate_content(
                model=VERIFY_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0,
                ),
            )
            data = json.loads(res.text)
            scores = [float(score) for score in data["scores"]]
            if len(scores) != len(drafts) or any(s < 0 or s > 10 for s in scores):
                raise ValueError("Gemini returned invalid candidate scores")

            # pick the draft with the highest score (on a tie, the longer answer)
            def answer_key(text):
                return re.sub(r"\s+", " ", text.lower()).strip()

            top = max(scores)
            tied = [i for i, s in enumerate(scores) if s == top]
            votes = Counter(answer_key(drafts[i]["answer"]) for i in tied)
            best = max(tied, key=lambda i: (votes[answer_key(drafts[i]["answer"])], len(drafts[i]["answer"])))

            logger.info("Verification scores: %s, best draft: %d", scores, best + 1)
            return scores, best, "", drafts[best]["answer"]
        except Exception as e:
            msg = str(e)
            last_error = e
            if "RESOURCE_EXHAUSTED" in msg or "current quota" in msg.lower():
                _quota_exhausted.set()
                logger.error("Gemini quota is exhausted: %s", msg[:200])
                raise RuntimeError(
                    "Gemini quota is exhausted. Evaluation stopped to avoid recording "
                    "unverified answers as zero-score results. Check the Gemini API quota "
                    "or configure a model with available quota, then rerun."
                ) from e
            # a wrong model name will never fix itself, so stop immediately
            if "404" in msg or "NOT_FOUND" in msg:
                logger.error("Gemini model %r was not found: %s", VERIFY_MODEL, msg[:200])
                raise RuntimeError(f"Gemini model {VERIFY_MODEL!r} was not found: {msg}")
            match = re.search(r"retry in ([\d.]+)s", msg)
            wait = float(match.group(1)) + 1 if match else 2 ** attempt
            if attempt + 1 < retries:
                print(f"Verification failed ({msg[:200]}). Retrying in {wait:.0f}s...")
                logger.warning(
                    "Verification failed (attempt %d/%d): %s. Retrying in %.0fs",
                    attempt + 1, retries, msg[:200], wait,
                )
                time.sleep(wait)

    logger.error("Gemini verification failed after %d attempts: %s", retries, last_error)
    raise RuntimeError(
        f"Gemini verification failed after {retries} attempts: {last_error}"
    ) from last_error
# --------------------------------------------------------------------------- #
# Speculative RAG pipeline
# --------------------------------------------------------------------------- #

# if even the best draft scores below this (0-10), the documents don't answer the question
MIN_ACCEPT_SCORE = 4
NO_ANSWER_MESSAGE = "I couldn't find the answer to this question in your documents."

def answer_question(question, k=10, n_subsets=3):
    """Run the full pipeline. Returns (final_answer, list_of_source_file_names)."""
    print("  Retrieving relevant document chunks...", flush=True)
    logger.info("Retrieving relevant document chunks")
    docs, metas, embs = retrieve(question, k=k)
    if not docs:
        logger.warning("No documents were retrieved for question: %s", question)
        return (
            "No documents are indexed yet. Upload a PDF from the sidebar and "
            "I'll be able to answer questions about it.",
            [],
        )

    labels = cluster(embs, n_clusters=n_subsets)
    subsets = build_subsets(docs, labels, n_subsets=n_subsets)

    # debug: show the start of each chunk in each subset
    for i, s in enumerate(subsets, start=1):
        print(f"  Subset {i}: " + " | ".join(c[:40] for c in s))

    print(f"  Generating {n_subsets} answer drafts with Ollama...", flush=True)
    logger.info("Generating %d answer drafts with Ollama", n_subsets)
    with ThreadPoolExecutor(max_workers=n_subsets) as pool:
        results = list(pool.map(lambda s: draft(question, s), subsets))

    # keep only the drafts that worked, together with their own subsets
    ok = [(d, s) for d, s in zip(results, subsets) if d is not None]
    if not ok:
        logger.error("All drafts failed for question: %s", question)
        return "The local model failed to generate answers, please try again.", []
    drafts = [d for d, _ in ok]
    subsets = [s for _, s in ok]

    print("  Comparing drafts with Gemini...", flush=True)
    logger.info("Comparing drafts with Gemini")
    scores, best, evidence, final_answer = verify(question, drafts, subsets)

    for i, (d, score) in enumerate(zip(drafts, scores), start=1):
        print(f"\n--- Draft {i} (score: {score:.1f}) ---")
        print(f"Answer: {d['answer']}")
        print(f"Rationale: {d['rationale']}")
        logger.debug("Draft %d (score: %.1f) answer: %s", i, score, d["answer"])
        logger.debug("Draft %d rationale: %s", i, d["rationale"])
    if evidence:
        print(f"Verifier evidence: {evidence}")

    if max(scores) < MIN_ACCEPT_SCORE:
        logger.info("Best draft scored %.1f (below %d), answering that the documents have no answer",
                    max(scores), MIN_ACCEPT_SCORE)
        return NO_ANSWER_MESSAGE, []
    
    sources = list(dict.fromkeys(m["source"] for m in metas))
    logger.info("Answer ready from sources: %s", sources)
    return final_answer, sources


def speculative_rag(question, k=10, n_subsets=3):
    """Backward-compatible wrapper that returns just the answer text."""
    return answer_question(question, k=k, n_subsets=n_subsets)[0]


if __name__ == "__main__":
    build_index()
    print("Speculative RAG ready. Type 'exit' or 'quit' to stop.")
    logger.info("Speculative RAG ready")

    while True:
        try:
            question = input("\nAsk a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            logger.info("Session ended by the user")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            logger.info("Session ended by the user")
            break

        logger.info("New question: %s", question)
        try:
            answer = speculative_rag(question)
            print(f"\nFinal Answer: {answer}")
            logger.info("Final answer: %s", answer)
        except Exception as e:
            # keep the loop alive if one question fails (network, model, etc.)
            print(f"\nSomething went wrong: {e}")
            logger.error("Something went wrong: %s", e, exc_info=True)