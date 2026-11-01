import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
# Folder that holds vectors.npy and embeddings.json (set RAG_INDEX_DIR to point elsewhere).
INDEX_DIR = Path(os.environ.get("RAG_INDEX_DIR", BASE_DIR.parent / "Conversational_system"))

CHAT_MODEL = "qwen2.5:1.5b"           # local model (Ollama)
USE_OPENROUTER = True                 # False = everything stays local
OPENROUTER_MODEL = "openrouter/free"  # picks a free model automatically
MAX_TURNS = 3
MAX_SUBQUESTIONS = 3
MAX_RETRIES = 1
TOP_K = 8             # chunks fetched per search
MAX_CHUNKS = 6        # chunks sent to the model (shared between the sub-questions)
MIN_SCORE = 0.30      # below this, a chunk is never used
REL_MARGIN = 0.20     # keep chunks within this distance of the best score
DUP_THRESHOLD = 0.85
SHOW_CONTEXT = True   # print the text sent to the model (turn off when done testing)
NOT_FOUND = "I couldn't find this in the document."
NOT_UNDERSTOOD = "I didn't understand that. Could you write your question in a full sentence?"

GREETING_WORDS = {"hello", "hi", "hey", "thanks", "thank", "bye", "goodbye"}
OK_SHORT = {"why", "how", "who", "when", "where", "what", "more", "explain", "details"}

REWRITE_PROMPT = """You resolve follow-up questions for a document question-answering assistant.

Use the recent conversation to understand the user's current question.
- If the question refers to something mentioned earlier (for example "it", "that one",
  "the second one", "its name", a numbered item), rewrite it as one complete,
  standalone question that names the topic explicitly.
- If the question is already standalone or clearly changes to a new topic, return it
  unchanged. Do not force the previous topic onto an unrelated question.
- Do not answer the question and do not invent details.
Return only the rewritten question."""

DECOMPOSE_PROMPT = f"""Split the question only when it clearly asks for multiple distinct pieces of information.
For every sub-question:
- Preserve all important names, places, numbers, identifiers, and ordinal words (first/second/third).
- Replace pronouns such as his/her/it/their with the correct named entity from the original question.
- Never remove a key term from the question.
- Do not add facts or change the topic.
If the question asks for only one thing, return the original question unchanged on one line.
Return at most {MAX_SUBQUESTIONS} lines, no numbering, no explanation.

Example:
Tell me about Alice Smith and her favourite food.
Tell me about Alice Smith.
What is Alice Smith's favourite food?

Example:
Which documents mention London?
Which documents mention London?"""

CHECK_PROMPT = """You check whether the supplied context contains enough direct evidence to answer the question.
Reply YES only if the context clearly supports the requested answer.
Reply NO if it is merely related to the topic, discusses something nearby, or does not identify the requested person/name/fact.
Reply with one word only: YES or NO."""

SEARCH_REWRITE_PROMPT = """The document search did not find enough evidence to answer the question.
Write a different short search query (3 to 8 keywords) for the SAME question and SAME topic.
Preserve every important name, place, number, and identifier from the question.
Never replace the subject with an entity from another topic.
Return only the query."""

ANSWER_PROMPT = """You answer questions using only the supplied document passages.

Rules:
- Use only facts directly supported by the context. Do not use outside knowledge or conversation history as evidence.
- If the context does not give the requested name or fact, say that you could not find it in the document.
- Never invent reasons, benefits, or explanations that are not written in the context. If you are not sure about a detail, say so.
- When the question asks for a list (models, steps, components, features), list every item that appears in the context with what it does. Do not say how many there are unless the context says so.
- Answer every part of a multi-part question, and clearly say when one part is not found.
- Write at most 4 short sentences in simple clear English.
- Ignore table-of-contents entries, page numbers, and dotted lines.
- Passages begin with the document name and page number.
- If documents disagree, explain which document says what.
- Answer in your own words; do not quote long passages."""

CHAT_PROMPT = (
    "You are a friendly assistant for a document question-answering app. "
    "Reply briefly and do not invent facts."
)


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


@lru_cache(maxsize=1)
def get_cloud_client():
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return None
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def ask(system, user, history=None, cloud=False):
    """One LLM call. cloud=True sends it to OpenRouter, with a local fallback."""
    _, _, _, llm = load_index()
    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})

    if cloud and USE_OPENROUTER:
        client = get_cloud_client()
        if client:
            try:
                response = client.chat.completions.create(
                    model=OPENROUTER_MODEL, temperature=0, messages=messages
                )
                text = (response.choices[0].message.content or "").strip()
                if text:
                    print("[answer from OpenRouter]")
                    return text
            except Exception as exc:
                print(f"Warning: OpenRouter failed ({exc}); using the local model.")

    response = llm.chat.completions.create(
        model=CHAT_MODEL, temperature=0, messages=messages
    )
    return (response.choices[0].message.content or "").strip()


def route(question):
    """Route short greetings to chat; send everything else to retrieval."""
    words = re.findall(r"[a-z']+", question.lower())
    if words and len(words) <= 4 and words[0] in GREETING_WORDS:
        return "CHAT"
    return "SEARCH"


def _critical_terms(question):
    """Terms that must not disappear when an LLM rewrites a question."""
    terms = []
    patterns = [
        r"\b(?:model|option|step)\s*(?:#\s*)?(?:\d+|one|two|three|first|second|third)\b",
        r"\b(?:first|second|third|fourth|fifth)\s+(?:model|option|step|one)\b",
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:I|II|III|IV|V|VI|VII|VIII|IX|X|\d+))?\b",
    ]
    skip = {
        "what", "which", "who", "tell", "me", "how", "where", "when", "why",
        "the", "and", "about", "is", "are", "does",
    }
    for pattern in patterns:
        flags = re.IGNORECASE if ("model" in pattern or "first" in pattern) else 0
        for match in re.finditer(pattern, question, flags=flags):
            value = match.group(0).strip()
            if value.lower() not in skip:
                terms.append(value)

    # Capitalized words after the first word (names of people, places, projects).
    for word in question.split()[1:]:
        cleaned = re.sub(r"[^A-Za-z0-9'-]", "", word)
        if cleaned and cleaned[0].isupper() and cleaned.lower() != "i":
            terms.append(cleaned)

    # Words that change the meaning if the rewriter drops them.
    meaning_terms = [
        r"\bonly\b", r"\ball\b", r"\bboth\b", r"\bnot\b",
        r"\bnever\b", r"\bexcept\b", r"\bbefore\b", r"\bafter\b",
        r"\bproblem\b", r"\bsolve(?:s|d)?\b", r"\bwhy\b",
    ]
    for pattern in meaning_terms:
        for match in re.finditer(pattern, question, flags=re.IGNORECASE):
            terms.append(match.group(0))

    unique, seen = [], set()
    for term in terms:
        if term.lower() not in seen:
            seen.add(term.lower())
            unique.append(term)
    return unique


def _last_user_question(history):
    """The most recent thing the user asked (used as a generic context hint)."""
    for message in reversed(history):
        if message.get("role") == "user":
            return message.get("content", "").strip()
    return ""

REFERENCE_WORDS = {
    "it", "its", "he", "she", "his", "her", "they", "them", "their",
    "this", "that", "these", "those", "one", "ones", "former", "latter",
    "same", "both", "another", "previous", "first", "second", "third",
}

def needs_history(question):
    """True when the question depends on the earlier conversation."""
    words = re.findall(r"[a-z']+", question.lower())
    return (
        len(words) <= 12
        and any(w in REFERENCE_WORDS for w in words)
    ) or question.lower().startswith(("and ", "what about", "how about"))

def make_standalone(question, history):
    """Resolve references, then guard against dropped names/numbers."""
    if not history or not needs_history(question):
        return question

    recent = history[-MAX_TURNS * 2:]
    chat_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
    try:
        rewritten = ask(
            REWRITE_PROMPT,
            f"Recent conversation:\n{chat_text}\n\nCurrent user question: {question}",
        )
        rewritten = (rewritten or "").strip().strip('"').strip("'")
    except Exception as exc:
        print(f"Warning: follow-up rewriting failed; using original question ({exc}).")
        rewritten = ""

    if not rewritten or len(rewritten) > max(500, len(question) * 8):
        rewritten = question

    lowered = rewritten.lower()
    missing_terms = [t for t in _critical_terms(question) if t.lower() not in lowered]
    if missing_terms:
        # the rewriter dropped something: trust the user's own words
        rewritten = question
    return rewritten or question


def _extract_named_subject(question):
    """Best-effort extraction of a named subject, including regnal numerals."""
    stopwords = {"tell", "what", "which", "who", "where", "when", "how", "why"}
    pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:I|II|III|IV|V|VI|VII|VIII|IX|X|\d+))?)\b"
    for match in re.finditer(pattern, question):
        candidate = match.group(1)
        if candidate.lower() not in stopwords:
            return candidate
    return None


def repair_subquestions(parent_question, subquestions):
    """Replace possessive pronouns in decomposed questions with the named subject."""
    subject = _extract_named_subject(parent_question)
    repaired = []
    for sub in subquestions:
        candidate = sub.strip()
        if subject:
            candidate = re.sub(r"\bhis\b", f"{subject}'s", candidate, flags=re.IGNORECASE)
            candidate = re.sub(r"\bher\b", f"{subject}'s", candidate, flags=re.IGNORECASE)
            candidate = re.sub(r"\bits\b", f"{subject}'s", candidate, flags=re.IGNORECASE)
        repaired.append(candidate)
    return repaired


def dedupe(subquestions):
    """Remove near-duplicate sub-questions using embedding similarity."""
    if len(subquestions) < 2:
        return subquestions

    _, _, embed_model, _ = load_index()
    vectors = embed_model.encode(subquestions, normalize_embeddings=True)
    keep = []
    for i in range(len(subquestions)):
        if all(float(vectors[i] @ vectors[j]) < DUP_THRESHOLD for j in keep):
            keep.append(i)
    return [subquestions[i] for i in keep]


def split_by_and(question):
    """Split 'A, and who B?' into two questions using a simple rule."""
    parts = re.split(
        r",?\s+and\s+(?=(?:what|why|how|where|when|who|which)\b)",
        question,
        flags=re.IGNORECASE,
    )
    parts = [p.strip(" ,?") for p in parts if len(p.strip(" ,?")) > 8]
    return [p + "?" for p in parts]


def decompose(question):
    """Only split questions that visibly contain multiple asks."""
    has_multiple_asks = bool(re.search(r"\band\b|[,;].*\b(and|also)\b", question, re.IGNORECASE))
    if not has_multiple_asks:
        return [question]

    # simple rule first: "A, and who/what/how B" splits without asking the model
    rule_parts = split_by_and(question)
    if len(rule_parts) >= 2:
        return dedupe(repair_subquestions(question, rule_parts[:MAX_SUBQUESTIONS]))

    # otherwise fall back to the model
    try:
        output = ask(DECOMPOSE_PROMPT, question)
    except Exception:
        return [question]

    lines = [
        re.sub(r"^\s*(\d+[\.\)]|[-*•])\s*", "", line).strip()
        for line in output.splitlines()
    ]
    lines = [line for line in lines if len(line) > 8 and "->" not in line]

    seen, subquestions = set(), []
    for line in lines:
        key = line.lower()
        if key not in seen:
            seen.add(key)
            subquestions.append(line)

    subquestions = subquestions[:MAX_SUBQUESTIONS]
    if not subquestions:
        return [question]

    combined = " ".join(subquestions).lower()
    if any(term.lower() not in combined for term in _critical_terms(question)):
        return [question]

    # Two explicit question words in the original but only one sub-question back:
    # keep the full original rather than silently losing half of it.
    question_words = re.findall(
        r"\b(?:what|why|how|where|when|who|which)\b", question, flags=re.IGNORECASE
    )
    if len(question_words) >= 2 and len(subquestions) < 2:
        return [question]

    return dedupe(repair_subquestions(question, subquestions)) or [question]


def retrieve(query, source=None):
    """Return relevant (chunk_index, cosine_similarity) pairs."""
    vectors, data, embed_model, _ = load_index()
    query_vector = embed_model.encode([query], normalize_embeddings=True)[0]
    scores = vectors @ query_vector

    if source:
        for i, item in enumerate(data):
            if item["source"] != source:
                scores[i] = -1

    ranked = np.argsort(scores)[::-1][:TOP_K]
    if len(ranked) == 0:
        return []

    best_score = float(scores[ranked[0]])
    if best_score < MIN_SCORE:
        return []

    return [
        (int(i), float(scores[i]))
        for i in ranked
        if scores[i] >= best_score - REL_MARGIN and scores[i] >= MIN_SCORE
    ]


def format_context(indices):
    _, data, _, _ = load_index()
    return "\n\n".join(
        f"[{data[i]['source']}, page {data[i]['page']}] {data[i]['text']}"
        for i in indices
    )


def is_sufficient(question, indices):
    """Never accept a chunk based on similarity score alone."""
    if not indices:
        return False
    try:
        output = ask(
            CHECK_PROMPT,
            f"Context:\n{format_context(indices)}\n\nQuestion: {question}",
        ).upper()
        return output.startswith("YES")
    except Exception:
        return False  # fail closed


def search_with_retry(subquestion, source, trace):
    """Retrieve, verify evidence, and retry once with a different query."""
    query = subquestion
    tried = set()

    for attempt in range(MAX_RETRIES + 1):
        tried.add(query.lower())
        hits = retrieve(query, source)
        best_score = hits[0][1] if hits else 0.0
        trace.append(f"  search '{query}' -> {len(hits)} chunks, best score {best_score:.2f}")

        if hits:
            indices = [index for index, _ in hits]
            if is_sufficient(subquestion, indices):
                return indices
            trace.append("  context judged not enough")

        if attempt == MAX_RETRIES:
            break

        try:
            new_query = ask(
                SEARCH_REWRITE_PROMPT,
                f"Question to answer: {subquestion}\nFirst search query: {query}",
            )
            new_query = (new_query or "").strip().strip('"').strip("'")
        except Exception:
            new_query = ""

        required = _critical_terms(subquestion)
        if (not new_query
                or new_query.lower() in tried
                or any(t.lower() not in new_query.lower() for t in required)):
            trace.append("  no better query found, stopping")
            break

        query = new_query
        trace.append(f"  retry with new query: '{query}'")

    return []


def answer_question(question, history=None, source=None, verbose=False):
    """Return (answer, source_pages). History is a list of user/assistant messages."""
    question = question.strip()
    if not question:
        raise ValueError("Please enter a question.")

    history = history or []
    trace = []

    if route(question) == "CHAT":
        trace.append("route: CHAT (no search)")
        if verbose:
            print("\n".join(trace))
        return ask(CHAT_PROMPT, question, history[-MAX_TURNS * 2:]), []

    # A single unclear word is not a question.
    words = re.findall(r"[A-Za-z']+", question)
    if len(words) < 2 and (not words or words[0].lower() not in OK_SHORT):
        return NOT_UNDERSTOOD, []

    trace.append("route: SEARCH")

    

    standalone = make_standalone(question, history)
    trace.append(f"standalone: {standalone}")

    subquestions = decompose(standalone)
    trace.append(f"sub-questions: {subquestions}")

    # search every sub-question and keep its own list of chunks
    per_sub, missing = [], []
    for subquestion in subquestions:
        trace.append(f"* {subquestion}")
        indices = search_with_retry(subquestion, source, trace)
        if indices:
            per_sub.append(indices)
        else:
            missing.append(subquestion)

    if verbose:
        print("\n".join(trace))

    if not per_sub:
        return NOT_FOUND, []

    # take chunks in turns (best of each sub-question first), so every
    # sub-question gets a place in the context
    used_indices, rank = [], 0
    while len(used_indices) < MAX_CHUNKS and any(rank < len(x) for x in per_sub):
        for indices in per_sub:
            if (rank < len(indices)
                    and indices[rank] not in used_indices
                    and len(used_indices) < MAX_CHUNKS):
                used_indices.append(indices[rank])
        rank += 1

    context = format_context(used_indices)

    if verbose and SHOW_CONTEXT:
        print("\n--- CONTEXT SENT TO THE MODEL ---")
        print(context)
        print("--- END CONTEXT ---")

    # The document passages are the evidence; the standalone question already
    # contains the resolved references, so no chat history is sent here.
    answer = ask(
        ANSWER_PROMPT,
        f"Context:\n{context}\n\nQuestion: {standalone}",
        cloud=True,
    ) or NOT_FOUND

    if missing:
        answer += "\n\nI couldn't find information about: " + "; ".join(missing)

    _, data, _, _ = load_index()
    pages_by_file = {}
    for index in used_indices:
        pages_by_file.setdefault(data[index]["source"], set()).add(data[index]["page"])

    pages = [
        f"{filename}: pages {', '.join(str(page) for page in sorted(page_numbers))}"
        for filename, page_numbers in pages_by_file.items()
    ]
    return answer, pages


if __name__ == "__main__":
    print("Commands: 'new' = clear memory, 'sources' = list pdfs, 'exit' = quit")
    history = []

    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue

        command = question.lower()
        if command in ("exit", "quit", "q"):
            break
        if command == "new":
            history = []
            print("History cleared.")
            continue
        if command.startswith("new "):
            history = []
            question = question[4:].strip()
            print("History cleared.")
            if not question:
                continue
            command = question.lower()
        if command == "sources":
            for filename in list_sources():
                print(" -", filename)
            continue

        try:
            answer, pages = answer_question(question, history, verbose=True)
            print("\n" + answer)
            for page_line in pages:
                print(" -", page_line)
            history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])
        except Exception as exc:
            print(f"\nError: {exc}")
            print("Check that Ollama is running, the model is available, and the index files exist.")