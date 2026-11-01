"""AI Assistant: a Streamlit chat UI for the Speculative RAG pipeline."""

import importlib
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

RAG_DIR = APP_DIR.parent / "speculative_system"   
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[1]                   # src/ (holds the shared logging_config.py)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from logging_config import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger("app")

BACKEND_MODULE = "Speculative_RAG"
HISTORY_PATH = APP_DIR / "chat-history" / "chat_history.json"
HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
SETTINGS_PATH = APP_DIR / "settings.json"
DEFAULT_SETTINGS = {"top_k": 10, "n_subsets": 3, "show_sources": True}
RECENT_LIMIT = 5


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_backend():
    logger.info("Loading RAG backend")
    module = importlib.import_module(BACKEND_MODULE)
    module.build_index()  # fast no-op when nothing changed
    logger.info("RAG backend ready")
    return module


def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        if isinstance(exc, FileNotFoundError):
            logger.debug("%s not found, using default", path)
        else:
            logger.warning("Could not read %s (%s), using default", path, exc)
        return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def pretty_time(iso):
    try:
        return datetime.fromisoformat(iso).strftime("%b %d, %H:%M")
    except (TypeError, ValueError):
        return ""


def new_chat_id():
    return uuid.uuid4().hex[:12]


def save_chats():
    save_json(HISTORY_PATH, st.session_state.chats)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
ss = st.session_state
if "chats" not in ss:
    ss.chats = load_json(HISTORY_PATH, {})
if "chat_id" not in ss:
    ss.chat_id = new_chat_id()
if "settings" not in ss:
    ss.settings = {**DEFAULT_SETTINGS, **load_json(SETTINGS_PATH, {})}
if "show_history" not in ss:
    ss.show_history = False
if "uploader_key" not in ss:
    ss.uploader_key = 0


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
:root { --ink:#1d315b; --muted:#687fae; --blue:#4f70ff; --line:#e3eaff; }
html, body, [class*="css"], [data-testid="stApp"] { font-family:"Inter","Segoe UI",Arial,sans-serif !important; }
.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"],section.main {
  background:#fff !important; color:var(--ink);
}
header[data-testid="stHeader"] { background:transparent; }
[data-testid="stToolbar"] { background:transparent; }
[data-testid="stSidebar"] {
  background:linear-gradient(180deg,#f5f8ff 0%,#f8faff 100%);
  border-right:1px solid #e4eaf7; min-width:350px !important; max-width:350px !important;
}
[data-testid="stSidebarContent"] { padding:1.15rem 1.05rem 6rem; }
[data-testid="stSidebar"] * { color:#20375f; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { font-size:.9rem; font-weight:500; }
[data-testid="stMainBlockContainer"] { max-width:1200px; padding:1rem 2rem 7rem; }
.brand { display:flex; gap:15px; align-items:center; margin:0 0 1.5rem .3rem; }
.brand-icon { color:#4d68ff; line-height:1; display:flex; }
.brand-icon svg { width:54px; height:54px; }
.brand-title { font-size:1.22rem; font-weight:700; line-height:1.4; letter-spacing:-.02em; }
.brand-sub { color:#7388b4; font-size:.84rem; font-weight:450; }
.welcome { text-align:center; margin:6.2rem auto 2.4rem; max-width:680px; }
.bot { color:#9aafff; display:flex; justify-content:center; line-height:1; margin:0 auto .6rem; }
.bot svg { width:76px; height:76px; }
.welcome h1 { color:#1d315b; font-size:2.65rem; letter-spacing:-.035em; margin:.6rem 0 .75rem; }
.welcome p { color:#7185b2; font-size:1.08rem; line-height:1.7; }
.welcome .hint { margin-top:1.2rem; font-size:.95rem; color:#4d68ff; }

/* ---- chat bubbles: dark navy ---- */
[data-testid="stChatMessage"] { width:fit-content!important; max-width:min(78%, 720px)!important; box-sizing:border-box!important; overflow-wrap:anywhere; margin:.9rem 0; padding:.85rem 1.15rem; border:0; border-radius:18px 18px 18px 4px; background:#1c3461; box-shadow:0 2px 10px #1c346122; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { width:fit-content!important; max-width:min(78%, 720px)!important; margin-left:auto!important; margin-right:0!important; border-radius:18px 18px 4px 18px; padding:.65rem 1.05rem; background:#3b55d9; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) { margin-right:auto; margin-left:0; }
[data-testid="stMain"] [class*="st-key-chat_row_"] { width:min(calc(100vw - 2rem), 880px)!important; max-width:880px!important; box-sizing:border-box!important; margin-left:auto!important; margin-right:auto!important; padding-left:0!important; padding-right:0!important; }
[data-testid="stMain"] [class*="st-key-chat_row_user_"] [data-testid="stChatMessage"] { margin-left:auto!important; margin-right:0!important; }
[data-testid="stMain"] [class*="st-key-chat_row_assistant_"] [data-testid="stChatMessage"] { margin-left:0!important; margin-right:auto!important; }
[data-testid="stChatMessage"] p { margin:.15rem 0; line-height:1.55; }
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] h1,[data-testid="stChatMessage"] h2,[data-testid="stChatMessage"] h3 { color:#f4f7ff!important; }
[data-testid="stChatMessage"] [data-testid="stCaptionContainer"],
[data-testid="stChatMessage"] [data-testid="stCaptionContainer"] * { color:#aebfe8!important; }
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"] { display:none; }

[data-testid="stBottom"],[data-testid="stBottom"]>div,[data-testid="stBottomBlockContainer"],
[data-testid="stBottom"]::before,[data-testid="stBottomBlockContainer"]::before { background:transparent!important; }
[data-testid="stBottomBlockContainer"] { width:min(calc(100vw - 2rem), 1100px)!important; max-width:1100px!important; box-sizing:border-box!important; margin:0 auto!important; padding-bottom:1rem; }
[data-testid="stChatInput"] { background:#fff!important; border:1px solid #c7d3f5!important; border-radius:36px!important; padding:4px 10px!important; box-shadow:0 4px 18px #6480cd14!important; }
[data-testid="stChatInput"]>div { background:transparent!important; border:0!important; box-shadow:none!important; }
[data-testid="stChatInput"] textarea { color:#263d67!important; min-height:32px!important; font-size:1rem!important; font-weight:550!important; line-height:1.4!important; }
[data-testid="stChatInput"] textarea::placeholder { color:#657da9!important; opacity:1; }
[data-testid="stChatInput"] button { background:#3b55d9!important; color:white!important; border:0!important; border-radius:50%!important; min-width:38px!important; min-height:38px!important; width:38px!important; height:38px!important; }

/* ---- sidebar buttons ---- */
[data-testid="stSidebar"] [data-testid="stButton"] button { text-align:left!important; justify-content:flex-start!important; background:transparent; border:0; border-radius:10px; color:#304a78; padding-left:.75rem; min-height:40px; }
[data-testid="stSidebar"] [data-testid="stButton"] button > div { width:100%; display:flex; justify-content:flex-start!important; text-align:left!important; }
[data-testid="stSidebar"] [data-testid="stButton"] button [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stButton"] button p { width:100%; text-align:left!important; }
[data-testid="stSidebar"] [data-testid="stButton"] button:hover { background:#e9efff; }
[data-testid="stSidebar"] [data-testid="stButton"] button p { font-size:.91rem; font-weight:550; }
[data-testid="stSidebar"] .st-key-new_chat button { background:#e5ecff; color:#1458f5; }
[data-testid="stSidebar"] .st-key-new_chat button p { color:#1458f5; font-weight:650; }
[data-testid="stSidebar"] .st-key-history_btn.active button { background:#e5ecff; }
/* ---- sidebar: uploader + expander (force light colors) ---- */
[data-testid="stSidebar"] [data-testid="stFileUploader"] section,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
  background:#fff!important; border:1px dashed #b9c8f2!important; border-radius:12px!important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button,
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
  background:#e5ecff!important; color:#1458f5!important; border:1px solid #c7d3f5!important; border-radius:10px!important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover { background:#d6e1ff!important; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] button *,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span { color:#304a78!important; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] button * { color:#1458f5!important; }

[data-testid="stSidebar"] [data-testid="stExpander"],
[data-testid="stSidebar"] [data-testid="stExpander"] details {
  background:#eef2ff!important; border:1px solid #dbe4ff!important; border-radius:12px!important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary { background:transparent!important; color:#20375f!important; }
[data-testid="stSidebar"] [data-testid="stExpander"] summary *,
[data-testid="stSidebar"] [data-testid="stExpander"] summary svg { color:#20375f!important; fill:#20375f!important; }
.side-heading { color:#61769f; font-size:.9rem; margin:1rem 0 .4rem; font-weight:600; }
.doc-chip { display:flex; justify-content:space-between; gap:.5rem; font-size:.8rem; padding:.2rem .1rem; }
.doc-chip span:first-child { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.doc-chip span:last-child { color:#8295bd; white-space:nowrap; }
.model-badge { position:fixed; top:1.1rem; right:180px; z-index:1000000; display:flex; align-items:center; gap:.7rem; color:#344e7e; font-size:.9rem; font-weight:600; white-space:nowrap; }
@media(max-width:700px) { [data-testid="stSidebar"]{min-width:260px!important;max-width:260px!important} [data-testid="stMainBlockContainer"]{padding:1rem 1rem 6rem} .welcome{margin:3rem auto 1.5rem}.welcome h1{font-size:2rem} }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Settings dialog
# --------------------------------------------------------------------------- #
@st.dialog("Settings")
def settings_dialog():
    s = ss.settings
    top_k = st.slider(
        "Chunks retrieved per question", 3, 20, int(s["top_k"]),
        help="More chunks = more context for the answer, but slower.",
    )
    n_subsets = st.slider(
        "Number of drafts compared", 2, 5, int(s["n_subsets"]),
        help="How many draft answers the local model writes before Gemini picks the best one.",
    )
    show_sources = st.toggle("Show sources under answers", value=bool(s["show_sources"]))

    if st.button("Save settings", type="primary", use_container_width=True):
        ss.settings = {"top_k": top_k, "n_subsets": n_subsets, "show_sources": show_sources}
        save_json(SETTINGS_PATH, ss.settings)
        logger.info("Settings saved: %s", ss.settings)
        st.rerun()

    st.divider()
    st.markdown("**Danger zone**")
    confirm = st.checkbox("I want to delete all saved chats")
    if st.button("Clear all chat history", disabled=not confirm, use_container_width=True):
        ss.chats = {}
        ss.chat_id = new_chat_id()
        save_chats()
        logger.warning("All chat history cleared")
        st.rerun()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def ingest_uploads(files):
    """Save uploaded PDFs into the docs folder and index them in ChromaDB."""
    try:
        backend = load_backend()
    except Exception as exc:
        logger.exception("Could not start the RAG backend")
        st.error(f"Could not start the RAG backend: {exc}")
        return
    for uploaded in files:
        name = Path(uploaded.name).name
        target = backend.DOCS_DIR / name
        target.write_bytes(uploaded.getbuffer())
        logger.info("Indexing uploaded PDF %s", name)
        bar = st.progress(0.0, text=f"Indexing {name}…")
        try:
            status, chunks = backend.ingest_pdf(
                target, name,
                progress_callback=lambda done, total, n=name: bar.progress(done / total, text=f"Indexing {n}…"),
            )
        except Exception as exc:
            logger.exception("Failed to index %s", name)
            bar.empty()
            st.error(f"Failed to index {name}: {exc}")
            continue
        bar.empty()
        logger.info("%s indexed: status=%s, chunks=%s", name, status, chunks)
        if status == "added":
            st.success(f"{name}: {chunks} chunks indexed.")
        elif status == "unchanged":
            st.info(f"{name} was already indexed.")
        else:
            st.warning(f"No readable text found in {name} (scanned PDF?).")
    ss.uploader_key += 1  # resets the uploader widget


def render_chat_list(limit=None):
    """Clickable list of saved chats, newest first."""
    ordered = sorted(
        ((cid, c) for cid, c in ss.chats.items() if c.get("messages")),
        key=lambda item: item[1].get("updated", ""),
        reverse=True,
    )
    if not ordered:
        st.caption("No saved chats yet.")
        return
    for cid, chat in ordered[:limit]:
        title = chat.get("title", "New conversation")
        label = f"{title[:26]}{'…' if len(title) > 26 else ''}"
        cols = st.columns([5, 1], vertical_alignment="center")
        with cols[0]:
            if st.button(label, key=f"open_{cid}", use_container_width=True,
                         help=f"{title}\n{pretty_time(chat.get('updated'))}",
                         disabled=(cid == ss.chat_id)):
                ss.chat_id = cid
                st.rerun()
        with cols[1]:
            if st.button("", icon=":material/delete:", key=f"del_{cid}", help="Delete this chat"):
                logger.info("Deleting chat %s", cid)
                ss.chats.pop(cid, None)
                if cid == ss.chat_id:
                    ss.chat_id = new_chat_id()
                save_chats()
                st.rerun()


with st.sidebar:
    st.markdown('''<div class="brand"><div class="brand-icon"><svg viewBox="0 0 80 80" fill="none" xmlns="http://www.w3.org/2000/svg"><g stroke="#4d68ff" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"><path d="M40 8v8M22 16l4 5M58 16l-4 5"/><rect x="18" y="26" width="44" height="36" rx="15"/><path d="M13 39v10M67 39v10"/></g><circle cx="31" cy="43" r="3" fill="#4d68ff"/><circle cx="49" cy="43" r="3" fill="#4d68ff"/><path d="M32 53c4 3 12 3 16 0" stroke="#4d68ff" stroke-width="3" stroke-linecap="round"/></svg></div><div><div class="brand-title">AI Assistant</div><div class="brand-sub">Ask anything, get things done.</div></div></div>''', unsafe_allow_html=True)

    if st.button("New Chat", icon=":material/chat_bubble_outline:", key="new_chat", use_container_width=True):
        ss.chat_id = new_chat_id()
        ss.show_history = False
        st.rerun()

    if st.button("History", icon=":material/schedule:", key="history_btn", use_container_width=True):
        ss.show_history = not ss.show_history
        st.rerun()

    if st.button("Settings", icon=":material/settings:", key="settings_nav", use_container_width=True):
        settings_dialog()

    # ---- documents ----
    st.markdown('<div class="side-heading">Your Documents</div>', unsafe_allow_html=True)
    uploads = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True,
        key=f"uploader_{ss.uploader_key}", label_visibility="collapsed",
    )
    if uploads and st.button("Add to knowledge base", icon=":material/upload_file:",
                             key="ingest_btn", use_container_width=True):
        ingest_uploads(uploads)

    try:
        indexed_docs = load_backend().list_documents()
    except Exception:
        indexed_docs = {}
    if indexed_docs:
        with st.expander(f"Indexed documents ({len(indexed_docs)})", expanded=False):
            for i, (doc_name, n_chunks) in enumerate(indexed_docs.items()):
                cols = st.columns([5, 1], vertical_alignment="center")
                cols[0].markdown(
                    f'<div class="doc-chip"><span title="{doc_name}">{doc_name}</span>'
                    f'<span>{n_chunks} chunks</span></div>', unsafe_allow_html=True)
                if cols[1].button("", icon=":material/delete:", key=f"rmdoc_{i}", help=f"Remove {doc_name}"):
                    logger.info("Removing document %s", doc_name)
                    load_backend().delete_document(doc_name)
                    st.rerun()
    else:
        st.caption("No documents yet. Upload a PDF to get started.")

    # ---- chats ----
    if ss.show_history:
        st.markdown('<div class="side-heading">Chat History</div>', unsafe_allow_html=True)
        render_chat_list()
    else:
        st.markdown('<div class="side-heading">Recent Chats</div>', unsafe_allow_html=True)
        render_chat_list(limit=RECENT_LIMIT)

st.markdown('<div class="model-badge"><span>🤖</span><span>Speculative RAG</span></div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Main chat area
# --------------------------------------------------------------------------- #
def render_chat_message(role, content, sources=None, row_id=0):
    """Render user messages on the right and assistant messages on the left."""
    with st.container(key=f"chat_row_{role}_{row_id}"):
        avatar = "🔵" if role == "assistant" else None
        with st.chat_message(role, avatar=avatar):
            st.markdown(content)
            if role == "assistant" and sources and ss.settings["show_sources"]:
                st.caption("Sources · " + " · ".join(map(str, sources)))


messages = ss.chats.get(ss.chat_id, {}).get("messages", [])

welcome_area = st.empty()
if not messages:
    with welcome_area.container():
        st.markdown('''<div class="welcome"><div class="bot"><svg viewBox="0 0 80 80" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="Assistant robot"><g stroke="#9aafff" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"><path d="M40 8v8M22 16l4 5M58 16l-4 5"/><rect x="18" y="26" width="44" height="36" rx="15"/><path d="M13 39v10M67 39v10"/></g><circle cx="31" cy="43" r="3" fill="#9aafff"/><circle cx="49" cy="43" r="3" fill="#9aafff"/><path d="M32 53c4 3 12 3 16 0" stroke="#9aafff" stroke-width="3" stroke-linecap="round"/></svg></div><h1>AI Assistant</h1><p>This is your AI Assistant that can help you summarize documents and search for specific small details in your document.<br>What do you have in mind?</p><p class="hint">Upload a PDF from the sidebar, then ask away.</p></div>''', unsafe_allow_html=True)

for message_index, message in enumerate(messages):
    render_chat_message(
        message["role"], message["content"], message.get("sources"), row_id=message_index
    )

question = st.chat_input("Type your message here...")
if question and question.strip():
    logger.info("Question received in chat %s (%d chars)", ss.chat_id, len(question))
    chat = ss.chats.setdefault(
        ss.chat_id,
        {"title": question.strip()[:60], "created": now_iso(), "updated": now_iso(), "messages": []},
    )
    messages = chat["messages"]
    messages.append({"role": "user", "content": question})
    welcome_area.empty()
    render_chat_message("user", question, row_id=len(messages) - 1)

    answered = False
    try:
        with st.container(key=f"chat_row_assistant_{len(messages)}"):
            with st.chat_message("assistant", avatar="🔵"):
                with st.spinner("Searching your documents…"):
                    backend = load_backend()
                    answer, sources = backend.answer_question(
                        question,
                        k=int(ss.settings["top_k"]),
                        n_subsets=int(ss.settings["n_subsets"]),
                    )
                st.markdown(answer)
                if sources and ss.settings["show_sources"]:
                    st.caption("Sources · " + " · ".join(map(str, sources)))
        messages.append({"role": "assistant", "content": answer, "sources": sources})
        answered = True
        logger.info("Answer ready (%d chars, %d sources)", len(answer), len(sources or []))
    except Exception as exc:
        logger.exception("The RAG system could not answer")
        st.error(f"The RAG system could not answer: {exc}")

    chat["updated"] = now_iso()
    save_chats()
    if answered:
        st.rerun()