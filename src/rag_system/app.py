from pathlib import Path

import streamlit as st

from ask import answer_question

LOGO_PATH = Path(__file__).with_name("logo.png")

st.set_page_config(page_title="FitStyle AI | Document Chat", page_icon=str(LOGO_PATH), layout="centered")

st.markdown("""
<style>
    :root {
        --ink: #39213e;
        --muted: #8f748b;
        --accent: #a82771;
        --accent-deep: #61256d;
        --paper: #fff9fb;
        --line: #f0dde6;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 12% 2%, rgba(248, 221, 230, .8), transparent 34%),
            radial-gradient(ellipse at 94% 9%, rgba(232, 218, 241, .72), transparent 32%),
            var(--paper);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] { padding-top: 2.4rem; max-width: 900px; }
    .hero { padding: .7rem 0 1.6rem; }
    .eyebrow {
        display: inline-flex; align-items: center; gap: .45rem;
        padding: .38rem .72rem; border: 1px solid #efd6e2; border-radius: 999px;
        background: rgba(255,255,255,.7); color: var(--accent-deep);
        font-size: .78rem; font-weight: 650; letter-spacing: .07em; text-transform: uppercase;
    }
    .hero h1 { color: var(--ink); font-size: 2.55rem; letter-spacing: -.045em; margin: .9rem 0 .35rem; }
    .hero p { color: var(--muted); font-size: 1.04rem; margin: 0; }
    .assistant-overview {
        margin: .2rem 0 1.4rem; padding: 1rem 1.2rem;
        border: 1px solid #efd6e2; border-radius: 14px;
        background: rgba(255,255,255,.72);
        box-shadow: 0 7px 22px rgba(93, 38, 82, .045);
    }
    .assistant-overview h3 { color: #61256d; margin: 0 0 .35rem; font-size: 1.08rem; }
    .assistant-overview p { color: #766879; margin: 0; line-height: 1.55; }
    .source-note {
        display: inline-block; margin-top: .7rem; color: #a82771;
        font-size: .82rem; font-weight: 600; letter-spacing: .015em;
    }
    [data-testid="stChatMessage"] {
        border: 1px solid var(--line); border-radius: 18px;
        background: rgba(255,255,255,.86); padding: .8rem 1rem;
        box-shadow: 0 8px 24px rgba(57, 33, 62, .045);
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #a82771, #61256d);
        border-color: transparent; color: #fff;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p { color: #fff; }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        border-left: 3px solid #d35c92;
    }
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] {
        width: calc(100% - 3rem) !important; max-width: 900px !important;
        box-sizing: border-box; margin-left: auto !important; margin-right: auto !important;
    }
    [data-testid="stBottom"] [data-testid="stChatInput"] {
        width: 100% !important; max-width: none !important; box-sizing: border-box;
    }    [data-testid="stChatInput"] {
        background: #fff !important; border: 1px solid #dfb8ca !important;
        border-radius: 12px !important; padding: 7px 9px !important;
        box-shadow: 0 5px 18px rgba(93, 38, 82, .08) !important;
    }
    [data-testid="stChatInput"] > div {
        background: transparent !important; border: 0 !important;
        box-shadow: none !important; padding: 0 !important;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important; border: 0 !important;
        border-radius: 0 !important; box-shadow: none !important;
        color: #39213e !important; min-height: 40px;
    }
    [data-testid="stChatInput"] textarea:focus {
        border: 0 !important; box-shadow: none !important; outline: none !important;
    }
    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #a82771, #61256d) !important;
        color: #fff !important; border: 0 !important; border-radius: 9px !important;
        min-width: 40px; min-height: 40px;
    }
    [data-testid="stChatInput"] button:hover { filter: brightness(1.08); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #b16d8f 0%, #dba8be 52%, #f8e9ef 100%);
    }
    [data-testid="stSidebar"] * { color: #482b49; }
    .sidebar-brand {
        display: flex; align-items: center; width: 100%; box-sizing: border-box;
        min-height: 58px; margin: 0 0 1rem; padding: 0 .6rem;
    }
    .sidebar-brand span {
        color: #fff8fc !important; -webkit-text-fill-color: #fff8fc !important;
        font-size: 2.1rem; line-height: 1.2; font-weight: 750;
        text-shadow: 0 1px 3px rgba(66, 34, 62, .3);
    }
    [data-testid="stSidebar"] p { font-size: 1.08rem; line-height: 1.6; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        font-size: .98rem; color: #5d3b5a;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        display: block; width: 100%; box-sizing: border-box;
        border: 1px solid #dfbdd0; border-radius: 12px;
        background: rgba(255,255,255,.82); color: #482b49;
        font-size: 1.08rem; font-weight: 650; padding: .68rem .85rem;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button:hover {
        background: #fff; border-color: #c997b2;
    }    [data-testid="stAlert"] { border-radius: 14px; }
    .stCaption, [data-testid="stCaptionContainer"] { color: var(--muted); }
    @media (max-width: 640px) {
        [data-testid="stMainBlockContainer"] { padding-top: 1.4rem; }
        .hero h1 { font-size: 2rem; }
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <span class="eyebrow">✦ FitStyle AI · Knowledge Assistant</span>
  <h1>Ask the documentation</h1>
  <p>Get clear answers grounded in the FitStyle AI project document.</p>
</div>
<div class="assistant-overview">
  <h3>Your document assistant</h3>
  <p>Ask about the FitStyle AI project and get clear answers grounded in its documentation.</p>
  <span class="source-note">✦ Source page references included</span>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div class=\"sidebar-brand\"><span>✦ FitStyle AI</span></div>", unsafe_allow_html=True)
    if st.button("＋  Start a new chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.info("Try asking: “What are the main components of FitStyle AI?”", icon="💡")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("pages"):
            st.caption("Source pages · " + ", ".join(map(str, message["pages"])))

question = st.chat_input("Ask a question about the FitStyle AI document...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document..."):
            try:
                answer, pages = answer_question(question)
                st.markdown(answer)
                st.caption("Source pages · " + ", ".join(map(str, pages)))
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "pages": pages}
                )
            except Exception as error:
                st.error(str(error))
