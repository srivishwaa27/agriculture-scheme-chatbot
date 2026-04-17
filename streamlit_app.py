"""
Streamlit Chatbot App — Agriculture Scheme RAG Chatbot
Run:
    streamlit run streamlit_app.py
"""

import os
import json
import base64
import streamlit as st
from rag_pipeline import rag_answer, get_retriever          # pre-warm on startup

HISTORY_FILE = "chat_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f)

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="🌾 Agriculture Scheme Chatbot",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# CUSTOM CSS & BACKGROUND
# ─────────────────────────────────────────
def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            /* Blurred and Dimmed Background Image */
            .stApp::before {{
                content: "";
                position: fixed;
                top: 0; left: 0; width: 100vw; height: 100vh;
                background-image: url(data:image/jpeg;base64,{encoded_string});
                background-size: cover;
                background-position: center;
                filter: blur(8px) brightness(0.4); /* Applies blur and darkening */
                z-index: -1;
            }}
            .stApp {{
                background-color: transparent !important;
            }}
            
            /* Glassmorphism for chat messages */
            [data-testid="stChatMessage"] {{
                background: rgba(30, 40, 30, 0.7) !important;
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border-radius: 12px;
                padding: 15px;
                border: 1px solid rgba(255, 255, 255, 0.15);
                color: #ffffff !important;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            /* Transparent Header */
            header[data-testid="stHeader"] {{
                background-color: transparent !important;
            }}
            /* Sidebar Dark Glass */
            [data-testid="stSidebar"] {{
                background-color: rgba(15, 25, 15, 0.85) !important;
                backdrop-filter: blur(15px);
                border-right: 1px solid rgba(255,255,255,0.1);
            }}

            /* Reduce Sidebar Font Size */
            [data-testid="stSidebar"] .stButton button p {{
                font-size: 0.85rem !important;
            }}
            [data-testid="stSidebar"] h3 {{
                font-size: 1.05rem !important;
            }}
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p, [data-testid="stSidebar"] .stCaptionContainer {{
                font-size: 0.85rem !important;
            }}
            /* Text elements to stand out against background */
            .main-header {{
                text-align: center;
                margin-top: 15vh;
                font-size: 2.8rem !important;
                font-weight: 800;
                color: #ffffff;
                text-shadow: 2px 2px 6px rgba(0,0,0,0.8);
                margin-bottom: 0.2rem;
                letter-spacing: 1px;
                line-height: 1.1;
            }}
            .sub-header {{
                color: #e0e0e0;
                font-size: 1.1rem;
                text-shadow: 1px 1px 4px rgba(0,0,0,0.8);
                margin-bottom: 1.5rem;
            }}
            .source-tag {{
                background: rgba(46, 125, 50, 0.8);
                color: #ffffff;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 0.8rem;
                margin-right: 4px;
                border: 1px solid rgba(255,255,255,0.3);
            }}
            /* Bottom Input Box and Wrappers */
            [data-testid="stBottom"] {{
                background-color: transparent !important;
                position: static !important;
            }}
            [data-testid="stBottom"] > div, [data-testid="stBottomBlock"] {{
                background-color: transparent !important;
                background: transparent !important;
            }}

            /* Center the search bar */
            .stChatFloatingInputContainer {{
                background-color: transparent !important;
                background: transparent !important;
                max-width: 650px !important;
                margin: 0 auto !important;
            }}

            [data-testid="stChatInput"] {{
                background: rgba(20, 30, 20, 0.7) !important;
                border: 1px solid rgba(255, 255, 255, 0.3) !important;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
                backdrop-filter: blur(10px);
                border-radius: 24px;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        pass

add_bg_from_local("bg_image.jpg")

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
# Hardcoded API Key as requested
hf_api_key = "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"

with st.sidebar:

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.subheader("💬 Sample questions")
    examples = [
        "What is PM Kisan scheme?",
        "Eligibility for Kisan Credit Card?",
        "Agriculture schemes in Tamil Nadu?",
        "How to apply for crop insurance?",
        "PM Fasal Bima Yojana benefits?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.prefill = ex

    st.divider()
    st.subheader("🕒 Chat History")
    history_list = load_history()
    if history_list:
        for i, hq in enumerate(reversed(history_list[-10:])):
            if st.button(hq, key=f"hist_{i}", use_container_width=True):
                st.session_state.prefill = hq
    else:
        st.caption("No past chats yet.")

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    has_new_input = st.session_state.get("chat_input") or st.session_state.get("prefill")
    if not has_new_input:
        st.markdown('<p class="main-header">Agriculture Scheme Chatbot</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# PRE-WARM (cache the retriever on first load)
# ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base …")
def warm_up():
    return get_retriever()

warm_up()

# ─────────────────────────────────────────
# CHAT STATE
# ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "prefill" not in st.session_state:
    st.session_state.prefill = ""

# ─────────────────────────────────────────
# DISPLAY HISTORY
# ─────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])



# ─────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────
user_input = st.chat_input(
    "Ask about any agricultural scheme …",
    key="chat_input",
)

# Allow prefill from sidebar buttons
if st.session_state.prefill:
    user_input = st.session_state.prefill
    st.session_state.prefill = ""

if user_input:
    hist = load_history()
    if user_input not in hist:
        hist.append(user_input)
        save_history(hist)

    # Guard: API key required
    if not hf_api_key:
        st.error("⚠️  Please enter your Hugging Face API Key in the sidebar.")
        st.stop()

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and generating answer …"):
            try:
                result = rag_answer(user_input, hf_api_key)
                answer  = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                if sources:

                    with st.expander("🔍 View retrieved context"):
                        st.text(result["context"])

            except Exception as e:
                answer  = f"❌ Error: {e}"
                sources = []
                st.error(answer)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
    })
