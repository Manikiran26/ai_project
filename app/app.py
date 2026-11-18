"""
app.py - AI Customer Support Assistant
======================================
Clean, focused customer support triage application powered by NLP and Supervised ML.
Features:
- Instant query classification into support categories
- Predefined actionable response generation
- Native Streamlit chat interface (zero HTML leakage)
- 1-click test chips for fast demonstration
"""

import os
import sys
import streamlit as st

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.predict import load_model, predict_and_respond


# ====================================================================
# PAGE CONFIGURATION
# ====================================================================
st.set_page_config(
    page_title="AI Customer Support Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ====================================================================
# CLEAN STYLING (FORCED LIGHT THEME, PROPER CONTAINER SPACING)
# ====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }

    /* Clean Streamlit default header */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 1.5rem !important;
    }

    /* Completely hide sidebar */
    [data-testid="stSidebar"] {
        display: none !important;
    }

    /* Main container constraints */
    .block-container {
        max-width: 820px !important;
        padding-top: 2rem !important;
        padding-bottom: 3.5rem !important;
        margin: 0 auto !important;
    }

    /* Header styling */
    .app-header {
        text-align: center;
        padding: 0.5rem 0 1.2rem 0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 1.2rem;
    }
    .app-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.025em;
    }
    .app-subtitle {
        font-size: 1.05rem;
        color: #64748b;
        margin: 0 0 0.75rem 0;
        font-weight: 400;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background-color: #f0fdf4;
        color: #166534;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        border: 1px solid #bbf7d0;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #22c55e;
    }

    /* Clear Chat Button */
    div.stButton > button {
        background-color: #ffffff !important;
        color: #475569 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        padding: 0.35rem 0.85rem !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
        transition: all 0.15s ease-in-out !important;
    }
    div.stButton > button:hover {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        border-color: #94a3b8 !important;
    }

    /* Chat message styles */
    div[data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 1rem 1.2rem !important;
        margin-bottom: 0.85rem !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02) !important;
    }

    /* Chat Input Styling */
    div[data-testid="stChatInput"] {
        border-radius: 12px !important;
    }
    div[data-testid="stChatInput"] > div {
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 12px !important;
        background: #ffffff !important;
        box-shadow: 0 3px 10px rgba(0,0,0,0.04) !important;
    }
    div[data-testid="stChatInput"] textarea {
        color: #0f172a !important;
        font-size: 0.95rem !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #94a3b8 !important;
    }
</style>
""", unsafe_allow_html=True)


# ====================================================================
# MODEL LOADING (CACHED FOR INSTANT RESPONSE)
# ====================================================================
@st.cache_resource(show_spinner=False)
def load_support_pipeline():
    models_dir = os.path.join(project_root, 'models')
    try:
        model, vectorizer, info = load_model(models_dir)
        return model, vectorizer
    except Exception as e:
        st.error(f"Failed to load ML model: {e}")
        return None, None

model, vectorizer = load_support_pipeline()


# ====================================================================
# HEADER
# ====================================================================
st.markdown("""
<div class="app-header">
    <h1 class="app-title">AI Customer Support Assistant</h1>
    <p class="app-subtitle">Automated Query Classification & Response System</p>
    <div class="status-badge">
        <span class="status-dot"></span>
        AI System Active & Ready
    </div>
</div>
""", unsafe_allow_html=True)


# ====================================================================
# SESSION INITIALIZATION
# ====================================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "category": None,
            "confidence": None,
            "response": (
                "Hello! Welcome to the Customer Support Assistant. "
                "Please describe your issue (e.g. delivery delays, payment failures, "
                "refund requests, account login, or hardware issues), and I will identify "
                "the category and provide step-by-step assistance."
            )
        }
    ]


# ====================================================================
# TOOLBAR & QUICK CHIPS
# ====================================================================
col_info, col_clear = st.columns([4, 1.2])
with col_info:
    msg_count = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.markdown(
        f"<div style='padding-top:6px; font-size:0.86rem; color:#475569; font-weight:600;'>"
        f"💬 Active Conversation &nbsp;•&nbsp; <strong>{msg_count}</strong> query handled"
        f"</div>",
        unsafe_allow_html=True
    )
with col_clear:
    if st.button("🗑️ Clear Chat", key="clear_chat_action"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "category": None,
                "confidence": None,
                "response": (
                    "Hello! Welcome to the Customer Support Assistant. "
                    "Please describe your issue, and I will identify "
                    "the category and provide step-by-step assistance."
                )
            }
        ]
        st.rerun()

# 1-Click Quick Test Queries
st.markdown(
    "<div style='font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#64748b; margin:0.8rem 0 0.4rem 0;'>"
    "⚡ Quick Test Queries:"
    "</div>",
    unsafe_allow_html=True
)

chip_cols = st.columns(4)
sample_queries = [
    ("📦 Delivery Delay", "i didnt received my order"),
    ("💳 Payment Deducted", "my payment failed but money was deducted"),
    ("🔄 Request Refund", "i want to request a full refund for my order"),
    ("🔐 Account Login", "cannot login to my account need to reset password"),
]

selected_query = None
for i, (label, text) in enumerate(sample_queries):
    with chip_cols[i]:
        if st.button(label, key=f"quick_btn_{i}"):
            selected_query = text

st.markdown("<hr style='margin:0.8rem 0 1.2rem 0; border:none; border-top:1px solid #e2e8f0;'>", unsafe_allow_html=True)


# ====================================================================
# CHAT STREAM RENDERING (ZERO HTML LEAKS)
# ====================================================================
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            category = msg.get("category")
            confidence = msg.get("confidence")
            response = msg.get("response", "")
            priority = msg.get("priority", "Normal")
            priority_color = msg.get("priority_color", "#16a34a")
            sentiment = msg.get("sentiment", "Neutral")
            sentiment_icon = msg.get("sentiment_icon", "😐")

            if category:
                # Render clean category, priority, and sentiment badges
                conf_text = f"{confidence:.0%}" if confidence else "95%"
                badge_html = (
                    f"<div style='display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-bottom:12px;'>"
                    f"<span style='background-color:#eff6ff; color:#1d4ed8; font-weight:700; font-size:0.85rem; padding:4px 12px; border-radius:6px; border:1px solid #bfdbfe;'>"
                    f"🎯 Category: {category} &nbsp;•&nbsp; Confidence: {conf_text}"
                    f"</span>"
                    f"<span style='background-color:#fef2f2; color:{priority_color}; font-weight:700; font-size:0.80rem; padding:4px 10px; border-radius:6px; border:1px solid #fecaca;'>"
                    f"⚡ Priority: {priority}"
                    f"</span>"
                    f"<span style='background-color:#f1f5f9; color:#475569; font-weight:700; font-size:0.80rem; padding:4px 10px; border-radius:6px; border:1px solid #cbd5e1;'>"
                    f"{sentiment_icon} Sentiment: {sentiment}"
                    f"</span>"
                    f"</div>"
                )
                st.markdown(badge_html, unsafe_allow_html=True)

                # Render response cleanly
                st.markdown(response)
            else:
                st.write(response)


# ====================================================================
# CHAT INPUT & EXECUTION
# ====================================================================
user_input = st.chat_input("Type your customer support query...")

if user_input:
    selected_query = user_input

if selected_query:
    # 1. Append user message
    st.session_state.messages.append({
        "role": "user",
        "content": selected_query
    })

    # 2. Run prediction with full conversation history
    if model is not None and vectorizer is not None:
        prediction_result = predict_and_respond(
            selected_query, model, vectorizer, history=st.session_state.messages
        )
        cat = prediction_result.get("predicted_category")
        conf = prediction_result.get("confidence")
        resp = prediction_result.get("response")
        priority = prediction_result.get("priority", "Normal")
        priority_color = prediction_result.get("priority_color", "#16a34a")
        sentiment = prediction_result.get("sentiment", "Neutral")
        sentiment_icon = prediction_result.get("sentiment_icon", "😐")
    else:
        cat = "General Inquiry"
        conf = 0.50
        resp = "We have received your query. Our team will review it shortly."
        priority = "Normal"
        priority_color = "#16a34a"
        sentiment = "Neutral"
        sentiment_icon = "😐"

    # 3. Append assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "category": cat,
        "confidence": conf,
        "response": resp,
        "priority": priority,
        "priority_color": priority_color,
        "sentiment": sentiment,
        "sentiment_icon": sentiment_icon,
    })

    st.rerun()
