# modules/ui_components.py
import streamlit as st
from contextlib import contextmanager
from html import escape

def set_page_config():
    st.set_page_config(page_title="Vitality AI", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

def init_session():
    # robust initialization of session keys used across the app
    keys = {
        "history": [],
        "health_context": "",
        "uploaded_files": [],
        "theme": "dark",
        "sidebar_open": True,
        "model_provider": "OpenAI (GPT)",
        "api_key": "",
        "input_text": "",
        "thinking": False,
        "last_transcript": "",
        "show_recorder": False,
        "show_full_app": False
    }
    for k, v in keys.items():
        if k not in st.session_state:
            st.session_state[k] = v

def inject_css(theme: str = "dark"):
    # ChatGPT-like CSS while keeping a visible left sidebar (white on light theme)
    if theme == "dark":
        bg = "#0b1220"; text = "#e6eef3"; muted = "#94A3B8"; panel = "#071024"; card = "#0f1724"
    else:
        bg = "#F6F9FB"; text = "#0F172A"; muted = "#6B7280"; panel = "#FFFFFF"; card = "#FFFFFF"

    css = f"""
    <style>
    .stApp {{ background: {bg}; color: {text}; font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; }}
    #MainMenu {{ visibility: hidden; }} footer {{ visibility: hidden; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
      background: {panel} !important;
      color: {text} !important;
      padding: 18px;
      min-width: 260px;
      border-right: 1px solid rgba(255,255,255,0.03);
    }}

    /* center chat column */
    .v-center-wrap {{ display:flex; justify-content:center; width:100%; }}
    .v-chat-column {{ width: min(980px, 92%); padding: 24px 12px 140px 12px; box-sizing:border-box; }}

    /* header */
    .v-chat-title {{ font-weight:700; font-size:28px; margin-bottom:6px; color:{text}; }}
    .v-chat-sub {{ color:{muted}; margin-bottom:18px; }}

    /* message bubbles */
    .v-msg-row {{ display:flex; gap:12px; margin:10px 0; align-items:flex-end; }}
    .v-msg-row.user {{ justify-content:flex-end; }}
    .v-msg-bubble {{ max-width:74%; padding:12px 14px; border-radius:14px; line-height:1.4; font-size:1rem; }}
    .v-msg-bubble.user {{ background: linear-gradient(90deg,#3b82f6,#6c5ce7); color:white; border-bottom-right-radius:6px; }}
    .v-msg-bubble.assistant {{ background: rgba(255,255,255,0.03); color:{text}; border-bottom-left-radius:6px; }}
    .v-avatar {{ width:36px; height:36px; border-radius:8px; display:inline-block; background: linear-gradient(135deg,#7C3AED,#10a37f); }}

    /* make Streamlit buttons used in bottom bar match input height and center */
    div.stButton > button, div.stButton > button svg {{
      height: 44px !important;
      min-height: 44px !important;
      display: inline-flex !important;
      align-items: center !important;
      justify-content: center !important;
    }}

    /* fixed bottom input visuals */
    .v-bottom-bar {{ position: fixed; left: 50%; transform: translateX(-50%); bottom: 18px; width: min(980px, 92%); z-index:9999; pointer-events: none; }}
    .v-input-visual {{ pointer-events: auto; width:100%; display:flex; gap:8px; align-items:center; background:{card}; border-radius:28px; padding:8px; box-shadow: 0 6px 30px rgba(2,6,23,0.6); border:1px solid rgba(255,255,255,0.03); }}
    .v-icon-btn {{ width:44px; height:44px; border-radius:10px; display:inline-grid; place-items:center; background: rgba(255,255,255,0.02); cursor:pointer; border:none; }}

    /* search / big input variant used in landing vs chat (kept simple) */
    .big-search {{ display:flex; gap:8px; align-items:center; background:{card}; border-radius:999px; padding:10px 14px; }}
    .big-search input {{ border:none; outline:none; font-size:18px; width:100%; background:transparent; color:{text}; }}

    .small-muted {{ color:{muted}; font-size:0.9rem; }}

    @media (max-width:900px) {{
      .v-msg-bubble {{ max-width:86%; }}
      .v-chat-column {{ padding-bottom: 180px; }}
      .v-bottom-bar {{ width: calc(100% - 28px); left: 14px; transform:none; }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# render helpers (keep simple & safe)
def render_user_message(txt: str):
    safe = escape(txt).replace("\n", "<br/>")
    html = f"""<div class="v-msg-row user"><div class="v-msg-bubble user">{safe}</div><div class="v-avatar" title="You"></div></div>"""
    st.markdown(html, unsafe_allow_html=True)

def render_assistant_message(txt: str):
    safe = escape(txt).replace("\n", "<br/>")
    html = f"""<div class="v-msg-row"><div class="v-avatar" title="Vitality"></div><div class="v-msg-bubble assistant">{safe}</div></div>"""
    st.markdown(html, unsafe_allow_html=True)

def show_buffering(message: str = "Thinking...", helper: str = ""):
    ph = st.empty()
    ph.markdown(f"<div style='padding:12px;border-radius:12px;background:rgba(255,255,255,0.02)'><strong>{escape(message)}</strong><div style='color:var(--muted);font-size:0.9rem'>{escape(helper)}</div></div>", unsafe_allow_html=True)
    return ph

@contextmanager
def thinking(message: str = "Thinking...", helper: str = ""):
    st.session_state.thinking = True
    ph = show_buffering(message, helper)
    try:
        yield ph
    finally:
        ph.empty()
        st.session_state.thinking = False

def render_bottom_input_placeholder():
    # small markup so CSS reserves space for the fixed bottom bar
    html = """<div class="v-bottom-bar"><div class="v-input-visual"><div style="width:12px"></div><div style="flex:1"></div><div style="width:12px"></div></div></div>"""
    st.markdown(html, unsafe_allow_html=True)