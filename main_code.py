# main_code.py
# Vitality AI - Personal Health OS with LLMs and RAG

import warnings

# Suppress PyTorch “torch.classes” warnings / diagnostics
warnings.filterwarnings("ignore", message=".*torch.classes.*")

# Optional: silence PyTorch internal debug messages completely

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime

from modules.ui_components import (
    set_page_config,
    inject_css,
    init_session,
    render_user_message,
    render_assistant_message,
    render_bottom_input_placeholder,
)
from modules.llm_factory import load_model
from modules.bio_tools import get_health_tools
from modules.agent_engine import create_health_agent
from modules.file_handler import extract_pdf_data


# --- loading / thinking animation helper ---
def _thinking_html() -> str:
    return """
    <div style="display:flex;justify-content:flex-start;margin:8px 4px;">
      <div style="
        padding:8px 12px;
        border-radius:16px;
        background:rgba(148,163,184,0.14);
        color:rgba(148,163,184,0.95);
        font-size:0.9rem;
        display:flex;
        align-items:center;
        gap:6px;
      ">
        <span>Vitality is thinking</span>
        <span class="v-dots">
          <span>.</span><span>.</span><span>.</span>
        </span>
      </div>
    </div>
    <style>
    .v-dots span {
      animation: v-bounce 1s infinite;
      display:inline-block;
      margin-right:2px;
    }
    .v-dots span:nth-child(2) { animation-delay: 0.15s; }
    .v-dots span:nth-child(3) { animation-delay: 0.30s; }
    @keyframes v-bounce {
      0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
      40% { transform: translateY(-4px); opacity: 1; }
    }
    </style>
    """


# ---------------- init ----------------
set_page_config()
init_session()
if "tavily_used" not in st.session_state:
    st.session_state["tavily_used"] = False

# Ensure bottom_input always exists and is never None
if "bottom_input" not in st.session_state or st.session_state.get("bottom_input") is None:
    st.session_state["bottom_input"] = ""

inject_css(st.session_state.theme)

# thread-aware store
if "all_threads" not in st.session_state:
    # each thread: {"messages": [...]} 
    st.session_state["all_threads"] = []

# lazy optional modules
_VECTOR_AVAILABLE = False
_vector_module = None
_TAVILY_AVAILABLE = False
_tavily_module = None


def _lazy_load_vector():
    global _VECTOR_AVAILABLE, _vector_module
    if _vector_module is not None:
        return _VECTOR_AVAILABLE
    try:
        import modules.vector_store as vs

        _vector_module = vs
        _VECTOR_AVAILABLE = True
    except Exception:
        _VECTOR_AVAILABLE = False
    return _VECTOR_AVAILABLE


def _lazy_load_tavily():
    global _TAVILY_AVAILABLE, _tavily_module
    if _tavily_module is not None:
        return _TAVILY_AVAILABLE
    try:
        import modules.tavily_tool as tt

        _tavily_module = tt
        _TAVILY_AVAILABLE = True
    except Exception:
        _TAVILY_AVAILABLE = False
    return _TAVILY_AVAILABLE


# read query param STT safely (harmless)
try:
    qp = st.query_params
    if "vitality_stt" in qp:
        v = qp.get("vitality_stt")
        st.session_state["last_transcript"] = (
            v[0] if isinstance(v, list) and v else (v or "")
        )
        try:
            st.query_params.clear()
        except Exception:
            try:
                st.experimental_set_query_params()
            except Exception:
                pass
except Exception:
    pass

# init vector if available
if _lazy_load_vector():
    try:
        _vector_module.init_store()
    except Exception as e:
        st.warning(f"Vector DB init warning: {e}")


# ---------------- helper: append history ----------------
def _append_history(role: str, content: str):
    st.session_state.history.append(
        {"role": role, "content": content, "time": str(datetime.utcnow())}
    )


# ---------------- layout: sidebar + centered chat ----------------
# ---------- SIDEBAR (cleaned single copy) ----------
with st.sidebar:
    # Theme selector
    st.markdown("### Choose Theme")
    theme_choice = st.radio(
        "Theme",
        ["Light", "Dark"],
        index=0 if st.session_state.get("theme", "dark") == "light" else 1,
        format_func=lambda x: " ☀️ Light " if x == "Light" else " 🌙 Dark ",
        label_visibility="collapsed",
        key="sidebar_theme_radio",
    )
    new_theme = "light" if theme_choice == "Light" else "dark"
    if new_theme != st.session_state.get("theme", "dark"):
        st.session_state["theme"] = new_theme
        inject_css(st.session_state.theme)

    st.markdown("## 🩺 Vitality AI")
    st.caption("Your Personal Health OS")
    st.divider()

    # Provider + API key
    provider = st.selectbox(
        "AI Model",
        ["OpenAI (GPT)", "Google Gemini", "Groq"],
        index=(
            0
            if st.session_state.get("model_provider", "OpenAI (GPT)")
            == "OpenAI (GPT)"
            else 1
            if st.session_state.get("model_provider") == "Google Gemini"
            else 2
        ),
        key="sidebar_provider_select",
    )
    st.session_state.model_provider = provider
    st.session_state.api_key = st.text_input(
        f"{provider} API Key",
        type="password",
        value=st.session_state.get("api_key", ""),
        help="Enter API key for selected provider",
        key="sidebar_api_key",
    )

    tavily_key = st.text_input(
        "Tavily API Key (optional)",
        type="password",
        value=os.environ.get("TAVILY_API_KEY", ""),
        key="sidebar_tavily_key",
    )
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

    st.divider()
    st.markdown("### 📂 Health Data")
    st.markdown("Upload PDFs (text-layer extraction) below:", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a PDF", type=["pdf"], key="side_pdf_picker"
    )
    if uploaded_file:
        txt = extract_pdf_data(uploaded_file) or ""
        # ensure uploaded_files exists
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []
        st.session_state.uploaded_files.insert(0, uploaded_file)
        st.session_state.health_context = (
            st.session_state.get("health_context", "") + f"\n\n[{uploaded_file.name}]\n" + txt
        )
        st.success("Uploaded & indexed (text layer).")
        if _lazy_load_vector():
            try:
                _vector_module.ingest_text(
                    source_name=uploaded_file.name.replace(" ", "_"),
                    text=txt,
                )
                st.success(f"Indexed {uploaded_file.name} into VectorDB.")
            except Exception as e:
                st.warning(f"Vector ingestion failed: {e}")

    st.divider()

    # Chat history card (single)
    st.markdown(
        """
        <div style="
            padding: 10px 10px 8px 10px;
            border-radius: 14px;
            background: rgba(15,23,42,0.95);
            border: 1px solid rgba(148,163,184,0.45);
            margin-bottom: 10px;
        ">
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### 📂 Chat History")

    # ONE New chat button (unique key)
    if st.button("➕ New chat", use_container_width=True, key="sidebar_new_chat"):
        cur = st.session_state.get("history", [])
        if cur:
            # save only if not identical to last thread
            if not st.session_state.all_threads or st.session_state.all_threads[-1] != cur:
                st.session_state.all_threads.append(cur.copy())
        st.session_state.history = []
        st.session_state.health_context = ""
        st.session_state.uploaded_files = []
        st.session_state["bottom_input"] = ""
        st.session_state.last_transcript = ""

    st.markdown("<hr style='margin:8px 0;border-color:rgba(51,65,85,0.9);'>", unsafe_allow_html=True)

    # Build the list of threads to display (saved threads + optionally current unsaved chat)
    threads = list(st.session_state.get("all_threads", []))
    if st.session_state.get("history"):
        # show current history as well, but do not mutate the saved list
        threads = threads + [st.session_state.history]

    if threads:
        # newest first
        for idx, thread in enumerate(reversed(threads), start=1):
            chat_num = len(threads) - idx + 1
            # use unique key per row (chat number + idx ensures uniqueness)
            if st.button(f"Chat {chat_num}", key=f"thread_btn_{chat_num}_{idx}", use_container_width=True):
                st.session_state.history = thread.copy()
    else:
        st.markdown(
            "<div style='color:rgba(148,163,184,0.9);font-size:0.9rem;'>No chats yet.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # Download current chat if present
    if st.session_state.get("history"):
        full_txt = "\n\n".join([f"{m['role'].upper()}:\n{m['content']}" for m in st.session_state.history])
        st.download_button(
            "⬇ Download Chat (.txt)",
            data=full_txt,
            file_name="vitality_chat_history.txt",
            mime="text/plain",
            key="sidebar_download_history",
        )
# ---------- END SIDEBAR ----------


# ---------------- main chat content (center) ----------------
st.markdown(
    '<div class="v-center-wrap"><div class="v-chat-column">',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="v-chat-title">👋 Hey, I\'m Vitality.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="v-chat-sub">I can calculate your macros, read your lab reports, '
    "triage symptoms, or plan workouts.</div>",
    unsafe_allow_html=True,
)

_badge_map = {
    "Google Gemini": "💠 Gemini",
    "Groq": "⚡ Groq",
    "OpenAI (GPT)": "🔵 GPT",
}
badge = _badge_map.get(
    st.session_state.get("model_provider", "OpenAI (GPT)"),
    st.session_state.get("model_provider"),
)
st.markdown(
    "<div style='margin-top:10px;margin-bottom:8px; "
    "color:rgba(255,255,255,0.85)'>Using: "
    f"<span style='padding:6px 10px;border-radius:999px;"
    "background:rgba(255,255,255,0.02);'>"
    f"{badge}</span></div>",
    unsafe_allow_html=True,
)

# Render history / welcome
if not st.session_state.history:
    st.markdown(
        "<div style='padding:18px;border-radius:12px;"
        "background:rgba(255,255,255,0.02)'>Welcome — ask about BMI, labs, "
        "or workouts. Use the sidebar to upload PDFs and change AI model."
        "</div>",
        unsafe_allow_html=True,
    )

for msg in st.session_state.history:
    if msg["role"] == "user":
        render_user_message(msg["content"])
    else:
        render_assistant_message(msg["content"])

st.markdown('<div style="height:160px"></div>', unsafe_allow_html=True)
st.markdown("</div></div>", unsafe_allow_html=True)

# render visual placeholder (keeps same look)
render_bottom_input_placeholder()

# ---------------- Bottom input UI (using a form with clear_on_submit=True) ----------------
prefill = st.session_state.get("last_transcript", "")

if prefill:
    try:
        if not st.session_state.get("bottom_input"):
            st.session_state["bottom_input"] = prefill
    except Exception:
        pass
    finally:
        try:
            st.session_state["last_transcript"] = ""
        except Exception:
            pass

# Initialize defensive session keys
if "_last_sent_prompt" not in st.session_state:
    st.session_state["_last_sent_prompt"] = None
if "_processing_send" not in st.session_state:
    st.session_state["_processing_send"] = False

with st.container():
    cols = st.columns([0.02, 0.96, 0.02])
    with cols[1]:
        with st.form(key="chat_form", clear_on_submit=True):
            prompt = st.text_input(
                "Message",
                placeholder=(
                    " Ask anything like ' Calculate my BMI if my weight is 75 kg and height is 1.8 m ' or "
                    "' Plan a 4-day workout split '... "
                ),
                key="bottom_input",
                label_visibility="collapsed",
            )
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            send_submitted = st.form_submit_button("Send")


# ---------------- Handle Send (form submission) ----------------

if send_submitted:
    prompt = (prompt or "").strip()

    if not st.session_state.get("api_key", "").strip():
        st.warning("⚠️ Please enter your API key in the sidebar before sending a message.")
    elif not prompt:
        st.info("Please type a message before sending.")
    elif st.session_state.get("_processing_send"):
        st.info("Processing previous message — please wait a moment.")
    elif st.session_state.get("_last_sent_prompt") == prompt:
        st.info("This message appears to be already sent. Edit and resend if needed.")
    else:
        st.session_state["_processing_send"] = True
        st.session_state["_last_sent_prompt"] = prompt
        try:
            ph = st.empty()
            ph.markdown(_thinking_html(), unsafe_allow_html=True)

            _append_history("user", prompt)

            # Simple context (no vector RAG for now)
            context = st.session_state.get("health_context", "No reports uploaded.")[:3000]

            # Load LLM for selected provider
            llm = load_model(
                st.session_state.get("model_provider", "OpenAI (GPT)"),
                st.session_state.get("api_key", ""),
            )
            if llm is None:
                raise ValueError("LLM failed to load. Check API key / provider.")

            # Build tools list
            tools = get_health_tools()
            if not isinstance(tools, list):
                tools = list(tools.values()) if isinstance(tools, dict) else [tools]

            # Add Tavily tool if available
            if _lazy_load_tavily():
                try:
                    tav_tool = _tavily_module.get_tavily_tool()

                    # Wrapper that flips a flag, then calls the real tool
                    def _tavily_wrapper(query: str, max_results: int = 5):
                        st.session_state["tavily_used"] = True
                        func = getattr(tav_tool, "func", tav_tool)  # Tool.func or bare callable
                        try:
                            return func(query) if func.__code__.co_argcount == 1 else func(query, max_results)
                        except Exception:
                            return func(query)

                    try:
                        from langchain.tools import Tool as LC_Tool  # type: ignore

                        wrapped_tav = LC_Tool(
                            name=getattr(tav_tool, "name", "tavily_search"),
                            func=_tavily_wrapper,
                            description=getattr(tav_tool, "description", "Tavily search tool (wrapped)"),
                        )
                    except Exception:
                        class _LocalSimple:
                            def __init__(self, name, func, description=""):
                                self.name = name
                                self.func = func
                                self.description = description

                            def __call__(self, x):
                                return self.func(x)

                        wrapped_tav = _LocalSimple(
                            name=getattr(tav_tool, "name", "tavily_search"),
                            func=_tavily_wrapper,
                            description=getattr(tav_tool, "description", "Tavily search tool (wrapped)"),
                        )

                        tools.append(wrapped_tav)
                except Exception:
                    st.warning("Tavily tool load failed; continuing without Tavily.")

            # Reset flag for this turn
            st.session_state["tavily_used"] = False

            # Create agent and invoke
            agent = create_health_agent(llm, tools)
            chat_history = "\n".join(
                f"{m['role']}: {m['content']}" for m in st.session_state.history[-8:]
            )
            response = agent.invoke(
                {
                    "input": prompt,
                    "context": context,
                    "chat_history": chat_history,
                }
            )
            output = response.get("output") if isinstance(response, dict) else str(response)
            output = output or "No response."
            _append_history("assistant", output)

            # If Tavily ran, append evidence note
            if st.session_state.get("tavily_used"):
                evidence_text = "EVIDENCE: Results retrieved via Tavily (external web search)."
                _append_history("assistant", evidence_text)
        except Exception as e:
            _append_history("assistant", f"Error: {e}")
        finally:
            ph.empty()
            st.session_state["_processing_send"] = False
