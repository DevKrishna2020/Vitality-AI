# main_code.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime

from modules.ui_components import (
    set_page_config, inject_css, init_session,
    render_user_message, render_assistant_message, render_bottom_input_placeholder
)
from modules.llm_factory import load_model
from modules.bio_tools import get_health_tools
from modules.agent_engine import create_health_agent
from modules.file_handler import extract_pdf_data

# --- Safe rerun helper (minimal) ---
try:
    from streamlit.runtime.scriptrunner.script_runner import RerunException as _RerunException
except Exception:
    _RerunException = None

def safe_rerun_minimal():
    """
    Single, guarded rerun helper. Use sparingly and prefer a single call at the end of heavy handlers.
    """
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
            return
    except Exception:
        pass
    if _RerunException is not None:
        raise _RerunException("safe_rerun triggered")
    st.session_state["_safe_rerun_toggle"] = not st.session_state.get("_safe_rerun_toggle", False)

# --- New: safe, simple toggle-style rerun (avoid experimental_rerun) ---
def trigger_rerun():
    """
    Toggle a session key to force a rerun without calling experimental_rerun.
    This avoids touching Streamlit internal rerun objects that sometimes cause crashes.
    """
    st.session_state["_trigger_rerun_toggle"] = not st.session_state.get("_trigger_rerun_toggle", False)


# ---------------- init ----------------
set_page_config()
init_session()
# Ensure bottom_input always exists and is never None
if "bottom_input" not in st.session_state or st.session_state.get("bottom_input") is None:
    st.session_state["bottom_input"] = ""
inject_css(st.session_state.theme)

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
        st.session_state["last_transcript"] = v[0] if isinstance(v, list) and v else (v or "")
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

# ---------------- layout: sidebar + centered chat ----------------
with st.sidebar:
    st.markdown("## 🩺 Vitality AI")
    st.caption("Your Personal Health OS")
    st.divider()

    provider = st.selectbox(
        "AI Model",
        ["OpenAI (GPT)", "Google Gemini", "Groq"],
        index=0 if st.session_state.get("model_provider", "OpenAI (GPT)") == "OpenAI (GPT)" else (1 if st.session_state.get("model_provider") == "Google Gemini" else 2)
    )
    st.session_state.model_provider = provider
    st.session_state.api_key = st.text_input(f"{provider} API Key", type="password", value=st.session_state.get("api_key", ""), help="Enter API key for selected provider")

    tavily_key = st.text_input("Tavily API Key (optional)", type="password", value=os.environ.get("TAVILY_API_KEY", ""))
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

    st.divider()
    st.markdown("### 📂 Health Data")
    st.markdown("Upload PDFs (text-layer extraction) below:")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"], key="side_pdf_picker")
    if uploaded_file:
        txt = extract_pdf_data(uploaded_file) or ""
        st.session_state.uploaded_files.insert(0, uploaded_file)
        st.session_state.health_context = (st.session_state.health_context or "") + f"\n\n[{uploaded_file.name}]\n" + txt
        st.success("Uploaded & indexed (text layer).")
        if _lazy_load_vector():
            try:
                _vector_module.ingest_text(source_name=uploaded_file.name.replace(" ", "_"), text=txt)
                st.success(f"Indexed {uploaded_file.name} into VectorDB.")
            except Exception as e:
                st.warning(f"Vector ingestion failed: {e}")

    st.divider()
    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        # keep UI control but DO NOT trigger experimental reruns here; set a request flag instead
        if st.button("Toggle theme"):
            st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
            inject_css(st.session_state.theme)
            st.session_state["_request_rerun"] = True
    with c2:
        if st.button("New chat"):
            # Clear UI state immediately (avoid experimental_rerun during heavy work)
            st.session_state.history = []
            st.session_state.health_context = ""
            st.session_state.uploaded_files = []
            st.session_state["bottom_input"] = ""
            st.session_state.last_transcript = ""
            st.session_state["_request_rerun"] = True

    st.divider()
    st.markdown("### 🔁 Chat History")
    if st.session_state.history:
        for i, m in enumerate(reversed(st.session_state.history[-30:])):
            who = "You" if m["role"] == "user" else "Vitality"
            preview = m["content"].replace("\n", " ")[:70]
            if st.button(f"{who}: {preview}", key=f"hist_{i}"):
                # set the bottom_input session key and request a rerun
                st.session_state["bottom_input"] = m["content"]
                st.session_state["_request_rerun"] = True
    else:
        st.info("No chats yet. Start below.")

    st.divider()
    if st.session_state.history:
        full_txt = "\n\n".join([f"{m['role'].upper()}:\n{m['content']}" for m in st.session_state.history])
        st.download_button("⬇ Download Chat (.txt)", data=full_txt, file_name="vitality_chat_history.txt", mime="text/plain")

# If any sidebar action set _request_rerun, perform it now (single controlled rerun)
if st.session_state.pop("_request_rerun", False):
    trigger_rerun()

# ---------------- main chat content (center) ----------------
st.markdown('<div class="v-center-wrap"><div class="v-chat-column">', unsafe_allow_html=True)
st.markdown('<div class="v-chat-title">👋 Hey, I\'m Vitality.</div>', unsafe_allow_html=True)
st.markdown('<div class="v-chat-sub">I can calculate your macros, read your lab reports, triage symptoms, or plan workouts.</div>', unsafe_allow_html=True)

_badge_map = {"Google Gemini": "💠 Gemini", "Groq": "⚡ Groq", "OpenAI (GPT)": "🔵 GPT"}
badge = _badge_map.get(st.session_state.get("model_provider", "OpenAI (GPT)"), st.session_state.get("model_provider"))
st.markdown(f"<div style='margin-top:10px;margin-bottom:8px; color:rgba(255,255,255,0.85)'>Using: <span style='padding:6px 10px;border-radius:999px;background:rgba(255,255,255,0.02);'>{badge}</span></div>", unsafe_allow_html=True)

# Render history / welcome
if not st.session_state.history:
    st.markdown("<div style='padding:18px;border-radius:12px;background:rgba(255,255,255,0.02)'>Welcome — ask about BMI, labs, or workouts. Use the sidebar to upload PDFs and change AI model.</div>", unsafe_allow_html=True)

for msg in st.session_state.history:
    if msg["role"] == "user":
        render_user_message(msg["content"])
    else:
        render_assistant_message(msg["content"])

st.markdown('<div style="height:160px"></div>', unsafe_allow_html=True)
st.markdown('</div></div>', unsafe_allow_html=True)

# render visual placeholder (keeps same look)
render_bottom_input_placeholder()

# ---------------- Bottom input UI (using a form with clear_on_submit=True) ----------------
# Prefill logic: prefer last_transcript only (consume it once)
prefill = st.session_state.get("last_transcript", "")

# DEBUG
print("DEBUG prefill computed -> prefill:", repr(prefill), " session bottom_input (BEFORE set):", repr(st.session_state.get("bottom_input")), " last_transcript:", repr(st.session_state.get("last_transcript")))

# Apply and clear last_transcript so it won't stick around
if prefill:
    try:
        if not st.session_state.get("bottom_input"):
            st.session_state["bottom_input"] = prefill
            print("DEBUG prefill applied -> bottom_input set to:", repr(prefill))
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

# Use a centered container and a form (Enter submits; clear_on_submit clears widget)
with st.container():
    cols = st.columns([0.02, 0.96, 0.02])
    with cols[1]:
        with st.form(key="chat_form", clear_on_submit=True):
            user_text = st.text_input(
                "Message",
                placeholder="Ask anything — e.g. 'Calculate my BMI 75 1.8' or 'Plan a 4-day workout split'...",
                key="bottom_input",
                label_visibility="collapsed"
            )
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            send_submitted = st.form_submit_button("Send")

# thinking animation helper remains as-is (_thinking_html)

def _append_history(role: str, content: str):
    st.session_state.history.append({"role": role, "content": content, "time": str(datetime.utcnow())})

# ---------------- Handle Send (form submission) ----------------
if send_submitted:
    # Read current input from session_state
    prompt = (st.session_state.get("bottom_input") or "").strip()
    print("DEBUG form submit -> prompt:", repr(prompt), " _last_sent_prompt:", repr(st.session_state.get("_last_sent_prompt")), " _processing_send:", st.session_state.get("_processing_send"))

    # Basic validation
    if not st.session_state.get("api_key", "").strip():
        st.warning("⚠️ Please enter your API key in the sidebar before sending a message.")
    elif not prompt:
        st.info("Please type a message before sending.")
    else:
        # Debounce / duplicate guard:
        # If this exact prompt was just sent (session key), ignore — prevents double-append across reruns.
        if st.session_state.get("_last_sent_prompt") == prompt:
            st.info("This message appears to be already sent. Edit and resend if needed.")
        elif st.session_state.get("_processing_send"):
            # Another send is in-flight (shouldn't normally occur) — ignore to be safe
            st.info("Processing previous message — please wait a moment.")
        else:
            # Mark processing
            st.session_state["_processing_send"] = True
            try:
                # show thinking placeholder
                ph = st.empty()
                ph.markdown(_thinking_html(), unsafe_allow_html=True)

                # append user message (only once)
                st.session_state.history.append({"role": "user", "content": prompt, "time": str(datetime.utcnow())})
                # record last sent prompt so immediate reruns won't double-append
                st.session_state["_last_sent_prompt"] = prompt
                print("DEBUG appended -> prompt used:", repr(prompt), " history_last_user:", repr(st.session_state.history[-1]))

                # quick red-flag safety
                red_flags = ["chest pain", "severe shortness of breath", "can't breathe", "loss of consciousness", "severe bleeding", "stroke"]
                if any(k in prompt.lower() for k in red_flags):
                    urgent = ("⚠️ **Red flag detected.** If you experience life-threatening symptoms, call emergency services immediately.")
                    st.session_state.history.append({"role": "assistant", "content": urgent, "time": str(datetime.utcnow())})
                    ph.empty()
                    # request rerun to show messages
                    st.session_state["_request_rerun"] = True
                else:
                    # Build RAG context
                    context = ""
                    try:
                        if _lazy_load_vector():
                            try:
                                hits = _vector_module.query(prompt, top_k=4)
                            except Exception:
                                hits = []
                            if hits:
                                pieces = []
                                for h in hits:
                                    src = h.get("metadata", {}).get("source", "unknown")
                                    txt = h.get("text", "")[:800]
                                    pieces.append(f"[{src}] {txt}")
                                context = "\n\n".join(pieces)
                        if not context:
                            context = st.session_state.get("health_context", "No reports uploaded.")[:3000]
                    except Exception:
                        context = st.session_state.get("health_context", "No reports uploaded.")[:3000]

                    try:
                        llm = load_model(st.session_state.get("model_provider", "OpenAI (GPT)"), st.session_state.get("api_key", ""))
                        if llm is None:
                            raise ValueError("LLM failed to load. Check API key / provider.")
                        tools = get_health_tools()
                        if _lazy_load_tavily():
                            try:
                                tav_tool = _tavily_module.get_tavily_tool()
                                tools.append(tav_tool)
                            except Exception:
                                pass

                        agent = create_health_agent(llm, tools)
                        chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.history[-8:]])
                        response = agent.invoke({"input": prompt, "context": context, "chat_history": chat_history})
                        output = response.get("output") if isinstance(response, dict) else str(response)
                        output = output or "No response."
                        st.session_state.history.append({"role": "assistant", "content": output, "time": str(datetime.utcnow())})
                    except Exception as e:
                        st.session_state.history.append({"role": "assistant", "content": f"Error: {e}", "time": str(datetime.utcnow())})
                        print("DEBUG agent error:", repr(e))
                    finally:
                        ph.empty()
                        st.session_state["_request_rerun"] = True
            finally:
                # Done processing
                st.session_state["_processing_send"] = False

# If send or sidebar set a rerun request, do one controlled rerun
if st.session_state.pop("_request_rerun", False):
    # clear last_transcript defensively; bottom_input should already be cleared by the form
    try:
        st.session_state["last_transcript"] = ""
    except Exception:
        pass
    # trigger the safe minimal rerun you already have (or the toggle)
    try:
        safe_rerun_minimal()
    except Exception:
        # fallback: toggle a session key to force UI update if safe_rerun_minimal fails
        st.session_state["_safe_rerun_toggle"] = not st.session_state.get("_safe_rerun_toggle", False)