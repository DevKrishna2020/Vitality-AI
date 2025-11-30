# modules/agent_engine.py
"""
Create a single Health Advisor agent wrapper that adapts to both:
 - a simple tools dict of callables (from get_health_tools())
 - or a list of LangChain Tool objects (if using LangChain Tool wrappers)

The returned object exposes:
    agent = create_health_agent(llm, tools)
    response = agent.invoke({"input": "...", "context": "...", "chat_history": "..."}))
    # response is a dict with at least {"output": "..."} (string)
"""
from typing import Any, Dict, Iterable, List, Optional
import json
import traceback

# NOTE: avoid importing langchain at module-import time — lazy-load it when needed
_HAS_LANGCHAIN = False
_langchain_objects = {}

def _lazy_load_langchain():
    """
    Attempt to import the parts of langchain we need.
    Sets _HAS_LANGCHAIN and caches the imported names in _langchain_objects.
    Returns True if langchain imports succeeded.
    """
    global _HAS_LANGCHAIN, _langchain_objects
    if _HAS_LANGCHAIN:
        return True
    try:
        # Import only inside this function to avoid import-time side effects (torch, etc.)
        from langchain.agents import initialize_agent, AgentType
        from langchain.tools import Tool as LC_Tool
        # import PromptTemplate from langchain_core if available; try both locations
        try:
            from langchain_core.prompts import PromptTemplate
        except Exception:
            # fallback to langchain.prompts if present
            from langchain import PromptTemplate  # type: ignore
        _langchain_objects = {
            "initialize_agent": initialize_agent,
            "AgentType": AgentType,
            "LC_Tool": LC_Tool,
            "PromptTemplate": PromptTemplate,
        }
        _HAS_LANGCHAIN = True
        return True
    except Exception:
        _HAS_LANGCHAIN = False
        _langchain_objects = {}
        return False

# ---------------------------
# Safety red-flag check
# ---------------------------
_RED_FLAG_TERMS = [
    "chest pain", "can't breathe", "cannot breathe", "severe shortness", "loss of consciousness",
    "severe bleeding", "stroke", "blue lips", "sudden weakness", "slurred speech", "not breathing"
]

def _contains_red_flag(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(term in t for term in _RED_FLAG_TERMS)

# ---------------------------
# Core wrapper class
# ---------------------------

class LangChainHealthAgent:
    def __init__(self, executor, prompt_template: Optional[Any] = None):
        """
        executor: result from initialize_agent(...) (AgentExecutor-like) OR a callable fallback.
        prompt_template: Optional PromptTemplate used to format the full prompt passed to the agent
        """
        self.executor = executor
        self.prompt_template = prompt_template

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        payload keys:
          - input: str (user message)
          - context: str (uploaded reports / extracted text)
          - chat_history: str (recent chat history)
          - uploaded_files: Optional[List[file-like]]
        Returns: dict {"output": str, ...}
        """
        try:
            user_input = payload.get("input", "")
            context = payload.get("context", "")
            chat_history = payload.get("chat_history", "")

            # Quick red-flag safety check (do not call models for emergencies)
            if _contains_red_flag(user_input) or _contains_red_flag(context):
                urgent_msg = (
                    "⚠️ **Red flag detected.** If you are experiencing life-threatening symptoms "
                    "— chest pain, severe difficulty breathing, sudden weakness, loss of consciousness, "
                    "severe bleeding, or signs of stroke — call emergency services or go to the nearest emergency department immediately."
                )
                return {"output": urgent_msg, "meta": {"red_flag": True}}

            # Format prompt using template if available
            if self.prompt_template:
                try:
                    # PromptTemplate may have .format or .format_prompt depending on implementation;
                    # try a few common shapes.
                    if hasattr(self.prompt_template, "format"):
                        prompt_text = self.prompt_template.format(
                            input=user_input,
                            context=context,
                            chat_history=chat_history
                        )
                    elif hasattr(self.prompt_template, "format_prompt"):
                        # some langchain versions use format_prompt -> .to_string() or similar
                        pf = self.prompt_template.format_prompt(
                            input=user_input,
                            context=context,
                            chat_history=chat_history
                        )
                        prompt_text = str(pf)
                    else:
                        raise AttributeError("Unsupported PromptTemplate shape")
                except Exception:
                    # fallback to a simple combined prompt
                    prompt_text = (
                        f"Context:\n{context}\n\nChat history:\n{chat_history}\n\nUser Input:\n{user_input}"
                    )
            else:
                prompt_text = (
                    f"Context:\n{context}\n\nChat history:\n{chat_history}\n\nUser Input:\n{user_input}"
                )

            # If executor is a callable (fallback) — call directly
            if callable(self.executor) and not _lazy_load_langchain():
                try:
                    resp = self.executor(prompt_text)
                    # Normalize: if dict-like, return text key
                    if isinstance(resp, dict):
                        text = resp.get("text") or resp.get("output") or str(resp)
                    else:
                        text = str(resp)
                    return {"output": text}
                except Exception as e:
                    return {"output": f"Error in fallback executor: {e}", "error": str(e)}

            # Otherwise expect an AgentExecutor-like object from LangChain with .run(...) or .invoke(...)
            # Try run() first, then invoke(), then __call__
            try:
                if hasattr(self.executor, "run"):
                    result = self.executor.run(prompt_text)
                    return {"output": result}
                elif hasattr(self.executor, "invoke"):
                    result = self.executor.invoke({"input": prompt_text})
                    if isinstance(result, dict):
                        return {"output": result.get("output") or result.get("text") or str(result)}
                    return {"output": str(result)}
                elif callable(self.executor):
                    result = self.executor(prompt_text)
                    if isinstance(result, dict):
                        return {"output": result.get("text") or result.get("output") or str(result)}
                    return {"output": str(result)}
                else:
                    return {"output": "Agent executor does not provide run/invoke/call interface."}
            except Exception as e:
                # If the agent runtime fails (e.g. langchain internals), return a helpful error
                traceback_str = traceback.format_exc()
                return {"output": f"Agent runtime error: {e}", "error": str(e), "trace": traceback_str}

        except Exception as e:
            traceback_str = traceback.format_exc()
            return {"output": f"Agent error: {e}", "error": str(e), "trace": traceback_str}

# ---------------------------
# Factory function
# ---------------------------

def create_health_agent(llm, tools) -> LangChainHealthAgent:
    """
    Build and return a LangChainHealthAgent wrapper around a LangChain AgentExecutor.

    llm: an LLM object compatible with LangChain (or a simple callable fallback)
    tools: either
      - a dict mapping name -> callable (simple tools), or
      - an iterable/list of LangChain Tool objects

    The returned object's .invoke(payload) returns dict {"output": str}
    """

    # Build the prompt template (conservative/system instructions)

    template = """

You are Vitality AI — a professional, safety-first Health & Fitness Assistant.
Your role is to provide clear, evidence-aware medical education, guideline-informed triage, risk estimation,
interpretation of uploaded reports, and practical next steps — while never diagnosing, prescribing, or replacing a licensed clinician.

You MUST follow these rules:
1. Do not provide definitive diagnoses. Always present possible causes or differential considerations and explain uncertainty.
2. Do not prescribe medications, dosages, or perform treatment planning. If treatment options are discussed, present them as general standard-of-care approaches and instruct the user to consult a clinician for prescriptions and individualized dosing.
3. Perform triage, not diagnosis. Classify urgency as one of: Emergency / Urgent / See primary care / Self-care at home and explain the reasoning.
4. Always include red-flag checks. If red-flag symptoms are present, instruct immediate emergency care and do not attempt to answer further clinical details.
5. When factual claims are made, prefer evidence. If an external evidence tool (e.g., Tavily) is available and relevant, use it and cite brief source labels/links in the EVIDENCE section.

If the user asks for up-to-date information, guidelines, or explicitly says
"search the web", "use Tavily", "find recent", "latest", or "2024/2025", you
SHOULD call the "tavily_search" tool with a short query that captures their request
before answering.

When the user explicitly asks you to "search the web", "give sources", "cite", "find studies", "find guidelines", or "show sources",
CALL the tool named "tavily_search" with the user's search query (a short plain-text query). The tavily_search tool returns summarized results (titles, short snippets, and URLs).
If the tool is used, always include an **EVIDENCE** section listing the top Tavily results (title + short snippet + URL) and label the tool used (e.g., "Evidence: [Tavily] — <title> (url)").

Guidance for using Tavily:
- Only query Tavily for requests that explicitly ask for sources, citations, recent guidelines, drug labels, or when you (the model) are unsure / need up-to-date verification.
- Do NOT fabricate citations. If Tavily returns no results, state "No external sources found." and proceed with conservative guidance.
- Use at most 1–3 Tavily results to support a claim; synthesize them concisely in the EVIDENCE section.

6. Be explicit about uncertainty. Use phrases like "may", "could", "possible", and provide relative likelihood where helpful.
7. Give actionable, low-risk next steps (monitoring, symptom relief, when to seek care). Avoid anything that looks like a prescription.
8. Always end with a concise disclaimer that you are not a substitute for professional medical care; for emergencies, call local emergency services.

Input variables available:
- {context} : extracted text / uploaded reports
- {chat_history} : recent conversation history
- {input} : the user's current message

Required response format (plain text, short sections):
SUMMARY: 1-2 sentence lay summary of what you understand from the user.

ASSESSMENT: short list of possible causes or considerations (use conditional/likelihood language).

TRIAGE: one-line categorical recommendation (Emergency / Urgent / See primary care / Self-care at home) and brief rationale.

NEXT STEPS: 3–6 concrete, safe actions the user can take now (monitoring, symptom relief, seek care, prepare questions for clinician). Do NOT prescribe or give doses.

EVIDENCE: If external tools (e.g., tavily_search) were used, list short citations or source labels (e.g., "Tavily: FDA label — DrugX, 2024"). If none used, write "No external sources used."

DISCLAIMER: One clear sentence: you are not a substitute for professional medical care. For emergencies, call local emergency services.

Tone:
- Empathetic, calm, non-alarming, plain language.
- Conservative phrasing ("may", "could", "possible"), explicit uncertainty.
- Keep responses concise (aim < 350 words). Prioritize safety and clarity.

"""

    # Attempt to lazy-load langchain (imports may fail if langchain or its deps are not installed)
    has_langchain = _lazy_load_langchain()

    PromptTemplate = _langchain_objects.get("PromptTemplate") if has_langchain else None

    prompt_obj = None
    if has_langchain and PromptTemplate is not None:
        try:
            # Try common factory method names
            if hasattr(PromptTemplate, "from_template"):
                prompt_obj = PromptTemplate.from_template(template)
            else:
                # some older/newer versions accept the template directly
                prompt_obj = PromptTemplate(template)
        except Exception:
            prompt_obj = None

    # Normalize tools into LangChain Tool objects if needed
    lc_tools: List[Any] = []
    if has_langchain:
        LC_Tool = _langchain_objects.get("LC_Tool")
        try:
            # If tools is an iterable of LangChain Tool objects, accept it as-is
            if isinstance(tools, Iterable):
                sample = None
                try:
                    sample = next(iter(tools))
                except Exception:
                    sample = None

                if sample is not None and hasattr(sample, "name") and hasattr(sample, "func"):
                    lc_tools = list(tools)
                else:
                    # dict-like mapping
                    if isinstance(tools, dict):
                        for name, fn in tools.items():
                            try:
                                lc_tools.append(LC_Tool(name=name, func=fn, description=f"Tool: {name}"))
                            except Exception:
                                # fallback thin wrapper
                                def _wrap(f):
                                    return lambda x: f(x) if isinstance(x, str) else f(*x)
                                lc_tools.append(LC_Tool(name=name, func=_wrap(fn), description=f"Tool: {name}"))
                    else:
                        # list of callables without names -> auto-name them
                        for i, fn in enumerate(tools):
                            nm = getattr(fn, "__name__", f"tool_{i}")
                            try:
                                lc_tools.append(LC_Tool(name=nm, func=fn, description=f"Tool: {nm}"))
                            except Exception:
                                lc_tools.append(LC_Tool(name=nm, func=lambda x, f=fn: f(x), description=f"Tool: {nm}"))
            elif isinstance(tools, dict):
                for name, fn in tools.items():
                    try:
                        lc_tools.append(LC_Tool(name=name, func=fn, description=f"Tool: {name}"))
                    except Exception:
                        lc_tools.append(LC_Tool(name=name, func=lambda x, f=fn: f(x), description=f"Tool: {name}"))
        except Exception:
            # Something about converting tools failed — fall back to None so we use simple fallback
            lc_tools = None
    else:
        lc_tools = None

    # If LangChain available and we have converted tools, try initialize_agent
    if has_langchain and lc_tools is not None:
        try:
            initialize_agent = _langchain_objects.get("initialize_agent")
            AgentType = _langchain_objects.get("AgentType")
            agent_executor = initialize_agent(
                tools=lc_tools,
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # switched to Zero-Shot ReAct-style agent
                verbose=False,
                handle_parsing_errors=True,
            )
            return LangChainHealthAgent(agent_executor, prompt_template=prompt_obj)
        except Exception:
            # initialization failed — fall back to simple executor below
            pass

    # Fallback executor (no langchain or initialization failed)
    def fallback_executor(prompt_text: str):
        try:
            if hasattr(llm, "call"):
                out = llm.call({"input": prompt_text})
                if isinstance(out, dict):
                    return out.get("text") or out.get("output") or str(out)
                return str(out)
            elif callable(llm):
                r = llm(prompt_text)
                if isinstance(r, dict):
                    return r.get("text") or r.get("output") or str(r)
                return str(r)
            else:
                return "No LLM callable available."
        except Exception as e:
            return f"LLM fallback error: {e}"

    return LangChainHealthAgent(fallback_executor, prompt_template=prompt_obj)