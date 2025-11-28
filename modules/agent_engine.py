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
You are Vitality AI — a cautious, safety-first Health & Fitness Advisor.
You provide educational information, triage guidance, lifestyle recommendations, and calculations.
You MUST NOT provide definitive diagnoses. Always include a short disclaimer and escalate to emergency care for red-flag symptoms.

Context from Reports:
{context}

Chat History:
{chat_history}

User Input: {input}

Provide:
1) A brief summary of findings or understanding.
2) Conservative triage recommendation (self-care / see primary care / urgent / emergency).
3) Clear next steps the user can take.
4) If relevant, credit any tools used to obtain factual information (e.g., BMI calculator).
Keep the response concise and safe.
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
                agent=AgentType.OPENAI_FUNCTIONS,  # safe default; can be changed if needed
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