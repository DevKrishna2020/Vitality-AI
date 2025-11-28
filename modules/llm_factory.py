# modules/llm_factory.py

from typing import Optional, Dict, Generator, Any
from types import SimpleNamespace

# Dummy fallback
class DummyLLM:
    def __init__(self, provider: str = "Dummy"):
        self.provider = provider

    def call(self, payload: Any) -> Dict[str, str]:
        prompt = payload.get("input") if isinstance(payload, dict) else str(payload)
        return {"text": f"[{self.provider} stub] No API key configured or provider not supported. Prompt received: {prompt[:400]}"}

    def stream(self, payload: Any) -> Generator[str, None, None]:
        yield self.call(payload)["text"]


def _wrap_llm_with_interface(llm_obj, provider_name: str):
    """
    Return an adapter that exposes .call(payload) and .stream(payload) for diverse SDK objects.
    If llm_obj already has these methods, return it directly.
    """

    # If object already provides call/stream, return as-is
    if hasattr(llm_obj, "call") and hasattr(llm_obj, "stream"):
        return llm_obj

    # Import lightly for message-shaped calls when available (optional)
    try:
        from langchain.schema import HumanMessage, SystemMessage  # type: ignore
    except Exception:
        HumanMessage = None
        SystemMessage = None

    class Adapter:
        def __init__(self, obj):
            self._obj = obj
            self.provider = provider_name

        def _extract_text_from_result(self, res):
            """
            Robustly extract text from a variety of result shapes:
            - dict with choices / choices[0].message.content
            - LangChain LLMResult-like with .generations
            - strings
            - objects with .text or .content
            """
            try:
                if res is None:
                    return ""
                # If it's already a string
                if isinstance(res, str):
                    return res

                # Common OpenAI-style dict
                if isinstance(res, dict):
                    # ChatCompletion shape
                    if "choices" in res and len(res["choices"]) > 0:
                        c = res["choices"][0]
                        if isinstance(c, dict):
                            if "message" in c and isinstance(c["message"], dict):
                                return c["message"].get("content", "") or str(res)
                            return c.get("text", "") or str(res)
                    # new shape (maybe openai.chat)
                    if "output_text" in res:
                        return res["output_text"]

                # LangChain LLMResult-like (generations)
                if hasattr(res, "generations"):
                    gens = getattr(res, "generations")
                    # sometimes it's a list of lists
                    try:
                        first = gens[0]
                        if isinstance(first, (list, tuple)):
                            cand = first[0]
                        else:
                            cand = first
                        # candidate may have .text
                        if hasattr(cand, "text"):
                            return cand.text
                        if isinstance(cand, dict) and "text" in cand:
                            return cand["text"]
                    except Exception:
                        pass

                # If it's an object with .text or .content
                for attr in ("text", "content"):
                    if hasattr(res, attr):
                        val = getattr(res, attr)
                        if isinstance(val, str):
                            return val

                # otherwise fall back to str()
                return str(res)
            except Exception:
                return str(res)

        def call(self, payload):
            # Accept either a string or a dict payload.
            prompt = payload.get("input") if isinstance(payload, dict) else str(payload)

            # If the caller provided chat history (string), try to pass it as messages
            chat_history = None
            if isinstance(payload, dict):
                chat_history = payload.get("chat_history") or payload.get("history") or payload.get("chat") or None

            # Build a messages list if we have history (system + user)
            messages = None
            if chat_history:
                messages = [
                    {"role": "system", "content": str(chat_history)},
                    {"role": "user", "content": str(prompt)}
                ]

            # 0) quick helpers
            def try_and_extract(fn, *args, **kwargs):
                try:
                    res = fn(*args, **kwargs)
                    return {"text": self._extract_text_from_result(res)}
                except Exception:
                    return None

            # 1) Try direct invoke() (sync) - common in some LangChain wrappers
            if hasattr(self._obj, "invoke"):
                # If messages available, prefer invoking with messages
                if messages is not None:
                    try:
                        res = self._obj.invoke({"messages": messages})
                        return {"text": self._extract_text_from_result(res)}
                    except Exception:
                        try:
                            res = self._obj.invoke(messages=messages)
                            return {"text": self._extract_text_from_result(res)}
                        except Exception:
                            pass
                # fallback to passing raw prompt
                try:
                    res = self._obj.invoke(prompt)
                    return {"text": self._extract_text_from_result(res)}
                except Exception:
                    try:
                        res = self._obj.invoke({"input": prompt})
                        return {"text": self._extract_text_from_result(res)}
                    except Exception:
                        pass

            # 2) Try async forms (ainvoke / agenerate / astream)
            try:
                import asyncio
                # ainvoke
                if hasattr(self._obj, "ainvoke"):
                    try:
                        if messages is not None:
                            res = asyncio.run(self._obj.ainvoke({"messages": messages}))
                        else:
                            res = asyncio.run(self._obj.ainvoke(prompt))
                        return {"text": self._extract_text_from_result(res)}
                    except Exception:
                        try:
                            if messages is not None:
                                res = asyncio.run(self._obj.ainvoke(messages=messages))
                                return {"text": self._extract_text_from_result(res)}
                        except Exception:
                            pass
                # agenerate
                if hasattr(self._obj, "agenerate"):
                    try:
                        if messages is not None:
                            res = asyncio.run(self._obj.agenerate(messages=messages))
                        else:
                            res = asyncio.run(self._obj.agenerate(prompt))
                        return {"text": self._extract_text_from_result(res)}
                    except Exception:
                        try:
                            res = asyncio.run(self._obj.agenerate(messages=[{"role":"user","content":prompt}]))
                            return {"text": self._extract_text_from_result(res)}
                        except Exception:
                            pass
            except Exception:
                pass

            # 3) generate() (sync)
            if hasattr(self._obj, "generate"):
                try:
                    if messages is not None:
                        out = self._obj.generate(messages=messages)
                    else:
                        out = self._obj.generate(prompt)
                    return {"text": self._extract_text_from_result(out)}
                except Exception:
                    try:
                        out = self._obj.generate(messages=[{"role": "user", "content": prompt}])
                        return {"text": self._extract_text_from_result(out)}
                    except Exception:
                        pass

            # 4) __call__ style
            if hasattr(self._obj, "__call__"):
                try:
                    if messages is not None:
                        out = self._obj(messages)
                    else:
                        out = self._obj(prompt)
                    return {"text": self._extract_text_from_result(out)}
                except Exception:
                    try:
                        out = self._obj({"messages":[{"role":"user","content":prompt}]})
                        return {"text": self._extract_text_from_result(out)}
                    except Exception:
                        pass

            # 5) predictable single-call methods
            for method in ("predict", "send", "complete", "predict_text"):
                if hasattr(self._obj, method):
                    try:
                        if messages is not None:
                            res = getattr(self._obj, method)(messages)
                        else:
                            res = getattr(self._obj, method)(prompt)
                        return {"text": self._extract_text_from_result(res)}
                    except Exception:
                        pass

            # 6) Chat-like nested API
            try:
                if hasattr(self._obj, "chat") and hasattr(self._obj.chat, "completions") and hasattr(self._obj.chat.completions, "create"):
                    if messages is not None:
                        resp = self._obj.chat.completions.create(messages=messages)
                    else:
                        resp = self._obj.chat.completions.create(messages=[{"role":"user","content":prompt}])
                    return {"text": self._extract_text_from_result(resp)}
            except Exception:
                pass

            # 7) Last resort: debug info
            try:
                attrs = ", ".join(sorted([a for a in dir(self._obj) if not a.startswith("_")])[:80])
            except Exception:
                attrs = "unavailable"
            return {"text": f"[{self.provider} adapter] Could not call provider directly. Prompt: {prompt[:400]}\n\nAvailable attrs: {attrs}"}

        def stream(self, payload):
            # Try streaming entrypoints if they exist: astream, astream_events, stream, astream
            prompt = payload.get("input") if isinstance(payload, dict) else str(payload)

            # 1) If object has astream (async streaming), run and yield chunks
            try:
                import asyncio
                if hasattr(self._obj, "astream"):
                    try:
                        async def _run_and_yield():
                            async for chunk in self._obj.astream(prompt):
                                yield chunk
                        # run coroutine generator with asyncio and yield results
                        # note: we cannot yield from async generator directly here; fallback to single call
                        chunks = asyncio.run(self._obj.astream(prompt))
                        # if we got an iterable, yield each element
                        try:
                            for c in chunks:
                                yield self._extract_text_from_result(c)
                            return
                        except Exception:
                            pass
                    except Exception:
                        pass
            except Exception:
                pass

            # 2) If object has astream_events (some SDKs)
            if hasattr(self._obj, "astream_events"):
                try:
                    import asyncio
                    events = asyncio.run(self._obj.astream_events(prompt))
                    for e in events:
                        yield self._extract_text_from_result(e)
                    return
                except Exception:
                    pass

            # 3) If object has 'stream' or 'send_stream', try those
            for method in ("stream", "send_stream", "stream_text"):
                if hasattr(self._obj, method):
                    try:
                        gen = getattr(self._obj, method)(prompt)
                        # if it's an iterator/generator, yield items
                        if hasattr(gen, "__iter__") or hasattr(gen, "__next__"):
                            for part in gen:
                                yield self._extract_text_from_result(part)
                            return
                        # else if single result, yield once
                        yield self._extract_text_from_result(gen)
                        return
                    except Exception:
                        pass

            # fallback: single call and yield result
            yield self.call(payload)["text"]

    return Adapter(llm_obj)


def load_model(provider: str, api_key: Optional[str], **opts):
    """
    Factory entry point.

    provider: case-insensitive string, e.g. "Google Gemini", "Groq", "OpenAI (GPT)"
    api_key: API key or credential token (or None)
    opts: optional overrides like model, temperature, project, location
    """
    if not provider:
        return DummyLLM("NoProvider")

    p = provider.lower().strip()

    # ---------- Google Gemini (langchain_google_genai) ----------
    if "google" in p or "gemini" in p:
        if not api_key:
            # No key provided -> can't instantiate
            return DummyLLM("GoogleGemini (no key)")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except Exception as e:
            # package not installed
            raise ImportError("langchain_google_genai package not found. Install it (pip install langchain-google-genai) or ensure your environment provides it.") from e

        # prefer an up-to-date Gemini model; older 1.x/1.5 models may be retired.
        preferred_models = [
            opts.get("model"),           # allow user override (None safe)
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
        ]
        # normalize & remove None, duplicates
        seen = set()
        models_to_try = []
        for m in preferred_models:
            if not m:
                continue
            name = str(m).strip()
            if name and name not in seen:
                seen.add(name)
                models_to_try.append(name)

        temperature = float(opts.get("temperature", 0.7))

        last_exc = None
        for model_name in models_to_try:
            try:
                llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=temperature)
                return _wrap_llm_with_interface(llm, "Google Gemini")
            except Exception as e:
                # remember last error and try next model name
                last_exc = e
                # continue to next candidate model
                continue

        # if none worked, raise the last exception with helpful guidance
        raise RuntimeError(
            "Failed to instantiate Google Gemini client. Tried models: "
            f"{', '.join(models_to_try)}. Last error: {last_exc}. "
            "Ensure your API key and billing are valid, the Generative API is enabled, "
            "and try a supported model id (check Google GenAI docs / ListModels)."
        ) from last_exc

    # ---------- Groq ----------
    if "groq" in p:
        if not api_key:
            return DummyLLM("Groq (no key)")
        try:
            from langchain_groq import ChatGroq
        except Exception as e:
            raise ImportError("langchain_groq package not found. Install it (pip install langchain-groq) or ensure your env has it.") from e

        model_name = opts.get("model", "llama3-8b-8192")
        temperature = float(opts.get("temperature", 0.7))
        try:
            llm = ChatGroq(model_name=model_name, groq_api_key=api_key, temperature=temperature)
            return _wrap_llm_with_interface(llm, "Groq")
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate Groq client: {e}") from e

    # ---------- OpenAI via openai (optional) ----------
    if "openai" in p or "gpt" in p:
        # allow user to supply None (means not configured)
        if not api_key:
            return DummyLLM("OpenAI (no key)")
        try:
            import openai
        except Exception as e:
            raise ImportError("openai package required for OpenAI provider. Install with: pip install openai") from e

        model = opts.get("model", "gpt-4o-mini")
        temperature = float(opts.get("temperature", 0.2))

        class OpenAIWrapper:
            def __init__(self, api_key, model, temperature):
                self.api_key = api_key
                self.model = model
                self.temperature = temperature
                openai.api_key = api_key
                self._openai = openai

            def call(self, payload):
                prompt = payload.get("input") if isinstance(payload, dict) else str(payload)
                # Ensure compatible call for varying SDK versions
                try:
                    resp = self._openai.ChatCompletion.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=1024,
                    )
                    # extract text robustly
                    text = ""
                    if resp and "choices" in resp and len(resp["choices"]) > 0:
                        choice = resp["choices"][0]
                        if "message" in choice:
                            text = choice["message"].get("content", "")
                        else:
                            text = choice.get("text", "")
                    return {"text": text}
                except Exception as e:
                    # Try new shape if available
                    try:
                        resp2 = self._openai.chat.completions.create(model=self.model, messages=[{"role":"user","content":prompt}], temperature=self.temperature)
                        text = resp2.choices[0].message.content
                        return {"text": text}
                    except Exception as e2:
                        raise RuntimeError(f"OpenAI call failed: {e} | {e2}")

            def stream(self, payload):
                prompt = payload.get("input") if isinstance(payload, dict) else str(payload)
                try:
                    # ChatCompletion with stream
                    for event in self._openai.ChatCompletion.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        stream=True,
                        temperature=self.temperature,
                    ):
                        # event shapes vary; attempt to extract incremental content
                        try:
                            choices = event.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})
                            content = delta.get("content") or choices[0].get("text")
                            if content:
                                yield content
                        except Exception:
                            yield str(event)
                except Exception:
                    # fallback to single call
                    yield self.call(payload)["text"]

        return OpenAIWrapper(api_key, model, temperature)

    # ---------- Fallback ----------
    return DummyLLM(provider)