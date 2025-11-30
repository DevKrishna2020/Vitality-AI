from typing import Optional, Dict, Generator, Any


# Dummy fallback
class DummyLLM:
    def __init__(self, provider: str = "Dummy"):
        self.provider = provider

    def call(self, payload: Any) -> Dict[str, str]:
        prompt = payload.get("input") if isinstance(payload, dict) else str(payload)
        return {
            "text": f"[{self.provider} stub] No API key configured or provider not supported. "
                    f"Prompt received: {prompt[:400]}"
        }

    def stream(self, payload: Any) -> Generator[str, None, None]:
        yield self.call(payload)["text"]


def _wrap_llm_with_interface(llm_obj, provider_name: str):
    # If object already provides call/stream, return as-is
    if hasattr(llm_obj, "call") and hasattr(llm_obj, "stream"):
        return llm_obj

    class Adapter:
        def __init__(self, obj):
            self._obj = obj
            self.provider = provider_name

        def _extract_text_from_result(self, res):
            try:
                if res is None:
                    return ""
                if isinstance(res, str):
                    return res

                if isinstance(res, dict):
                    if "choices" in res and res["choices"]:
                        c = res["choices"][0]
                        if isinstance(c, dict):
                            if "message" in c and isinstance(c["message"], dict):
                                return c["message"].get("content", "") or str(res)
                            return c.get("text", "") or str(res)
                    if "output_text" in res:
                        return res["output_text"]

                if hasattr(res, "generations"):
                    gens = getattr(res, "generations")
                    try:
                        first = gens[0]
                        cand = first[0] if isinstance(first, (list, tuple)) else first
                        if hasattr(cand, "text"):
                            return cand.text
                        if isinstance(cand, dict) and "text" in cand:
                            return cand["text"]
                    except Exception:
                        pass

                for attr in ("text", "content"):
                    if hasattr(res, attr):
                        val = getattr(res, attr)
                        if isinstance(val, str):
                            return val

                return str(res)
            except Exception:
                return str(res)

        def call(self, payload):
            prompt = payload.get("input") if isinstance(payload, dict) else str(payload)
            chat_history = None
            if isinstance(payload, dict):
                chat_history = (
                    payload.get("chat_history")
                    or payload.get("history")
                    or payload.get("chat")
                    or None
                )

            messages = None
            if chat_history:
                messages = [
                    {"role": "system", "content": str(chat_history)},
                    {"role": "user", "content": str(prompt)},
                ]

            if hasattr(self._obj, "invoke"):
                try:
                    if messages is not None:
                        res = self._obj.invoke({"messages": messages})
                    else:
                        res = self._obj.invoke(prompt)
                    return {"text": self._extract_text_from_result(res)}
                except Exception:
                    try:
                        res = self._obj.invoke({"input": prompt})
                        return {"text": self._extract_text_from_result(res)}
                    except Exception:
                        pass

            if hasattr(self._obj, "generate"):
                try:
                    res = (
                        self._obj.generate(messages=messages)
                        if messages is not None
                        else self._obj.generate(prompt)
                    )
                    return {"text": self._extract_text_from_result(res)}
                except Exception:
                    pass

            if hasattr(self._obj, "__call__"):
                try:
                    res = self._obj(messages) if messages is not None else self._obj(prompt)
                    return {"text": self._extract_text_from_result(res)}
                except Exception:
                    pass

            for method in ("predict", "send", "complete", "predict_text"):
                if hasattr(self._obj, method):
                    try:
                        fn = getattr(self._obj, method)
                        res = fn(messages) if messages is not None else fn(prompt)
                        return {"text": self._extract_text_from_result(res)}
                    except Exception:
                        pass

            return {
                "text": f"[{self.provider} adapter] Provider error while handling the request. "
                        "Please check API key / model id and try again."
            }

        def stream(self, payload):
            prompt = payload.get("input") if isinstance(payload, dict) else str(payload)
            yield self.call({"input": prompt})["text"]

    return Adapter(llm_obj)


def load_model(provider: str, api_key: Optional[str], **opts):
    if not provider:
        return DummyLLM("NoProvider")

    p = provider.lower().strip()

    # ---------- Google Gemini ----------
   
    if "google" in p or "gemini" in p:
        if not api_key:
            return DummyLLM("GoogleGemini (no key)")

        # Only treat ImportError as "package missing"
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            return DummyLLM("GoogleGemini (missing langchain_google_genai)")

        preferred_models = [
            opts.get("model"),
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
        ]
        seen, models_to_try = set(), []
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
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key,
                    temperature=temperature,
                )
                return _wrap_llm_with_interface(llm, "Google Gemini")
            except Exception as e:
                last_exc = e
                continue

        # Fall back to a clear dummy message instead of the long torch trace
        return DummyLLM(
            f"GoogleGemini (failed to init; tried {', '.join(models_to_try)}; last error: {type(last_exc).__name__})"
        )
    # ---------- Groq ----------
    if "groq" in p:
        if not api_key:
            return DummyLLM("Groq (no key)")
        try:
            from langchain_groq import ChatGroq
        except Exception as e:
            raise ImportError(
                "langchain_groq package not found. Install it (pip install langchain-groq)."
            ) from e

        # Use new recommended default model id
        model_name = opts.get("model", "llama-3.1-8b-instant")
        temperature = float(opts.get("temperature", 0.7))
        try:
            llm = ChatGroq(
                model_name=model_name,
                groq_api_key=api_key,
                temperature=temperature,
            )
            return _wrap_llm_with_interface(llm, "Groq")
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate Groq client: {e}") from e

    # ---------- OpenAI ----------
    if "openai" in p or "gpt" in p:
        if not api_key:
            return DummyLLM("OpenAI (no key)")
        try:
            import openai
        except Exception as e:
            raise ImportError(
                "openai package required for OpenAI provider. Install with: pip install openai"
            ) from e

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
                try:
                    resp = self._openai.ChatCompletion.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=1024,
                    )
                    text = ""
                    if resp and "choices" in resp and resp["choices"]:
                        choice = resp["choices"][0]
                        if "message" in choice:
                            text = choice["message"].get("content", "")
                        else:
                            text = choice.get("text", "")
                    return {"text": text}
                except Exception as e:
                    try:
                        resp2 = self._openai.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=self.temperature,
                        )
                        text = resp2.choices[0].message.content
                        return {"text": text}
                    except Exception as e2:
                        raise RuntimeError(f"OpenAI call failed: {e} | {e2}")

            def stream(self, payload):
                prompt = payload.get("input") if isinstance(payload, dict) else str(payload)
                try:
                    for event in self._openai.ChatCompletion.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        stream=True,
                        temperature=self.temperature,
                    ):
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
                    yield self.call(payload)["text"]

        return OpenAIWrapper(api_key, model, temperature)

    # ---------- Fallback ----------
    return DummyLLM(provider)