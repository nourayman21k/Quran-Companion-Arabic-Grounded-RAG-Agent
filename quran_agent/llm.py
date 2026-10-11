"""LLMs with automatic fallback.

Groq's free tier limits every model separately (8k tokens/min, 200k tokens/day each). So:
- the many small calls (enhancer, researchers, graders) run on FAST_LLM_MODEL,
- the few important ones (synthesizer, reflector) run on LLM_MODEL,
- and when a model hits a rate limit (or Groq rejects a call), the next model in FALLBACK_LLM_MODELS
  takes over -- ending with a local Ollama model if one is running.
"""
from dataclasses import dataclass
from functools import lru_cache

from . import config


def _fallback_errors() -> tuple:
    import groq
    return (groq.APIError,)   # rate limits (429), request too large (413), failed tool calls (400), network


def groq_chat(model: str):
    from langchain_groq import ChatGroq
    kwargs = {"reasoning_effort": config.REASONING_EFFORT} if model.startswith("openai/gpt-oss") else {}
    return ChatGroq(model=model, max_retries=config.LLM_MAX_RETRIES, temperature=0, **kwargs)


def ollama_chat(model: str):
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model, base_url=config.OLLAMA_BASE_URL, temperature=0, reasoning=False, num_ctx=8192)


@lru_cache(maxsize=1)
def ollama_available() -> bool:
    """True if Ollama is running and has OLLAMA_MODEL pulled (checked once per process)."""
    if not config.OLLAMA_MODEL:
        return False
    try:
        import requests
        import langchain_ollama  # noqa: F401
        tags = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=1.5).json()
        return any(m["name"].split(":latest")[0] == config.OLLAMA_MODEL.split(":latest")[0]
                   for m in tags.get("models", []))
    except Exception:
        return False


class ModelChain:
    """A primary model plus fallbacks, usable as plain chat, structured output, or with tools."""

    def __init__(self, models: list, names: list[str]):
        self.models, self.names = models, names

    def _chain(self, make):
        runnables = [make(m) for m in self.models]
        chain = runnables[0]
        if len(runnables) > 1:
            chain = chain.with_fallbacks(runnables[1:], exceptions_to_handle=_fallback_errors())
        return chain.with_retry(**config.RETRY)

    def plain(self):
        return self._chain(lambda m: m)

    def structured(self, schema):
        # native JSON-schema output is more reliable on Groq than the default forced tool call
        return self._chain(lambda m: m.with_structured_output(schema, method="json_schema"))

    def with_tools(self, tools: list):
        return self._chain(lambda m: m.bind_tools(tools))


@dataclass
class Models:
    smart: ModelChain   # synthesizer, reflector
    fast: ModelChain    # enhancer, researchers, graders


def _chain_for(primary: str) -> ModelChain:
    if config.LLM_PROVIDER == "ollama":
        return ModelChain([ollama_chat(config.OLLAMA_MODEL)], [config.OLLAMA_MODEL])
    names = [primary] + [m for m in config.FALLBACK_LLM_MODELS if m != primary]
    models = [groq_chat(n) for n in names]
    if ollama_available():
        names.append(f"{config.OLLAMA_MODEL} (local)")
        models.append(ollama_chat(config.OLLAMA_MODEL))
    return ModelChain(models, names)


def build_models() -> Models:
    return Models(smart=_chain_for(config.LLM_MODEL), fast=_chain_for(config.FAST_LLM_MODEL))
