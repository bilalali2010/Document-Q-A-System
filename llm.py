import os
from langchain.llms import OpenAI

def get_llm(model_name: str | None = None, temperature: float = 0.0):
    """
    Configure LangChain's OpenAI wrapper to use OpenRouter key.
    Expects user to set OPENROUTER_API_KEY in environment or Streamlit secrets.

    Behavior:
      - If OPENROUTER_API_KEY found, it's copied to OPENAI_API_KEY env var (used by openai libs).
      - Optionally uses OPENAI_API_BASE if provided in env (recommended: OpenRouter base).
      - model_name defaults to env OPENAI_MODEL or "gpt-4o-mini".
    """
    # Map secret name to what the OpenAI client expects
    router_key = os.getenv("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if router_key:
        os.environ["OPENAI_API_KEY"] = router_key

    # If user provided explicit base for OpenRouter, set OPENAI_API_BASE
    # (Streamlit secrets can include OPENAI_API_BASE or OPENROUTER_API_BASE)
    base = os.getenv("OPENAI_API_BASE") or os.getenv("OPENROUTER_API_BASE") or os.environ.get("OPENAI_API_BASE")
    if base:
        os.environ["OPENAI_API_BASE"] = base

    model = model_name or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    llm = OpenAI(model_name=model, temperature=temperature)
    return llm
