import os
from langchain.llms import OpenAI  # works with OpenRouter API key

def get_openrouter_llm(model_name: str | None = None, temperature: float = 0.0):
    """
    Returns a LangChain LLM that points to OpenRouter using OPENROUTER_API_KEY.
    """
    router_key = os.getenv("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not router_key:
        raise ValueError("OPENROUTER_API_KEY not set in environment or Streamlit secrets!")

    os.environ["OPENAI_API_KEY"] = router_key
    os.environ["OPENAI_API_BASE"] = os.getenv("OPENROUTER_API_BASE") or "https://api.openrouter.ai/v1"

    model = model_name or os.getenv("OPENAI_MODEL") or "x-ai/grok-4.1-fast:free"

    llm = OpenAI(model_name=model, temperature=temperature)
    return llm
