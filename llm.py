import os
from langchain.chat_models import ChatOpenAI  # updated import for latest LangChain

def get_openrouter_llm(model_name: str | None = None, temperature: float = 0.0):
    """
    Returns a LangChain ChatOpenAI LLM configured to use OpenRouter API.
    """
    router_key = os.getenv("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not router_key:
        raise ValueError("OPENROUTER_API_KEY not set in environment or Streamlit secrets!")

    os.environ["OPENAI_API_KEY"] = router_key
    os.environ["OPENAI_API_BASE"] = os.getenv("OPENROUTER_API_BASE") or "https://api.openrouter.ai/v1"

    model = model_name or os.getenv("OPENAI_MODEL") or "x-ai/grok-4.1-fast:free"

    llm = ChatOpenAI(model_name=model, temperature=temperature)
    return llm
